// src/nemotron_h/nh_dflash.cu
// CUDA implementation of the LiLiCorr DFlash speculative drafter for Nemotron-H / Bolt.
#include "nh_dflash.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::nemotron_h {

#define CUDA_CHECK(x)                                                                                     \
    do {                                                                                                  \
        cudaError_t err_ = (x);                                                                           \
        if (err_ != cudaSuccess) {                                                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            std::exit(1);                                                                                 \
        }                                                                                                 \
    } while (0)

#define CUBLAS_CHECK(x)                                                                                   \
    do {                                                                                                  \
        cublasStatus_t st_ = (x);                                                                         \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                               \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int) st_);          \
            std::exit(1);                                                                                 \
        }                                                                                                 \
    } while (0)

namespace {

// NVFP4 constants
constexpr int QK_NVFP4 = 64;
constexpr int QK_NVFP4_SUB = 16;

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

__global__ void dequantize_q8_0_embed_kernel(const uint8_t* __restrict__ table,
                                            const int32_t* __restrict__ tokens,
                                            int n_tokens,
                                            int n_embd,
                                            float* __restrict__ out) {
    int tok_idx = blockIdx.y;
    if (tok_idx >= n_tokens) return;
    int32_t token = tokens[tok_idx];

    int nb = n_embd / 32;
    size_t row_bytes = (size_t) nb * sizeof(block_q8_0);
    const block_q8_0* row = reinterpret_cast<const block_q8_0*>(table + (size_t) token * row_bytes);
    float* out_row = out + (size_t) tok_idx * n_embd;

    int b_idx = blockIdx.x;
    int lane = threadIdx.x;

    if (b_idx < nb && lane < 32) {
        float d = __half2float(row[b_idx].d);
        float val = d * (float) row[b_idx].qs[lane];
        out_row[b_idx * 32 + lane] = val;
    }
}

struct block_nvfp4 {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB]; // 4 scales (UE4M3)
    uint8_t qs[QK_NVFP4 / 2];           // 32 packed 4-bit nibbles
};

__device__ __constant__ float c_kvalues_fp4[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

__device__ __forceinline__ float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    float raw = (exp == 0) ? ldexpf((float) man, -9) : ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    return raw * 0.5f;
}

__global__ void dequantize_nvfp4_kernel(const block_nvfp4* __restrict__ vx, half* __restrict__ yy, int64_t ne, float tensor_scale) {
    int64_t i = blockIdx.x;
    int tid = threadIdx.x;
    int64_t base = i * QK_NVFP4;
    if (base >= ne) return;

    const block_nvfp4& xb = vx[i];
    int sub = tid / (QK_NVFP4_SUB / 2); // 0..3
    int j = tid % (QK_NVFP4_SUB / 2);   // 0..7

    float d = ue4m3_to_fp32(xb.d[sub]) * tensor_scale;
    uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    float v0 = d * c_kvalues_fp4[q & 0x0F];
    float v1 = d * c_kvalues_fp4[q >> 4];

    yy[y0] = __float2half(v0);
    yy[y1] = __float2half(v1);
}

/// y [T][n_out] = W x for an NVFP4 weight [n_out][n_in] kept packed (64-element blocks along each row, tensor
/// scale s) and T <= 8 FP16 rows x [T][n_in]: 8 warps a block, 4 output rows a warp, 8 lanes a row (one 64-block
/// each per 512-column chunk of x, staged through shared memory with each block padded off the others' banks)
__device__ __forceinline__ float ue4m3_to_fp32_fast(uint32_t x) {   // ue4m3_to_fp32 without ldexpf
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 0xF, mn = x & 0x7;
    if (e == 0) return (float) mn * (1.0f / 1024.0f);
    return __uint_as_float(((e + 119) << 23) | (mn << 20));   // (1 + mn/8) 2^(e-7) * 0.5
}

/// y [T][n_out] = W x for an NVFP4 weight [n_out][n_in] kept packed (64-element blocks along each row, tensor
/// scale s) and T <= 8 FP16 rows x [T][n_in]: 8 warps a block, 4 output rows a warp, 8 lanes a row (one 64-block
/// each per 512-column chunk of x, staged through shared memory, each block's 64 columns padded to 72 so the 8
/// lanes of a 16-byte load phase hit all 32 banks once).  FP32 math throughout (FP16 sums overflow on the drafter's
/// activations).
template <typename TOut, int LPR>
__global__ void __launch_bounds__(256) nvfp4_mm8_kernel(const block_nvfp4* __restrict__ W, float s,
                                                        const half* __restrict__ x, TOut* __restrict__ y, int n_in,
                                                        int n_out, int T) {
    constexpr int kChunk = LPR * QK_NVFP4, kPad = QK_NVFP4 + 8;
    __shared__ __align__(16) half xs[8][LPR * kPad];
    __shared__ float lut[16];   // the FP4 values (constant memory would serialize the lanes' different indices)
    if (threadIdx.x < 16) lut[threadIdx.x] = c_kvalues_fp4[threadIdx.x];
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (8 * 32 / LPR) + (threadIdx.x / 32) * (32 / LPR) + lane / LPR;
    const int sub = lane % LPR;
    const int nb_row = n_in / QK_NVFP4;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int c0 = 0; c0 < n_in; c0 += kChunk) {
        __syncthreads();
        for (int i = threadIdx.x; i < 8 * kChunk / 8; i += blockDim.x) {   // 16-byte pieces
            const int t = i / (kChunk / 8), j = (i % (kChunk / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < T && c0 + j < n_in) v = *reinterpret_cast<const uint4*>(x + (size_t) t * n_in + c0 + j);
            *reinterpret_cast<uint4*>(&xs[t][(j / QK_NVFP4) * kPad + j % QK_NVFP4]) = v;
        }
        __syncthreads();
        const int b = c0 / QK_NVFP4 + sub;
        if (row >= n_out || b >= nb_row) continue;
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + (size_t) row * nb_row + b);   // 36 bytes, 4-aligned
        uint32_t w[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(wp + i);
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = ue4m3_to_fp32_fast((w[0] >> (8 * sb)) & 0xFF) * s;
            // element e < 8: byte e's low nibble; e >= 8: byte e-8's high nibble.  Pair k = elements 2k, 2k+1.
            float2 wv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int e0 = 2 * k, e1 = 2 * k + 1;
                const uint32_t b0 = (w[1 + sb * 2 + (e0 % 8) / 4] >> (8 * ((e0 % 8) % 4))) & 0xFF;
                const uint32_t b1 = (w[1 + sb * 2 + (e1 % 8) / 4] >> (8 * ((e1 % 8) % 4))) & 0xFF;
                const uint32_t q0 = e0 < 8 ? (b0 & 0xF) : (b0 >> 4), q1 = e1 < 8 ? (b1 & 0xF) : (b1 >> 4);
                wv[k] = make_float2(lut[q0], lut[q1]);
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint4* xr = reinterpret_cast<const uint4*>(&xs[t][sub * kPad + sb * 16]);
                const uint4 xa = xr[0], xb = xr[1];
                const half2* ha = reinterpret_cast<const half2*>(&xa);
                const half2* hb = reinterpret_cast<const half2*>(&xb);
                float dot = 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 fa = __half22float2(ha[k]), fb = __half22float2(hb[k]);
                    dot = fmaf(wv[k].x, fa.x, fmaf(wv[k].y, fa.y, dot));
                    dot = fmaf(wv[k + 4].x, fb.x, fmaf(wv[k + 4].y, fb.y, dot));
                }
                acc[t] = fmaf(d, dot, acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 1);
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 2);
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 4);
        if (LPR == 16) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 8);
    }
    if (sub != 0 || row >= n_out) return;
#pragma unroll
    for (int t = 0; t < 8; ++t) {   // constant indices: acc stays in registers
        if (t >= T) break;
        if constexpr (std::is_same<TOut, half>::value) y[(size_t) t * n_out + row] = __float2half(acc[t]);
        else y[(size_t) t * n_out + row] = acc[t];
    }
}

template <typename TOut>
static void nvfp4_mm8(const block_nvfp4* W, float s, const half* x, TOut* y, int n_in, int n_out, int T,
                      cudaStream_t stream) {
    nvfp4_mm8_kernel<TOut, 16><<<(n_out + 15) / 16, 256, 0, stream>>>(W, s, x, y, n_in, n_out, T);
}

__global__ void dequantize_nvfp4_f32_kernel(const block_nvfp4* __restrict__ vx, float* __restrict__ yy, int64_t ne, float tensor_scale) {
    int64_t i = blockIdx.x;
    int tid = threadIdx.x;
    int64_t base = i * QK_NVFP4;
    if (base >= ne) return;

    const block_nvfp4& xb = vx[i];
    int sub = tid / (QK_NVFP4_SUB / 2);
    int j = tid % (QK_NVFP4_SUB / 2);

    float d = ue4m3_to_fp32(xb.d[sub]) * tensor_scale;
    uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    float v0 = d * c_kvalues_fp4[q & 0x0F];
    float v1 = d * c_kvalues_fp4[q >> 4];

    yy[y0] = v0;
    yy[y1] = v1;
}

__global__ void convert_bf16_to_half_kernel(const uint16_t* __restrict__ src, half* __restrict__ dst, int64_t ne) {
    int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        uint32_t u = ((uint32_t) src[idx]) << 16;
        float f = __uint_as_float(u);
        dst[idx] = __float2half(f);
    }
}

__global__ void convert_f32_to_half_kernel(const float* __restrict__ src, half* __restrict__ dst, int64_t ne) {
    int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void convert_half_to_f32_kernel(const half* __restrict__ src, float* __restrict__ dst, int64_t ne) {
    int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        dst[idx] = __half2float(src[idx]);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void rms_norm_kernel(const half* __restrict__ x, const float* __restrict__ w, half* __restrict__ y, int dim, int ncols, float eps) {
    int t = blockIdx.x;
    if (t >= ncols) return;
    const half* xt = x + t * dim;
    half* yt = y + t * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(xt[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        s_scale = rsqrtf(sum_sq / (float) dim + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(xt[i]) * scale * w[i];
        yt[i] = __float2half(v);
    }
}

__global__ void rms_norm_f32_to_half_kernel(const float* __restrict__ x, const float* __restrict__ w, half* __restrict__ y, int dim, int ncols, float eps) {
    int t = blockIdx.x;
    if (t >= ncols) return;
    const float* xt = x + t * dim;
    half* yt = y + t * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xt[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        s_scale = rsqrtf(sum_sq / (float) dim + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xt[i] * scale * w[i];
        yt[i] = __float2half(v);
    }
}

__global__ void rms_norm_f32_to_f32_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int dim, int ncols, float eps) {
    int t = blockIdx.x;
    if (t >= ncols) return;
    const float* xt = x + t * dim;
    float* yt = y + t * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xt[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        s_scale = rsqrtf(sum_sq / (float) dim + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xt[i] * scale * w[i];
        yt[i] = v;
    }
}

__global__ void rms_norm_per_head_kernel(half* __restrict__ x, const float* __restrict__ w, int head_dim, int n_heads, int ncols, float eps) {
    // 1 warp per (head, token)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int total_heads = n_heads * ncols;
    if (wid >= total_heads) return;

    int t = wid / n_heads;
    int h = wid % n_heads;
    half* xh = x + (t * n_heads + h) * head_dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
        float v = __half2float(xh[i]);
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float scale = rsqrtf(sum_sq / (float) head_dim + eps);
    scale = __shfl_sync(0xFFFFFFFF, scale, 0);

    for (int i = lane; i < head_dim; i += 32) {
        float v = __half2float(xh[i]) * scale * w[i];
        xh[i] = __float2half(v);
    }
}

__global__ void neox_rope_kernel(half* __restrict__ q, half* __restrict__ k, int n_head_q, int n_head_kv, int head_dim, int pos0, int ncols, float freq_base) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int half_dim = head_dim / 2; // 64
    int i = tid % half_dim;
    int wid = tid / half_dim;
    int total_heads = (n_head_q + n_head_kv) * ncols;
    if (wid >= total_heads) return;

    int t = wid / (n_head_q + n_head_kv);
    int h_idx = wid % (n_head_q + n_head_kv);
    int pos = pos0 + t;

    float theta = (float) pos * powf(freq_base, -2.0f * (float) i / (float) head_dim);
    float cos_th = cosf(theta);
    float sin_th = sinf(theta);

    if (h_idx < n_head_q) {
        int h = h_idx;
        half* qh = q + (t * n_head_q + h) * head_dim;
        float x0 = __half2float(qh[i]);
        float x1 = __half2float(qh[i + half_dim]);
        qh[i]            = __float2half(x0 * cos_th - x1 * sin_th);
        qh[i + half_dim] = __float2half(x0 * sin_th + x1 * cos_th);
    } else {
        int h = h_idx - n_head_q;
        half* kh = k + (t * n_head_kv + h) * head_dim;
        float x0 = __half2float(kh[i]);
        float x1 = __half2float(kh[i + half_dim]);
        kh[i]            = __float2half(x0 * cos_th - x1 * sin_th);
        kh[i + half_dim] = __float2half(x0 * sin_th + x1 * cos_th);
    }
}

__global__ void dflash_conv_kernel(const half* __restrict__ x, const half* __restrict__ dynamic, const float* __restrict__ base,
                                        half* __restrict__ y, int hidden_size, int n_groups, int ncols, int side) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size * ncols) return;
    int c = idx % hidden_size;
    int t = idx / hidden_size;

    int g = c / (hidden_size / n_groups); // c / 16
    float dyn0 = __half2float(dynamic[(g + 0 * n_groups + side * (2 * n_groups)) + t * (4 * n_groups)]);
    float dyn1 = __half2float(dynamic[(g + 1 * n_groups + side * (2 * n_groups)) + t * (4 * n_groups)]);

    float base0 = base[c + 0 * hidden_size + side * (2 * hidden_size)];
    float base1 = base[c + 1 * hidden_size + side * (2 * hidden_size)];

    float w0 = dyn0 + base0;
    float w1 = dyn1 + base1;

    float val0 = __half2float(x[c + t * hidden_size]);
    float val1 = (t > 0) ? __half2float(x[c + (t - 1) * hidden_size]) : 0.0f;

    y[idx] = __float2half(w0 * val0 + w1 * val1);
}

__global__ void dflash_conv_f32_to_f32_kernel(const float* __restrict__ x, const half* __restrict__ dynamic, const float* __restrict__ base,
                                              float* __restrict__ y, int hidden_size, int n_groups, int ncols, int side) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size * ncols) return;
    int c = idx % hidden_size;
    int t = idx / hidden_size;

    int g = c / (hidden_size / n_groups); // c / 16
    float dyn0 = __half2float(dynamic[(g + 0 * n_groups + side * (2 * n_groups)) + t * (4 * n_groups)]);
    float dyn1 = __half2float(dynamic[(g + 1 * n_groups + side * (2 * n_groups)) + t * (4 * n_groups)]);

    float base0 = base[c + 0 * hidden_size + side * (2 * hidden_size)];
    float base1 = base[c + 1 * hidden_size + side * (2 * hidden_size)];

    float w0 = dyn0 + base0;
    float w1 = dyn1 + base1;

    float val0 = x[c + t * hidden_size];
    float val1 = (t > 0) ? x[c + (t - 1) * hidden_size] : 0.0f;

    y[idx] = w0 * val0 + w1 * val1;
}

__global__ void add_residual_f32_kernel(float* __restrict__ a, const float* __restrict__ b, int64_t ne) {
    int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        a[idx] += b[idx];
    }
}

__global__ void swiglu_kernel(const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ out, int64_t ne) {
    int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu * u);
    }
}

__global__ void draft_argmax_kernel(const float* __restrict__ logits, int n_vocab, int n_cols, int32_t* __restrict__ out_tokens) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    const float* col_logits = logits + (size_t) col * n_vocab;
    int tid = threadIdx.x;

    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float v = col_logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    __shared__ float s_warp_max[32];
    __shared__ int s_warp_idx[32];
    int lane = tid % 32;
    int warp = tid / 32;
    if (lane == 0) {
        s_warp_max[warp] = max_val;
        s_warp_idx[warp] = max_idx;
    }
    __syncthreads();

    if (tid == 0) {
        float b_max = s_warp_max[0];
        int b_idx = s_warp_idx[0];
        int num_warps = blockDim.x / 32;
        for (int w = 1; w < num_warps; ++w) {
            if (s_warp_max[w] > b_max) {
                b_max = s_warp_max[w];
                b_idx = s_warp_idx[w];
            }
        }
        out_tokens[col] = b_idx;
    }
}

__global__ void find_top8_and_lse_kernel(const float* __restrict__ logits,
                                        int n_vocab,
                                        int32_t* __restrict__ cand_ids,
                                        float* __restrict__ cand_logits,
                                        float* __restrict__ lse_out,
                                        float* __restrict__ probs_out) {
    int s = blockIdx.x;
    int tid = threadIdx.x;
    const float* row = logits + (size_t) s * n_vocab;

    float t_val[8];
    int32_t t_id[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        t_val[k] = -1e30f;
        t_id[k] = -1;
    }

    float t_max = -1e30f;
    float t_sum = 0.0f;

    for (int idx = tid; idx < n_vocab; idx += blockDim.x) {
        float v = row[idx];
        if (v > t_max) {
            t_sum = t_sum * expf(t_max - v) + 1.0f;
            t_max = v;
        } else {
            t_sum += expf(v - t_max);
        }

        if (v > t_val[7]) {
            int p = 6;
            while (p >= 0 && v > t_val[p]) {
                t_val[p + 1] = t_val[p];
                t_id[p + 1] = t_id[p];
                p--;
            }
            t_val[p + 1] = v;
            t_id[p + 1] = idx;
        }
    }

    __shared__ float s_val[256 * 8];
    __shared__ int32_t s_id[256 * 8];
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        s_val[tid * 8 + k] = t_val[k];
        s_id[tid * 8 + k] = t_id[k];
    }
    s_max[tid] = t_max;
    s_sum[tid] = t_sum;
    __syncthreads();

    for (int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            float m_val[8];
            int32_t m_id[8];
            int p0 = 0, p1 = 0;
            const float* ptr0 = &s_val[tid * 8];
            const int32_t* id0 = &s_id[tid * 8];
            const float* ptr1 = &s_val[(tid + step) * 8];
            const int32_t* id1 = &s_id[(tid + step) * 8];

            #pragma unroll
            for (int out = 0; out < 8; ++out) {
                if (p0 < 8 && (p1 >= 8 || ptr0[p0] >= ptr1[p1])) {
                    m_val[out] = ptr0[p0];
                    m_id[out] = id0[p0];
                    p0++;
                } else {
                    m_val[out] = ptr1[p1];
                    m_id[out] = id1[p1];
                    p1++;
                }
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                s_val[tid * 8 + k] = m_val[k];
                s_id[tid * 8 + k] = m_id[k];
            }

            float m0 = s_max[tid], m1 = s_max[tid + step];
            float sum0 = s_sum[tid], sum1 = s_sum[tid + step];
            if (m1 > m0) {
                s_sum[tid] = sum0 * expf(m0 - m1) + sum1;
                s_max[tid] = m1;
            } else {
                s_sum[tid] = sum0 + sum1 * expf(m1 - m0);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = s_max[0] + logf(s_sum[0]);
        lse_out[s] = lse;
        for (int k = 0; k < 8; ++k) {
            cand_ids[s * 8 + k] = s_id[k];
            cand_logits[s * 8 + k] = s_val[k];
            probs_out[s * 8 + k] = expf(s_val[k] - lse);
        }
    }
}

__global__ void compute_feature_mlp1_kernel(const float* __restrict__ cand_logits,
                                           const float* __restrict__ lse,
                                           const float* __restrict__ probs,
                                           const float* __restrict__ fnorm_w,
                                           const float* __restrict__ fnorm_b,
                                           const float* __restrict__ mlp1_w,
                                           const float* __restrict__ mlp1_b,
                                           float* __restrict__ out) {
    int node = blockIdx.x;
    int s = node / 8;
    int k = node % 8;

    float lp = logf(fmaxf(probs[node], 1e-20f));
    float top_lp = logf(fmaxf(probs[s * 8 + 0], 1e-20f));

    float f[5];
    f[0] = lp;
    f[1] = probs[node];
    f[2] = lp - top_lp;
    f[3] = (float) k / 7.0f;
    f[4] = (k == 0) ? 1.0f : 0.0f;

    float m = (f[0] + f[1] + f[2] + f[3] + f[4]) * 0.2f;
    float var = ((f[0] - m) * (f[0] - m) +
                 (f[1] - m) * (f[1] - m) +
                 (f[2] - m) * (f[2] - m) +
                 (f[3] - m) * (f[3] - m) +
                 (f[4] - m) * (f[4] - m)) * 0.2f;
    float inv_std = rsqrtf(var + 1.0e-5f);
    float fn[5];
    #pragma unroll
    for (int c = 0; c < 5; ++c) {
        fn[c] = (f[c] - m) * inv_std * fnorm_w[c] + fnorm_b[c];
    }

    for (int h = threadIdx.x; h < 1024; h += blockDim.x) {
        float sum = mlp1_b[h];
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            sum += mlp1_w[h * 5 + c] * fn[c];
        }
        out[node * 1024 + h] = sum / (1.0f + expf(-sum));
    }
}

__global__ void combine_inputs_kernel(const float* __restrict__ token_states,
                                     const float* __restrict__ pass_proj,
                                     const float* __restrict__ pass_b,
                                     const float* __restrict__ feature_states,
                                     const float* __restrict__ feature_mlp3_b,
                                     const float* __restrict__ slot_embd,
                                     const float* __restrict__ rank_embd,
                                     float* __restrict__ x) {
    int node = blockIdx.x;
    int s = node / 8;
    int k = node % 8;

    for (int h = threadIdx.x; h < 1024; h += blockDim.x) {
        float tok = token_states[node * 1024 + h];
        float pass = pass_proj[s * 1024 + h] + pass_b[h];
        float feat = feature_states[node * 1024 + h] + feature_mlp3_b[h];
        float pos = slot_embd[s * 1024 + h] + rank_embd[k * 1024 + h];
        x[node * 1024 + h] = tok + pass + feat + pos;
    }
}

__global__ void rms_norm_f32_scaled_kernel(const float* __restrict__ x,
                                          const float* __restrict__ w,
                                          float* __restrict__ y,
                                          int dim,
                                          int ncols,
                                          float eps) {
    int t = blockIdx.x;
    if (t >= ncols) return;
    const float* xt = x + t * dim;
    float* yt = y + t * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xt[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        s_scale = rsqrtf(sum_sq / (float) dim + eps);
    }
    __syncthreads();
    float scale = s_scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        yt[i] = xt[i] * scale * w[i];
    }
}

__global__ void add_bias_f32_kernel(float* __restrict__ x, const float* __restrict__ b, int dim, int ncols) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    float* xt = x + col * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        xt[i] += b[i];
    }
}

__global__ void lilicorr_attn_kernel(const float* __restrict__ qkv,
                                    const float* __restrict__ attn_bias,
                                    float* __restrict__ attn_out) {
    int h = blockIdx.x;
    int q = threadIdx.x;

    if (q >= 56) return;

    float my_q[128];
    const float* q_ptr = qkv + q * 3072 + h * 128;
    #pragma unroll
    for (int d = 0; d < 128; ++d) {
        my_q[d] = q_ptr[d];
    }

    float scores[56];
    float max_s = -1e30f;
    const float scale = 0.08838834764831845f;
    const float* bias_ptr = attn_bias + h * 56 * 56 + q * 56;

    for (int k = 0; k < 56; ++k) {
        const float* k_ptr = qkv + k * 3072 + 1024 + h * 128;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < 128; ++d) {
            dot += my_q[d] * k_ptr[d];
        }
        float s = dot * scale + bias_ptr[k];
        scores[k] = s;
        if (s > max_s) max_s = s;
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < 56; ++k) {
        float ex = expf(scores[k] - max_s);
        scores[k] = ex;
        sum_exp += ex;
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < 56; ++k) {
        scores[k] *= inv_sum;
    }

    float* out_ptr = attn_out + q * 1024 + h * 128;
    for (int d = 0; d < 128; ++d) {
        float val = 0.0f;
        for (int k = 0; k < 56; ++k) {
            val += scores[k] * qkv[k * 3072 + 2048 + h * 128 + d];
        }
        out_ptr[d] = val;
    }
}

__global__ void add_bias_and_residual_kernel(const float* __restrict__ proj,
                                            const float* __restrict__ bias,
                                            float* __restrict__ x,
                                            int dim,
                                            int ncols) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    const float* p = proj + col * dim;
    float* xt = x + col * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        xt[i] += p[i] + bias[i];
    }
}


__global__ void build_factor_in_kernel(const float* __restrict__ x_norm,
                                      const float* __restrict__ anchor_norm,
                                      float* __restrict__ factor_in,
                                      int dim,
                                      int ncols) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    const float* xn = x_norm + col * dim;
    float* fi = factor_in + col * (3 * dim);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float x_val = xn[d];
        float a_val = anchor_norm[d];
        fi[d] = x_val;
        fi[dim + d] = a_val;
        fi[2 * dim + d] = x_val * a_val;
    }
}

__global__ void silu_f32_in_place_kernel(float* __restrict__ x, const float* __restrict__ bias, int dim, int ncols) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    float* xt = x + col * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = xt[d] + bias[d];
        xt[d] = v / (1.0f + expf(-v));
    }
}

__global__ void add_bias_and_l2_norm_kernel(float* __restrict__ x,
                                           const float* __restrict__ bias,
                                           int dim,
                                           int ncols,
                                           float eps) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    float* xt = x + col * dim;
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = xt[d] + bias[d];
        xt[d] = v;
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        s_scale = rsqrtf(fmaxf(sum_sq, eps * eps));
    }
    __syncthreads();
    float scale = s_scale;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        xt[d] *= scale;
    }
}

__global__ void build_lilicorr_lattice_kernel(const int32_t* __restrict__ cand_ids,
                                             const float* __restrict__ anchor_out,
                                             const float* __restrict__ in_vecs,
                                             const float* __restrict__ out_vecs,
                                             float* __restrict__ lattice,
                                             float logit_scale) {
    int pos = blockIdx.x;
    float* row = lattice + (size_t) pos * 4096;

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        row[i] = 0.0f;
    }
    __syncthreads();

    if (pos == 0) return;

    int s = pos - 1;

    if (threadIdx.x < 8) {
        row[threadIdx.x] = (float) cand_ids[s * 8 + threadIdx.x];
    }
    __syncthreads();

    if (pos == 1) {
        for (int k = 0; k < 8; ++k) {
            const float* in_col = in_vecs + k * 1024;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < 1024; d += blockDim.x) {
                dot += anchor_out[d] * in_col[d];
            }
            dot = block_reduce_sum(dot);
            if (threadIdx.x == 0) {
                row[8 + k] = dot * logit_scale;
            }
            __syncthreads();
        }
    } else {
        const float* prev_out = out_vecs + (s - 1) * 8 * 1024;
        const float* cur_in = in_vecs + s * 8 * 1024;
        for (int pair = 0; pair < 64; ++pair) {
            int j = pair / 8;
            int k = pair % 8;
            const float* out_col = prev_out + j * 1024;
            const float* in_col = cur_in + k * 1024;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < 1024; d += blockDim.x) {
                dot += out_col[d] * in_col[d];
            }
            dot = block_reduce_sum(dot);
            if (threadIdx.x == 0) {
                row[8 + j * 8 + k] = dot * logit_scale;
            }
            __syncthreads();
        }
    }
}

// Simple flexible attention kernel for decode/draft where n_block=8 and total_keys <= 1024
__global__ void dflash_attn_block_kernel(const half* __restrict__ q, const half* __restrict__ k_block, const half* __restrict__ v_block,
                                        const half* __restrict__ k_cache, const half* __restrict__ v_cache, const float* __restrict__ sinks,
                                        half* __restrict__ out, int head_dim, int n_head_q, int n_head_kv, int n_past, int n_block, int cache_capacity) {
    int h = blockIdx.x;
    int t_q = blockIdx.y;
    int h_kv = h / (n_head_q / n_head_kv);
    int total_keys = n_past + n_block;
    int tid = threadIdx.x;

    const half* q_vec = q + (t_q * n_head_q + h) * head_dim;
    float scale = 1.0f / sqrtf((float) head_dim);

    extern __shared__ float s_mem[]; // [total_keys] scores, followed by warp buffers
    float* s_scores = s_mem;

    float my_max = (sinks && tid == 0) ? sinks[h] : -1e30f;
    for (int k_idx = tid; k_idx < total_keys; k_idx += blockDim.x) {
        const half* k_vec = nullptr;
        if (k_idx < n_past) {
            int slot = (n_past <= cache_capacity) ? k_idx : (k_idx % cache_capacity);
            k_vec = k_cache + (slot * n_head_kv + h_kv) * head_dim;
        } else {
            int slot = k_idx - n_past;
            k_vec = k_block + (slot * n_head_kv + h_kv) * head_dim;
        }
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        float sc = dot * scale;
        s_scores[k_idx] = sc;
        if (sc > my_max) my_max = sc;
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, my_max, offset);
        if (other > my_max) my_max = other;
    }
    __shared__ float s_warp_max[32];
    if ((tid % 32) == 0) s_warp_max[tid / 32] = my_max;
    __syncthreads();
    float global_max = s_warp_max[0];
    for (int i = 1; i < blockDim.x / 32; ++i) {
        if (s_warp_max[i] > global_max) global_max = s_warp_max[i];
    }
    __syncthreads();

    // Exp and sum
    float my_sum = (sinks && tid == 0) ? expf(sinks[h] - global_max) : 0.0f;
    for (int k_idx = tid; k_idx < total_keys; k_idx += blockDim.x) {
        float e = expf(s_scores[k_idx] - global_max);
        s_scores[k_idx] = e;
        my_sum += e;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }
    __shared__ float s_warp_sum[32];
    if ((tid % 32) == 0) s_warp_sum[tid / 32] = my_sum;
    __syncthreads();
    float global_sum = s_warp_sum[0];
    for (int i = 1; i < blockDim.x / 32; ++i) {
        global_sum += s_warp_sum[i];
    }
    float inv_sum = 1.0f / (global_sum + 1e-10f);
    __syncthreads();

    // Normalize weights
    for (int k_idx = tid; k_idx < total_keys; k_idx += blockDim.x) {
        s_scores[k_idx] *= inv_sum;
    }
    __syncthreads();

    // Compute output values
    half* out_vec = out + (t_q * n_head_q + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < total_keys; ++k_idx) {
            const half* v_vec = nullptr;
            if (k_idx < n_past) {
                int slot = (n_past <= cache_capacity) ? k_idx : (k_idx % cache_capacity);
                v_vec = v_cache + (slot * n_head_kv + h_kv) * head_dim;
            } else {
                int slot = k_idx - n_past;
                v_vec = v_block + (slot * n_head_kv + h_kv) * head_dim;
            }
            acc += s_scores[k_idx] * __half2float(v_vec[d]);
        }
        out_vec[d] = __float2half(acc);
    }
}

#ifdef DFLASH_DEBUG
static void check_nan(const char* label, const half* d_ptr, size_t n, cudaStream_t stream) {
    std::vector<half> h(std::min(n, (size_t) 1024));
    cudaMemcpyAsync(h.data(), d_ptr, h.size() * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int nans = 0;
    float min_v = 1e30f, max_v = -1e30f;
    for (size_t i = 0; i < h.size(); ++i) {
        float f = __half2float(h[i]);
        if (std::isnan(f) || std::isinf(f)) nans++;
        if (f < min_v) min_v = f;
        if (f > max_v) max_v = f;
    }
    std::printf("[CHECK] %s: size=%zu, nans=%d, min=%f, max=%f, val[0]=%f\n",
                label, n, nans, min_v, max_v, __half2float(h[0]));
}

static void check_nan_f32(const char* label, const float* d_ptr, size_t n, cudaStream_t stream) {
    std::vector<float> h(std::min(n, (size_t) 1024));
    cudaMemcpyAsync(h.data(), d_ptr, h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int nans = 0;
    float min_v = 1e30f, max_v = -1e30f;
    for (size_t i = 0; i < h.size(); ++i) {
        float f = h[i];
        if (std::isnan(f) || std::isinf(f)) nans++;
        if (f < min_v) min_v = f;
        if (f > max_v) max_v = f;
    }
    std::printf("[CHECK] %s: size=%zu, nans=%d, min=%f, max=%f, val[0]=%f\n",
                label, n, nans, min_v, max_v, h[0]);
}
#else
static inline void check_nan(const char*, const half*, size_t, cudaStream_t) {}
static inline void check_nan_f32(const char*, const float*, size_t, cudaStream_t) {}
#endif

} // namespace

// --------------------------------------------------------------------------------------
// Implementation of DFlashDrafter::Impl
struct DFlashDrafter::Impl {
    DFlashConfig cfg;
    cublasHandle_t cublas = nullptr;
    float last_ms = 0.0f;

    // Weight buffers on GPU
    half* d_fc_weight = nullptr; // [24576, 4096]
    float* d_enc_output_norm = nullptr; // [4096]

    struct LayerWeights {
        float* attn_norm = nullptr; // [4096]
        half* attn_conv_proj = nullptr; // [4096, 1024]
        float* attn_conv_base = nullptr; // [4096, 2, 2]
        half* wq = nullptr; // [4096, 4096]
        half* wk = nullptr; // [4096, 256]
        half* wv = nullptr; // [4096, 256]
        float* attn_q_norm = nullptr; // [128]
        float* attn_k_norm = nullptr; // [128]
        float* attn_sinks = nullptr; // [32]
        half* wo = nullptr; // [4096, 4096]
        float* ffn_norm = nullptr; // [4096]
        half* ffn_conv_proj = nullptr; // [4096, 1024]
        float* ffn_conv_base = nullptr; // [4096, 2, 2]
        block_nvfp4* ffn_gate = nullptr; // [4096, 12288], NVFP4 kept packed (nvfp4_mm8)
        block_nvfp4* ffn_up = nullptr; // [4096, 12288]
        block_nvfp4* ffn_down = nullptr; // [12288, 4096]
        float ffn_gate_s = 1.0f, ffn_up_s = 1.0f, ffn_down_s = 1.0f; // their tensor scales
    };
    std::vector<LayerWeights> layers;
    float* d_output_norm = nullptr; // [4096]

    // Target weights
    const void* d_token_embd_q8 = nullptr;
    const void* d_output_q8 = nullptr;
    bool own_target_weights = false;
    void* d_owned_token_embd = nullptr;
    void* d_owned_output = nullptr;

    // LiLiCorr weights
    float* d_lilicorr_token_proj = nullptr; // [4096, 1024]
    float* d_lilicorr_token_proj_b = nullptr; // [1024]
    float* d_lilicorr_pass_hidden_proj = nullptr; // [4096, 1024]
    float* d_lilicorr_pass_hidden_proj_b = nullptr; // [1024]
    float* d_lilicorr_feature_norm = nullptr; // [5]
    float* d_lilicorr_feature_norm_b = nullptr; // [5]
    float* d_lilicorr_feature_mlp_1 = nullptr; // [5, 1024]
    float* d_lilicorr_feature_mlp_1_b = nullptr; // [1024]
    float* d_lilicorr_feature_mlp_3 = nullptr; // [1024, 1024]
    float* d_lilicorr_feature_mlp_3_b = nullptr; // [1024]
    float* d_lilicorr_slot_embedding = nullptr; // [1024, 7]
    float* d_lilicorr_rank_embedding = nullptr; // [1024, 8]
    float* d_lilicorr_attention_bias = nullptr; // [56, 56, 8]
    float* d_lilicorr_context_proj = nullptr; // [4096, 1024]
    float* d_lilicorr_context_proj_b = nullptr; // [1024]

    struct LiLiCorrLayerWeights {
        float* attn_norm = nullptr; // [1024]
        float* mlp_norm = nullptr; // [1024]
        float* attn_in_proj = nullptr; // [1024, 3072]
        float* attn_in_proj_b = nullptr; // [3072]
        float* attn_out_proj = nullptr; // [1024, 1024]
        float* attn_out_proj_b = nullptr; // [1024]
        float* mlp_0 = nullptr; // [1024, 2048]
        float* mlp_0_b = nullptr; // [2048]
        float* mlp_2 = nullptr; // [2048, 1024]
        float* mlp_2_b = nullptr; // [1024]
    };
    std::vector<LiLiCorrLayerWeights> lilicorr_layers;
    float* d_lilicorr_output_norm = nullptr; // [1024]
    float* d_lilicorr_anchor_norm = nullptr; // [1024]
    float* d_lilicorr_factor_input_proj = nullptr; // [3072, 1024]
    float* d_lilicorr_factor_input_proj_b = nullptr; // [1024]
    float* d_lilicorr_out_head = nullptr; // [1024, 1024]
    float* d_lilicorr_out_head_b = nullptr; // [1024]
    float* d_lilicorr_in_head = nullptr; // [1024, 1024]
    float* d_lilicorr_in_head_b = nullptr; // [1024]
    float* d_lilicorr_anchor_out_head = nullptr; // [1024, 1024]
    float* d_lilicorr_anchor_out_head_b = nullptr; // [1024]

    // LiLiCorr runtime activation and lattice buffers
    int32_t* d_lilicorr_cand_ids = nullptr;
    float* d_lilicorr_cand_logits = nullptr;
    float* d_lilicorr_lse = nullptr;
    float* d_lilicorr_probs = nullptr;
    float* d_lilicorr_feat_mlp1_out = nullptr;
    float* d_lilicorr_feature_states = nullptr;
    float* d_lilicorr_embeds_f32 = nullptr;
    float* d_lilicorr_token_states = nullptr;
    float* d_lilicorr_pass_proj = nullptr;
    float* d_lilicorr_anchor_state_proj = nullptr;
    float* d_lilicorr_x = nullptr;
    float* d_lilicorr_norm_x = nullptr;
    float* d_lilicorr_qkv = nullptr;
    float* d_lilicorr_attn_out = nullptr;
    float* d_lilicorr_attn_proj = nullptr;
    float* d_lilicorr_ffn_norm_f32 = nullptr;
    float* d_lilicorr_mlp_gate = nullptr;
    float* d_lilicorr_mlp_out = nullptr;
    float* d_lilicorr_norm_anchor = nullptr;
    float* d_lilicorr_factor_in = nullptr;
    float* d_lilicorr_factors = nullptr;
    float* d_lilicorr_out = nullptr;
    float* d_lilicorr_in = nullptr;
    float* d_lilicorr_anchor_out = nullptr;
    float* d_lilicorr_lattice = nullptr;
    std::vector<float> h_lattice;
    const float* d_lilicorr_logits_override = nullptr;

    // KV cache per layer: [sliding_window=1024, n_head_kv=2, head_dim=128]
    struct LayerKVCache {
        half* k_cache = nullptr;
        half* v_cache = nullptr;
    };
    std::vector<LayerKVCache> kv_caches;
    int kv_len = 0; // current valid tokens in KV cache
    // tokens injected since reset(): the ring slot of the next one is n_injected % sliding_window, so a sequence that
    // starts (or restarts) at any position fills slots 0, 1, ... and the attention reads exactly the kv_len valid ones
    int64_t n_injected = 0;
    const half* enc_f16_input = nullptr;   // encode_and_inject_f16's input for the shared encode path
    float* d_anchor_state = nullptr; // [4096] anchor context features from last prompt position

    // Workspace device memory
    half* d_work_half_1 = nullptr;
    half* d_work_half_2 = nullptr;
    half* d_work_half_3 = nullptr;
    float* d_work_float_1 = nullptr;
    void* d_q8_scratch = nullptr; // scratch for native_mmvq

    half* d_enc_inp_half = nullptr;
    size_t enc_inp_half_cap = 0;
    float* d_enc_fc_out = nullptr;
    half* d_enc_norm_out = nullptr;
    size_t enc_fc_out_cap = 0;
    cudaStream_t default_stream = nullptr;

    Impl() {
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUDA_CHECK(cudaStreamCreateWithFlags(&default_stream, cudaStreamNonBlocking));
    }

    ~Impl() {
        if (cublas) cublasDestroy(cublas);
        if (default_stream) cudaStreamDestroy(default_stream);
        // Free allocated device buffers
        auto free_buf = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
        free_buf((void*&) d_fc_weight);
        free_buf((void*&) d_enc_output_norm);
        for (auto& l : layers) {
            free_buf((void*&) l.attn_norm);
            free_buf((void*&) l.attn_conv_proj);
            free_buf((void*&) l.attn_conv_base);
            free_buf((void*&) l.wq);
            free_buf((void*&) l.wk);
            free_buf((void*&) l.wv);
            free_buf((void*&) l.attn_q_norm);
            free_buf((void*&) l.attn_k_norm);
            free_buf((void*&) l.attn_sinks);
            free_buf((void*&) l.wo);
            free_buf((void*&) l.ffn_norm);
            free_buf((void*&) l.ffn_conv_proj);
            free_buf((void*&) l.ffn_conv_base);
            free_buf((void*&) l.ffn_gate);
            free_buf((void*&) l.ffn_up);
            free_buf((void*&) l.ffn_down);
        }
        free_buf((void*&) d_output_norm);
        if (own_target_weights) {
            free_buf(d_owned_token_embd);
            free_buf(d_owned_output);
        }
        free_buf((void*&) d_lilicorr_token_proj);
        free_buf((void*&) d_lilicorr_token_proj_b);
        free_buf((void*&) d_lilicorr_pass_hidden_proj);
        free_buf((void*&) d_lilicorr_pass_hidden_proj_b);
        free_buf((void*&) d_lilicorr_feature_norm);
        free_buf((void*&) d_lilicorr_feature_norm_b);
        free_buf((void*&) d_lilicorr_feature_mlp_1);
        free_buf((void*&) d_lilicorr_feature_mlp_1_b);
        free_buf((void*&) d_lilicorr_feature_mlp_3);
        free_buf((void*&) d_lilicorr_feature_mlp_3_b);
        free_buf((void*&) d_lilicorr_slot_embedding);
        free_buf((void*&) d_lilicorr_rank_embedding);
        free_buf((void*&) d_lilicorr_attention_bias);
        free_buf((void*&) d_lilicorr_context_proj);
        free_buf((void*&) d_lilicorr_context_proj_b);
        for (auto& l : lilicorr_layers) {
            free_buf((void*&) l.attn_norm);
            free_buf((void*&) l.mlp_norm);
            free_buf((void*&) l.attn_in_proj);
            free_buf((void*&) l.attn_in_proj_b);
            free_buf((void*&) l.attn_out_proj);
            free_buf((void*&) l.attn_out_proj_b);
            free_buf((void*&) l.mlp_0);
            free_buf((void*&) l.mlp_0_b);
            free_buf((void*&) l.mlp_2);
            free_buf((void*&) l.mlp_2_b);
        }
        free_buf((void*&) d_lilicorr_output_norm);
        free_buf((void*&) d_lilicorr_anchor_norm);
        free_buf((void*&) d_lilicorr_factor_input_proj);
        free_buf((void*&) d_lilicorr_factor_input_proj_b);
        free_buf((void*&) d_lilicorr_out_head);
        free_buf((void*&) d_lilicorr_out_head_b);
        free_buf((void*&) d_lilicorr_in_head);
        free_buf((void*&) d_lilicorr_in_head_b);
        free_buf((void*&) d_lilicorr_anchor_out_head);
        free_buf((void*&) d_lilicorr_anchor_out_head_b);
        free_buf((void*&) d_lilicorr_cand_ids);
        free_buf((void*&) d_lilicorr_cand_logits);
        free_buf((void*&) d_lilicorr_lse);
        free_buf((void*&) d_lilicorr_probs);
        free_buf((void*&) d_lilicorr_feat_mlp1_out);
        free_buf((void*&) d_lilicorr_feature_states);
        free_buf((void*&) d_lilicorr_embeds_f32);
        free_buf((void*&) d_lilicorr_token_states);
        free_buf((void*&) d_lilicorr_pass_proj);
        free_buf((void*&) d_lilicorr_anchor_state_proj);
        free_buf((void*&) d_lilicorr_x);
        free_buf((void*&) d_lilicorr_norm_x);
        free_buf((void*&) d_lilicorr_qkv);
        free_buf((void*&) d_lilicorr_attn_out);
        free_buf((void*&) d_lilicorr_attn_proj);
        free_buf((void*&) d_lilicorr_ffn_norm_f32);
        free_buf((void*&) d_lilicorr_mlp_gate);
        free_buf((void*&) d_lilicorr_mlp_out);
        free_buf((void*&) d_lilicorr_norm_anchor);
        free_buf((void*&) d_lilicorr_factor_in);
        free_buf((void*&) d_lilicorr_factors);
        free_buf((void*&) d_lilicorr_out);
        free_buf((void*&) d_lilicorr_in);
        free_buf((void*&) d_lilicorr_anchor_out);
        free_buf((void*&) d_lilicorr_lattice);
        for (auto& c : kv_caches) {
            free_buf((void*&) c.k_cache);
            free_buf((void*&) c.v_cache);
        }
        free_buf((void*&) d_anchor_state);
        free_buf((void*&) d_work_half_1);
        free_buf((void*&) d_work_half_2);
        free_buf((void*&) d_work_half_3);
        free_buf((void*&) d_work_float_1);
        free_buf(d_q8_scratch);
        free_buf((void*&) d_enc_inp_half);
        free_buf((void*&) d_enc_fc_out);
        free_buf((void*&) d_enc_norm_out);
    }
};

DFlashDrafter::DFlashDrafter() : impl_(std::make_unique<Impl>()) {}
DFlashDrafter::~DFlashDrafter() = default;

const DFlashConfig& DFlashDrafter::config() const {
    return impl_->cfg;
}

float DFlashDrafter::last_draft_ms() const {
    return impl_->last_ms;
}

void DFlashDrafter::get_lattice(float* h_lattice_out) const {
    if (h_lattice_out && !impl_->h_lattice.empty()) {
        std::memcpy(h_lattice_out, impl_->h_lattice.data(), impl_->h_lattice.size() * sizeof(float));
    }
}

void DFlashDrafter::set_lilicorr_logits_override(const float* d_logits) {
    impl_->d_lilicorr_logits_override = d_logits;
}

const float* DFlashDrafter::lattice_device() const {
    return impl_->d_lilicorr_lattice;
}

void DFlashDrafter::set_target_weights(const void* d_token_embd_q8, const void* d_output_q8, int vocab_size) {
    impl_->d_token_embd_q8 = d_token_embd_q8;
    impl_->d_output_q8 = d_output_q8;
    impl_->cfg.n_vocab = vocab_size;
}

void DFlashDrafter::reset() {
    impl_->kv_len = 0;
    impl_->n_injected = 0;
}

size_t DFlashDrafter::state_bytes() const {
    const auto& c = impl_->cfg;
    return (size_t) c.n_layers * 2 * c.sliding_window * c.n_head_kv * c.head_dim * sizeof(half) + c.n_embd * sizeof(float) +
           2 * sizeof(int64_t);
}

void DFlashDrafter::copy_state(void* host, bool save, cudaStream_t stream) {
    auto& m = *impl_;
    if (!stream) stream = m.default_stream;
    const size_t kvb = (size_t) m.cfg.sliding_window * m.cfg.n_head_kv * m.cfg.head_dim * sizeof(half);
    uint8_t* h = static_cast<uint8_t*>(host);
    const auto kind = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    for (auto& c : m.kv_caches) {
        CUDA_CHECK(save ? cudaMemcpyAsync(h, c.k_cache, kvb, kind, stream) : cudaMemcpyAsync(c.k_cache, h, kvb, kind, stream));
        CUDA_CHECK(save ? cudaMemcpyAsync(h + kvb, c.v_cache, kvb, kind, stream) : cudaMemcpyAsync(c.v_cache, h + kvb, kvb, kind, stream));
        h += 2 * kvb;
    }
    const size_t ab = (size_t) m.cfg.n_embd * sizeof(float);
    CUDA_CHECK(save ? cudaMemcpyAsync(h, m.d_anchor_state, ab, kind, stream) : cudaMemcpyAsync(m.d_anchor_state, h, ab, kind, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int64_t* tail = reinterpret_cast<int64_t*>(h + ab);
    if (save) {
        tail[0] = m.kv_len;
        tail[1] = m.n_injected;
    } else {
        m.kv_len = (int) tail[0];
        m.n_injected = tail[1];
    }
}

bool DFlashDrafter::load(const std::string& dflash_gguf_path,
                         const std::string& target_gguf_path,
                         std::string& err) {
    try {
        std::vector<std::string> dft_shards = {dflash_gguf_path};
        strata::GgufModel gm_dft(dft_shards);

        for (const auto& [k, v] : gm_dft.shard(0).metadata()) {
            if (k == "tokenizer.chat_template") continue;
            if (v.is_num()) {
                std::printf("DFlash meta: %-35s = %f\n", k.c_str(), v.num());
            } else if (v.type == strata::MetaType::STRING) {
                std::printf("DFlash meta: %-35s = \"%s\"\n", k.c_str(), v.s.c_str());
            }
        }

        // Helper to load and dequantize
        auto load_nvfp4 = [&](const std::string& name, const std::string& scale_name, half*& d_ptr, int64_t n_elems) {
            size_t s = 0;
            const strata::TensorInfo* t = gm_dft.find(name, &s);
            if (!t) throw std::runtime_error("missing tensor: " + name);
            float scale_val = 1.0f;
            if (!scale_name.empty()) {
                size_t ss = 0;
                const strata::TensorInfo* ts = gm_dft.find(scale_name, &ss);
                if (ts) {
                    const void* s_raw = gm_dft.shard(ss).tensor_data(*ts);
                    std::memcpy(&scale_val, s_raw, sizeof(float));
                }
            }
            CUDA_CHECK(cudaMalloc(&d_ptr, n_elems * sizeof(half)));
            // Upload raw blocks
            const void* raw_data = gm_dft.shard(s).tensor_data(*t);
            void* d_raw = nullptr;
            size_t raw_bytes = (size_t) (n_elems / QK_NVFP4) * sizeof(block_nvfp4);
            CUDA_CHECK(cudaMalloc(&d_raw, raw_bytes));
            CUDA_CHECK(cudaMemcpy(d_raw, raw_data, raw_bytes, cudaMemcpyHostToDevice));
            int nb = (int) (n_elems / QK_NVFP4);
            dequantize_nvfp4_kernel<<<nb, 32>>>(reinterpret_cast<const block_nvfp4*>(d_raw), d_ptr, n_elems, scale_val);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_raw));
        };

        auto load_nvfp4_packed = [&](const std::string& name, const std::string& scale_name, block_nvfp4*& d_ptr,
                                     float& scale_val, int64_t n_elems) {
            size_t s = 0;
            const strata::TensorInfo* t = gm_dft.find(name, &s);
            if (!t) throw std::runtime_error("missing tensor: " + name);
            scale_val = 1.0f;
            size_t ss = 0;
            if (const strata::TensorInfo* ts = gm_dft.find(scale_name, &ss))
                std::memcpy(&scale_val, gm_dft.shard(ss).tensor_data(*ts), sizeof(float));
            const size_t raw_bytes = (size_t) (n_elems / QK_NVFP4) * sizeof(block_nvfp4);
            CUDA_CHECK(cudaMalloc(&d_ptr, raw_bytes));
            CUDA_CHECK(cudaMemcpy(d_ptr, gm_dft.shard(s).tensor_data(*t), raw_bytes, cudaMemcpyHostToDevice));
        };

        auto load_nvfp4_f32 = [&](const std::string& name, const std::string& scale_name, float*& d_ptr, int64_t n_elems) {
            size_t s = 0;
            const strata::TensorInfo* t = gm_dft.find(name, &s);
            if (!t) throw std::runtime_error("missing tensor: " + name);
            float scale_val = 1.0f;
            if (!scale_name.empty()) {
                size_t ss = 0;
                const strata::TensorInfo* ts = gm_dft.find(scale_name, &ss);
                if (ts) {
                    const void* s_raw = gm_dft.shard(ss).tensor_data(*ts);
                    std::memcpy(&scale_val, s_raw, sizeof(float));
                }
            }
            CUDA_CHECK(cudaMalloc(&d_ptr, n_elems * sizeof(float)));
            const void* raw_data = gm_dft.shard(s).tensor_data(*t);
            void* d_raw = nullptr;
            size_t raw_bytes = (size_t) (n_elems / QK_NVFP4) * sizeof(block_nvfp4);
            CUDA_CHECK(cudaMalloc(&d_raw, raw_bytes));
            CUDA_CHECK(cudaMemcpy(d_raw, raw_data, raw_bytes, cudaMemcpyHostToDevice));
            int nb = (int) (n_elems / QK_NVFP4);
            dequantize_nvfp4_f32_kernel<<<nb, 32>>>(reinterpret_cast<const block_nvfp4*>(d_raw), d_ptr, n_elems, scale_val);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_raw));
        };

        auto load_bf16 = [&](const std::string& name, half*& d_ptr, int64_t n_elems) {
            size_t s = 0;
            const strata::TensorInfo* t = gm_dft.find(name, &s);
            if (!t) throw std::runtime_error("missing tensor: " + name);
            CUDA_CHECK(cudaMalloc(&d_ptr, n_elems * sizeof(half)));
            const void* raw_data = gm_dft.shard(s).tensor_data(*t);
            void* d_raw = nullptr;
            CUDA_CHECK(cudaMalloc(&d_raw, n_elems * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemcpy(d_raw, raw_data, n_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
            int block = 256;
            int grid = (int) ((n_elems + block - 1) / block);
            convert_bf16_to_half_kernel<<<grid, block>>>(reinterpret_cast<const uint16_t*>(d_raw), d_ptr, n_elems);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_raw));
        };

        auto load_f32 = [&](const std::string& name, float*& d_ptr, int64_t n_elems) {
            size_t s = 0;
            const strata::TensorInfo* t = gm_dft.find(name, &s);
            if (!t) throw std::runtime_error("missing tensor: " + name);
            CUDA_CHECK(cudaMalloc(&d_ptr, n_elems * sizeof(float)));
            const void* raw_data = gm_dft.shard(s).tensor_data(*t);
            CUDA_CHECK(cudaMemcpy(d_ptr, raw_data, n_elems * sizeof(float), cudaMemcpyHostToDevice));
        };


        // 1. Encoder
        load_nvfp4("fc.weight", "fc.scale", impl_->d_fc_weight, (int64_t) impl_->cfg.n_embd_inp * impl_->cfg.n_embd);
        load_f32("enc.output_norm.weight", impl_->d_enc_output_norm, impl_->cfg.n_embd);

        // 2. Transformer layers 0..4
        impl_->layers.resize(impl_->cfg.n_layers);
        for (int i = 0; i < impl_->cfg.n_layers; ++i) {
            std::string p = "blk." + std::to_string(i) + ".";
            auto& l = impl_->layers[i];
            load_f32(p + "attn_norm.weight", l.attn_norm, impl_->cfg.n_embd);
            load_bf16(p + "attn_conv_proj.weight", l.attn_conv_proj, (int64_t) impl_->cfg.n_embd * 1024);
            load_f32(p + "attn_conv_base", l.attn_conv_base, (int64_t) impl_->cfg.n_embd * 2 * 2);
            load_bf16(p + "attn_q.weight", l.wq, (int64_t) impl_->cfg.n_embd * impl_->cfg.n_embd);
            load_bf16(p + "attn_k.weight", l.wk, (int64_t) impl_->cfg.n_embd * (impl_->cfg.n_head_kv * impl_->cfg.head_dim));
            load_bf16(p + "attn_v.weight", l.wv, (int64_t) impl_->cfg.n_embd * (impl_->cfg.n_head_kv * impl_->cfg.head_dim));
            load_f32(p + "attn_q_norm.weight", l.attn_q_norm, impl_->cfg.head_dim);
            load_f32(p + "attn_k_norm.weight", l.attn_k_norm, impl_->cfg.head_dim);
            load_f32(p + "attn_sinks.weight", l.attn_sinks, impl_->cfg.n_head);
            load_bf16(p + "attn_output.weight", l.wo, (int64_t) impl_->cfg.n_embd * impl_->cfg.n_embd);
            load_f32(p + "ffn_norm.weight", l.ffn_norm, impl_->cfg.n_embd);
            load_bf16(p + "ffn_conv_proj.weight", l.ffn_conv_proj, (int64_t) impl_->cfg.n_embd * 1024);
            load_f32(p + "ffn_conv_base", l.ffn_conv_base, (int64_t) impl_->cfg.n_embd * 2 * 2);
            load_nvfp4_packed(p + "ffn_gate.weight", p + "ffn_gate.scale", l.ffn_gate, l.ffn_gate_s, (int64_t) impl_->cfg.n_embd * impl_->cfg.n_ff);
            load_nvfp4_packed(p + "ffn_up.weight", p + "ffn_up.scale", l.ffn_up, l.ffn_up_s, (int64_t) impl_->cfg.n_embd * impl_->cfg.n_ff);
            load_nvfp4_packed(p + "ffn_down.weight", p + "ffn_down.scale", l.ffn_down, l.ffn_down_s, (int64_t) impl_->cfg.n_ff * impl_->cfg.n_embd);
        }
        load_f32("output_norm.weight", impl_->d_output_norm, impl_->cfg.n_embd);

        // 3. LiLiCorr Tensors
        load_nvfp4_f32("lilicorr.token_proj.weight", "lilicorr.token_proj.scale", impl_->d_lilicorr_token_proj, (int64_t) impl_->cfg.n_embd * impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.token_proj.bias", impl_->d_lilicorr_token_proj_b, impl_->cfg.lilicorr_hidden);
        load_nvfp4_f32("lilicorr.pass_hidden_proj.weight", "lilicorr.pass_hidden_proj.scale", impl_->d_lilicorr_pass_hidden_proj, (int64_t) impl_->cfg.n_embd * impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.pass_hidden_proj.bias", impl_->d_lilicorr_pass_hidden_proj_b, impl_->cfg.lilicorr_hidden);
        load_f32("blk.7.ffn_norm.weight", impl_->d_lilicorr_feature_norm, 5);
        load_f32("blk.7.ffn_norm.bias", impl_->d_lilicorr_feature_norm_b, 5);
        load_f32("blk.7.ffn_up.weight", impl_->d_lilicorr_feature_mlp_1, 5 * impl_->cfg.lilicorr_hidden);
        load_f32("blk.7.ffn_up.bias", impl_->d_lilicorr_feature_mlp_1_b, impl_->cfg.lilicorr_hidden);
        load_f32("blk.7.ffn_down.weight", impl_->d_lilicorr_feature_mlp_3, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_hidden);
        load_f32("blk.7.ffn_down.bias", impl_->d_lilicorr_feature_mlp_3_b, impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.slot_embedding.weight", impl_->d_lilicorr_slot_embedding, impl_->cfg.lilicorr_hidden * (impl_->cfg.block_size - 1));
        load_f32("lilicorr.rank_embedding.weight", impl_->d_lilicorr_rank_embedding, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_top_k);
        load_f32("lilicorr.attention_bias.bias", impl_->d_lilicorr_attention_bias, 56 * 56 * impl_->cfg.lilicorr_heads);
        load_nvfp4_f32("lilicorr.context_proj.weight", "lilicorr.context_proj.scale", impl_->d_lilicorr_context_proj, (int64_t) impl_->cfg.n_embd * impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.context_proj.bias", impl_->d_lilicorr_context_proj_b, impl_->cfg.lilicorr_hidden);

        // LiLiCorr layers 0, 1 (blk.5, blk.6)
        impl_->lilicorr_layers.resize(impl_->cfg.lilicorr_layers);
        for (int i = 0; i < impl_->cfg.lilicorr_layers; ++i) {
            std::string bp = "blk." + std::to_string(impl_->cfg.n_layers + i) + ".";
            std::string lp = "lilicorr.layers." + std::to_string(i) + ".";
            auto& l = impl_->lilicorr_layers[i];
            load_f32(bp + "attn_norm.weight", l.attn_norm, impl_->cfg.lilicorr_hidden);
            load_f32(bp + "ffn_norm.weight", l.mlp_norm, impl_->cfg.lilicorr_hidden);
            load_f32(lp + "attn.in_proj.weight", l.attn_in_proj, impl_->cfg.lilicorr_hidden * 3 * impl_->cfg.lilicorr_hidden);
            load_f32(lp + "attn.in_proj.bias", l.attn_in_proj_b, 3 * impl_->cfg.lilicorr_hidden);
            load_nvfp4_f32(bp + "attn_output.weight", bp + "attn_output.scale", l.attn_out_proj, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_hidden);
            load_f32(bp + "attn_output.bias", l.attn_out_proj_b, impl_->cfg.lilicorr_hidden);
            load_nvfp4_f32(bp + "ffn_up.weight", bp + "ffn_up.scale", l.mlp_0, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_mlp);
            load_f32(bp + "ffn_up.bias", l.mlp_0_b, impl_->cfg.lilicorr_mlp);
            load_nvfp4_f32(bp + "ffn_down.weight", bp + "ffn_down.scale", l.mlp_2, impl_->cfg.lilicorr_mlp * impl_->cfg.lilicorr_hidden);
            load_f32(bp + "ffn_down.bias", l.mlp_2_b, impl_->cfg.lilicorr_hidden);
        }
        load_f32("lilicorr.output_norm.weight", impl_->d_lilicorr_output_norm, impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.anchor_norm.weight", impl_->d_lilicorr_anchor_norm, impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.factor_input_proj.weight", impl_->d_lilicorr_factor_input_proj, 3 * impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.factor_input_proj.bias", impl_->d_lilicorr_factor_input_proj_b, impl_->cfg.lilicorr_hidden);
        load_f32("lilicorr.out_head.weight", impl_->d_lilicorr_out_head, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_factor_dim);
        load_f32("lilicorr.out_head.bias", impl_->d_lilicorr_out_head_b, impl_->cfg.lilicorr_factor_dim);
        load_f32("lilicorr.in_head.weight", impl_->d_lilicorr_in_head, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_factor_dim);
        load_f32("lilicorr.in_head.bias", impl_->d_lilicorr_in_head_b, impl_->cfg.lilicorr_factor_dim);
        load_f32("lilicorr.anchor_out_head.weight", impl_->d_lilicorr_anchor_out_head, impl_->cfg.lilicorr_hidden * impl_->cfg.lilicorr_factor_dim);
        load_f32("lilicorr.anchor_out_head.bias", impl_->d_lilicorr_anchor_out_head_b, impl_->cfg.lilicorr_factor_dim);

        // 4. Target weights (if target_gguf_path is given)
        if (!target_gguf_path.empty()) {
            std::vector<std::string> tgt_shards = {target_gguf_path};
            strata::GgufModel gm_tgt(tgt_shards);
            size_t s_embd = 0, s_out = 0;
            const strata::TensorInfo* t_embd = gm_tgt.find("token_embd.weight", &s_embd);
            const strata::TensorInfo* t_out = gm_tgt.find("output.weight", &s_out);
            if (!t_embd || !t_out) {
                throw std::runtime_error("target model missing token_embd.weight or output.weight");
            }
            std::printf("target model token_embd.weight type=%d shape=[%lld, %lld]\n", (int) t_embd->type, (long long) t_embd->shape[0], (long long) t_embd->shape[1]);
            std::printf("target model output.weight type=%d shape=[%lld, %lld]\n", (int) t_out->type, (long long) t_out->shape[0], (long long) t_out->shape[1]);
            impl_->cfg.n_vocab = (int) t_embd->shape[1];
            size_t bytes_embd = strata::tensor_payload_bytes(*t_embd);
            size_t bytes_out = strata::tensor_payload_bytes(*t_out);

            CUDA_CHECK(cudaMalloc(&impl_->d_owned_token_embd, bytes_embd));
            CUDA_CHECK(cudaMemcpy(impl_->d_owned_token_embd, gm_tgt.shard(s_embd).tensor_data(*t_embd), bytes_embd, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&impl_->d_owned_output, bytes_out));
            CUDA_CHECK(cudaMemcpy(impl_->d_owned_output, gm_tgt.shard(s_out).tensor_data(*t_out), bytes_out, cudaMemcpyHostToDevice));

            impl_->d_token_embd_q8 = impl_->d_owned_token_embd;
            impl_->d_output_q8 = impl_->d_owned_output;
            impl_->own_target_weights = true;
        }

        // 5. KV Cache initialization
        impl_->kv_caches.resize(impl_->cfg.n_layers);
        for (int i = 0; i < impl_->cfg.n_layers; ++i) {
            size_t cache_bytes = (size_t) impl_->cfg.sliding_window * impl_->cfg.n_head_kv * impl_->cfg.head_dim * sizeof(half);
            CUDA_CHECK(cudaMalloc(&impl_->kv_caches[i].k_cache, cache_bytes));
            CUDA_CHECK(cudaMalloc(&impl_->kv_caches[i].v_cache, cache_bytes));
            CUDA_CHECK(cudaMemset(impl_->kv_caches[i].k_cache, 0, cache_bytes));
            CUDA_CHECK(cudaMemset(impl_->kv_caches[i].v_cache, 0, cache_bytes));
        }
        CUDA_CHECK(cudaMalloc(&impl_->d_anchor_state, impl_->cfg.n_embd * sizeof(float)));

        // 6. Scratch workspaces
        size_t max_half_work = 2097152; // 2M elements (4 MB) to comfortably hold FFN and LiLiCorr embeddings
        CUDA_CHECK(cudaMalloc(&impl_->d_work_half_1, max_half_work * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&impl_->d_work_half_2, max_half_work * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&impl_->d_work_half_3, max_half_work * sizeof(half)));
        size_t max_float_work = (size_t) impl_->cfg.n_vocab * impl_->cfg.block_size + 1048576;
        CUDA_CHECK(cudaMalloc(&impl_->d_work_float_1, max_float_work * sizeof(float)));
        size_t q8_bytes = strata::kernels::native_q8_1_bytes(impl_->cfg.n_embd, impl_->cfg.block_size);
        CUDA_CHECK(cudaMalloc(&impl_->d_q8_scratch, q8_bytes));

        // 7. LiLiCorr runtime buffers
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_cand_ids, 56 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_cand_logits, 56 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_lse, 7 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_probs, 56 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_feat_mlp1_out, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_feature_states, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_embeds_f32, 56 * impl_->cfg.n_embd * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_token_states, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_pass_proj, 7 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_anchor_state_proj, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_x, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_norm_x, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_qkv, 56 * 3072 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_attn_out, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_attn_proj, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_ffn_norm_f32, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_mlp_gate, 56 * 2048 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_mlp_out, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_norm_anchor, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_factor_in, 56 * 3072 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_factors, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_out, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_in, 56 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_anchor_out, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&impl_->d_lilicorr_lattice, 8 * 4096 * sizeof(float)));
        impl_->h_lattice.resize(8 * 4096, 0.0f);

        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void DFlashDrafter::encode_and_inject_f16(const __half* d_target_features, int n_tokens, int pos0, cudaStream_t stream) {
    impl_->enc_f16_input = d_target_features;
    encode_and_inject(nullptr, n_tokens, pos0, stream);
    impl_->enc_f16_input = nullptr;
}

void DFlashDrafter::encode_and_inject(const float* d_target_features,
                                     int n_tokens,
                                     int pos0,
                                     cudaStream_t stream) {
    auto& m = *impl_;
    if (!stream) stream = m.default_stream;
    cublasSetStream(m.cublas, stream);

    size_t needed_inp = (size_t) n_tokens * m.cfg.n_embd_inp;
    if (!m.enc_f16_input && needed_inp > m.enc_inp_half_cap) {
        if (m.d_enc_inp_half) cudaFree(m.d_enc_inp_half);
        CUDA_CHECK(cudaMalloc(&m.d_enc_inp_half, needed_inp * sizeof(half)));
        m.enc_inp_half_cap = needed_inp;
    }
    size_t needed_fc = (size_t) n_tokens * m.cfg.n_embd;
    if (needed_fc > m.enc_fc_out_cap) {
        if (m.d_enc_fc_out) cudaFree(m.d_enc_fc_out);
        if (m.d_enc_norm_out) cudaFree(m.d_enc_norm_out);
        CUDA_CHECK(cudaMalloc(&m.d_enc_fc_out, needed_fc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m.d_enc_norm_out, needed_fc * sizeof(half)));
        m.enc_fc_out_cap = needed_fc;
    }

    // 1. Convert input features [n_tokens, 24576] from float to half (unless they come as FP16 already)
    const half* d_inp_half = m.enc_f16_input;
    if (!d_inp_half) {
        int b = 256;
        int g = (int) ((needed_inp + b - 1) / b);
        convert_f32_to_half_kernel<<<g, b, 0, stream>>>(d_target_features, m.d_enc_inp_half, needed_inp);
        d_inp_half = m.d_enc_inp_half;
    }

    // 2. Encoder projection: fc(inp_target) -> [4096, n_tokens] (FP32)
    float* d_fc_out = m.d_enc_fc_out;
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             m.cfg.n_embd, n_tokens, m.cfg.n_embd_inp,
                             &alpha, m.d_fc_weight, CUDA_R_16F, m.cfg.n_embd_inp,
                             d_inp_half, CUDA_R_16F, m.cfg.n_embd_inp,
                             &beta, d_fc_out, CUDA_R_32F, m.cfg.n_embd,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    check_nan_f32("enc: d_fc_out", d_fc_out, (size_t) m.cfg.n_embd * n_tokens, stream);

    // 3. Encoder norm: rms_norm(fc_out, enc_output_norm)
    half* d_inp_g = m.d_enc_norm_out;
    rms_norm_f32_to_half_kernel<<<n_tokens, 256, 0, stream>>>(d_fc_out, m.d_enc_output_norm, d_inp_g, m.cfg.n_embd, n_tokens, m.cfg.eps);
    check_nan("enc: d_inp_g", d_inp_g, (size_t) m.cfg.n_embd * n_tokens, stream);

    // Save the last token's context embedding for LiLiCorr anchor in FP32
    int last_idx = n_tokens - 1;
    rms_norm_f32_to_f32_kernel<<<1, 256, 0, stream>>>(d_fc_out + last_idx * m.cfg.n_embd, m.d_enc_output_norm, m.d_anchor_state, m.cfg.n_embd, 1, m.cfg.eps);
    check_nan_f32("enc: d_anchor_state", m.d_anchor_state, m.cfg.n_embd, stream);

    // 4. Inject K and V into KV cache for layers 0..4
    for (int il = 0; il < m.cfg.n_layers; ++il) {
        auto& l = m.layers[il];
        auto& c = m.kv_caches[il];
        int kv_dim = m.cfg.n_head_kv * m.cfg.head_dim; // 256

        half* d_k_proj = m.d_work_half_2;
        half* d_v_proj = m.d_work_half_3;

        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 kv_dim, n_tokens, m.cfg.n_embd,
                                 &alpha, l.wk, CUDA_R_16F, m.cfg.n_embd,
                                 d_inp_g, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_k_proj, CUDA_R_16F, kv_dim,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 kv_dim, n_tokens, m.cfg.n_embd,
                                 &alpha, l.wv, CUDA_R_16F, m.cfg.n_embd,
                                 d_inp_g, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_v_proj, CUDA_R_16F, kv_dim,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        // RMS norm on K per head
        int total_k_heads = m.cfg.n_head_kv * n_tokens;
        int threads = 32 * total_k_heads;
        int blocks = (threads + 255) / 256;
        rms_norm_per_head_kernel<<<blocks, 256, 0, stream>>>(d_k_proj, l.attn_k_norm, m.cfg.head_dim, m.cfg.n_head_kv, n_tokens, m.cfg.eps);

        // RoPE on K (64 threads per head)
        int rope_threads = 64 * total_k_heads;
        int rope_blocks = (rope_threads + 255) / 256;
        neox_rope_kernel<<<rope_blocks, 256, 0, stream>>>(nullptr, d_k_proj, 0, m.cfg.n_head_kv, m.cfg.head_dim, pos0, n_tokens, m.cfg.rope_freq_base);

        // Into the ring, in at most two contiguous runs (only the last sliding_window tokens of a long chunk matter)
        const int W = m.cfg.sliding_window;
        const int skip = std::max(0, n_tokens - W);
        int t = skip;
        int slot = (int) ((m.n_injected + skip) % W);
        while (t < n_tokens) {
            const int run = std::min(n_tokens - t, W - slot);
            CUDA_CHECK(cudaMemcpyAsync(c.k_cache + (size_t) slot * kv_dim, d_k_proj + (size_t) t * kv_dim, (size_t) run * kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(c.v_cache + (size_t) slot * kv_dim, d_v_proj + (size_t) t * kv_dim, (size_t) run * kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            t += run;
            slot = (slot + run) % W;
        }
    }

    m.n_injected += n_tokens;
    m.kv_len = (int) std::min<int64_t>(m.cfg.sliding_window, m.n_injected);
}

std::vector<int32_t> DFlashDrafter::draft(int32_t id_last,
                                         int32_t pos0,
                                         float* d_logits_out,
                                         cudaStream_t stream) {
    auto& m = *impl_;
    if (!stream) stream = m.default_stream;
    cublasSetStream(m.cublas, stream);

    auto t_start = std::chrono::steady_clock::now();

    const int block_size = m.cfg.block_size; // 8
    const int n_draft = m.cfg.n_draft;       // 7

    // 1. Gather token embeddings for [id_last, 990, ..., 990] (8 tokens)
    int32_t h_tokens[8];
    h_tokens[0] = id_last;
    for (int i = 1; i < block_size; ++i) h_tokens[i] = m.cfg.mask_token_id;

    int32_t* d_tok_ids = reinterpret_cast<int32_t*>(m.d_work_half_1 + 100000); // temporary slot
    CUDA_CHECK(cudaMemcpyAsync(d_tok_ids, h_tokens, 8 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    float* d_x            = m.d_work_float_1 + 0;      // [4096, 8]
    float* d_attn_proj    = m.d_work_float_1 + 32768;  // [4096, 8]
    float* d_cur          = m.d_work_float_1 + 65536;  // [4096, 8]
    float* d_ffn_inp      = m.d_work_float_1 + 98304;  // [4096, 8]
    float* d_ffn_out      = m.d_work_float_1 + 131072; // [4096, 8]
    float* d_ffn_conv_out = m.d_work_float_1 + 163840; // [4096, 8]
    float* d_t_embd_f32   = m.d_work_float_1 + 196608; // [4096, 8]

    // Dequantize embeddings directly from Q8_0 token_embd.weight into F32 d_x
    dequantize_q8_0_embed_kernel<<<dim3(m.cfg.n_embd / 32, block_size), 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(m.d_token_embd_q8), d_tok_ids, block_size, m.cfg.n_embd, d_x);

    check_nan_f32("draft: initial d_x", d_x, (size_t) m.cfg.n_embd * block_size, stream);

    float alpha = 1.0f, beta = 0.0f;

    // Disjoint half workspace offsets:
    half* d_conv_in     = m.d_work_half_1 + 0;      // [4096, 8]
    half* d_attn_out    = m.d_work_half_1 + 32768;  // [4096, 8]
    half* d_t_embd      = m.d_work_half_1 + 65536;  // [4096, 8]

    half* d_norm        = m.d_work_half_2 + 0;      // [4096, 8]
    half* d_attn_dyn    = m.d_work_half_2 + 32768;  // [1024, 8]
    half* d_q           = m.d_work_half_2 + 40960;  // [4096, 8]
    int kv_dim = m.cfg.n_head_kv * m.cfg.head_dim;  // 256
    half* d_k           = m.d_work_half_2 + 73728;  // [256, 8]
    half* d_v           = m.d_work_half_2 + 75776;  // [256, 8]
    half* d_ffn_norm    = m.d_work_half_2 + 80000;  // [4096, 8]
    half* d_ffn_dyn     = m.d_work_half_2 + 120000; // [1024, 8]
    half* d_ffn_conv_in = m.d_work_half_2 + 130000; // [4096, 8]

    half* d_gate        = m.d_work_half_3 + 0;      // [12288, 8]
    half* d_up          = m.d_work_half_3 + 100000; // [12288, 8]

    // 2. Transformer layers 0..4
    for (int il = 0; il < m.cfg.n_layers; ++il) {
        auto& l = m.layers[il];
        auto& c = m.kv_caches[il];

        // a) Attention norm: F32 d_x -> half d_norm
        rms_norm_f32_to_half_kernel<<<block_size, 256, 0, stream>>>(d_x, l.attn_norm, d_norm, m.cfg.n_embd, block_size, m.cfg.eps);

        // b) Attn conv in
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 1024, block_size, m.cfg.n_embd,
                                 &alpha, l.attn_conv_proj, CUDA_R_16F, m.cfg.n_embd,
                                 d_norm, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_attn_dyn, CUDA_R_16F, 1024,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        dflash_conv_kernel<<<128, 256, 0, stream>>>(d_norm, d_attn_dyn, l.attn_conv_base, d_conv_in, m.cfg.n_embd, 256, block_size, 0);

        // c) Q, K, V projections
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 m.cfg.n_embd, block_size, m.cfg.n_embd,
                                 &alpha, l.wq, CUDA_R_16F, m.cfg.n_embd,
                                 d_conv_in, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_q, CUDA_R_16F, m.cfg.n_embd,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 kv_dim, block_size, m.cfg.n_embd,
                                 &alpha, l.wk, CUDA_R_16F, m.cfg.n_embd,
                                 d_conv_in, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_k, CUDA_R_16F, kv_dim,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 kv_dim, block_size, m.cfg.n_embd,
                                 &alpha, l.wv, CUDA_R_16F, m.cfg.n_embd,
                                 d_conv_in, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_v, CUDA_R_16F, kv_dim,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        // RMS norm per head on Q and K
        int q_heads = m.cfg.n_head * block_size;
        rms_norm_per_head_kernel<<<(q_heads * 32 + 255) / 256, 256, 0, stream>>>(d_q, l.attn_q_norm, m.cfg.head_dim, m.cfg.n_head, block_size, m.cfg.eps);

        int k_heads = m.cfg.n_head_kv * block_size;
        rms_norm_per_head_kernel<<<(k_heads * 32 + 255) / 256, 256, 0, stream>>>(d_k, l.attn_k_norm, m.cfg.head_dim, m.cfg.n_head_kv, block_size, m.cfg.eps);

        // RoPE on Q and K
        int total_qk_heads = (m.cfg.n_head + m.cfg.n_head_kv) * block_size;
        neox_rope_kernel<<<(total_qk_heads * 64 + 255) / 256, 256, 0, stream>>>(d_q, d_k, m.cfg.n_head, m.cfg.n_head_kv, m.cfg.head_dim, pos0, block_size, m.cfg.rope_freq_base);

        if (il == 0) {
            check_nan("draft: l0 d_q", d_q, (size_t) m.cfg.n_embd * block_size, stream);
            check_nan("draft: l0 d_k", d_k, (size_t) kv_dim * block_size, stream);
            check_nan("draft: l0 d_v", d_v, (size_t) kv_dim * block_size, stream);
            check_nan_f32("draft: l0 attn_sinks", l.attn_sinks, m.cfg.n_head, stream);
        }

        // d) Attention over [K_cache, K_block]
        dim3 grid_attn(m.cfg.n_head, block_size);
        int total_keys = m.kv_len + block_size;
        size_t shmem_bytes = (size_t) total_keys * sizeof(float);
        dflash_attn_block_kernel<<<grid_attn, 256, shmem_bytes, stream>>>(d_q, d_k, d_v, c.k_cache, c.v_cache, l.attn_sinks,
                                                                         d_attn_out, m.cfg.head_dim, m.cfg.n_head, m.cfg.n_head_kv,
                                                                         m.kv_len, block_size, m.cfg.sliding_window);
        if (il == 0) check_nan("draft: layer 0 d_attn_out", d_attn_out, (size_t) m.cfg.n_embd * block_size, stream);

        // Attn out projection: half d_attn_out -> F32 d_attn_proj
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 m.cfg.n_embd, block_size, m.cfg.n_embd,
                                 &alpha, l.wo, CUDA_R_16F, m.cfg.n_embd,
                                 d_attn_out, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_attn_proj, CUDA_R_32F, m.cfg.n_embd,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        if (il == 0) check_nan_f32("draft: l0 d_attn_proj", d_attn_proj, (size_t) m.cfg.n_embd * block_size, stream);

        // Attn conv out (side = 1): F32 d_attn_proj -> F32 d_cur
        dflash_conv_f32_to_f32_kernel<<<(m.cfg.n_embd * block_size + 255) / 256, 256, 0, stream>>>(
            d_attn_proj, d_attn_dyn, l.attn_conv_base, d_cur, m.cfg.n_embd, 256, block_size, 1);
        if (il == 0) check_nan_f32("draft: l0 d_cur", d_cur, (size_t) m.cfg.n_embd * block_size, stream);

        // Residual: ffn_inp = cur + x (F32)
        add_residual_f32_kernel<<<(m.cfg.n_embd * block_size + 255) / 256, 256, 0, stream>>>(
            d_cur, d_x, m.cfg.n_embd * block_size);
        CUDA_CHECK(cudaMemcpyAsync(d_ffn_inp, d_cur, m.cfg.n_embd * block_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (il == 0) check_nan_f32("draft: l0 d_ffn_inp", d_ffn_inp, (size_t) m.cfg.n_embd * block_size, stream);

        // FFN norm: F32 d_ffn_inp -> half d_ffn_norm
        rms_norm_f32_to_half_kernel<<<block_size, 256, 0, stream>>>(
            d_ffn_inp, l.ffn_norm, d_ffn_norm, m.cfg.n_embd, block_size, m.cfg.eps);
        if (il == 0) check_nan("draft: l0 d_ffn_norm", d_ffn_norm, (size_t) m.cfg.n_embd * block_size, stream);

        // FFN conv in
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 1024, block_size, m.cfg.n_embd,
                                 &alpha, l.ffn_conv_proj, CUDA_R_16F, m.cfg.n_embd,
                                 d_ffn_norm, CUDA_R_16F, m.cfg.n_embd,
                                 &beta, d_ffn_dyn, CUDA_R_16F, 1024,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        dflash_conv_kernel<<<128, 256, 0, stream>>>(
            d_ffn_norm, d_ffn_dyn, l.ffn_conv_base, d_ffn_conv_in, m.cfg.n_embd, 256, block_size, 0);
        if (il == 0) check_nan("draft: l0 d_ffn_conv_in", d_ffn_conv_in, (size_t) m.cfg.n_embd * block_size, stream);

        // SwiGLU FFN: gate & up
        nvfp4_mm8(l.ffn_gate, l.ffn_gate_s, d_ffn_conv_in, d_gate, m.cfg.n_embd, m.cfg.n_ff, block_size, stream);
        if (il == 0) check_nan("draft: l0 d_gate", d_gate, (size_t) m.cfg.n_ff * block_size, stream);

        nvfp4_mm8(l.ffn_up, l.ffn_up_s, d_ffn_conv_in, d_up, m.cfg.n_embd, m.cfg.n_ff, block_size, stream);
        if (il == 0) check_nan("draft: l0 d_up", d_up, (size_t) m.cfg.n_ff * block_size, stream);

        // SiLU(gate) * up
        swiglu_kernel<<<(m.cfg.n_ff * block_size + 255) / 256, 256, 0, stream>>>(
            d_gate, d_up, d_gate, m.cfg.n_ff * block_size);
        if (il == 0) check_nan("draft: l0 swiglu", d_gate, (size_t) m.cfg.n_ff * block_size, stream);

        // FFN down: half d_gate -> F32 d_ffn_out
        nvfp4_mm8(l.ffn_down, l.ffn_down_s, d_gate, d_ffn_out, m.cfg.n_ff, m.cfg.n_embd, block_size, stream);
        if (il == 0) check_nan_f32("draft: l0 d_ffn_out", d_ffn_out, (size_t) m.cfg.n_embd * block_size, stream);

        // FFN conv out (side = 1): F32 d_ffn_out -> F32 d_ffn_conv_out
        dflash_conv_f32_to_f32_kernel<<<(m.cfg.n_embd * block_size + 255) / 256, 256, 0, stream>>>(
            d_ffn_out, d_ffn_dyn, l.ffn_conv_base, d_ffn_conv_out, m.cfg.n_embd, 256, block_size, 1);
        if (il == 0) check_nan_f32("draft: l0 d_ffn_conv_out", d_ffn_conv_out, (size_t) m.cfg.n_embd * block_size, stream);

        // Residual: x = ffn_conv_out + ffn_inp (F32)
        add_residual_f32_kernel<<<(m.cfg.n_embd * block_size + 255) / 256, 256, 0, stream>>>(
            d_ffn_conv_out, d_ffn_inp, m.cfg.n_embd * block_size);
        CUDA_CHECK(cudaMemcpyAsync(d_x, d_ffn_conv_out, m.cfg.n_embd * block_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (il == 0) check_nan_f32("draft: layer 0 d_x after FFN", d_x, (size_t) m.cfg.n_embd * block_size, stream);
        if (il == 4) check_nan_f32("draft: layer 4 d_x after FFN", d_x, (size_t) m.cfg.n_embd * block_size, stream);
    }

    // Output norm: F32 d_x -> F32 d_t_embd_f32
    rms_norm_f32_to_f32_kernel<<<block_size, 256, 0, stream>>>(
        d_x, m.d_output_norm, d_t_embd_f32, m.cfg.n_embd, block_size, m.cfg.eps);
    check_nan_f32("draft: d_t_embd_f32", d_t_embd_f32, (size_t) m.cfg.n_embd * block_size, stream);

    // 3. LM Head for draft positions 1..7 (7 tokens)
    float* d_logits_7 = d_logits_out ? d_logits_out : (m.d_work_float_1 + 262144);
    // Quantize 7 columns of activations to Q8_1
    strata::kernels::native_quantize_q8_1(d_t_embd_f32 + m.cfg.n_embd, m.d_q8_scratch, m.cfg.n_embd, n_draft, stream);
    // Matrix multiplication: Q8_0 weights * Q8_1 activations -> F32 logits [131072, 7]
    strata::kernels::native_mmvq(8, m.d_output_q8, m.d_q8_scratch, d_logits_7, m.cfg.n_embd, m.cfg.n_vocab, n_draft, stream);
    check_nan_f32("draft: d_logits_7", d_logits_7, (size_t) n_draft * m.cfg.n_vocab, stream);

    // 4. LiLiCorr Candidate Selection and Feature Computation
    find_top8_and_lse_kernel<<<7, 256, 0, stream>>>(
        m.d_lilicorr_logits_override ? m.d_lilicorr_logits_override : d_logits_7, m.cfg.n_vocab, m.d_lilicorr_cand_ids, m.d_lilicorr_cand_logits, m.d_lilicorr_lse, m.d_lilicorr_probs);

    // Feature MLP 1 + SiLU
    compute_feature_mlp1_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_cand_logits, m.d_lilicorr_lse, m.d_lilicorr_probs,
        m.d_lilicorr_feature_norm, m.d_lilicorr_feature_norm_b,
        m.d_lilicorr_feature_mlp_1, m.d_lilicorr_feature_mlp_1_b,
        m.d_lilicorr_feat_mlp1_out);

    // Feature MLP 3: GEMM [1024, 1024] x [1024, 56] -> [1024, 56]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 56, 1024,
                             &alpha, m.d_lilicorr_feature_mlp_3, CUDA_R_32F, 1024,
                             m.d_lilicorr_feat_mlp1_out, CUDA_R_32F, 1024,
                             &beta, m.d_lilicorr_feature_states, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // Token Projections:
    // Dequantize 56 candidate token embeddings from target Q8_0 token_embd into F32
    dequantize_q8_0_embed_kernel<<<dim3(m.cfg.n_embd / 32, 56), 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(m.d_token_embd_q8), m.d_lilicorr_cand_ids, 56, m.cfg.n_embd, m.d_lilicorr_embeds_f32);

    // Token projection GEMM: [1024, 4096] x [4096, 56] -> [1024, 56]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 56, m.cfg.n_embd,
                             &alpha, m.d_lilicorr_token_proj, CUDA_R_32F, m.cfg.n_embd,
                             m.d_lilicorr_embeds_f32, CUDA_R_32F, m.cfg.n_embd,
                             &beta, m.d_lilicorr_token_states, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    add_bias_f32_kernel<<<56, 256, 0, stream>>>(m.d_lilicorr_token_states, m.d_lilicorr_token_proj_b, 1024, 56);

    // Pass hidden projections: GEMM on d_t_embd_f32 slice for positions 1..7: [1024, 4096] x [4096, 7] -> [1024, 7]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 7, m.cfg.n_embd,
                             &alpha, m.d_lilicorr_pass_hidden_proj, CUDA_R_32F, m.cfg.n_embd,
                             d_t_embd_f32 + m.cfg.n_embd, CUDA_R_32F, m.cfg.n_embd,
                             &beta, m.d_lilicorr_pass_proj, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // Combine inputs into x: token_states + (pass_proj + pass_b) + (feature_states + feature_mlp3_b) + (slot_embd + rank_embd)
    combine_inputs_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_token_states, m.d_lilicorr_pass_proj, m.d_lilicorr_pass_hidden_proj_b,
        m.d_lilicorr_feature_states, m.d_lilicorr_feature_mlp_3_b,
        m.d_lilicorr_slot_embedding, m.d_lilicorr_rank_embedding,
        m.d_lilicorr_x);

    // Context projection on anchor context features: [1024, 4096] x [4096, 1] -> [1024, 1]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 1, m.cfg.n_embd,
                             &alpha, m.d_lilicorr_context_proj, CUDA_R_32F, m.cfg.n_embd,
                             m.d_anchor_state, CUDA_R_32F, m.cfg.n_embd,
                             &beta, m.d_lilicorr_anchor_state_proj, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    add_bias_f32_kernel<<<1, 256, 0, stream>>>(m.d_lilicorr_anchor_state_proj, m.d_lilicorr_context_proj_b, 1024, 1);

    // 5. LiLiCorr Transformer Layers (2 layers)
    for (int il = 0; il < m.cfg.lilicorr_layers; ++il) {
        auto& lw = m.lilicorr_layers[il];

        // RMS norm
        rms_norm_f32_scaled_kernel<<<56, 256, 0, stream>>>(
            m.d_lilicorr_x, lw.attn_norm, m.d_lilicorr_norm_x, 1024, 56, m.cfg.eps);

        // Attn in proj: [3072, 1024] x [1024, 56] -> [3072, 56]
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 3072, 56, 1024,
                                 &alpha, lw.attn_in_proj, CUDA_R_32F, 1024,
                                 m.d_lilicorr_norm_x, CUDA_R_32F, 1024,
                                 &beta, m.d_lilicorr_qkv, CUDA_R_32F, 3072,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        add_bias_f32_kernel<<<56, 256, 0, stream>>>(m.d_lilicorr_qkv, lw.attn_in_proj_b, 3072, 56);

        // Attention over 56 nodes across 8 heads with bias
        lilicorr_attn_kernel<<<8, 56, 0, stream>>>(
            m.d_lilicorr_qkv, m.d_lilicorr_attention_bias, m.d_lilicorr_attn_out);

        // Attn out proj: [1024, 1024] x [1024, 56] -> [1024, 56]
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 1024, 56, 1024,
                                 &alpha, lw.attn_out_proj, CUDA_R_32F, 1024,
                                 m.d_lilicorr_attn_out, CUDA_R_32F, 1024,
                                 &beta, m.d_lilicorr_attn_proj, CUDA_R_32F, 1024,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        add_bias_and_residual_kernel<<<56, 256, 0, stream>>>(
            m.d_lilicorr_attn_proj, lw.attn_out_proj_b, m.d_lilicorr_x, 1024, 56);

        // MLP RMS norm
        rms_norm_f32_scaled_kernel<<<56, 256, 0, stream>>>(
            m.d_lilicorr_x, lw.mlp_norm, m.d_lilicorr_ffn_norm_f32, 1024, 56, m.cfg.eps);

        // MLP 0: [2048, 1024] x [1024, 56] -> [2048, 56]
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 2048, 56, 1024,
                                 &alpha, lw.mlp_0, CUDA_R_32F, 1024,
                                 m.d_lilicorr_ffn_norm_f32, CUDA_R_32F, 1024,
                                 &beta, m.d_lilicorr_mlp_gate, CUDA_R_32F, 2048,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        silu_f32_in_place_kernel<<<56, 256, 0, stream>>>(
            m.d_lilicorr_mlp_gate, lw.mlp_0_b, 2048, 56);

        // MLP 2: [1024, 2048] x [2048, 56] -> [1024, 56]
        CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 1024, 56, 2048,
                                 &alpha, lw.mlp_2, CUDA_R_32F, 2048,
                                 m.d_lilicorr_mlp_gate, CUDA_R_32F, 2048,
                                 &beta, m.d_lilicorr_mlp_out, CUDA_R_32F, 1024,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        add_bias_and_residual_kernel<<<56, 256, 0, stream>>>(
            m.d_lilicorr_mlp_out, lw.mlp_2_b, m.d_lilicorr_x, 1024, 56);
    }

    // 6. Output Norms and Factor Heads
    rms_norm_f32_scaled_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_x, m.d_lilicorr_output_norm, m.d_lilicorr_norm_x, 1024, 56, m.cfg.eps);
    rms_norm_f32_scaled_kernel<<<1, 256, 0, stream>>>(
        m.d_lilicorr_anchor_state_proj, m.d_lilicorr_anchor_norm, m.d_lilicorr_norm_anchor, 1024, 1, m.cfg.eps);

    build_factor_in_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_norm_x, m.d_lilicorr_norm_anchor, m.d_lilicorr_factor_in, 1024, 56);

    // Factor input proj: [1024, 3072] x [3072, 56] -> [1024, 56]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 56, 3072,
                             &alpha, m.d_lilicorr_factor_input_proj, CUDA_R_32F, 3072,
                             m.d_lilicorr_factor_in, CUDA_R_32F, 3072,
                             &beta, m.d_lilicorr_factors, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    silu_f32_in_place_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_factors, m.d_lilicorr_factor_input_proj_b, 1024, 56);

    // Out head: [1024, 1024] x [1024, 56] -> [1024, 56]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 56, 1024,
                             &alpha, m.d_lilicorr_out_head, CUDA_R_32F, 1024,
                             m.d_lilicorr_factors, CUDA_R_32F, 1024,
                             &beta, m.d_lilicorr_out, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    add_bias_and_l2_norm_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_out, m.d_lilicorr_out_head_b, 1024, 56, 1e-4f);

    // In head: [1024, 1024] x [1024, 56] -> [1024, 56]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 56, 1024,
                             &alpha, m.d_lilicorr_in_head, CUDA_R_32F, 1024,
                             m.d_lilicorr_factors, CUDA_R_32F, 1024,
                             &beta, m.d_lilicorr_in, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    add_bias_and_l2_norm_kernel<<<56, 256, 0, stream>>>(
        m.d_lilicorr_in, m.d_lilicorr_in_head_b, 1024, 56, 1e-4f);

    // Anchor out head: [1024, 1024] x [1024, 1] -> [1024, 1]
    CUBLAS_CHECK(cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             1024, 1, 1024,
                             &alpha, m.d_lilicorr_anchor_out_head, CUDA_R_32F, 1024,
                             m.d_lilicorr_norm_anchor, CUDA_R_32F, 1024,
                             &beta, m.d_lilicorr_anchor_out, CUDA_R_32F, 1024,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    add_bias_and_l2_norm_kernel<<<1, 256, 0, stream>>>(
        m.d_lilicorr_anchor_out, m.d_lilicorr_anchor_out_head_b, 1024, 1, 1e-4f);

    // 7. Build Lattice
    build_lilicorr_lattice_kernel<<<8, 256, 0, stream>>>(
        m.d_lilicorr_cand_ids, m.d_lilicorr_anchor_out, m.d_lilicorr_in, m.d_lilicorr_out,
        m.d_lilicorr_lattice, 8.0f);

    // Copy lattice to host and greedily walk from predecessor 0
    CUDA_CHECK(cudaMemcpyAsync(m.h_lattice.data(), m.d_lilicorr_lattice, 8 * 4096 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> draft_tokens(n_draft);
    int32_t predecessor = 0;
    for (int i = 1; i <= n_draft; ++i) {
        const float* row = m.h_lattice.data() + (size_t) i * 4096;
        const float* scores = row + 8 + predecessor * 8;
        int best_k = 0;
        float best_s = scores[0];
        for (int k = 1; k < 8; ++k) {
            if (scores[k] > best_s) {
                best_s = scores[k];
                best_k = k;
            }
        }
        predecessor = best_k;
        draft_tokens[i - 1] = (int32_t) row[predecessor];
    }

    auto t_end = std::chrono::steady_clock::now();
    m.last_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    return draft_tokens;
}

} // namespace strata::nemotron_h

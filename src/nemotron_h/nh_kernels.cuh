// src/nemotron_h/nh_kernels.cuh - the Nemotron-H engine path's own kernels (src/nemotron_h/nh_engine.cu includes it
// once).  Every formula is llama.cpp's for the same tensors (src/models/nemotron-h.cpp, mamba-base.cpp,
// nemotron-h-moe.cpp; ggml-cpu ssm_conv / ssm_scan): the parity tools in tools/nemotron_h compare against it.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "strata/kernels/dp4a.hpp"

namespace nh {

constexpr int kThreads = 256;
constexpr int kNormThreads = 1024;
constexpr int kChunkCells = 256;   // decode attention split-K chunk
constexpr int kHD = 128;           // attention head dim (key_length == value_length == 128)
constexpr int kSsmP = 64;          // Mamba-2 head dim (ssm.inner_size / ssm.time_step_rank)
constexpr int kSsmN = 128;         // Mamba-2 state size
constexpr int kMaxTopK = 32;

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }   // ggml's

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int MaxWarps = 32>
__device__ __forceinline__ float block_reduce_sum(float val, float* sh) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();   // a previous call's sh[0] may still be read by other warps
    val = warp_reduce_sum(val);
    if (lane == 0) sh[warp] = val;
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    float warp_sum = (threadIdx.x < num_warps) ? sh[lane] : 0.0f;
    if (warp == 0) {
        warp_sum = warp_reduce_sum(warp_sum);
        if (lane == 0) sh[0] = warp_sum;
    }
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ void warp_max_pair(float& v, int& i) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, v, m);
        int oi = __shfl_xor_sync(0xffffffff, i, m);
        if (ov > v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
}

template <int B>
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
template <int B>
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// ------------------------------------------------------------------------------------------------ norms
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w, one block per row of n.
__global__ void __launch_bounds__(1024) k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    const int n4 = n / 4;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 xv = xr4[i];
        s += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_reduce_sum<32>(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    float4* yr4 = reinterpret_cast<float4*>(yr);
    const float4* wr4 = reinterpret_cast<const float4*>(w);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 xv = xr4[i];
        float4 wv = wr4[i];
        float4 out_v;
        out_v.x = xv.x * r * wv.x;
        out_v.y = xv.y * r * wv.y;
        out_v.z = xv.z * r * wv.z;
        out_v.w = xv.w * r * wv.w;
        yr4[i] = out_v;
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += a[r] (+ b[r]); h[r] = norm(x[r]) * w: RMS, or (layer_norm) LayerNorm without a bias (the MTP head's
/// shared_head_norm, ggml_norm then the weight).  One block per row of n.
__global__ void __launch_bounds__(1024) k_add_norm(float* x, const float* a, const float* b, const float* w, float* h, int n, float eps,
                           bool layer_norm) {
    __shared__ float sh[32];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* ar = a + (size_t) blockIdx.x * n;
    const float* br = b ? b + (size_t) blockIdx.x * n : nullptr;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f, s1 = 0.f;

    const int n4 = n / 4;
    float4* xr4 = reinterpret_cast<float4*>(xr);
    const float4* ar4 = reinterpret_cast<const float4*>(ar);
    const float4* br4 = reinterpret_cast<const float4*>(br);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 xv = xr4[i];
        float4 av = ar4[i];
        float4 v;
        if (br4) {
            float4 bv = br4[i];
            v.x = xv.x + (av.x + bv.x);
            v.y = xv.y + (av.y + bv.y);
            v.z = xv.z + (av.z + bv.z);
            v.w = xv.w + (av.w + bv.w);
        } else {
            v.x = xv.x + av.x;
            v.y = xv.y + av.y;
            v.z = xv.z + av.z;
            v.w = xv.w + av.w;
        }
        xr4[i] = v;
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        s1 += v.x + v.y + v.z + v.w;
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
        const float v = br ? xr[i] + (ar[i] + br[i]) : xr[i] + ar[i];
        xr[i] = v;
        s += v * v;
        s1 += v;
    }

    if (!layer_norm) {
        s = block_reduce_sum<32>(s, sh);
        const float r = rsqrtf(s / (float) n + eps);
        float4* hr4 = reinterpret_cast<float4*>(hr);
        const float4* wr4 = reinterpret_cast<const float4*>(w);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 xv = xr4[i];
            float4 wv = wr4[i];
            float4 out_v;
            out_v.x = xv.x * r * wv.x;
            out_v.y = xv.y * r * wv.y;
            out_v.z = xv.z * r * wv.z;
            out_v.w = xv.w * r * wv.w;
            hr4[i] = out_v;
        }
        for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
        return;
    }
    const float mean = block_reduce_sum<32>(s1, sh) / (float) n;
    float v2 = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 xv = xr4[i];
        float dx = xv.x - mean;
        float dy = xv.y - mean;
        float dz = xv.z - mean;
        float dw = xv.w - mean;
        v2 += dx * dx + dy * dy + dz * dz + dw * dw;
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        v2 += d * d;
    }
    v2 = block_reduce_sum<32>(v2, sh);
    const float r = rsqrtf(v2 / (float) n + eps);
    float4* hr4 = reinterpret_cast<float4*>(hr);
    const float4* wr4 = reinterpret_cast<const float4*>(w);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 xv = xr4[i];
        float4 wv = wr4[i];
        float4 out_v;
        out_v.x = (xv.x - mean) * r * wv.x;
        out_v.y = (xv.y - mean) * r * wv.y;
        out_v.z = (xv.z - mean) * r * wv.z;
        out_v.w = (xv.w - mean) * r * wv.w;
        hr4[i] = out_v;
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) hr[i] = (xr[i] - mean) * r * w[i];
}

// ------------------------------------------------------------------------------------------------ small matmuls
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in % 4 == 0: 2 rows per warp, sh_x cached.
__global__ void __launch_bounds__(256) k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    constexpr int kMaxIn4 = 1024;
    __shared__ float4 sh_x[kMaxIn4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_in4 = n_in / 4;
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    if (n_in4 <= kMaxIn4) {
        for (int i = threadIdx.x; i < n_in4; i += blockDim.x) {
            sh_x[i] = xr[i];
        }
        __syncthreads();
    }
    const int o0 = blockIdx.x * 16 + warp * 2;
    const int o1 = o0 + 1;
    const float4* wr0 = (o0 < n_out) ? (const float4*) (W + (size_t) o0 * n_in) : nullptr;
    const float4* wr1 = (o1 < n_out) ? (const float4*) (W + (size_t) o1 * n_in) : nullptr;
    float acc0 = 0.f, acc1 = 0.f;
    if (n_in4 <= kMaxIn4) {
        for (int i = lane; i < n_in4; i += 32) {
            const float4 b = sh_x[i];
            if (wr0) {
                const float4 a0 = wr0[i];
                acc0 += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
            }
            if (wr1) {
                const float4 a1 = wr1[i];
                acc1 += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
            }
        }
    } else {
        for (int i = lane; i < n_in4; i += 32) {
            const float4 b = xr[i];
            if (wr0) {
                const float4 a0 = wr0[i];
                acc0 += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
            }
            if (wr1) {
                const float4 a1 = wr1[i];
                acc1 += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
            }
        }
    }
    for (int m = 16; m > 0; m >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, m);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, m);
    }
    if (lane == 0) {
        if (o0 < n_out) y[(size_t) blockIdx.y * n_out + o0] = acc0;
        if (o1 < n_out) y[(size_t) blockIdx.y * n_out + o1] = acc1;
    }
}
/// FP32 out = x . W^T for the prompt's router (src/qwen35's k_sgemm_nt): (16 TM) x 64 tiles, split-K partials.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
__global__ void k_scale(float* x, float s, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
/// Same per-lane multiply as k_scale, four floats at a time.  Prompt outputs only (n4 = n/4, 16-byte aligned).
__global__ void k_scale4(float* x, float s, int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = reinterpret_cast<float4*>(x)[i];
    v.x *= s;
    v.y *= s;
    v.z *= s;
    v.w *= s;
    reinterpret_cast<float4*>(x)[i] = v;
}
/// x = relu(x)^2 in place (the shared expert's hidden).
__global__ void k_relu2(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = fmaxf(x[i], 0.f);
    x[i] = __fmul_rn(u, u);
}
/// y16 = relu(x)^2 as FP16 (the prompt experts' hidden for the down GEMM).
__global__ void k_relu2_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = fmaxf(x[i], 0.f);
    y[i] = __float2half(__fmul_rn(u, u));
}

// ------------------------------------------------------------------------------------------------ Mamba-2
/// The causal conv over T tokens (one launch): channel c's 3-entry history in registers, out = SiLU(the 4-tap sum +
/// bias) - ggml_ssm_conv's order (oldest tap first), then the bias add, then SiLU.  in rows `ld` floats apart.
/// `write`: the history after the last token is stored (a verify window leaves it for the commit).  grid C / 128.
__global__ void __launch_bounds__(128) k_conv(float* hist, const float* in, int ld, const float* w, const float* bias,
                                              float* out, int C, int T, bool write) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const float b = bias ? bias[c] : 0.f;
    for (int t = 0; t < T; ++t) {
        const float x = in[(size_t) t * ld + c];
        float sum = 0.f;
        sum += v0 * w0;
        sum += v1 * w1;
        sum += v2 * w2;
        sum += x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        const float y = sum + b;
        out[(size_t) t * C + c] = y / (1.0f + expf(-y));
    }
    if (!write) return;
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// Prompt conv (T > 64): the same 4-tap SiLU as k_conv, but one tile of TILE tokens per block so the grid fills the
/// GPU.  Tile 0 starts from `hist`; later tiles read the three inputs before the tile (the window is raw inputs).
/// History is not stored here — k_conv_save_hist does that when the caller wants it.  Decode stays on k_conv.
template <int TILE>
__global__ void __launch_bounds__(128) k_conv_prompt(const float* hist, const float* in, int ld, const float* w,
                                                    const float* bias, float* out, int C, int T) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    if (c >= C) return;
    const int t0 = (int) blockIdx.y * TILE;
    if (t0 >= T) return;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const float b = bias ? bias[c] : 0.f;
    float xm3, xm2, xm1;
    if (t0 == 0) {
        xm3 = hist[c * 3];
        xm2 = hist[c * 3 + 1];
        xm1 = hist[c * 3 + 2];
    } else {
        xm3 = in[(size_t) (t0 - 3) * ld + c];
        xm2 = in[(size_t) (t0 - 2) * ld + c];
        xm1 = in[(size_t) (t0 - 1) * ld + c];
    }
    const int t1 = t0 + TILE < T ? t0 + TILE : T;
    for (int t = t0; t < t1; ++t) {
        const float x = in[(size_t) t * ld + c];
        float sum = 0.f;
        sum += xm3 * w0;
        sum += xm2 * w1;
        sum += xm1 * w2;
        sum += x * w3;
        xm3 = xm2;
        xm2 = xm1;
        xm1 = x;
        const float y = sum + b;
        out[(size_t) t * C + c] = y / (1.0f + expf(-y));
    }
}
/// Last three conv inputs into the history, matching k_conv's write when T >= 3.
__global__ void __launch_bounds__(128) k_conv_save_hist(float* hist, const float* in, int ld, int C, int T) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    if (c >= C || T < 3) return;
    hist[c * 3] = in[(size_t) (T - 3) * ld + c];
    hist[c * 3 + 1] = in[(size_t) (T - 2) * ld + c];
    hist[c * 3 + 2] = in[(size_t) (T - 1) * ld + c];
}

/// The Mamba-2 selective scan over T tokens (ggml_ssm_scan, scalar A per head): per head h, dt = softplus(dt_raw +
/// dt_bias), dA = exp(dt * A); s = s * dA + B * (x * dt); y = C . s, then y += D * x.  State [H][P=64][N=128] fp32.
/// Block = one head, 256 threads: thread (p = tid / 4, q = tid % 4) keeps states n in [32 q, 32 q + 32) in
/// registers.  xc rows: x [0, H P) | B [H P, H P + G N) | C [.., + G N), `C_` floats apart.  dt rows `ld_dt` apart.
/// y (T, H P) may be null (a commit).  grid H.  Decode and short chunks; long prompt chunks use nh_ssm_scan.cuh.
__global__ void __launch_bounds__(256) k_ssm(float* state, const float* xc, int C_, const float* dt, int ld_dt,
                                            const float* dt_bias, const float* A, const float* D, float* y, int H,
                                            int G, int T, bool write) {
    __shared__ float sB[kSsmN], sC[kSsmN];
    const int head = blockIdx.x, tid = threadIdx.x;
    const int p = tid >> 2, q = tid & 3;
    const int g = head / (H / G);
    const int HP = H * kSsmP;
    float s[32];
    float* base = state + ((size_t) head * kSsmP + p) * kSsmN + q * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) s[i] = base[i];
    const float a = A[head], dd = D[head], db = dt_bias[head];
    for (int t = 0; t < T; ++t) {
        const float* row = xc + (size_t) t * C_;
        if (tid < kSsmN) sB[tid] = row[HP + g * kSsmN + tid];
        else sC[tid - kSsmN] = row[HP + G * kSsmN + g * kSsmN + tid - kSsmN];
        __syncthreads();
        const float dts = softplus(dt[(size_t) t * ld_dt + head] + db);
        const float dA = expf(dts * a);
        const float x = row[head * kSsmP + p];
        const float xdt = x * dts;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float v = s[i] * dA + sB[q * 32 + i] * xdt;
            s[i] = v;
            acc += v * sC[q * 32 + i];
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (y != nullptr && q == 0) y[(size_t) t * HP + head * kSsmP + p] = acc + x * dd;
        __syncthreads();   // sB / sC are the next token's
    }
    if (!write) return;
#pragma unroll
    for (int i = 0; i < 32; ++i) base[i] = s[i];
}

/// Same recurrence as k_ssm, coalesced.  One warp owns one (head, p) and 128 state lanes as float4 at n = 4 * lane,
/// so the [H][P][N] load and store is a single 512-byte warp transaction instead of a 128-byte stride.  A block is
/// 8 warps (8 values of p of one head); grid (P/8, H) = (8, H).  B and C (128 floats, shared by every p of the head)
/// are staged once per token.  P = 64 and N = 128, the geometry the engine accepts.  y's 128-term dot is a warp sum,
/// so its rounding can differ from k_ssm; each state element is still `s * dA + B * (x * dt)`.
__global__ void __launch_bounds__(256, 4) k_ssm_fast(float* state, const float* xc, int C_, const float* dt, int ld_dt,
                                                    const float* dt_bias, const float* A, const float* D, float* y,
                                                    int H, int G, int T, bool write) {
    __shared__ float4 sB[32], sC[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = (int) blockIdx.y;
    const int p = (int) blockIdx.x * 8 + warp;
    const int g = head / (H / G);
    const int HP = H * kSsmP;
    float4 st = *reinterpret_cast<float4*>(state + ((size_t) head * kSsmP + p) * kSsmN + lane * 4);
    const float a = A[head], dd = D[head], db = dt_bias[head];
    for (int t = 0; t < T; ++t) {
        const float* row = xc + (size_t) t * C_;
        if (warp == 0) {
            sB[lane] = *reinterpret_cast<const float4*>(row + HP + g * kSsmN + lane * 4);
            sC[lane] = *reinterpret_cast<const float4*>(row + HP + G * kSsmN + g * kSsmN + lane * 4);
        }
        __syncthreads();
        const float dts = softplus(dt[(size_t) t * ld_dt + head] + db);
        const float dA = expf(dts * a);
        const float x = row[head * kSsmP + p];
        const float xdt = x * dts;
        const float4 b = sB[lane], c = sC[lane];
        st.x = st.x * dA + b.x * xdt;
        st.y = st.y * dA + b.y * xdt;
        st.z = st.z * dA + b.z * xdt;
        st.w = st.w * dA + b.w * xdt;
        float acc = st.x * c.x + st.y * c.y + st.z * c.z + st.w * c.w;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (y != nullptr && lane == 0) y[(size_t) t * HP + head * kSsmP + p] = acc + x * dd;
        __syncthreads();
    }
    if (!write) return;
    *reinterpret_cast<float4*>(state + ((size_t) head * kSsmP + p) * kSsmN + lane * 4) = st;
}

/// y = rms_group(y_ssm * SiLU(z)) * w: the gate (ggml_swiglu_split(z, y)) then the grouped RMS norm over
/// group_size channels (Zamba2RMSNormGated).  z rows `ld_z` floats apart.  grid (n_group, T), block 256.
__global__ void k_gate_norm(const float* yin, const float* z, int ld_z, const float* w, float* out, int d_inner,
                            int group, float eps) {
    __shared__ float sh[kThreads];
    const int g = blockIdx.x, t = blockIdx.y;
    const float* yr = yin + (size_t) t * d_inner + g * group;
    const float* zr = z + (size_t) t * ld_z + g * group;
    float* o = out + (size_t) t * d_inner + g * group;
    float s = 0.f;
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = yr[i] * silu(zr[i]);
        o[i] = v;
        s += v * v;
    }
    s = block_sum<kThreads>(s, sh);
    const float r = rsqrtf(s / (float) group + eps);
    for (int i = threadIdx.x; i < group; i += blockDim.x) o[i] = o[i] * r * w[g * group + i];
}

// ------------------------------------------------------------------------------------------------ attention
/// k / v of T rows into the FP16 cache [kvh][ctx][HD] at positions *pos0 + t (no RoPE: Nemotron-H's attention is
/// NoPE).  grid (HKV, T), block HD.
__global__ void k_kv_store(const float* kx, const float* vx, __half* kc, __half* vc, int HKV, const int32_t* pos0,
                           int ctx) {
    const int kh = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = *pos0 + t;
    const size_t src = ((size_t) t * HKV + kh) * kHD + i, dst = ((size_t) kh * ctx + pos) * kHD + i;
    kc[dst] = __float2half(kx[src]);
    vc[dst] = __float2half(vx[src]);
}
/// Split-K decode attention: block (chunk c of 256 cells, query row = t * H + h) sees cells [0, *pos0 + t].  The
/// chunk's partial accumulator / max / sum.  grid (n_chunk, rows), block 256.
__global__ void k_attn_part(const float* q, const __half* kc, const __half* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    __shared__ float qs[kHD], sc[kChunkCells], sh[kThreads], part[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    if (j < kHD) qs[j] = q[((size_t) t * H + h) * kHD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const __half2* kp = (const __half2*) (kc + ((size_t) kvh * ctx + p) * kHD);
        float a = 0.f;
#pragma unroll 8
        for (int i = 0; i < kHD / 2; ++i) {
            const float2 kv = __half22float2(kp[i]);
            a += qs[2 * i] * kv.x + qs[2 * i + 1] * kv.y;
        }
        s = a * scale;
    }
    const float m = block_max<kThreads>(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum<kThreads>(e, sh);
    const int n = min(kChunkCells, npos - p0);
    // two halves of the cells, 128 output dims each
    const int d = j & (kHD - 1), half = j >> 7;
    const int k0 = half * (kChunkCells / 2), k1 = min(n, k0 + kChunkCells / 2);
    float acc = 0.f;
    const __half* vp = vc + ((size_t) kvh * ctx + p0) * kHD + d;
    for (int k = k0; k < k1; ++k) acc += sc[k] * __half2float(vp[(size_t) k * kHD]);
    part[j] = acc;
    __syncthreads();
    const size_t o = (size_t) row * n_chunk + c;
    if (half == 0) pacc[o * kHD + d] = part[d] + part[d + kHD];
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const int32_t* pos0, int t0, int H,
                             int n_chunk, float* out) {
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * kHD + d];
        den += w * pl[o];
    }
    out[((size_t) t * H + h) * kHD + d] = num / den;
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> FP16 probabilities over [0, pos0 + t].
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max<kThreads>(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum<kThreads>(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}

// ------------------------------------------------------------------------------------------------ MoE
/// The router (build_moe_ffn with SIGMOID gating, exp_probs_b, n_group 1): p = sigmoid(logit), the top K by p + bias
/// (lowest id on ties), weights = p / max(sum p, 2^-14) * scale.  One block of 512 threads per token (E <= 512).
/// The router (build_moe_ffn with SIGMOID gating, exp_probs_b, n_group 1): p = sigmoid(logit), the top K by p + bias
/// (lowest id on ties), weights = p / max(sum p, 2^-14) * scale.  One block of 512 threads per token (E <= 512).
__global__ void __launch_bounds__(512) k_route(const float* logits, const float* bias, int E, int K, float scale,
                                               bool norm, int32_t* ids, float* wts) {
    __shared__ float s_warp_v[16];
    __shared__ int s_warp_i[16];
    __shared__ int s_win;
    __shared__ float all_p[512];
    __shared__ float pick_p[kMaxTopK];
    __shared__ int pick_i[kMaxTopK];

    const int t = blockIdx.x, e = threadIdx.x;
    const float p = e < E ? sigm(logits[(size_t) t * E + e]) : 0.f;
    all_p[e] = p;
    float sel = e < E ? p + bias[e] : -INFINITY;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int k = 0; k < K; ++k) {
        float cur_v = sel;
        int cur_i = e;
        warp_max_pair(cur_v, cur_i);
        if (lane == 0) {
            s_warp_v[warp] = cur_v;
            s_warp_i[warp] = cur_i;
        }
        __syncthreads();
        if (warp == 0) {
            float wv = (lane < 16) ? s_warp_v[lane] : -INFINITY;
            int wi = (lane < 16) ? s_warp_i[lane] : 9999;
            warp_max_pair(wv, wi);
            if (lane == 0) {
                s_win = wi;
                pick_i[k] = wi;
                pick_p[k] = all_p[wi];
            }
        }
        __syncthreads();
        if (e == s_win) {
            sel = -INFINITY;
        }
    }
    if (e == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += pick_p[k];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int k = 0; k < K; ++k) {
            ids[(size_t) t * K + k] = pick_i[k];
            float w = norm ? pick_p[k] / sum : pick_p[k];
            wts[(size_t) t * K + k] = w * scale;
        }
    }
}
/// The routed entries with a VRAM slot (slot_of[x] >= 0) grouped per expert (ascending id), each group's blob at
/// blobs + slot * stride (src/qwen35's k_group_hy).  One block of 512 threads.
__global__ void __launch_bounds__(512) k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long stride, unsigned long long* grp_ptr,
                           int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512], s_slot[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) {
        cnt[x] = 0;
        s_slot[x] = slot_of[x];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int id = ids[e];
        if (s_slot[id] >= 0) atomicAdd(&cnt[id], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) s_slot[x] * stride);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int id = ids[e];
            if (s_slot[id] < 0) continue;
            const int p = off[id]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// out[t] = sum_k w[t][k] * parts[row(t, k)] (in pick order, FMA after the first product); row = prow[t K + k] (the
/// prompt's expert-sorted rows) or t K + k.  grid T, block 256.
__global__ void k_combine(const float* parts, const int32_t* prow, const float* wts, int K_, float* out, int n) {
    __shared__ const float* pk[kMaxTopK];
    const int t = blockIdx.x;
    if ((int) threadIdx.x < K_)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const int n4 = n / 4;
    if ((n & 3) == 0) {
        float4* out4 = reinterpret_cast<float4*>(out + (size_t) t * n);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4* p0 = reinterpret_cast<const float4*>(pk[0]);
            float4 p = p0[i];
            float4 acc;
            acc.x = wr[0] * p.x;
            acc.y = wr[0] * p.y;
            acc.z = wr[0] * p.z;
            acc.w = wr[0] * p.w;
#pragma unroll
            for (int k = 1; k < K_; ++k) {
                const float4* pk_ptr = reinterpret_cast<const float4*>(pk[k]);
                float4 pv = pk_ptr[i];
                acc.x = fmaf(wr[k], pv.x, acc.x);
                acc.y = fmaf(wr[k], pv.y, acc.y);
                acc.z = fmaf(wr[k], pv.z, acc.z);
                acc.w = fmaf(wr[k], pv.w, acc.w);
            }
            out4[i] = acc;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float acc = wr[0] * pk[0][i];
            for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
            out[(size_t) t * n + i] = acc;
        }
    }
}
/// The prompt's routed entries sorted by expert (src/qwen35's three kernels): bounds[x] .. bounds[x + 1] are expert
/// x's sorted rows; sorted row r holds entry dst[r] of token src[r]; inv[e] is entry e's sorted row.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur) {
    if (threadIdx.x != 0) return;
    int pos = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
    }
    bounds[n_expert] = pos;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;
}
/// dst[r] = FP16 of x[src[r]] (n values per row): the prompt experts' activations in expert order.
__global__ void k_gather_f16(const float* x, const int32_t* src, __half* dst, int rows, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * n) return;
    const int64_t r = i / n, c = i % n;
    dst[i] = __float2half(x[(size_t) src[r] * n + c]);
}

// ------------------------------------------------------------------------------------------------ the head
/// Greedy pick of a logits row (lowest id on ties), two stages (src/qwen35's k_argmax).
constexpr int kArgBlocks = 128;
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * kArgBlocks;
    pi += (size_t) blockIdx.y * kArgBlocks;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}

/// Grouped prompt experts: expert e's `n4` uint4s at src[e] + src_off land in dst's slot e (`stride4` uint4s apart).
/// Up and down of an IQ expert are 16-byte aligned, so one copy covers up and, with src_off = down_off, down.
__global__ void k_copy_experts(const uint8_t* const* __restrict__ src, int src_off, uint4* __restrict__ dst, int n4,
                               int stride4) {
    const int e = (int) blockIdx.y;
    const uint4* s = (const uint4*) (src[e] + src_off);
    uint4* d = dst + (size_t) e * (size_t) stride4;
    for (int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x; i < n4; i += (int) blockDim.x * (int) gridDim.x)
        d[i] = s[i];
}

/// One group's expert pointers and row bounds. Passed by value so the launch copies them; a reused pinned
/// buffer would still be unread when the host fills the next group (the H2D stays queued behind the multiply).
constexpr int kMmqStage = 32;
struct MmqStage {
    const uint8_t* ptr[kMmqStage];
    int32_t bounds[kMmqStage + 1];
    int n;
};
__global__ void k_stage_mmq(MmqStage s, const uint8_t** dst_ptr, int32_t* dst_bounds) {
    const int i = (int) threadIdx.x;
    if (i < s.n) dst_ptr[i] = s.ptr[i];
    if (i <= s.n) dst_bounds[i] = s.bounds[i];
}
}  // namespace nh



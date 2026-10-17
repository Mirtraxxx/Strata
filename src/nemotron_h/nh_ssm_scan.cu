// src/nemotron_h/nh_ssm_scan.cu - chunked SSD scan for Mamba-2 prompt chunks (see nh_ssm_scan.cuh).
// Formulas match llama.cpp ggml-cuda/ssm-scan.cu (SSM_SSD_CHUNK_SIZE 256) and k_ssm: dt_bias is fused into
// softplus, y includes D * x.  Intermediates X_dt / B_weighted / M are FP16 for tensor-core GEMMs; state, y,
// C_scaled and CB stay FP32.
#include "nh_ssm_scan.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>

namespace nh {

// Defined in nh_kernels.cuh (one TU per binary: nh_engine.cu, or nh_ssm_scan_test.cu).  Do not include that header
// here: nh_engine.cu already compiles it, and a second include would duplicate every kernel.
extern __global__ void k_ssm(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias,
                             const float* A, const float* D, float* y, int H, int G, int T, bool write);
extern __global__ void k_ssm_fast(float* state, const float* xc, int C_, const float* dt, int ld_dt,
                                  const float* dt_bias, const float* A, const float* D, float* y, int H, int G, int T,
                                  bool write);

namespace {

#define SSM_CK(x)                                                                                                      \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "nh_ssm_scan: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,         \
                         __LINE__);                                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define SSM_BLAS(x)                                                                                                    \
    do {                                                                                                               \
        cublasStatus_t s_ = (x);                                                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                                             \
            std::fprintf(stderr, "nh_ssm_scan: %s failed: cublas %d (%s:%d)\n", #x, (int) s_, __FILE__, __LINE__);     \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

constexpr int kBlk = 256;

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

// Softplus(dt + bias) and inclusive prefix of dt per head.  One block per head, slabs of 256 tokens.
__global__ void k_ssd_prepare_dt(const float* __restrict__ dt, int ld_dt, const float* __restrict__ dt_bias,
                                 float* __restrict__ dt_sp, float* __restrict__ cs, int H, int T) {
    __shared__ float sdata[kBlk];
    const int h = blockIdx.x, tid = threadIdx.x;
    const float db = dt_bias[h];
    float running = 0.f;
    for (int t0 = 0; t0 < T; t0 += kBlk) {
        const int t = t0 + tid;
        float sp = 0.f;
        if (t < T) {
            const float val = dt[(size_t) t * ld_dt + h] + db;
            sp = val > 20.f ? val : log1pf(expf(val));
            dt_sp[(size_t) t * H + h] = sp;
        }
        sdata[tid] = sp;
        __syncthreads();
        for (int off = 1; off < kBlk; off <<= 1) {
            const float add = (tid >= off) ? sdata[tid - off] : 0.f;
            __syncthreads();
            sdata[tid] += add;
            __syncthreads();
        }
        if (t < T) cs[(size_t) t * H + h] = running + sdata[tid];
        running += sdata[kBlk - 1];
        __syncthreads();
    }
}

// X_dt = x * dt, B_weighted = B * exp(A*(cs_last - cs_t)), C_scaled = C * exp(A*cs_t),
// M[t_out, t_in] = exp(A*(cs_out - cs_in)) * (C_tout . B_tin) for t_in <= t_out.
__global__ void k_ssd_pre_matmul(const float* __restrict__ cs, const float* __restrict__ dt_sp,
                                 const float* __restrict__ A, const float* __restrict__ xc, int C_,
                                 const float* __restrict__ CB, __half* __restrict__ X_dt, __half* __restrict__ B_w,
                                 float* __restrict__ C_s, __half* __restrict__ M, int chunk_len, int H, int G, int P,
                                 int N, int chunk_off) {
    const int h = blockIdx.y;
    const int g = h / (H / G);
    const float A_h = A[h];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float cs_base = chunk_off > 0 ? cs[(size_t) (chunk_off - 1) * H + h] : 0.f;
    const float cs_last = cs[(size_t) (chunk_off + chunk_len - 1) * H + h] - cs_base;
    const int HP = H * P;

    const int n_xdt = chunk_len * P;
    if (idx < n_xdt) {
        const int p = idx % P, t = idx / P;
        const float x_val = xc[(size_t) (chunk_off + t) * C_ + h * P + p];
        const float dt_val = dt_sp[(size_t) (chunk_off + t) * H + h];
        X_dt[(size_t) h * n_xdt + t * P + p] = __float2half(x_val * dt_val);
    }

    const int n_bw = N * chunk_len;
    if (idx < n_bw) {
        const int n = idx % N, t = idx / N;
        const float cs_t = cs[(size_t) (chunk_off + t) * H + h] - cs_base;
        const float* row = xc + (size_t) (chunk_off + t) * C_;
        const float B_val = row[HP + g * N + n];
        const float C_val = row[HP + G * N + g * N + n];
        B_w[(size_t) h * n_bw + t * N + n] = __float2half(B_val * expf(A_h * (cs_last - cs_t)));
        C_s[(size_t) h * n_bw + t * N + n] = C_val * expf(A_h * cs_t);
    }

    const int n_M = chunk_len * chunk_len;
    if (idx < n_M) {
        const int t_out = idx % chunk_len, t_in = idx / chunk_len;
        half val;
        if (t_in <= t_out) {
            const float cs_out = cs[(size_t) (chunk_off + t_out) * H + h] - cs_base;
            const float cs_in = cs[(size_t) (chunk_off + t_in) * H + h] - cs_base;
            const float decay = expf(A_h * (cs_out - cs_in));
            const float* CB_g = CB + (size_t) g * n_M;
            val = __float2half(decay * CB_g[t_out + t_in * chunk_len]);
        } else {
            val = __float2half(0.f);
        }
        M[(size_t) h * n_M + t_in * chunk_len + t_out] = val;
    }
}

__global__ void k_ssd_scale_state(float* __restrict__ s, const float* __restrict__ cs, const float* __restrict__ A,
                                  int N, int P, int H, int chunk_off, int chunk_len) {
    const int h = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nsp = N * P;
    if (idx >= nsp) return;
    const float A_h = A[h];
    const float cs_base = chunk_off > 0 ? cs[(size_t) (chunk_off - 1) * H + h] : 0.f;
    const float cs_last = cs[(size_t) (chunk_off + chunk_len - 1) * H + h] - cs_base;
    s[(size_t) h * nsp + idx] *= expf(A_h * cs_last);
}

__global__ void k_ssd_dx(const float* __restrict__ xc, int C_, const float* __restrict__ D, float* __restrict__ y,
                         int H, int P, int T) {
    const int HP = H * P;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * HP) return;
    const int t = (int) (i / HP), hp = (int) (i % HP);
    const int h = hp / P, p = hp % P;
    y[i] += D[h] * xc[(size_t) t * C_ + h * P + p];
}

void ssm_ws_ensure(SsmWs& ws, int H, int P, int N, int G, int T) {
    const int L = kSsmSsdChunk;
    const int TT = std::max(T, 1);
    size_t off = 0;
    auto take = [&](size_t bytes) {
        const size_t a = align256(bytes);
        const size_t o = off;
        off += a;
        return o;
    };
    const size_t o_dt = take((size_t) TT * H * sizeof(float));
    const size_t o_cs = take((size_t) TT * H * sizeof(float));
    const size_t o_cb = take((size_t) L * L * G * sizeof(float));
    const size_t o_x = take((size_t) L * P * H * sizeof(__half));
    const size_t o_bw = take((size_t) N * L * H * sizeof(__half));
    const size_t o_csd = take((size_t) N * L * H * sizeof(float));
    const size_t o_m = take((size_t) L * L * H * sizeof(__half));
    const size_t o_st = take((size_t) H * P * N * sizeof(float));
    if (off > ws.cap) {
        if (ws.blob) SSM_CK(cudaFree(ws.blob));
        SSM_CK(cudaMalloc(&ws.blob, off));
        ws.cap = off;
    }
    char* b = (char*) ws.blob;
    ws.dt_sp = (float*) (b + o_dt);
    ws.cs = (float*) (b + o_cs);
    ws.CB = (float*) (b + o_cb);
    ws.X_dt = b + o_x;
    ws.B_w = b + o_bw;
    ws.C_s = (float*) (b + o_csd);
    ws.M = b + o_m;
    ws.s_tmp = (float*) (b + o_st);
}

}  // namespace

void ssm_ws_reserve(SsmWs& ws, int H, int P, int N, int G, int T) { ssm_ws_ensure(ws, H, P, N, G, T); }

void ssm_ws_free(SsmWs& ws) {
    if (ws.blob) cudaFree(ws.blob);
    ws = SsmWs{};
}

void ssm_scan_seq(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias,
                  const float* A, const float* D, float* y, int H, int G, int T, bool write, cudaStream_t st) {
    if (T <= 0) return;
    // k_ssm_fast: 8 warps cover 8 values of p, grid (8, H).  k_ssm (1 block / head) stays in the header unused.
    k_ssm_fast<<<dim3(8, H), 256, 0, st>>>(state, xc, C_, dt, ld_dt, dt_bias, A, D, y, H, G, T, write);
}

void ssm_scan_ssd(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias,
                  const float* A, const float* D, float* y, int H, int G, int P, int N, int T, bool write, SsmWs& ws,
                  cudaStream_t st, cublasHandle_t blas) {
    if (T <= 0) return;
    ssm_ws_ensure(ws, H, P, N, G, T);
    float* s_cur = state;
    if (!write) {
        SSM_CK(cudaMemcpyAsync(ws.s_tmp, state, (size_t) H * P * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
        s_cur = ws.s_tmp;
    }
    k_ssd_prepare_dt<<<H, kBlk, 0, st>>>(dt, ld_dt, dt_bias, ws.dt_sp, ws.cs, H, T);

    SSM_BLAS(cublasSetStream(blas, st));
    const float one = 1.f, zero = 0.f;
    const int d_inner = H * P;
    const int n_chunks = (T + kSsmSsdChunk - 1) / kSsmSsdChunk;
    auto* X_dt = (__half*) ws.X_dt;
    auto* B_w = (__half*) ws.B_w;
    auto* M = (__half*) ws.M;

    for (int k = 0; k < n_chunks; ++k) {
        const int off = k * kSsmSsdChunk;
        const int L = std::min(kSsmSsdChunk, T - off);
        const int n_M = L * L;
        const float* C_src = xc + (size_t) off * C_ + (size_t) H * P + (size_t) G * N;
        const float* B_src = xc + (size_t) off * C_ + (size_t) H * P;

        SSM_BLAS(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, N, &one, C_src, CUDA_R_32F, C_, N,
                                           B_src, CUDA_R_32F, C_, N, &zero, ws.CB, CUDA_R_32F, L, (long long) n_M, G,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        int64_t work = (int64_t) L * P;
        work = std::max(work, (int64_t) N * L);
        work = std::max(work, (int64_t) n_M);
        k_ssd_pre_matmul<<<dim3((unsigned) ((work + kBlk - 1) / kBlk), (unsigned) H), kBlk, 0, st>>>(
            ws.cs, ws.dt_sp, A, xc, C_, ws.CB, X_dt, B_w, ws.C_s, M, L, H, G, P, N, off);

        if (y) {
            float* y_ch = y + (size_t) off * d_inner;
            SSM_BLAS(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, L, N, &one, s_cur, CUDA_R_32F, N,
                                               (long long) N * P, ws.C_s, CUDA_R_32F, N, (long long) N * L, &zero, y_ch,
                                               CUDA_R_32F, d_inner, P, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            SSM_BLAS(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_T, P, L, L, &one, X_dt, CUDA_R_16F, P,
                                               (long long) L * P, M, CUDA_R_16F, L, n_M, &one, y_ch, CUDA_R_32F,
                                               d_inner, P, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }

        k_ssd_scale_state<<<dim3((unsigned) ((N * P + kBlk - 1) / kBlk), (unsigned) H), kBlk, 0, st>>>(
            s_cur, ws.cs, A, N, P, H, off, L);
        SSM_BLAS(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_T, N, P, L, &one, B_w, CUDA_R_16F, N,
                                           (long long) N * L, X_dt, CUDA_R_16F, P, (long long) L * P, &one, s_cur,
                                           CUDA_R_32F, N, (long long) N * P, H, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT));
    }
    if (y) {
        const size_t n = (size_t) T * H * P;
        k_ssd_dx<<<(unsigned) ((n + kBlk - 1) / kBlk), kBlk, 0, st>>>(xc, C_, D, y, H, P, T);
    }
}

void ssm_scan(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias, const float* A,
              const float* D, float* y, int H, int G, int P, int N, int T, bool write, SsmWs& ws, cudaStream_t st,
              cublasHandle_t blas) {
    if (T <= kSsmSsdMinTok && P == 64 && N == 128) {
        ssm_scan_seq(state, xc, C_, dt, ld_dt, dt_bias, A, D, y, H, G, T, write, st);
        return;
    }
    ssm_scan_ssd(state, xc, C_, dt, ld_dt, dt_bias, A, D, y, H, G, P, N, T, write, ws, st, blas);
}

}  // namespace nh

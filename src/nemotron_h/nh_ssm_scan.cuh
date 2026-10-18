// src/nemotron_h/nh_ssm_scan.cuh - Mamba-2 selective scan: the sequential kernel (nh_kernels.cuh k_ssm) for
// decode and short chunks, and a chunked SSD path (Mamba-2 paper / llama.cpp ssm-scan.cu / mamba_chunk_scan) for
// long prompt chunks.  Same recurrence as k_ssm: dt = softplus(dt_raw + dt_bias), dA = exp(dt * A),
// s = s * dA + B * (x * dt), y = C . s + D * x.  Heads, head_dim, state and groups are arguments (Bolt:
// 128 x 64, state 128, 8 groups).  State layout [H][P][N], xc rows x | B | C.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nh {

// Below this many tokens the sequential kernel has less launch/workspace cost than SSD.  llama.cpp uses 128.
constexpr int kSsmSsdMinTok = 128;
constexpr int kSsmSsdChunk = 256;

/// Growable scratch for the SSD path (dt prefix, CB / M, X_dt, B_weighted, C_scaled, a state copy when !write).
struct SsmWs {
    void* blob = nullptr;
    size_t cap = 0;
    float *dt_sp = nullptr, *cs = nullptr, *CB = nullptr, *C_s = nullptr, *s_tmp = nullptr;
    void *X_dt = nullptr, *B_w = nullptr, *M = nullptr;   // __half, tensor-core GEMM inputs
};

/// Allocates the SSD scratch for chunks of up to T tokens now (so VRAM sizing at load counts it).
void ssm_ws_reserve(SsmWs& ws, int H, int P, int N, int G, int T);
void ssm_ws_free(SsmWs& ws);

/// Decode / short chunks: k_ssm_fast (coalesced state, 8 warps per head-group of p).  P=64, N=128.
void ssm_scan_seq(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias,
                  const float* A, const float* D, float* y, int H, int G, int T, bool write, cudaStream_t st);

/// Chunked SSD: intra-chunk dual (causal M = decay (.) C B^T) and inter-chunk state via GEMMs.  Generic H, P, N, G.
void ssm_scan_ssd(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias,
                  const float* A, const float* D, float* y, int H, int G, int P, int N, int T, bool write, SsmWs& ws,
                  cudaStream_t st, cublasHandle_t blas);

/// Decode / short: sequential.  Long chunks: SSD.  P and N must match k_ssm when T is short (64 and 128).
void ssm_scan(float* state, const float* xc, int C_, const float* dt, int ld_dt, const float* dt_bias, const float* A,
              const float* D, float* y, int H, int G, int P, int N, int T, bool write, SsmWs& ws, cudaStream_t st,
              cublasHandle_t blas);

}  // namespace nh

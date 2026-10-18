// src/nemotron_h/nh_ssm_scan_test.cu - the chunked SSD Mamba-2 scan against the current sequential kernel (k_ssm).
// Synthetic inputs of Bolt's shape (128 heads x 64, state 128, 8 groups), a non-zero initial state, no model.
//
//   nh_ssm_scan_test            seq vs SSD at T = 1, 7, 64, 1000, 8192 (max |d|/max |v| < 1e-3 on y and state)
//                               then times both paths at T = 1, 2048 and 8192 (ms / layer)
#include "nh_kernels.cuh"
#include "nh_ssm_scan.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int H = 128, P = 64, N = 128, G = 8;
constexpr int HP = H * P;
constexpr int C_ = HP + 2 * G * N;   // 10240: x | B | C
constexpr double kTol = 1e-3;

// max |a-b| / max(|b|, eps): same scale as src/prefill/gdn_rec_parity.cu (chunked vs sequential scan).
double max_rel(const float* a, const float* b, size_t n, double* abs_out) {
    double ab = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ab = std::max(ab, std::abs((double) a[i] - (double) b[i]));
        mb = std::max(mb, std::abs((double) b[i]));
    }
    if (abs_out) *abs_out = ab;
    return ab / std::max(mb, 1e-6);
}

float softplus_h(float v) { return v > 20.f ? v : log1pf(expf(v)); }

void cpu_scan(const float* S0, const float* xc, const float* dt, const float* dt_b, const float* A, const float* D,
              float* y, float* S, int T) {
    std::memcpy(S, S0, (size_t) H * P * N * sizeof(float));
    for (int t = 0; t < T; ++t) {
        const float* row = xc + (size_t) t * C_;
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / G);
            const float dts = softplus_h(dt[t * H + h] + dt_b[h]);
            const float dA = expf(dts * A[h]);
            const float* B = row + HP + g * N;
            const float* C = row + HP + G * N + g * N;
            for (int p = 0; p < P; ++p) {
                const float x = row[h * P + p];
                const float xdt = x * dts;
                float acc = 0.f;
                float* s = S + ((size_t) h * P + p) * N;
                for (int n = 0; n < N; ++n) {
                    const float v = s[n] * dA + B[n] * xdt;
                    s[n] = v;
                    acc += v * C[n];
                }
                y[(size_t) t * HP + h * P + p] = acc + x * D[h];
            }
        }
    }
}

struct Dev {
    float *state = nullptr, *xc = nullptr, *dt = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *y = nullptr;
    void alloc(int T) {
        ck(cudaMalloc(&state, (size_t) H * P * N * sizeof(float)), "state");
        ck(cudaMalloc(&xc, (size_t) T * C_ * sizeof(float)), "xc");
        ck(cudaMalloc(&dt, (size_t) T * H * sizeof(float)), "dt");
        ck(cudaMalloc(&dt_b, (size_t) H * sizeof(float)), "dt_b");
        ck(cudaMalloc(&A, (size_t) H * sizeof(float)), "A");
        ck(cudaMalloc(&D, (size_t) H * sizeof(float)), "D");
        ck(cudaMalloc(&y, (size_t) T * HP * sizeof(float)), "y");
    }
    void free() {
        cudaFree(state);
        cudaFree(xc);
        cudaFree(dt);
        cudaFree(dt_b);
        cudaFree(A);
        cudaFree(D);
        cudaFree(y);
        *this = Dev{};
    }
};

void fill_host(int T, std::vector<float>& state, std::vector<float>& xc, std::vector<float>& dt, std::vector<float>& dt_b,
               std::vector<float>& A, std::vector<float>& D, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    state.assign((size_t) H * P * N, 0.f);
    xc.assign((size_t) T * C_, 0.f);
    dt.assign((size_t) T * H, 0.f);
    dt_b.assign(H, 0.f);
    A.assign(H, 0.f);
    D.assign(H, 0.f);
    for (auto& v : state) v = nd(rng) * 0.2f;
    for (auto& v : xc) v = nd(rng) * 0.5f;
    for (auto& v : dt) v = nd(rng);
    for (int h = 0; h < H; ++h) {
        dt_b[h] = nd(rng) * 0.2f;
        A[h] = -std::exp(nd(rng) * 0.5f - 0.5f);   // negative, as in the model
        D[h] = nd(rng) * 0.5f;
    }
}

int run_t(int T, cublasHandle_t blas, double* y_rel_out, double* s_rel_out) {
    std::vector<float> hS, hX, hDt, hDb, hA, hD;
    fill_host(T, hS, hX, hDt, hDb, hA, hD, 7u + (uint32_t) T);
    Dev a, b;
    a.alloc(T);
    b.alloc(T);
    auto upload = [&](Dev& d) {
        ck(cudaMemcpy(d.state, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice), "S");
        ck(cudaMemcpy(d.xc, hX.data(), hX.size() * 4, cudaMemcpyHostToDevice), "xc");
        ck(cudaMemcpy(d.dt, hDt.data(), hDt.size() * 4, cudaMemcpyHostToDevice), "dt");
        ck(cudaMemcpy(d.dt_b, hDb.data(), hDb.size() * 4, cudaMemcpyHostToDevice), "dt_b");
        ck(cudaMemcpy(d.A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice), "A");
        ck(cudaMemcpy(d.D, hD.data(), hD.size() * 4, cudaMemcpyHostToDevice), "D");
        ck(cudaMemset(d.y, 0, (size_t) T * HP * 4), "y0");
    };
    upload(a);
    upload(b);
    nh::ssm_scan_seq(a.state, a.xc, C_, a.dt, H, a.dt_b, a.A, a.D, a.y, H, G, T, true, 0);
    nh::SsmWs ws;
    nh::ssm_scan_ssd(b.state, b.xc, C_, b.dt, H, b.dt_b, b.A, b.D, b.y, H, G, P, N, T, true, ws, 0, blas);
    ck(cudaDeviceSynchronize(), "sync");
    std::vector<float> ya((size_t) T * HP), yb((size_t) T * HP), sa((size_t) H * P * N), sb((size_t) H * P * N);
    ck(cudaMemcpy(ya.data(), a.y, ya.size() * 4, cudaMemcpyDeviceToHost), "ya");
    ck(cudaMemcpy(yb.data(), b.y, yb.size() * 4, cudaMemcpyDeviceToHost), "yb");
    ck(cudaMemcpy(sa.data(), a.state, sa.size() * 4, cudaMemcpyDeviceToHost), "sa");
    ck(cudaMemcpy(sb.data(), b.state, sb.size() * 4, cudaMemcpyDeviceToHost), "sb");
    double yabs = 0, sabs = 0;
    const double yr = max_rel(yb.data(), ya.data(), ya.size(), &yabs);
    const double sr = max_rel(sb.data(), sa.data(), sa.size(), &sabs);
    std::printf("T=%-5d  y_rel=%.3e y_abs=%.3e  s_rel=%.3e s_abs=%.3e  %s\n", T, yr, yabs, sr, sabs,
                (yr < kTol && sr < kTol) ? "OK" : "FAIL");
    if (T <= 7) {
        std::vector<float> yc((size_t) T * HP), sc((size_t) H * P * N);
        cpu_scan(hS.data(), hX.data(), hDt.data(), hDb.data(), hA.data(), hD.data(), yc.data(), sc.data(), T);
        double ycs = 0, scs = 0;
        std::printf("       seq-vs-cpu y_rel=%.3e s_rel=%.3e  ssd-vs-cpu y_rel=%.3e  y0 seq=%.5f ssd=%.5f cpu=%.5f\n",
                    max_rel(ya.data(), yc.data(), yc.size(), &ycs), max_rel(sa.data(), sc.data(), sc.size(), &scs),
                    max_rel(yb.data(), yc.data(), yc.size(), nullptr), ya[0], yb[0], yc[0]);
    }
    if (y_rel_out) *y_rel_out = std::max(*y_rel_out, yr);
    if (s_rel_out) *s_rel_out = std::max(*s_rel_out, sr);
    nh::ssm_ws_free(ws);
    a.free();
    b.free();
    return (yr < kTol && sr < kTol) ? 0 : 1;
}

float median_ms(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void bench_t(int T, cublasHandle_t blas, int warmup, int iters) {
    std::vector<float> hS, hX, hDt, hDb, hA, hD;
    fill_host(T, hS, hX, hDt, hDb, hA, hD, 11u);
    Dev d;
    d.alloc(T);
    ck(cudaMemcpy(d.xc, hX.data(), hX.size() * 4, cudaMemcpyHostToDevice), "xc");
    ck(cudaMemcpy(d.dt, hDt.data(), hDt.size() * 4, cudaMemcpyHostToDevice), "dt");
    ck(cudaMemcpy(d.dt_b, hDb.data(), hDb.size() * 4, cudaMemcpyHostToDevice), "dt_b");
    ck(cudaMemcpy(d.A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice), "A");
    ck(cudaMemcpy(d.D, hD.data(), hD.size() * 4, cudaMemcpyHostToDevice), "D");
    nh::SsmWs ws;
    auto restore = [&] {
        ck(cudaMemcpy(d.state, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice), "S");
        ck(cudaMemset(d.y, 0, (size_t) T * HP * 4), "y0");
    };
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev0");
    ck(cudaEventCreate(&b), "ev1");
    auto time_path = [&](bool ssd) {
        for (int i = 0; i < warmup; ++i) {
            restore();
            if (ssd) nh::ssm_scan_ssd(d.state, d.xc, C_, d.dt, H, d.dt_b, d.A, d.D, d.y, H, G, P, N, T, true, ws, 0, blas);
            else nh::ssm_scan_seq(d.state, d.xc, C_, d.dt, H, d.dt_b, d.A, d.D, d.y, H, G, T, true, 0);
        }
        ck(cudaDeviceSynchronize(), "warm");
        std::vector<float> ms(iters);
        for (int i = 0; i < iters; ++i) {
            restore();
            ck(cudaEventRecord(a, 0), "rec0");
            if (ssd) nh::ssm_scan_ssd(d.state, d.xc, C_, d.dt, H, d.dt_b, d.A, d.D, d.y, H, G, P, N, T, true, ws, 0, blas);
            else nh::ssm_scan_seq(d.state, d.xc, C_, d.dt, H, d.dt_b, d.A, d.D, d.y, H, G, T, true, 0);
            ck(cudaEventRecord(b, 0), "rec1");
            ck(cudaEventSynchronize(b), "sync");
            ck(cudaEventElapsedTime(&ms[i], a, b), "el");
        }
        return median_ms(ms);
    };
    const float seq = time_path(false);
    const float ssd = time_path(true);
    std::printf("T=%-5d  seq=%.3f ms/layer  ssd=%.3f ms/layer  speedup=%.2fx\n", T, seq, ssd, seq / std::max(ssd, 1e-6f));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    nh::ssm_ws_free(ws);
    d.free();
}

}  // namespace

int main(int, char**) {
    int dev = 0;
    ck(cudaGetDevice(&dev), "dev");
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, dev), "prop");
    std::printf("nh_ssm_scan_test: %s  sequential path k_ssm_fast (coalesced state, 8 warps per head)\n", p.name);
    std::printf("Bolt shape: H=%d P=%d N=%d G=%d conv_ch=%d  SSD chunk=%d  min tok for dispatch=%d\n", H, P, N, G, C_,
                nh::kSsmSsdChunk, nh::kSsmSsdMinTok);
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 2;
    }
    const int ts[] = {1, 7, 64, 1000, 8192};
    double yworst = 0, sworst = 0;
    int fail = 0;
    for (int T : ts) fail += run_t(T, blas, &yworst, &sworst);
    std::printf("max relative error  y=%.3e  state=%.3e  tol=%.0e  %s\n", yworst, sworst, kTol,
                fail ? "FAIL" : "PASS");
    std::printf("microbench (median of 20 after 5 warmup, one scan = one layer):\n");
    std::printf("T=1 decode (median of 50 after 10 warmup):\n");
    bench_t(1, blas, 10, 50);
    bench_t(2048, blas, 5, 20);
    bench_t(8192, blas, 5, 20);
    cublasDestroy(blas);
    return fail ? 1 : 0;
}

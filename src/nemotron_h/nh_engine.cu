// src/nemotron_h/nh_engine.cu - strata-nh: the engine path for NVIDIA Nemotron-H MoE (GGUF architecture
// nemotron_h_moe: Nemotron 3.5 Super VL 120B-A12B, unsloth's GGUFs).  A driver of its own beside Flash-Next's and
// Qwen3.6's (their paths stay untouched), built from Strata's kernels plus its own (nh_kernels.cuh):
//
//   * 88 flat sublayers, each x += op(rms(x) * attn_norm): 40 Mamba-2 mixers, 8 attention mixers, 40 MoE blocks
//     (the GGUF's per-layer head_count_kv / feed_forward_length say which), then output_norm and the head;
//   * Mamba-2: in_proj (z | xBC | dt), the causal conv + bias + SiLU, the selective scan (128 heads x 64, state 128,
//     8 groups), y + D x, the SiLU(z) gate and the grouped RMS norm (1024 per group), out_proj;
//   * attention: 32 query heads over 2 KV heads, head 128, NO positional encoding (NoPE), an FP16 KV cache;
//   * MoE: the sigmoid router with the selection bias (512 experts, top 22, normalized, x 5.0), the latent
//     projections 4096 -> 1024 -> 4096 around the routed experts down(relu(up x)^2), the shared expert
//     down(relu(up h)^2) on the full width;
//   * the routed experts are 78 GB: the hybrid mode always - per MoE layer the same number of VRAM slots, a batch of
//     up to --hybrid-rows rows runs its resident experts on the GPU and the others on the CPU pool (ggml-cpu, read
//     straight from the mapped GGUF), a bigger prompt chunk streams every non-resident expert through a ring of
//     VRAM buffers (a copy thread feeding a copy stream) into grouped IQ MMQ (the weights stay quantized);
//   * the GGUF's MTP block (blk.88: eh_proj over [enorm(e) | hnorm(h)], attention, MoE, the LayerNorm head norm)
//     drafts for greedy requests (--mtp N), verified by the main model (the Mamba state is committed afterwards);
//   * or DFlash (--dflash GGUF, Bolt): the LiLiCorr block drafter (nh_dflash) reads the residual entering 6 target
//     sublayers and drafts up to 7 tokens a step, verified the same way;
//   * images (GENI): strata-vision's rows (llama.cpp mtmd, the mmproj's RADIO tower + projector) in place of the
//     prompt's <image> tokens (the server writes <img> <image> x n </img>).
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / GENI / T / DONE / STOP / PP, and VRAM for
// POST /v1/vram), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/nemotron_h's parity interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/nemotron_h/cpu_experts.hpp"
#include "strata/nemotron_h/engine.hpp"
#include "strata/prefill/gemm.hpp"
#if defined(STRATA_PREFILL_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#endif

#include "nh_dflash.hpp"
#include "nh_kernels.cuh"
#include "nh_ssm_scan.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

namespace K = strata::kernels;
#if defined(STRATA_PREFILL_MMQ)
namespace MMQ = strata::prefill::mmq;
#endif
using namespace nh;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-nh: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

/// Free and total VRAM in bytes: cudaMemGetInfo capped by NVML's device-wide free memory (on Windows WDDM,
/// cudaMemGetInfo reports this process's WDDM budget room and does not subtract all of the desktop's / other
/// processes' allocations that nvidia-smi counts).
void vram_info(size_t& fr, size_t& tot) {
    CK(cudaMemGetInfo(&fr, &tot));
    struct NvmlMem {
        unsigned long long total, free, used;
    };
    struct Nvml {
        using InitFn = int (*)();
        using ByPciFn = int (*)(const char*, void**);
        using ByIdxFn = int (*)(unsigned int, void**);
        using MemFn = int (*)(void*, NvmlMem*);
        ByPciFn by_pci = nullptr;
        ByIdxFn by_idx = nullptr;
        MemFn mem = nullptr;
        bool ok = false;
        Nvml() {
#if defined(_WIN32)
            HMODULE lib = LoadLibraryA("nvml.dll");
            if (!lib) {
                const char* pf = std::getenv("ProgramFiles");
                const std::string p = std::string(pf ? pf : "C:\\Program Files") + "\\NVIDIA Corporation\\NVSMI\\nvml.dll";
                lib = LoadLibraryA(p.c_str());
            }
            auto sym = [&](const char* n) -> void* { return lib ? (void*) GetProcAddress(lib, n) : nullptr; };
#else
            void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
            if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
            auto sym = [&](const char* n) -> void* { return lib ? dlsym(lib, n) : nullptr; };
#endif
            if (!lib) return;
            auto init = (InitFn) (sym("nvmlInit_v2") ? sym("nvmlInit_v2") : sym("nvmlInit"));
            by_pci = (ByPciFn) (sym("nvmlDeviceGetHandleByPciBusId_v2") ? sym("nvmlDeviceGetHandleByPciBusId_v2")
                                                                        : sym("nvmlDeviceGetHandleByPciBusId"));
            by_idx = (ByIdxFn) (sym("nvmlDeviceGetHandleByIndex_v2") ? sym("nvmlDeviceGetHandleByIndex_v2")
                                                                     : sym("nvmlDeviceGetHandleByIndex"));
            mem = (MemFn) sym("nvmlDeviceGetMemoryInfo");
            ok = init && mem && (by_pci || by_idx) && init() == 0;
        }
    };
    static Nvml nvml;
    if (!nvml.ok) return;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    void* h = nullptr;
    char pci[64] = {};
    if (cudaDeviceGetPCIBusId(pci, sizeof pci, dev) != cudaSuccess || !nvml.by_pci || nvml.by_pci(pci, &h) != 0) {
        (void) cudaGetLastError();
        if (!nvml.by_idx || nvml.by_idx((unsigned int) dev, &h) != 0) return;
    }
    NvmlMem m{};
    if (nvml.mem(h, &m) == 0 && m.total > 0) {
        const size_t nf = m.total > m.used ? std::min<size_t>((size_t) m.free, (size_t) (m.total - m.used)) : (size_t) m.free;
        fr = std::min<size_t>(fr, nf);
        tot = (size_t) m.total;
    }
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }

constexpr int kImagePad = 18;          // <image>: the rows strata-vision's embeddings take (between <img> and </img>)
constexpr int kEos = 11;               // <|im_end|> (tokenizer.ggml.eos_token_id)
constexpr int kEos2 = 2;               // </s>
constexpr size_t kTail = 4096;         // zeroed bytes after the last slot (a kernel reading a little past a matrix)
constexpr size_t kAllocGran = (size_t) 2 << 20;   // the slot buffers' planned cudaMalloc rounding

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
    float scale = 1.0f;           // companion .scale tensor (NVFP4), 1.0 when absent
};

enum class Kind { Mamba, Attn, Moe };

struct Moe {
    int E = 0, FF = 0;                                 // routed expert count and width of this layer
    const float *router = nullptr, *bias = nullptr;   // [E][N] f32, [E]
    Mat lat_down, lat_up, sh_up, sh_down;
    K::NativeExpertLayout L;                           // [up | down] blobs (ungated)
    size_t stride = 0;                                 // a slot's bytes (16-aligned)
    const uint8_t *src_up = nullptr, *src_down = nullptr;   // the mapped GGUF's expert tensors
    size_t up_bytes = 0, d_bytes = 0;                  // per expert
    uint8_t* host = nullptr;                           // the pinned arena's copy: blob e at host + e * stride (null: none)
    /// expert e's up / down rows in host memory: the arena's blob, else the mapped file's slices
    const uint8_t* up_of(int e) const { return host ? host + (size_t) e * stride : src_up + (size_t) e * up_bytes; }
    const uint8_t* down_of(int e) const { return host ? host + (size_t) e * stride + L.down_off : src_down + (size_t) e * d_bytes; }
    strata::nemotron_h::CpuFmt fmt;
    uint8_t* slots = nullptr;                          // n_slots blobs, `stride` apart
    int n_slots = 0;
};

struct Layer {
    Kind kind = Kind::Mamba;
    int idx = 0;                     // its index among the layers of its kind (state / KV / slot tables)
    const float* norm = nullptr;     // attn_norm (every sublayer's pre-norm)
    // Mamba-2
    Mat ssm_in, ssm_out;
    const float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    // MoE
    Moe moe;
};

struct Geo {
    int n_layers = 0, n_all = 0, N = 0, V = 0;
    int H = 0, HKV = 0;
    int E = 0, top_k = 0, FF = 0, FS = 0, LAT = 0;
    bool w_norm = true;
    float w_scale = 1.f;
    int ssm_conv = 4, ssm_N = 128, ssm_G = 8, ssm_inner = 8192, ssm_H = 128;
    int conv_ch = 0, in_proj = 0;
    float eps = 1e-5f;
};

struct Model {
    Geo g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are read from it
    std::vector<Layer> layers;
    int n_mamba = 0, n_attn = 0, n_moe = 0;
    // the MTP block (blk.<n_layers>): eh_proj over [enorm(e) | hnorm(h)], attention + MoE, shared_head_norm
    bool has_mtp = false;
    Layer mtp_attn, mtp_moe;          // its attention (KV index n_attn) and its MoE (slot index n_moe)
    const float* mtp_post_norm = nullptr;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    size_t dense_bytes = 0;
};

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) u = sign;
        else {
            uint32_t m = mantissa, shift = 0;
            while ((m & 0x400) == 0) { m <<= 1; ++shift; }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) u = sign | 0x7f800000u | (mantissa << 13);
    else u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else {
        std::fprintf(stderr, "strata-nh: %s is %s, expected F32, BF16 or F16\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

const strata::MetaValue* meta(const strata::GgufModel& gm, const std::string& key) { return gm.meta().get(key); }

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    const strata::MetaValue* arch = meta(gm, "general.architecture");
    if (!arch || arch->s != "nemotron_h_moe") {
        err = "strata-nh runs Nemotron-H MoE (nemotron_h_moe) files only";
        return false;
    }
    const std::string p = "nemotron_h_moe.";
    auto num = [&](const std::string& k, double dflt) {
        const strata::MetaValue* v = meta(gm, p + k);
        if (!v) return dflt;
        if (v->type == strata::MetaType::ARRAY) {
            double mx = 0;
            for (const strata::MetaValue& it : v->items) mx = std::max(mx, it.num());
            return mx > 0 ? mx : dflt;
        }
        return v->num();
    };
    Geo& g = m.g;
    g.n_all = (int) num("block_count", 0);
    const int nextn = (int) num("nextn_predict_layers", 0);
    g.n_layers = g.n_all - nextn;
    g.N = (int) num("embedding_length", 0);
    g.V = (int) num("vocab_size", 0);
    g.H = (int) num("attention.head_count", 0);
    g.E = 0;
    g.top_k = (int) num("expert_used_count", 0);
    g.FF = 0;
    g.FS = (int) num("expert_shared_feed_forward_length", 0);
    g.LAT = (int) num("moe_latent_size", 0);
    g.w_norm = num("expert_weights_norm", 1) != 0;
    g.w_scale = (float) num("expert_weights_scale", 1.0);
    g.ssm_conv = (int) num("ssm.conv_kernel", 4);
    g.ssm_N = (int) num("ssm.state_size", 128);
    g.ssm_G = (int) num("ssm.group_count", 8);
    g.ssm_inner = (int) num("ssm.inner_size", 8192);
    g.ssm_H = (int) num("ssm.time_step_rank", 128);
    g.eps = (float) num("attention.layer_norm_epsilon", 1e-5);
    if (const strata::MetaValue* v = meta(gm, p + "attention.layer_norm_rms_epsilon")) g.eps = (float) v->num();
    g.conv_ch = g.ssm_inner + 2 * g.ssm_G * g.ssm_N;
    g.in_proj = 2 * g.ssm_inner + 2 * g.ssm_G * g.ssm_N + g.ssm_H;
    const int key_len = (int) num("attention.key_length", 128);
    if (g.V == 0) {
        if (const strata::TensorInfo* t = gm.find("token_embd.weight")) g.V = (int) t->shape[1];
    }
    if (g.n_layers <= 0 || g.N % 256 || key_len != kHD || g.ssm_conv != 4 || g.ssm_N != kSsmN ||
        g.ssm_inner / g.ssm_H != kSsmP || g.ssm_H % g.ssm_G || g.top_k > kMaxTopK || g.top_k < 1 ||
        g.LAT % 256 || g.FS % 256 || g.ssm_inner % g.ssm_G) {
        err = "a geometry this driver's kernels do not take (head 128, conv 4, state 128, head_dim 64, <= 512 experts, "
              "top-k <= 32, widths in whole blocks)";
        return false;
    }
    // a sublayer's kind from its tensors (the per-layer arrays head_count_kv / feed_forward_length say the same, but the
    // reader keeps only an array's first 64 entries: there are 88 sublayers)
    auto has = [&](int64_t l, const char* s) { return gm.find("blk." + std::to_string(l) + "." + s) != nullptr; };
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-nh: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-nh: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        if (name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
            size_t ss = 0;
            if (const strata::TensorInfo* ts = gm.find(name.substr(0, name.size() - 7) + ".scale", &ss)) {
                if (ts->type == 0 && ts->elements() == 1)
                    std::memcpy(&w.scale, gm.shard(ss).tensor_data(*ts), 4);
            }
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    auto load_moe = [&](int64_t l, Moe& y) -> bool {
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.bias = f32(L(l, "exp_probs_b.bias"));
        y.lat_down = mat(L(l, "ffn_latent_down.weight"));
        y.lat_up = mat(L(l, "ffn_latent_up.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        size_t su = 0, sd = 0, sr = 0, sb = 0;
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        const strata::TensorInfo* tr = find(L(l, "ffn_gate_inp.weight"), sr);
        const strata::TensorInfo* tb = find(L(l, "exp_probs_b.bias"), sb);
        if (tu->shape.size() < 3 || td->shape.size() < 3) {
            err = L(l, "expert tensors are not 3D");
            return false;
        }
        y.FF = (int) tu->shape[1];
        y.E = (int) tu->shape[2];
        if (y.E < g.top_k || y.E > 512 || y.FF <= 0 || y.FF % 32 != 0) {
            err = "a geometry this driver's kernels do not take (head 128, conv 4, state 128, head_dim 64, <= 512 experts, "
                  "top-k <= 32, widths in whole blocks)";
            return false;
        }
        if ((int) tu->shape[0] != g.LAT || (int) td->shape[0] != y.FF || (int) td->shape[1] != g.LAT ||
            (int) td->shape[2] != y.E || (int) tr->elements() != g.N * y.E || (int) tb->elements() != y.E) {
            err = L(l, "expert tensors have an unexpected shape");
            return false;
        }
        if (!K::native_expert_ungated_supported((int) tu->type, (int) td->type, g.LAT, y.FF)) {
            err = L(l, "experts are ") + tu->type_name() + "/" + td->type_name() +
                  ", which the native expert kernels do not take";
            return false;
        }
        y.L = K::native_expert_layout_ungated((int) tu->type, (int) td->type, g.LAT, y.FF);
        y.stride = (y.L.bytes + 255) / 256 * 256;
        y.up_bytes = (size_t) y.FF * y.L.gu_row;
        y.d_bytes = (size_t) g.LAT * y.L.d_row;
        y.src_up = gm.shard(su).tensor_data(*tu);
        y.src_down = gm.shard(sd).tensor_data(*td);
        if (!strata::nemotron_h::cpu_fmt((int) tu->type, (int) td->type, g.LAT, y.FF, y.fmt, err)) return false;
        if (y.fmt.up_row != y.L.gu_row || y.fmt.d_row != y.L.d_row) {
            err = "the CPU and GPU expert row sizes differ";
            return false;
        }
        g.E = std::max(g.E, y.E);
        g.FF = std::max(g.FF, y.FF);
        return true;
    };
    m.layers.resize((size_t) g.n_layers);
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.norm = f32(L(l, "attn_norm.weight"));
        if (has(l, "attn_q.weight")) {
            y.kind = Kind::Attn;
            y.idx = m.n_attn++;
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            const int hkv = y.wk.n_out / kHD;
            if (g.HKV == 0) g.HKV = hkv;
            if (hkv != g.HKV || y.wq.n_out != g.H * kHD || y.wv.n_out != hkv * kHD) {
                err = L(l, "attention has an unexpected shape");
                return false;
            }
        } else if (has(l, "ffn_up_exps.weight")) {
            y.kind = Kind::Moe;
            y.idx = m.n_moe++;
            if (!load_moe(l, y.moe)) return false;
        } else {
            y.kind = Kind::Mamba;
            y.idx = m.n_mamba++;
            y.ssm_in = mat(L(l, "ssm_in.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.conv_w = f32(L(l, "ssm_conv1d.weight"));
            size_t s = 0;
            if (gm.find(L(l, "ssm_conv1d.bias"), &s)) y.conv_b = f32(L(l, "ssm_conv1d.bias"));
            y.dt_b = f32(L(l, "ssm_dt.bias"));
            y.A = f32(L(l, "ssm_a"));
            y.D = f32(L(l, "ssm_d"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
            if (y.ssm_in.n_out != g.in_proj) { err = L(l, "ssm_in has an unexpected width"); return false; }
        }
    }
    if (m.n_attn == 0 || g.H % std::max(1, g.HKV)) { err = "no attention layer, or heads not a multiple of KV heads"; return false; }
    m.has_mtp = nextn > 0;
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
        m.mtp_attn.kind = Kind::Attn;
        m.mtp_attn.idx = m.n_attn;
        m.mtp_attn.norm = f32(L(l, "attn_norm.weight"));
        m.mtp_attn.wq = mat(L(l, "attn_q.weight"));
        m.mtp_attn.wk = mat(L(l, "attn_k.weight"));
        m.mtp_attn.wv = mat(L(l, "attn_v.weight"));
        m.mtp_attn.wo = mat(L(l, "attn_output.weight"));
        m.mtp_post_norm = f32(L(l, "post_attention_norm.weight"));
        m.mtp_moe.kind = Kind::Moe;
        m.mtp_moe.idx = m.n_moe;
        m.mtp_moe.norm = m.mtp_post_norm;
        if (!load_moe(l, m.mtp_moe.moe)) return false;
    }
    m.out_norm = f32("output_norm.weight");
    {
        size_t s = 0;
        m.head = gm.find("output.weight", &s) ? mat("output.weight") : mat("token_embd.weight");
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        if (!K::embed_type_supported(m.embd_type)) { err = "token_embd's type is not one the embedding kernel takes"; return false; }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    m.dense_bytes = bytes;
    std::fprintf(stderr, "strata-nh: %s: %d sublayers (%d Mamba-2, %d attention, %d MoE)%s, %.2f GiB of dense weights on "
                 "the GPU in %.1f s (the routed experts stay in the mapped file)\n", path.c_str(), g.n_layers, m.n_mamba,
                 m.n_attn, m.n_moe, m.has_mtp ? " + the MTP block" : "", (double) bytes / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ the expert stream
/// The prompt path's expert stream: the non-resident experts of the forward's MoE layers, in order, through a ring
/// of `nr` VRAM buffers.  A copy thread reads each expert from the mapped GGUF (into a pinned bounce slot, or
/// straight when the mapping is registered with --pin-experts) and queues its upload on a copy stream; the engine's
/// stream waits for an item's upload before its GEMMs and records when it is done with the buffer.
struct ExpertStream {
    struct Item {
        const uint8_t *up, *down;
        size_t up_bytes, d_bytes;
    };
    int nr = 0;
    size_t slot = 0;
    uint8_t* ring = nullptr;     // device, nr * slot
    uint8_t* bounce = nullptr;   // pinned, nr * slot (null: the sources are registered)
    cudaStream_t cst = nullptr;
    std::vector<cudaEvent_t> ready, freed;
    std::vector<Item> plan;
    std::atomic<long> enq{0}, done_n{0};
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false;

    void init(int n, size_t slot_bytes, bool registered) {
        nr = n;
        slot = slot_bytes;
        CK(cudaMalloc(&ring, (size_t) nr * slot + kTail));
        CK(cudaMemset(ring, 0, (size_t) nr * slot + kTail));
        if (!registered) init_bounce();
        CK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
        ready.resize((size_t) nr);
        freed.resize((size_t) nr);
        for (int i = 0; i < nr; ++i) {
            CK(cudaEventCreateWithFlags(&ready[(size_t) i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&freed[(size_t) i], cudaEventDisableTiming));
        }
    }
    void init_bounce() {
        if (!bounce) CK(cudaHostAlloc((void**) &bounce, (size_t) nr * slot, cudaHostAllocDefault));
    }
    void begin(std::vector<Item> items) {
        finish();
        plan = std::move(items);
        enq = 0;
        done_n = 0;
        stop = false;
        if (plan.empty()) return;
        th = std::thread([this] { copier(); });
    }
    void copier() {
        CK(cudaSetDevice(dev));
        for (long i = 0; i < (long) plan.size(); ++i) {
            const int s = (int) (i % nr);
            if (i >= nr) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || done_n.load() > i - nr; });
                if (stop) return;
                lk.unlock();
                CK(cudaEventSynchronize(freed[(size_t) s]));
            }
            const Item& it = plan[(size_t) i];
            uint8_t* dst = ring + (size_t) s * slot;
            if (bounce) {
                uint8_t* b = bounce + (size_t) s * slot;
                std::memcpy(b, it.up, it.up_bytes);
                std::memcpy(b + it.up_bytes, it.down, it.d_bytes);
                CK(cudaMemcpyAsync(dst, b, it.up_bytes + it.d_bytes, cudaMemcpyHostToDevice, cst));
            } else if (it.down == it.up + it.up_bytes) {   // the arena's blob: one copy
                CK(cudaMemcpyAsync(dst, it.up, it.up_bytes + it.d_bytes, cudaMemcpyHostToDevice, cst));
            } else {
                CK(cudaMemcpyAsync(dst, it.up, it.up_bytes, cudaMemcpyHostToDevice, cst));
                CK(cudaMemcpyAsync(dst + it.up_bytes, it.down, it.d_bytes, cudaMemcpyHostToDevice, cst));
            }
            CK(cudaEventRecord(ready[(size_t) s], cst));
            {
                std::lock_guard<std::mutex> lk(mu);
                enq = i + 1;
            }
            cv.notify_all();
        }
    }
    /// item i's buffer, once its upload is queued (the engine's stream waits for it)
    const uint8_t* take(long i, cudaStream_t st) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return enq.load() > i; });
        }
        const int s = (int) (i % nr);
        CK(cudaStreamWaitEvent(st, ready[(size_t) s], 0));
        return ring + (size_t) s * slot;
    }
    /// Same host wait as take(), but no stream wait.  The prompt MMQ copier waits for ready[] on its own stream.
    const uint8_t* pointer(long i) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return enq.load() > i; });
        }
        return ring + (size_t) (i % nr) * slot;
    }
    int slot_index(long i) const { return (int) (i % nr); }
    void done(long i, cudaStream_t st) {
        CK(cudaEventRecord(freed[(size_t) (i % nr)], st));
        {
            std::lock_guard<std::mutex> lk(mu);
            done_n = i + 1;
        }
        cv.notify_all();
    }
    void finish() {
        if (!th.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        th.join();
    }
    int dev = 0;
};

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * 4));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

/// DFlash's target features: rows of the residual into their column block of the [rows, ld] FP16 matrix
__global__ void k_feat16(const float* __restrict__ x, int N, __half* __restrict__ dst, int ld) {
    const float* r = x + (size_t) blockIdx.x * N;
    __half* d = dst + (size_t) blockIdx.x * ld;
    for (int i = threadIdx.x; i < N; i += blockDim.x) d[i] = __float2half(r[i]);
}

/// STRATA_NH_PROF=1: cuda events on the prompt stream, resolved once per forward.  Off for speed runs.
struct Prof {
    enum Ph { MambaIn, MambaConv, MambaScan, MambaGate, MambaOut, AddNorm, Attn, MoePre, MoeShared, MoeExperts, MoeLat, Other, NPh };
    static constexpr int kMax = 1024;
    bool on = false;
    cudaEvent_t ev[kMax]{};
    int phase[kMax]{};
    int n = 0;
    static const char* name(int p) {
        static const char* nm[NPh] = {"mamba_in", "mamba_conv", "mamba_scan", "mamba_gate", "mamba_out", "add_norm",
                                      "attn", "moe_pre", "moe_shared", "moe_experts", "moe_lat", "other"};
        return nm[p];
    }
    void init() {
        const char* e = std::getenv("STRATA_NH_PROF");
        on = e != nullptr && std::strcmp(e, "1") == 0;
        if (!on) return;
        for (int i = 0; i < kMax; ++i) CK(cudaEventCreate(&ev[i]));
    }
    void begin(cudaStream_t st) {
        if (!on) return;
        CK(cudaEventRecord(ev[0], st));
        n = 1;
    }
    void mark(cudaStream_t st, int ph) {
        if (!on || n <= 0 || n >= kMax) return;
        phase[n] = ph;
        CK(cudaEventRecord(ev[n], st));
        ++n;
    }
    void finish() {
        if (!on || n < 2) { n = 0; return; }
        CK(cudaEventSynchronize(ev[n - 1]));
        double local[NPh] = {};
        double sum = 0;
        for (int i = 1; i < n; ++i) {
            float t = 0;
            CK(cudaEventElapsedTime(&t, ev[i - 1], ev[i]));
            local[phase[i]] += (double) t;
            sum += (double) t;
        }
        n = 0;
        std::fprintf(stderr, "strata-nh: prof");
        for (int i = 0; i < NPh; ++i)
            if (local[i] >= 0.05) std::fprintf(stderr, " %s %.1f", name(i), local[i]);
        std::fprintf(stderr, " (%.1f ms)\n", sum);
    }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;
    static constexpr int kVerify = 8;
    static constexpr int kGemmT = 8;   // a batch above this many rows runs its dense projections as GEMMs
    // activations
    float *x, *h, *y, *y2, *zx, *xc, *yr, *yg, *qx, *kx, *vx, *att, *logits_r, *ewts, *hl, *lat, *eparts, *su, *logits;
    float *hmain, *hlast, *seeds, *hmtp, *mtp_cat, *logits_m;
    float *ubuf = nullptr;
    __half *x16 = nullptr, *xg16 = nullptr, *hb16 = nullptr, *q16 = nullptr, *att_p16 = nullptr;
    float* att_s = nullptr;
    size_t kAttScoreFloats = (size_t) 32 << 20;
    void *q8, *xq, *escr;
    void* xq_l = nullptr;   // the latent rows' q8_1 image: the routed experts' input (every grouped call of a layer)
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick, *tok_m, *pos0_m, *pick_m;
    int32_t *rcnt, *rcur, *rbounds, *rinv;
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int32_t* hpin = nullptr;   // pinned staging: [0] token, [1] position, [2..9] picks, [16..] tokens
    int n_chunk = 0, att_rows = 0;
    std::vector<float*> conv, S, vbuf;   // per Mamba layer: conv history, state, a verify window's zx rows
    std::vector<__half*> kc, vc;         // per attention layer (+ the MTP block's at n_attn)
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    SsmWs ssm_ws;
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;
    int mtp_d = 0;
    bool prompt_mtp = true;   // the prompt also runs through the MTP block (its KV) when drafts are on
    // ---- DFlash (--dflash GGUF): the drafter reads the residual entering its target sublayers, FP16, the last
    // kFeatRows rows of a forward (its KV ring holds no more), and drafts df_d tokens a step
    std::unique_ptr<strata::nemotron_h::DFlashDrafter> df;
    int df_d = 0, df_ntgt = 0;
    std::vector<int> df_col;   // per sublayer: its column block in feat16 (-1: not a target)
    __half* feat16 = nullptr;
    static constexpr int kFeatRows = 1024;
    double df_ms = 0;          // drafting time (the drafter's own clock), summed
    long long df_steps = 0;
    // the hybrid mode
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance (the slots follow it)
        std::vector<double> total;   // never halved: the expert profile this engine saves (--expert-profile-save)
        int32_t* slot_of_d = nullptr;
    };
    // ---- transient GPU experts, Flash-Next's two ways of not computing a miss on the CPU: the PCIe share (a part of
    // a layer's misses uploaded when the routing is known, while the CPU computes the rest) and the prefetch (the next
    // MoE layer's likely experts, from its router applied to the hidden one sublayer early, uploaded while that
    // sublayer runs on the GPU and the CPU is idle).  Both into small VRAM buffers, run by one grouped call.
    float pcie_frac = 0.f;
    int pcie_cap = 48, pf_cap = 0;
    size_t tr_stride = 0;
    uint8_t* pcbuf = nullptr;
    uint8_t* pfbuf[2] = {};
    cudaStream_t cst2 = nullptr;
    cudaEvent_t ev_pc = nullptr, ev_trfree = nullptr, ev_pfroute = nullptr, ev_pf[2] = {}, ev_pffree[2] = {};
    unsigned long long* h_tgp = nullptr;
    int32_t *h_tgs = nullptr, *h_ted = nullptr, *h_tet = nullptr, *h_tng = nullptr;
    unsigned long long* d_tgp = nullptr;
    int32_t *d_tgs = nullptr, *d_ted = nullptr, *d_tet = nullptr, *d_tng = nullptr;
    float *logits_pf = nullptr, *wts_pf = nullptr;
    int32_t *ids_pf = nullptr, *h_pfids = nullptr;
    struct Pf {
        int mi = -1;                 // the MoE slot index whose prediction the buffer holds (-1: none)
        std::vector<int> slot;       // expert -> its place in the buffer (-1: not there)
    } pf[2];
    long long tr_pcie = 0, tr_pf_sent = 0, tr_pf_hit = 0;
    std::string profile_save;   // --expert-profile-save: the routing counts saved here (every 8 requests, at exit)
    // ---- turn checkpoints (Flash-Next's conversation checkpoints, reduced to what a recurrent state needs): the
    // Mamba states (conv history + scan state) and hlast at the end of a request's prompt, in pinned host memory.  The
    // state cannot step back, but the KV cache can (its cells past a position are just overwritten), so a request that
    // goes back to a checkpoint along the live sequence (the client regenerating or editing the last answer) restores
    // it and reads only the rest of its prompt.
    struct Ckpt {
        int64_t pos = 0;           // the tokens it holds: live[0 .. pos)
        float* host = nullptr;
        uint64_t used = 0;
        std::vector<uint8_t> df_state;   // the DFlash drafter's KV ring and anchor at pos
    };
    std::vector<Ckpt> ckpts;
    int max_ckpts = 4;
    uint64_t ckpt_clock = 0;
    size_t ckpt_floats() const {
        const Geo& g = m->g;
        return (size_t) m->n_mamba * ((size_t) g.conv_ch * (g.ssm_conv - 1) + (size_t) g.ssm_H * kSsmP * kSsmN) + g.N;
    }
    void ckpt_copy(float* host, bool save) {
        const Geo& g = m->g;
        const size_t nc = (size_t) g.conv_ch * (g.ssm_conv - 1), ns = (size_t) g.ssm_H * kSsmP * kSsmN;
        size_t off = 0;
        const auto kind = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        for (int i = 0; i < m->n_mamba; ++i) {
            if (save) {
                CK(cudaMemcpyAsync(host + off, conv[(size_t) i], nc * 4, kind, st));
                CK(cudaMemcpyAsync(host + off + nc, S[(size_t) i], ns * 4, kind, st));
            } else {
                CK(cudaMemcpyAsync(conv[(size_t) i], host + off, nc * 4, kind, st));
                CK(cudaMemcpyAsync(S[(size_t) i], host + off + nc, ns * 4, kind, st));
            }
            off += nc + ns;
        }
        if (save) CK(cudaMemcpyAsync(host + off, hlast, (size_t) g.N * 4, kind, st));
        else CK(cudaMemcpyAsync(hlast, host + off, (size_t) g.N * 4, kind, st));
        CK(cudaStreamSynchronize(st));
    }
    /// A checkpoint of the current state (pos tokens); an existing one at the same position is refreshed.
    void ckpt_save() {
        if (max_ckpts <= 0 || pos <= 0) return;
        Ckpt* c = nullptr;
        for (Ckpt& k : ckpts) if (k.pos == pos) c = &k;
        if (!c && (int) ckpts.size() < max_ckpts) {
            ckpts.push_back({});
            c = &ckpts.back();
            if (cudaHostAlloc((void**) &c->host, ckpt_floats() * 4, cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                ckpts.pop_back();
                return;
            }
        }
        if (!c) {   // the least recently used one
            c = &ckpts[0];
            for (Ckpt& k : ckpts) if (k.used < c->used) c = &k;
        }
        c->pos = pos;
        c->used = ++ckpt_clock;
        ckpt_copy(c->host, true);
        if (df) {
            c->df_state.resize(df->state_bytes());
            df->copy_state(c->df_state.data(), true, st);
        }
    }
    /// The checkpoints past `keep` tokens no longer describe the live sequence (its cells were rewritten from there).
    void ckpt_drop_past(int64_t keep) {
        for (Ckpt& k : ckpts) if (k.pos > keep) k.pos = -1;
    }
    /// The longest checkpoint at or below `limit` tokens: restored, its position returned (0: none).
    int64_t ckpt_restore(int64_t limit) {
        Ckpt* best = nullptr;
        for (Ckpt& k : ckpts) if (k.pos > 0 && k.pos <= limit && (!best || k.pos > best->pos)) best = &k;
        if (!best) return 0;
        ckpt_copy(best->host, false);
        if (df) {
            if (best->df_state.size() == df->state_bytes()) df->copy_state(best->df_state.data(), false, st);
            else df->reset();
        }
        best->used = ++ckpt_clock;
        pos = best->pos;
        return pos;
    }
    double t_route_wait_ms = 0;
    void init_transient() {
        const Geo& g = m->g;
        const size_t ent = (size_t) std::max(std::min(hybrid_t, tmax), kVerify) * g.top_k;
        CK(cudaStreamCreateWithFlags(&cst2, cudaStreamNonBlocking));
        for (cudaEvent_t* e : {&ev_pc, &ev_trfree, &ev_pf[0], &ev_pf[1], &ev_pffree[0], &ev_pffree[1]})
            CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_pfroute, cudaEventDisableTiming | cudaEventBlockingSync));
        CK(cudaHostAlloc((void**) &h_tgp, (ent + 1) * 8, cudaHostAllocDefault));
        for (int32_t** b : {&h_tgs, &h_ted, &h_tet, &h_tng}) CK(cudaHostAlloc((void**) b, (ent + 1) * 4, cudaHostAllocDefault));
        d_tgp = dalloc_t<unsigned long long>(ent + 1);
        d_tgs = dalloc_t<int32_t>(ent + 1); d_ted = dalloc_t<int32_t>(ent + 1); d_tet = dalloc_t<int32_t>(ent + 1);
        d_tng = dalloc_t<int32_t>(1);
        const size_t rows = (size_t) std::max(std::min(hybrid_t, tmax), kVerify);
        logits_pf = dalloc(rows * g.E); wts_pf = dalloc(rows * g.top_k); ids_pf = dalloc_t<int32_t>(rows * g.top_k);
        CK(cudaHostAlloc((void**) &h_pfids, rows * g.top_k * 4, cudaHostAllocDefault));
        for (Pf& P : pf) P.slot.assign((size_t) g.E, -1);
    }
    /// The next MoE layer's routing guessed from the current hidden (before the sublayer between runs): queued now,
    /// read by prefetch_issue once that sublayer's kernels are queued too.
    void prefetch_begin(const Layer& Ln, int T) {
        const Geo& g = m->g;
        const int E_ = Ln.moe.E;
        f32mm(Ln.moe.router, h, logits_pf, g.N, E_, T);
        k_route<<<T, 512, 0, st>>>(logits_pf, Ln.moe.bias, E_, g.top_k, g.w_scale, g.w_norm, ids_pf, wts_pf);
        CK(cudaMemcpyAsync(h_pfids, ids_pf, (size_t) T * g.top_k * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev_pfroute, st));
    }
    void prefetch_issue(const Layer& Ln, int T) {
        const Geo& g = m->g;
        const Moe& M = Ln.moe;
        const int mi = Ln.idx, b = mi & 1;
        CK(cudaEventSynchronize(ev_pfroute));
        Pf& P = pf[b];
        for (int x = 0; x < M.E; ++x) P.slot[(size_t) x] = -1;
        P.mi = mi;
        const Hy& Hh = hy[(size_t) mi];
        CK(cudaStreamWaitEvent(cst2, ev_pffree[b], 0));
        int n = 0;
        // rank order across the rows: every row's first pick, then every row's second, ...
        for (int k = 0; k < g.top_k && n < pf_cap; ++k)
            for (int t = 0; t < T && n < pf_cap; ++t) {
                const int x = h_pfids[t * g.top_k + k];
                if (Hh.slot_of[(size_t) x] >= 0 || P.slot[(size_t) x] >= 0) continue;
                P.slot[(size_t) x] = n;
                CK(cudaMemcpyAsync(pfbuf[b] + (size_t) n * tr_stride, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice, cst2));
                CK(cudaMemcpyAsync(pfbuf[b] + (size_t) n * tr_stride + M.L.down_off, M.down_of(x), M.d_bytes,
                                   cudaMemcpyHostToDevice, cst2));
                ++n;
            }
        if (!warming) tr_pf_sent += n;
        CK(cudaEventRecord(ev_pf[b], cst2));
    }
    std::vector<Hy> hy;   // per MoE slot index (the main layers, then the MTP block)
    int hybrid_t = 64;
    std::unique_ptr<strata::nemotron_h::CpuExperts> pool;
    std::vector<strata::nemotron_h::CpuJob> jobs;
    float *h_hl = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    cudaEvent_t ev_route = nullptr;
    ExpertStream xs;
    long xs_pos = 0;
#if defined(STRATA_PREFILL_MMQ)
    // Prompt chunks (T > hybrid_t): one MMQ launch per group of experts, weights read as IQ2_XXS.  STRATA_NH_EXPERT_MMQ=0
    // keeps the per-expert FP16 GEMMs.  The group buffers are allocated here, before place_experts sizes the slots.
    static constexpr int kMmqGroup = 32;
    static constexpr int kMmqIdPad = 160;   // MMQ reads up to J (<= 128) ids past the last row
    static constexpr int kMmqHold = 96;     // streamed experts held until a flush; capped again by the ring
    bool mmq_on = false;
    std::vector<char> mmq_ok;               // per MoE slot (main layers, then the MTP block)
    std::unique_ptr<MMQ::Context> mmq_ctx;
    uint8_t* mmq_up_b[2] = {};
    uint8_t* mmq_dn_b[2] = {};
    void* mmq_xq = nullptr;
    int32_t* mmq_ids = nullptr;
    int32_t* mmq_bounds_b[2] = {};
    const uint8_t** mmq_ptrs_b[2] = {};
    int mmq_rows_cap = 0;
    size_t mmq_max_bytes = 0;
    cudaEvent_t ev_moe_route = nullptr, ev_moe_cnt = nullptr;
    cudaStream_t cst_cp = nullptr;                 // copies the next expert group while st multiplies the previous one
    cudaEvent_t ev_cp[2] = {}, ev_mmq_done[2] = {};
    bool mmq_buf_live[2] = {};
    int mmq_buf = 0;
#endif
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0, hy_streamed = 0;
    double hy_cpu_ms = 0;
    bool warming = false;
    Prof prof;
    int adapt_every = 32, adapt_swaps = 8;   // --adapt-every / --adapt-swaps: the decode's slot replacements
    int swaps_after_prompt = 512;
    // POST /v1/vram (the server's `VRAM <reserve_mib>`): the startup slot count is the ceiling.  A shrink frees
    // slots, a regrow refills them up to that ceiling.  Dense weights, the KV cache and vram_reserve_mib stay.
    int vram_reserve_mib = 1024;
    int slots_cap = 0, slots_now = 0;
    int prompt_chunk = 2048;
    std::vector<Moe*> vram_moe;
    // GENI: rows that replace the embedding of an absolute position (host floats, N each)
    std::map<int64_t, const float*> inject;
    // --dump-layers: every main forward's sublayer outputs (the residual x), [pos][layer][N] f32 (parity.py)
    FILE* dump_f = nullptr;
    std::vector<float> dump_buf;

    const Layer& moe_layer(int mi) const {
        if (mi == m->n_moe) return m->mtp_moe;
        for (const Layer& L : m->layers)
            if (L.kind == Kind::Moe && L.idx == mi) return L;
        std::fprintf(stderr, "strata-nh: no MoE layer %d\n", mi);
        std::exit(1);
    }

    void init(const Model& model, int ctx_, int tmax_, int ring) {
        m = &model;
        ctx = ctx_;
        tmax = std::max(tmax_, kVerify);
        const Geo& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.N, Kp = (size_t) g.top_k;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); y2 = dalloc(T * N);
        hmain = dalloc(T * N); hlast = dalloc(N);
        if (m->has_mtp) {
            seeds = dalloc(T * N); hmtp = dalloc(T * N); mtp_cat = dalloc(T * 2 * N);
        }
        zx = dalloc(T * g.in_proj); xc = dalloc(T * g.conv_ch); yr = dalloc(T * g.ssm_inner); yg = dalloc(T * g.ssm_inner);
        qx = dalloc(T * g.H * kHD); kx = dalloc(T * g.HKV * kHD); vx = dalloc(T * g.HKV * kHD); att = dalloc(T * g.H * kHD);
        logits_r = dalloc(T * g.E); ewts = dalloc(T * Kp);
        hl = dalloc(T * g.LAT); lat = dalloc(T * g.LAT); eparts = dalloc(T * Kp * g.LAT); su = dalloc(T * g.FS);
        logits = dalloc((size_t) kVerify * g.V);
        if (m->has_mtp) logits_m = dalloc((size_t) g.V);
        ubuf = dalloc(T * g.FF);
        const int max_in = std::max({2 * g.N, g.ssm_inner, g.H * kHD, g.FS, g.LAT});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes(std::max(g.N, g.LAT), 1) * T));
        CK(cudaMalloc(&xq_l, K::native_q8_1_bytes(g.LAT, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        xg16 = dalloc_t<__half>(T * Kp * g.LAT);
        hb16 = dalloc_t<__half>(T * g.FF);
        q16 = dalloc_t<__half>(T * g.H * kHD);
        void* blas_ws = nullptr;
        CK(cudaMalloc(&blas_ws, (size_t) 8 << 20));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS ||
            cublasSetWorkspace(blas, blas_ws, (size_t) 8 << 20) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-nh: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * g.E);
        {
            std::string gerr;
            int64_t big = (int64_t) g.FF * g.LAT;
            auto take = [&](const Mat& w) { big = std::max<int64_t>(big, (int64_t) w.n_in * w.n_out); };
            for (const Layer& L : m->layers) {
                for (const Mat* w : {&L.ssm_in, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.moe.lat_down, &L.moe.lat_up,
                                     &L.moe.sh_up, &L.moe.sh_down})
                    take(*w);
            }
            if (m->has_mtp) {
                take(m->eh_proj);
                for (const Mat* w : {&m->mtp_attn.wq, &m->mtp_attn.wo, &m->mtp_moe.moe.sh_up, &m->mtp_moe.moe.sh_down}) take(*w);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-nh: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
            // touch both cuBLAS handles' FP16 GEMM paths now so their kernel modules are loaded before VRAM sizing
            gemm.f16((const uint16_t*) x16, (const uint16_t*) x16, y, 16, 64, 64, 64);
            const float one = 1.0f, zero = 0.0f;
            (void) cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 16, kHD, &one, x16, CUDA_R_16F, kHD, 0,
                                              q16, CUDA_R_16F, kHD, kHD, &zero, att_s, CUDA_R_32F, 64, 1024, 1,
                                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
        // the grouped kernels' scratch for a rows batch (at most hybrid_t rows: a bigger batch streams)
        const size_t rows_cap = (size_t) std::max(std::min(hybrid_t, tmax), kVerify);
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (rows_cap * Kp), (int64_t) g.FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.E) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.E) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        rinv = dalloc_t<int32_t>(T * Kp);
        rcnt = dalloc_t<int32_t>((size_t) g.E);
        rcur = dalloc_t<int32_t>((size_t) g.E);
        rbounds = dalloc_t<int32_t>((size_t) g.E + 1);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        if (m->has_mtp) {
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
        }
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (g.H * std::min<int64_t>((int64_t) T, 16));
        pacc = dalloc((size_t) att_rows * n_chunk * kHD);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        const int vcols = g.in_proj - g.ssm_inner;   // xBC | dt
        for (int i = 0; i < m->n_mamba; ++i) {
            conv.push_back(dalloc((size_t) g.conv_ch * (g.ssm_conv - 1)));
            S.push_back(dalloc((size_t) g.ssm_H * kSsmP * kSsmN));
            vbuf.push_back(dalloc((size_t) kVerify * vcols));
        }
        for (int i = 0; i < m->n_attn + (m->has_mtp ? 1 : 0); ++i) {
            kc.push_back(dalloc_t<__half>((size_t) g.HKV * ctx * kHD));
            vc.push_back(dalloc_t<__half>((size_t) g.HKV * ctx * kHD));
        }
        size_t max_stride = 0;
        for (const Layer& L : m->layers)
            if (L.kind == Kind::Moe) max_stride = std::max(max_stride, L.moe.stride);
        if (m->has_mtp) max_stride = std::max(max_stride, m->mtp_moe.moe.stride);
        xs.dev = 0;
        CK(cudaGetDevice(&xs.dev));
        xs.init(std::max(4, ring), max_stride, true);
        tr_stride = max_stride;
        if (pcie_frac <= 0.f) pcie_cap = 0;   // the share off: no buffer (its VRAM goes to the slots)
        if (pcie_cap > 0) pcbuf = dalloc_t<uint8_t>((size_t) pcie_cap * max_stride + kTail);
        if (pf_cap > 0)
            for (int b = 0; b < 2; ++b) pfbuf[b] = dalloc_t<uint8_t>((size_t) pf_cap * max_stride + kTail);
        CK(cudaHostAlloc((void**) &h_hl, T * g.LAT * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &cpu_out, T * Kp * g.LAT * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_ids, T * Kp * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_miss, T * Kp * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_cnt, (size_t) g.E * 4, cudaHostAllocDefault));
        ecpu = dalloc(T * Kp * g.LAT);
        d_miss = dalloc_t<int32_t>(T * Kp);
        init_transient();
        if (tmax > kSsmSsdMinTok) ssm_ws_reserve(ssm_ws, g.ssm_H, kSsmP, kSsmN, g.ssm_G, tmax);   // the prompt scan's scratch
#if defined(STRATA_PREFILL_MMQ)
        init_prompt_mmq();
#endif
        prof.init();
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming | cudaEventBlockingSync));
        CK(cudaDeviceSynchronize());
        size_t fr = 0, tot = 0;
        vram_info(fr, tot);
        std::fprintf(stderr, "strata-nh: context %d cells (FP16 KV, %.2f GiB), prompt chunk %d, %.2f GiB (%zu MiB) of VRAM free\n",
                     ctx, (double) kc.size() * 2 * g.HKV * ctx * kHD * 2 / (1 << 30), tmax, (double) fr / (1 << 30), fr >> 20);
    }

    void reset() {
        const Geo& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            CK(cudaMemsetAsync(conv[l], 0, (size_t) g.conv_ch * (g.ssm_conv - 1) * 4, st));
            CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_H * kSsmP * kSsmN * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.N * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
        if (df) df->reset();
    }

    // ---- matmuls: a decode batch on the native MMVQ (q8_1 activations), a prompt chunk on FP16 GEMMs
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void mmq(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
        if (w.scale != 1.0f) {
            const int64_t n = (int64_t) T * w.n_out;
            if (T > 64 && n % 4 == 0 && (reinterpret_cast<uintptr_t>(out) & 15) == 0)
                k_scale4<<<grid1(n / 4), kThreads, 0, st>>>(out, w.scale, n / 4);
            else
                k_scale<<<grid1(n), kThreads, 0, st>>>(out, w.scale, n);
        }
    }
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// out [T][n_out] = W [n_out][n_in] (f32) x: the router (FP32, as llama.cpp keeps it)
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3((n_out + 15) / 16, T), 256, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        const int tm = n_out > 64 ? 4 : 1;
        const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
        const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
        const int kz = (n_in / want + 15) / 16 * 16;
        const int splits = (n_in + kz - 1) / kz;
        float* dst = splits > 1 ? sk_part : out;
        if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
        else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
        if (splits > 1) k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
    }

    // ---- the sublayers: each reads h (the normed input) and writes its output to y (MoE: y + y2)
    void mamba(const Layer& L, int T, bool verify) {
        const Geo& g = m->g;
        const int li = L.idx;
        mm(L.ssm_in, h, zx, T);   // [T][z 8192 | xBC 10240 | dt 128]
        prof.mark(st, Prof::MambaIn);
        const int vcols = g.in_proj - g.ssm_inner;
        if (verify)   // the window's xBC | dt rows, for the commit
            CK(cudaMemcpy2DAsync(vbuf[(size_t) li], (size_t) vcols * 4, zx + g.ssm_inner, (size_t) g.in_proj * 4,
                                 (size_t) vcols * 4, (size_t) T, cudaMemcpyDeviceToDevice, st));
        if (T > 64) {
            k_conv_prompt<64><<<dim3(grid1(g.conv_ch, 128), (unsigned) ((T + 63) / 64)), 128, 0, st>>>(
                conv[(size_t) li], zx + g.ssm_inner, g.in_proj, L.conv_w, L.conv_b, xc, g.conv_ch, T);
            if (!verify)
                k_conv_save_hist<<<grid1(g.conv_ch, 128), 128, 0, st>>>(conv[(size_t) li], zx + g.ssm_inner, g.in_proj,
                                                                       g.conv_ch, T);
        } else {
            k_conv<<<grid1(g.conv_ch, 128), 128, 0, st>>>(conv[(size_t) li], zx + g.ssm_inner, g.in_proj, L.conv_w, L.conv_b,
                                                          xc, g.conv_ch, T, !verify);
        }
        prof.mark(st, Prof::MambaConv);
        ssm_scan(S[(size_t) li], xc, g.conv_ch, zx + g.ssm_inner + g.conv_ch, g.in_proj, L.dt_b, L.A, L.D, yr, g.ssm_H,
                 g.ssm_G, kSsmP, kSsmN, T, !verify, ssm_ws, st, blas);
        prof.mark(st, Prof::MambaScan);
        k_gate_norm<<<dim3(g.ssm_G, T), kThreads, 0, st>>>(yr, zx, g.in_proj, L.ssm_norm, yg, g.ssm_inner,
                                                           g.ssm_inner / g.ssm_G, g.eps);
        prof.mark(st, Prof::MambaGate);
        mm(L.ssm_out, yg, y, T);
        prof.mark(st, Prof::MambaOut);
    }

    void attention(const Layer& L, int T, int64_t p0, const int32_t* dpos) {
        const Geo& g = m->g;
        const int H = g.H, HKV = g.HKV, li = L.idx;
        prep(h, g.N, T, xq);
        apply(L.wq, xq, qx, T);
        apply(L.wk, xq, kx, T);
        apply(L.wv, xq, vx, T);
        k_kv_store<<<dim3(HKV, T), kHD, 0, st>>>(kx, vx, kc[(size_t) li], vc[(size_t) li], HKV, dpos, ctx);
        const float scale = 1.0f / sqrtf((float) kHD);
        if (T > kGemmT) {
            // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs per KV head over the FP16 cache, the causal
            // softmax in between; the queries in slices whose scores fit
            const int P = (int) p0 + T;
            const int ldp = (P + 7) / 8 * 8;
            const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
            const int G = H / HKV;
            const float one = 1.0f, zero = 0.0f;
            k_to_f16<<<grid1((int64_t) T * H * kHD), kThreads, 0, st>>>(qx, q16, (int64_t) T * H * kHD);
            for (int t0 = 0; t0 < T; t0 += tq) {
                const int nt = std::min(tq, T - t0);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Kh = kc[(size_t) li] + (size_t) kvh * ctx * kHD;
                    const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * kHD;
                    float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, kHD, &one, Kh, CUDA_R_16F, kHD, 0,
                                                   Qh, CUDA_R_16F, H * kHD, kHD, &zero, Sh, CUDA_R_32F, ldp,
                                                   (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-nh: attention scores GEMM failed\n");
                        std::exit(1);
                    }
                }
                k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0, scale);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Vh = vc[(size_t) li] + (size_t) kvh * ctx * kHD;
                    const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                    float* Oh = att + ((size_t) t0 * H + kvh * G) * kHD;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, kHD, nt, P, &one, Vh, CUDA_R_16F, kHD, 0,
                                                   Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                   H * kHD, kHD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-nh: attention values GEMM failed\n");
                        std::exit(1);
                    }
                }
            }
        } else {
            const int rows_per = att_rows / H;
            for (int t0 = 0; t0 < T; t0 += rows_per) {
                const int nt = std::min(rows_per, T - t0);
                const int nc = (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                k_attn_part<<<dim3(nc, nt * H), kThreads, 0, st>>>(qx, kc[(size_t) li], vc[(size_t) li], dpos, t0, H, HKV,
                                                                  ctx, scale, n_chunk, pacc, pm, pl);
                k_attn_merge<<<nt * H, kHD, 0, st>>>(pacc, pm, pl, dpos, t0, H, n_chunk, att);
            }
        }
        mm(L.wo, att, y, T);
    }

#if defined(STRATA_PREFILL_MMQ)
    /// Which layers can take the grouped IQ MMQ path (type, tile, byte layout).  A layer that fails stays on the
    /// per-expert FP16 GEMMs.
    void init_prompt_mmq() {
        const char* env = std::getenv("STRATA_NH_EXPERT_MMQ");
        const bool force_off = env != nullptr && std::strcmp(env, "0") == 0;
        const Geo& g = m->g;
        const int nmoe = m->n_moe + (m->has_mtp ? 1 : 0);
        mmq_ok.assign((size_t) nmoe, 0);
        if (!force_off && MMQ::built()) {
            auto ok = [&](const Moe& M) {
                if (!MMQ::supported(M.L.gu_type) || !MMQ::supported(M.L.d_type)) return false;
                if (!MMQ::fits(M.L.gu_type, M.FF) || !MMQ::fits(M.L.d_type, g.LAT)) return false;
                if (MMQ::matrix_bytes(M.L.gu_type, M.FF, g.LAT) != M.up_bytes) return false;
                if (MMQ::matrix_bytes(M.L.d_type, g.LAT, M.FF) != M.d_bytes) return false;
                if (((M.up_bytes | M.d_bytes | M.L.down_off) & 15) != 0) return false;
                if (M.L.down_off != M.up_bytes || M.up_bytes == 0) return false;
                return true;
            };
            for (const Layer& L : m->layers)
                if (L.kind == Kind::Moe && ok(L.moe)) mmq_ok[(size_t) L.idx] = 1;
            if (m->has_mtp && ok(m->mtp_moe.moe)) mmq_ok[(size_t) m->n_moe] = 1;
            for (char c : mmq_ok)
                if (c) mmq_on = true;
        }
        int n_ok = 0;
        for (char c : mmq_ok) n_ok += c != 0;
        if (!mmq_on) {
            std::fprintf(stderr, "strata-nh: prompt MoE: per-expert FP16 GEMM\n");
            return;
        }
        mmq_rows_cap = tmax;
        for (const Layer& L : m->layers)
            if (L.kind == Kind::Moe && mmq_ok[(size_t) L.idx])
                mmq_max_bytes = std::max(mmq_max_bytes, std::max(L.moe.up_bytes, L.moe.d_bytes));
        if (m->has_mtp && mmq_ok[(size_t) m->n_moe])
            mmq_max_bytes = std::max(mmq_max_bytes, std::max(m->mtp_moe.moe.up_bytes, m->mtp_moe.moe.d_bytes));
        const size_t wbytes = (size_t) kMmqGroup * mmq_max_bytes + kTail;
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&mmq_up_b[b], wbytes));
            CK(cudaMalloc(&mmq_dn_b[b], wbytes));
            mmq_bounds_b[b] = dalloc_t<int32_t>((size_t) kMmqGroup + 1);
            void* pptrs = nullptr;
            CK(cudaMalloc(&pptrs, (size_t) kMmqGroup * sizeof(void*)));
            mmq_ptrs_b[b] = static_cast<const uint8_t**>(pptrs);
            CK(cudaEventCreateWithFlags(&ev_cp[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_mmq_done[b], cudaEventDisableTiming));
        }
        CK(cudaStreamCreateWithFlags(&cst_cp, cudaStreamNonBlocking));
        CK(cudaMalloc(&mmq_xq, MMQ::q8_bytes(mmq_rows_cap, g.FF)));
        mmq_ids = dalloc_t<int32_t>((size_t) mmq_rows_cap + kMmqIdPad);
        CK(cudaEventCreateWithFlags(&ev_moe_route, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_moe_cnt, cudaEventDisableTiming));
        mmq_ctx = std::make_unique<MMQ::Context>();
        CK(cudaDeviceSynchronize());   // dalloc's legacy-stream memsets before iota on st
        MMQ::iota(mmq_ids, (int64_t) mmq_rows_cap + kMmqIdPad, st);
        std::fprintf(stderr,
                     "strata-nh: prompt MoE: grouped IQ MMQ (%d experts, weights stay quantized, copies double-buffered) on %d/%d MoE layers\n",
                     kMmqGroup, n_ok, nmoe);
    }

    /// T > hybrid_t, one MoE layer: route counts come back on cst2 while the shared expert runs on st, then experts
    /// with rows are grouped (at most kMmqGroup, and at most mmq_rows_cap rows).  Both matrices are copied out of the
    /// ring slot before done(), so the next uploads overlap the multiply.  Down writes sorted rows of eparts.
    void moe_stream_mmq(const Moe& M, int mi, int T) {
        const Geo& g = m->g;
        const int LAT = g.LAT, FF = M.FF, E_ = M.E;
        const int n_ent = T * g.top_k;
        Hy& Hh = hy[(size_t) mi];
        CK(cudaEventRecord(ev_moe_route, st));
        CK(cudaStreamWaitEvent(cst2, ev_moe_route, 0));
        CK(cudaMemcpyAsync(h_cnt, rcnt, (size_t) E_ * 4, cudaMemcpyDeviceToHost, cst2));
        CK(cudaEventRecord(ev_moe_cnt, cst2));
        shared(M, T);   // hides the count copy; y2 is independent of the routed experts
        prof.mark(st, Prof::MoeShared);
        CK(cudaEventSynchronize(ev_moe_cnt));

        struct Exp {
            const uint8_t* w;
            int c;
            long item;
        };
        Exp grp[kMmqGroup];
        long pend[kMmqHold];
        int ng = 0, np = 0, grows = 0, row0 = 0;
        const int hold_lim = std::max(1, std::min(kMmqHold, xs.nr - 1));

        auto flush = [&]() {
            if (ng > 0) {
                const uint8_t* hp[kMmqGroup];
                int32_t hb[kMmqGroup + 1];
                hb[0] = 0;
                int rows = 0, mr = 0;
                for (int i = 0; i < ng; ++i) {
                    hp[i] = grp[i].w;
                    rows += grp[i].c;
                    hb[i + 1] = rows;
                    if (grp[i].c > mr) mr = grp[i].c;
                }
                static_assert(kMmqGroup <= kMmqStage, "stage pack holds one group");
                MmqStage stage{};
                stage.n = ng;
                for (int i = 0; i < ng; ++i) {
                    stage.ptr[i] = hp[i];
                    stage.bounds[i] = hb[i];
                }
                stage.bounds[ng] = hb[ng];
                // Copies land in buffer b on cst_cp.  st waits for ev_cp[b] before the multiply, and the next user
                // of b waits for ev_mmq_done[b] first, so the multiply still owns the buffer.  done() is recorded on
                // cst_cp after that wait and the copy, or the ring would reuse a slot whose upload is still in flight.
                const int b = mmq_buf;
                if (mmq_buf_live[b]) CK(cudaStreamWaitEvent(cst_cp, ev_mmq_done[b], 0));
                for (int i = 0; i < np; ++i)
                    CK(cudaStreamWaitEvent(cst_cp, xs.ready[(size_t) xs.slot_index(pend[i])], 0));
                k_stage_mmq<<<1, kMmqGroup + 1, 0, cst_cp>>>(stage, mmq_ptrs_b[b], mmq_bounds_b[b]);
                const int n4u = (int) (M.up_bytes / 16);
                const int n4d = (int) (M.d_bytes / 16);
                auto launch_copy = [&](int src_off, uint8_t* dst, int n4) {
                    const unsigned gx = (unsigned) std::max(1, std::min(128, (n4 + kThreads - 1) / kThreads));
                    k_copy_experts<<<dim3(gx, (unsigned) ng), kThreads, 0, cst_cp>>>(mmq_ptrs_b[b], src_off,
                                                                                     reinterpret_cast<uint4*>(dst), n4, n4);
                };
                launch_copy(0, mmq_up_b[b], n4u);
                launch_copy((int) M.L.down_off, mmq_dn_b[b], n4d);
                CK(cudaMemsetAsync(mmq_up_b[b] + (size_t) ng * M.up_bytes, 0, kTail, cst_cp));
                CK(cudaMemsetAsync(mmq_dn_b[b] + (size_t) ng * M.d_bytes, 0, kTail, cst_cp));
                CK(cudaEventRecord(ev_cp[b], cst_cp));
                for (int i = 0; i < np; ++i) xs.done(pend[i], cst_cp);
                np = 0;
                CK(cudaGetLastError());

                MMQ::quantize(hl, ent_tok + row0, mmq_xq, M.L.gu_type, LAT, LAT, rows, st);
                CK(cudaStreamWaitEvent(st, ev_cp[b], 0));
                CK(cudaMemsetAsync(ubuf, 0, (size_t) rows * (size_t) FF * 4, st));
                MMQ::Product up;
                up.w = mmq_up_b[b];
                up.type = M.L.gu_type;
                up.w_rows = FF;
                up.w_cols = LAT;
                up.expert_bytes = M.up_bytes;
                up.n = ng;
                up.xq = mmq_xq;
                up.bounds = mmq_bounds_b[b];
                up.ids = mmq_ids;
                up.total_rows = rows;
                up.max_rows = mr;
                up.dst = ubuf;
                up.ld_dst = FF;
                mmq_ctx->run(up, st);
                k_relu2<<<grid1((int64_t) rows * FF), kThreads, 0, st>>>(ubuf, (int64_t) rows * FF);
                MMQ::quantize(ubuf, nullptr, mmq_xq, M.L.d_type, FF, FF, rows, st);
                CK(cudaMemsetAsync(eparts + (size_t) row0 * LAT, 0, (size_t) rows * (size_t) LAT * 4, st));
                MMQ::Product dn;
                dn.w = mmq_dn_b[b];
                dn.type = M.L.d_type;
                dn.w_rows = LAT;
                dn.w_cols = FF;
                dn.expert_bytes = M.d_bytes;
                dn.n = ng;
                dn.xq = mmq_xq;
                dn.bounds = mmq_bounds_b[b];
                dn.ids = mmq_ids;
                dn.total_rows = rows;
                dn.max_rows = mr;
                dn.dst = eparts + (size_t) row0 * LAT;
                dn.ld_dst = LAT;
                mmq_ctx->run(dn, st);
                CK(cudaEventRecord(ev_mmq_done[b], st));
                mmq_buf_live[b] = true;
                mmq_buf ^= 1;
                row0 += rows;
                ng = 0;
                grows = 0;
            } else if (np > 0) {
                for (int i = 0; i < np; ++i) {
                    CK(cudaStreamWaitEvent(cst_cp, xs.ready[(size_t) xs.slot_index(pend[i])], 0));
                    xs.done(pend[i], cst_cp);
                }
                np = 0;
            }
        };

        for (int xid = 0; xid < E_; ++xid) {
            const int c = h_cnt[xid];
            if (c > mmq_rows_cap) {
                std::fprintf(stderr, "strata-nh: expert %d has %d rows, the MMQ group holds %d\n", xid, c, mmq_rows_cap);
                std::exit(1);
            }
            if (!warming) {
                Hh.score[(size_t) xid] += c;
                Hh.total[(size_t) xid] += c;
            }
            const bool res = Hh.slot_of[(size_t) xid] >= 0;
            long item = -1;
            const uint8_t* w = nullptr;
            if (c <= 0) {
                if (!res) {
                    if (np >= hold_lim) flush();
                    item = xs_pos++;
                    (void) xs.pointer(item);
                    ++hy_streamed;
                    pend[np++] = item;
                }
                continue;
            }
            if (ng == kMmqGroup || (ng > 0 && grows + c > mmq_rows_cap) || (!res && np >= hold_lim)) flush();
            if (!res) {
                item = xs_pos++;
                w = xs.pointer(item);
                ++hy_streamed;
                pend[np++] = item;
            } else {
                w = M.slots + (size_t) Hh.slot_of[(size_t) xid] * M.stride;
            }
            grp[ng++] = {w, c, item};
            grows += c;
        }
        flush();
        prof.mark(st, Prof::MoeExperts);
        if (row0 != n_ent) {
            std::fprintf(stderr, "strata-nh: MMQ grouped %d rows, expected %d\n", row0, n_ent);
            std::exit(1);
        }
    }
#endif

    /// The MoE block for T rows of h: y = latent_up(sum_k w_k expert_k(latent_down h)), y2 = the shared expert.
    void moe(const Layer& Lr, int T) {
        const Geo& g = m->g;
        const Moe& M = Lr.moe;
        const int mi = Lr.idx, N = g.N, Kp = g.top_k, LAT = g.LAT, E_ = M.E, FF = M.FF;
        const int n_ent = T * Kp;
        f32mm(M.router, h, logits_r, N, E_, T);
        k_route<<<T, 512, 0, st>>>(logits_r, M.bias, E_, Kp, g.w_scale, g.w_norm, ids, ewts);
        prep(h, N, T, xq);                     // (h's image serves the latent and the shared expert's up)
        apply(M.lat_down, xq, hl, T);
        prof.mark(st, Prof::MoePre);
        const int32_t* prow = nullptr;
        if (T <= hybrid_t) {
            // ---- rows: the resident experts on the GPU, the others on the CPU pool, side by side
            Hy& Hh = hy[(size_t) mi];
            quant(hl, LAT, T, xq_l);
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_hl, hl, (size_t) T * LAT * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
            k_group_hy<<<1, 512, 0, st>>>(ids, n_ent, Kp, E_, Hh.slot_of_d, M.slots, M.stride, grp_ptr, grp_start,
                                          n_groups, ent_dst, ent_tok);
            K::native_expert_grouped_relu2(M.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                           std::min<int64_t>(n_ent, E_), n_ent, xq_l, escr, eparts, st);
            shared(M, T);
            const auto tw = Clock::now();
            CK(cudaEventSynchronize(ev_route));
            t_route_wait_ms += ms_since(tw);
            const auto t0 = Clock::now();
            // every entry: resident (0), prefetched (1), the PCIe share (2) or the CPU's (3)
            const int pb = (pf_cap > 0 && pf[mi & 1].mi == mi) ? (mi & 1) : -1;
            std::vector<int> cls((size_t) n_ent, 0), miss_x;
            std::vector<int> pc_slot;   // expert -> its place in pcbuf (lazily sized)
            for (int e = 0; e < n_ent; ++e) {
                const int xid = h_ids[e];
                if (!warming) { Hh.score[(size_t) xid] += 1.0; Hh.total[(size_t) xid] += 1.0; }
                if (Hh.slot_of[(size_t) xid] >= 0) cls[(size_t) e] = 0;
                else if (pb >= 0 && pf[pb].slot[(size_t) xid] >= 0) cls[(size_t) e] = 1;
                else {
                    cls[(size_t) e] = 3;
                    miss_x.push_back(xid);
                }
            }
            // the PCIe share: that part of the distinct missed experts, the ones with the most rows first (in pick order
            // on ties: the router's ranking), uploaded on the copy stream while the CPU computes the others
            int n_pc = 0;
            if (pcie_cap > 0 && pcie_frac > 0.f && !miss_x.empty()) {
                std::vector<int> cnt_x;
                std::vector<int> distinct;
                std::map<int, int> c;
                for (int x : miss_x) if (c[x]++ == 0) distinct.push_back(x);
                std::stable_sort(distinct.begin(), distinct.end(), [&](int a, int b) { return c[a] > c[b]; });
                const int want = std::min<int>(pcie_cap, (int) std::lround(pcie_frac * (double) distinct.size()));
                pc_slot.assign((size_t) E_, -1);
                CK(cudaStreamWaitEvent(cst2, ev_trfree, 0));
                for (int i = 0; i < want; ++i) {
                    const int x = distinct[(size_t) i];
                    pc_slot[(size_t) x] = n_pc;
                    CK(cudaMemcpyAsync(pcbuf + (size_t) n_pc * tr_stride, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice, cst2));
                    CK(cudaMemcpyAsync(pcbuf + (size_t) n_pc * tr_stride + M.L.down_off, M.down_of(x), M.d_bytes,
                                       cudaMemcpyHostToDevice, cst2));
                    ++n_pc;
                }
                CK(cudaEventRecord(ev_pc, cst2));
                for (int e = 0; e < n_ent; ++e)
                    if (cls[(size_t) e] == 3 && pc_slot[(size_t) h_ids[e]] >= 0) cls[(size_t) e] = 2;
            }
            // the transient experts' groups (prefetched + PCIe), built here, one grouped call after their uploads
            {
                std::map<int, std::vector<int>> grp;   // expert -> its entries (ascending id, entry order)
                for (int e = 0; e < n_ent; ++e)
                    if (cls[(size_t) e] == 1 || cls[(size_t) e] == 2) grp[h_ids[e]].push_back(e);
                if (!grp.empty()) {
                    int ng = 0, pos = 0;
                    for (const auto& [x, es] : grp) {
                        const bool pre = cls[(size_t) es[0]] == 1;
                        h_tgp[ng] = (unsigned long long) (pre ? pfbuf[pb] + (size_t) pf[pb].slot[(size_t) x] * tr_stride
                                                              : pcbuf + (size_t) pc_slot[(size_t) x] * tr_stride);
                        h_tgs[ng] = pos;
                        for (int e : es) {
                            h_ted[pos] = e;
                            h_tet[pos] = e / Kp;
                            ++pos;
                        }
                        ++ng;
                    }
                    h_tgs[ng] = pos;
                    *h_tng = ng;
                    CK(cudaMemcpyAsync(d_tgp, h_tgp, (size_t) ng * 8, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_tgs, h_tgs, (size_t) (ng + 1) * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_ted, h_ted, (size_t) pos * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_tet, h_tet, (size_t) pos * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_tng, h_tng, 4, cudaMemcpyHostToDevice, st));
                    if (n_pc > 0) CK(cudaStreamWaitEvent(st, ev_pc, 0));
                    if (pb >= 0) CK(cudaStreamWaitEvent(st, ev_pf[pb], 0));
                    K::native_expert_grouped_relu2(M.L, d_tgp, d_tgs, d_tng, d_ted, d_tet, ng, pos, xq_l, escr, eparts, st);
                }
                CK(cudaEventRecord(ev_trfree, st));
                if (pb >= 0) {
                    CK(cudaEventRecord(ev_pffree[pb], st));
                    pf[pb].mi = -1;
                }
            }
            jobs.clear();
            int n_miss = 0, n_res = 0, n_pf = 0;
            for (int e = 0; e < n_ent; ++e) {
                const int c = cls[(size_t) e];
                h_miss[e] = c == 3;
                n_res += c == 0;
                n_pf += c == 1;
                if (c != 3) continue;
                const int xid = h_ids[e];
                ++n_miss;
                strata::nemotron_h::CpuJob j;
                j.up = M.up_of(xid);
                j.down = M.down_of(xid);
                j.tok = e / Kp;
                j.out = cpu_out + (size_t) e * LAT;
                jobs.push_back(j);
            }
            if (!warming) {
                hy_cpu += n_miss;
                hy_gpu += n_res;
                tr_pf_hit += n_pf;
                tr_pcie += n_ent - n_miss - n_res - n_pf;
            }
            if (n_miss > 0) {
                pool->run(M.fmt, h_hl, T, jobs.data(), (int) jobs.size());
                hy_cpu_ms += ms_since(t0);
                CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * LAT * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
                k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, LAT);
            }
        } else {
            // ---- a prompt chunk: the entries sorted by expert, every expert's rows as two FP16 GEMMs, its weights from
            // its slot or from the stream
            CK(cudaMemsetAsync(rcnt, 0, (size_t) E_ * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, E_, rbounds, rcur);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
#if defined(STRATA_PREFILL_MMQ)
            if (mmq_on && mmq_ok[(size_t) mi]) {
                moe_stream_mmq(M, mi, T);
            } else
#endif
            {
            k_gather_f16<<<grid1((int64_t) n_ent * LAT), kThreads, 0, st>>>(hl, ent_tok, xg16, n_ent, LAT);
            CK(cudaMemcpyAsync(h_cnt, rcnt, (size_t) E_ * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            Hy& Hh = hy[(size_t) mi];
            int b0 = 0;
            for (int xid = 0; xid < E_; ++xid) {
                const int c = h_cnt[xid];
                if (!warming) { Hh.score[(size_t) xid] += c; Hh.total[(size_t) xid] += c; }
                const bool res = Hh.slot_of[(size_t) xid] >= 0;
                long item = -1;
                const uint8_t* w = nullptr;
                if (res) w = M.slots + (size_t) Hh.slot_of[(size_t) xid] * M.stride;
                else {
                    item = xs_pos++;
                    w = xs.take(item, st);
                    ++hy_streamed;
                }
                if (c > 0) {
                    gemm.native((const uint16_t*) (xg16 + (size_t) b0 * LAT), M.L.gu_type, w, ubuf, c, FF, LAT);
                    k_relu2_f16<<<grid1((int64_t) c * FF), kThreads, 0, st>>>(ubuf, hb16, (int64_t) c * FF);
                    gemm.native((const uint16_t*) hb16, M.L.d_type, w + M.L.down_off, eparts + (size_t) b0 * LAT, c, LAT, FF);
                }
                if (!res) xs.done(item, st);
                b0 += c;
            }
            prof.mark(st, Prof::MoeExperts);
            shared(M, T);
            prof.mark(st, Prof::MoeShared);
            }
            prow = rinv;
        }
        k_combine<<<T, kThreads, 0, st>>>(eparts, prow, ewts, Kp, lat, LAT);
        mm(M.lat_up, lat, y, T);
        prof.mark(st, Prof::MoeLat);
    }
    /// y2 = down(relu(up h)^2), h's image already in xq / x16 (prep in moe)
    void shared(const Moe& M, int T) {
        const Geo& g = m->g;
        prep(h, g.N, T, xq);
        apply(M.sh_up, xq, su, T);
        k_relu2<<<grid1((int64_t) T * g.FS), kThreads, 0, st>>>(su, (int64_t) T * g.FS);
        mm(M.sh_down, su, y2, T);
    }

    /// The prompt-path stream plan for a forward over the MoE slot indices `mis` (in order).
    void stream_plan(const std::vector<int>& mis) {
        std::vector<ExpertStream::Item> items;
        for (int mi : mis) {
            const Moe& M = moe_layer(mi).moe;
            const Hy& Hh = hy[(size_t) mi];
            for (int xid = 0; xid < M.E; ++xid)
                if (Hh.slot_of[(size_t) xid] < 0)
                    items.push_back({M.up_of(xid), M.down_of(xid), M.up_bytes, M.d_bytes});
        }
        xs_pos = 0;
        xs.begin(std::move(items));
    }

    void embed_inject(float* dst, int T, int64_t p0) {
        if (inject.empty()) return;
        const int N = m->g.N;
        for (auto it = inject.lower_bound(p0); it != inject.end() && it->first < p0 + T; ++it)
            CK(cudaMemcpyAsync(dst + (size_t) (it->first - p0) * N, it->second, (size_t) N * 4, cudaMemcpyHostToDevice, st));
    }

    /// The main model over T rows (device `tok`, positions p0 .., also *pos0): hmain = every row's final normed
    /// hidden; the last `heads` rows get logits and greedy picks.  `verify`: the Mamba state is left for commit().
    void forward(int T, int64_t p0, int heads, bool verify) {
        const Geo& g = m->g;
        const int N = g.N;
        if (T > hybrid_t) {
            std::vector<int> mis;
            for (int i = 0; i < m->n_moe; ++i) mis.push_back(i);
            stream_plan(mis);
            prof.begin(st);
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        embed_inject(x, T, p0);
        k_rms_ld<<<T, kNormThreads, 0, st>>>(x, N, m->layers[0].norm, h, N, N, g.eps);
        prof.mark(st, Prof::Other);
        const size_t nl = m->layers.size();
        if (dump_f) dump_buf.assign(nl * (size_t) T * N, 0.f);
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            const bool last = l + 1 == m->layers.size();
            if (feat16 && df_col[l] >= 0) {
                const int R = std::min(T, kFeatRows);
                k_feat16<<<R, 256, 0, st>>>(x + (size_t) (T - R) * N, N, feat16 + (size_t) df_col[l] * N, df_ntgt * N);
            }
            const float* wn = last ? m->out_norm : m->layers[l + 1].norm;
            float* hout = last ? hmain : h;
            // the prefetch: the next sublayer is a MoE block in the rows path -> guess its experts now, upload them while
            // this sublayer runs
            const bool pfn = pf_cap > 0 && T <= hybrid_t && !last && L.kind != Kind::Moe &&
                             m->layers[l + 1].kind == Kind::Moe;
            if (pfn) prefetch_begin(m->layers[l + 1], T);
            if (L.kind == Kind::Mamba) {
                mamba(L, T, verify);
                k_add_norm<<<T, kNormThreads, 0, st>>>(x, y, nullptr, wn, hout, N, g.eps, false);
                prof.mark(st, Prof::AddNorm);
                if (pfn) prefetch_issue(m->layers[l + 1], T);
            } else if (L.kind == Kind::Attn) {
                attention(L, T, p0, pos0);
                prof.mark(st, Prof::Attn);
                k_add_norm<<<T, kNormThreads, 0, st>>>(x, y, nullptr, wn, hout, N, g.eps, false);
                prof.mark(st, Prof::AddNorm);
                if (pfn) prefetch_issue(m->layers[l + 1], T);
            } else {
                moe(L, T);
                k_add_norm<<<T, kNormThreads, 0, st>>>(x, y, y2, wn, hout, N, g.eps, false);
                prof.mark(st, Prof::AddNorm);
            }
            if (dump_f)
                CK(cudaMemcpyAsync(dump_buf.data() + l * (size_t) T * N, x, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        }
        if (dump_f && !verify) {
            CK(cudaStreamSynchronize(st));
            for (int t = 0; t < T; ++t)
                for (size_t l = 0; l < nl; ++l) std::fwrite(dump_buf.data() + (l * (size_t) T + t) * N, 4, (size_t) N, dump_f);
        }
        if (T > hybrid_t) xs.finish();
        if (heads > 0) {
            head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
            prof.mark(st, Prof::Other);
        }
        if (T > hybrid_t) prof.finish();
    }
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, m->g.V, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t and the seed hidden sd[t]; hmtp gets
    /// each row's head-normed output.  `draft`: logits of the last row and its pick (pick_m).
    void forward_mtp(int T, int64_t p0, const float* sd, bool draft) {
        const Geo& g = m->g;
        const int N = g.N;
        if (T > hybrid_t) stream_plan({m->n_moe});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        embed_inject(y, T, p0);
        k_rms_ld<<<T, kNormThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, g.eps);        // [e | ...]
        k_rms_ld<<<T, kNormThreads, 0, st>>>(sd, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, g.eps);   // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms_ld<<<T, kNormThreads, 0, st>>>(x, N, m->mtp_attn.norm, h, N, N, g.eps);
        attention(m->mtp_attn, T, p0, pos0_m);
        k_add_norm<<<T, kNormThreads, 0, st>>>(x, y, nullptr, m->mtp_post_norm, h, N, g.eps, false);
        moe(m->mtp_moe, T);
        k_add_norm<<<T, kNormThreads, 0, st>>>(x, y, y2, m->mtp_shnorm, hmtp, N, g.eps, true);
        if (T > hybrid_t) xs.finish();
        if (draft) head_rows(hmtp + (size_t) (T - 1) * N, 1, logits_m, pick_m);
    }

    /// After a verify window: the Mamba state takes the first n_keep rows (the conv and the scan again over the
    /// window's kept inputs).
    void commit(int n_keep) {
        const Geo& g = m->g;
        const int vcols = g.in_proj - g.ssm_inner;
        for (const Layer& L : m->layers) {
            if (L.kind != Kind::Mamba) continue;
            const float* vb = vbuf[(size_t) L.idx];
            k_conv<<<grid1(g.conv_ch, 128), 128, 0, st>>>(conv[(size_t) L.idx], vb, vcols, L.conv_w, L.conv_b, xc, g.conv_ch,
                                                          n_keep, true);
            k_ssm_fast<<<dim3(8, g.ssm_H), 256, 0, st>>>(S[(size_t) L.idx], xc, g.conv_ch, vb + g.conv_ch, vcols, L.dt_b,
                                                        L.A, L.D, nullptr, g.ssm_H, g.ssm_G, n_keep, true);
        }
    }

    // ---- the hybrid mode's slots
    /// The slots to the experts the routing used most: at most max_swaps replacements over all layers, between
    /// forwards; the scores halve each time.
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_NH_FIXED_SLOTS") != nullptr;
        if (off || warming) return;
        int left = max_swaps;
        CK(cudaStreamSynchronize(st));
        for (size_t mi = 0; mi < hy.size() && left > 0; ++mi) {
            Hy& Hh = hy[mi];
            const Moe& M = moe_layer((int) mi).moe;
            const int ne = M.E, S_ = M.n_slots;
            if (S_ <= 0 || S_ >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x_ = 0; x_ < ne; ++x_) order[(size_t) x_] = x_;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return Hh.score[(size_t) a] > Hh.score[(size_t) b]; });
            std::vector<int> victims;
            for (int s = 0; s < S_; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return Hh.score[(size_t) Hh.exp_in_slot[(size_t) a]] < Hh.score[(size_t) Hh.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S_ && left > 0 && vi < victims.size(); ++i) {
                const int xid = order[(size_t) i];
                if (Hh.slot_of[(size_t) xid] >= 0) continue;
                const int s = victims[vi];
                const int yid = Hh.exp_in_slot[(size_t) s];
                if (Hh.score[(size_t) yid] >= Hh.score[(size_t) xid]) break;
                ++vi;
                Hh.slot_of[(size_t) yid] = -1;
                Hh.slot_of[(size_t) xid] = s;
                Hh.exp_in_slot[(size_t) s] = xid;
                uint8_t* dst = M.slots + (size_t) s * M.stride;
                CK(cudaMemcpyAsync(dst, M.up_of(xid), M.up_bytes, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(dst + M.L.down_off, M.down_of(xid), M.d_bytes, cudaMemcpyHostToDevice, st));
                --left;
                ++hy_swaps;
                changed = true;
            }
            if (changed) CK(cudaMemcpyAsync(Hh.slot_of_d, Hh.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : Hh.score) v *= 0.5;
            CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable
        }
    }

    /// The tokens at the current position: a prompt in chunks, one token as a decode step.  The last token's logits
    /// stay in `logits` (its pick in `pick`), hlast its normed hidden.  With drafts on, the prompt also runs through
    /// the MTP block (its KV for every position: row q reads h_{q-1} and token q).
    template <class F> void process(const int32_t* toks, int64_t n, F on_chunk) {
        const int N = m->g.N;
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            forward(T, pos, i + T == n ? 1 : 0, false);
            if (df) {
                const int R = std::min(T, kFeatRows);
                df->encode_and_inject_f16(feat16, R, (int) (pos + T - R), st);
            }
            if (m->has_mtp && mtp_d > 0 && prompt_mtp) {
                CK(cudaMemcpyAsync(seeds, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1) CK(cudaMemcpyAsync(seeds + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, seeds, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            on_chunk(i + T);
            if (T == 1) {
                if (pos % adapt_every == 0) rebalance(adapt_swaps);
            }
        }
        if (n > 1) rebalance(swaps_after_prompt);
    }

    /// One speculative step with D MTP drafts (greedy), src/qwen35's spec_step: `x_tok` is the next token (decided,
    /// not processed), `fix` the drafts the last window accepted (the MTP block redoes their rows first).  Returns
    /// the accepted count a; picks[0 .. a] are the window's picks.
    /// Sampled verification: given the D drafts (the window's D + 1 logits rows are in `logits`), returns the
    /// accepted count a and sets picks[0 .. a].  Null: greedy (the window's argmax picks).
    using Judge = std::function<int(int D, const int32_t* drafts, int32_t* picks)>;
    int judged(int D, const std::vector<int32_t>& drafts, int32_t* picks, const Judge* judge) {
        if (judge) return (*judge)(D, drafts.data(), picks);
        int a = 0;
        while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        return a;
    }
    /// The first `rows` rows of `logits` (a verify window's) on the host, row-major.
    void host_logits_rows(int rows, std::vector<float>& out) {
        out.resize((size_t) rows * m->g.V);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const Judge* judge = nullptr) {
        const int N = m->g.N;
        const int T0 = (int) fix.size() + 1;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        if (!fix.empty()) CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = hpin[2];
        }
        int last_row = T0 - 1;
        for (int d = 1; d < D; ++d) {
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            forward_mtp(1, pos + d, seeds, true);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = hpin[2];
        }
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        forward(T, pos, T, true);
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int a = judged(D, drafts, picks, judge);
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        const int64_t before = pos;
        pos += a + 1;
        if (before / adapt_every != pos / adapt_every) rebalance(adapt_swaps);
        return a;
    }

    /// One DFlash step: the drafter's block from `x_tok` (decided, not processed) at pos, D of its drafts verified
    /// in one window of D + 1 rows; the kept rows' features go into the drafter.  Returns the accepted count a;
    /// picks[0 .. a] are the window's picks.
    int dflash_step(int x_tok, int D, std::vector<int32_t>& drafts, int32_t* picks, const Judge* judge = nullptr) {
        const int N = m->g.N;
        CK(cudaStreamSynchronize(st));
        const std::vector<int32_t> blk = df->draft(x_tok, (int32_t) pos, nullptr, st);
        df_ms += df->last_draft_ms();
        ++df_steps;
        drafts.assign(blk.begin(), blk.begin() + D);
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        forward(T, pos, T, true);
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int a = judged(D, drafts, picks, judge);
        commit(a + 1);
        df->encode_and_inject_f16(feat16, a + 1, (int) pos, st);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        const int64_t before = pos;
        pos += a + 1;
        if (before / adapt_every != pos / adapt_every) rebalance(adapt_swaps);
        return a;
    }
    /// The drafts a speculative step offers (DFlash's, else the MTP block's; 0: no speculation)
    int spec_d() const { return df ? df_d : mtp_d; }
    int spec_any(int x_tok, std::vector<int32_t>& fix, std::vector<int32_t>& drafts, int32_t* picks,
                 const Judge* judge = nullptr) {
        if (df) return dflash_step(x_tok, df_d, drafts, picks, judge);
        return spec_step(x_tok, fix, mtp_d, drafts, picks, judge);
    }
    /// The drafter on the GPU before the expert slots are sized: its weights (FP16), the shared embedding and head
    /// (the engine's own Q8_0 copies), the feature buffer, and one inject + draft so its lazy buffers exist too.
    bool load_dflash(const std::string& path, int d, std::string& err) {
        const Geo& g = m->g;
        if (m->embd_type != 8 || m->head.type != 8) { err = "DFlash needs the target's token_embd and output in Q8_0"; return false; }
        df = std::make_unique<strata::nemotron_h::DFlashDrafter>();
        if (!df->load(path, "", err)) return false;
        df->set_target_weights(m->embd, m->head.d, g.V);
        const auto& cfg = df->config();
        if (cfg.n_embd != g.N) { err = "the DFlash GGUF's width is not the model's"; return false; }
        df_ntgt = (int) cfg.target_layers.size();
        df_col.assign(m->layers.size(), -1);
        for (int k = 0; k < df_ntgt; ++k) {
            const int l = cfg.target_layers[(size_t) k];
            if (l < 0 || l >= (int) m->layers.size()) { err = "a DFlash target layer past the model's"; return false; }
            df_col[(size_t) l] = k;
        }
        df_d = std::max(1, std::min(d, cfg.n_draft));
        feat16 = dalloc_t<__half>((size_t) kFeatRows * df_ntgt * g.N);
        CK(cudaMemsetAsync(feat16, 0, (size_t) kFeatRows * df_ntgt * g.N * 2, st));
        df->encode_and_inject_f16(feat16, std::min(tmax, kFeatRows), 0, st);
        (void) df->draft(0, std::min(tmax, kFeatRows), nullptr, st);
        df->reset();
        CK(cudaStreamSynchronize(st));
        return true;
    }

    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        // the rows path only (a chunk above --hybrid-rows would stream every non-resident expert: ~70 GB)
        const int32_t filler = 1010;
        std::vector<int32_t> toks((size_t) std::min(hybrid_t, kVerify), filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        process(&filler, 1, [](int64_t) {});
        if (spec_d() > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_any(filler, fix, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& Hh : hy) std::fill(Hh.score.begin(), Hh.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = hy_streamed = 0;
        tr_pcie = tr_pf_sent = tr_pf_hit = 0;
        hy_cpu_ms = t_route_wait_ms = 0;
        reset();
        std::fprintf(stderr, "strata-nh: warmed up in %.1f s\n", ms_since(t0) / 1000.0);
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.V);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// Bytes cudaMalloc takes for `s` slots on every MoE layer (one buffer per layer, 2 MiB rounding, the tail included).
size_t slots_bytes(const std::vector<Moe*>& all, int s) {
    size_t b = 0;
    for (Moe* M : all) {
        const size_t nb = (size_t) std::max(std::min(M->E, s), 1) * M->stride + kTail;
        b += (nb + kAllocGran - 1) & ~(kAllocGran - 1);
    }
    return b;
}

/// MemAvailable from /proc/meminfo in bytes (0 when unknown).
size_t mem_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) return (size_t) ms.ullAvailPhys;
    return 0;
#else
    std::ifstream f("/proc/meminfo");
    std::string k;
    size_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") return v * 1024;
    return 0;
#endif
}

/// The expert profile (Flash-Next's expert-profile.bin idea for this family): per MoE slot index (the main layers,
/// then the MTP block) every expert's routing count.  'NHP1', n_moe, n_expert, then f32 [n_moe][n_expert].  The slots
/// start with each layer's most-routed experts instead of 0 .. S - 1.
bool load_profile(const std::string& path, int n_moe, int E, std::vector<std::vector<double>>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hd[3];
    f.read((char*) hd, sizeof hd);
    if (!f || hd[0] != 0x3150484e || hd[1] != n_moe || hd[2] != E) return false;
    out.assign((size_t) n_moe, std::vector<double>((size_t) E, 0.0));
    std::vector<float> row((size_t) E);
    for (int i = 0; i < n_moe; ++i) {
        f.read((char*) row.data(), (std::streamsize) (E * 4));
        if (!f) return false;
        for (int x = 0; x < E; ++x) out[(size_t) i][(size_t) x] = row[(size_t) x];
    }
    return true;
}
bool save_profile(const std::string& path, const std::vector<Engine::Hy>& hy, int E) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) return false;
        const int32_t hd[3] = {0x3150484e, (int32_t) hy.size(), E};
        f.write((const char*) hd, sizeof hd);
        std::vector<float> row((size_t) E);
        for (const Engine::Hy& H : hy) {
            std::fill(row.begin(), row.end(), 0.f);
            for (size_t x = 0; x < H.total.size() && x < (size_t) E; ++x) row[x] = (float) H.total[x];
            f.write((const char*) row.data(), (std::streamsize) (E * 4));
        }
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

/// Every MoE layer (and the MTP block's) gets the same number of VRAM slots from the VRAM left after the engine's
/// buffers, the stream ring and the transient buffers (the PCIe share and the prefetch); `vram_gib` >= 0: that budget
/// instead, `slots` > 0: that many.  With `arena` (and the RAM for it) every expert is copied once into pinned host
/// memory, as Flash-Next's expert arena: the CPU pool reads it, and every upload (slots, the PCIe share, the prefetch,
/// the prompt stream) is a DMA at full PCIe speed.  `profile`: the slots start with its most-routed experts.
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int slots, int ring, bool arena, int reserve_mib,
                   const std::string& profile, std::string& err) {
    const auto t0 = Clock::now();
    const Geo& g = m.g;
    std::vector<Moe*> all;
    for (Layer& L : m.layers)
        if (L.kind == Kind::Moe) all.push_back(&L.moe);
    if (m.has_mtp) all.push_back(&m.mtp_moe.moe);
    size_t per_slot = 0, max_stride = 0, total = 0;
    int min_e = g.E;
    for (Moe* M : all) {
        per_slot += M->stride;
        max_stride = std::max(max_stride, M->stride);
        total += (size_t) M->E * M->stride;
        min_e = std::min(min_e, M->E);
    }
    // ---- the pinned arena
    bool have_arena = false;
    if (arena) {
        const size_t avail = mem_available();
        const size_t keep = (size_t) 6 << 30;   // the OS, the server, the vision encoder
        if (avail > total + keep) {
            uint8_t* base = nullptr;
            const char* how = "cudaHostAlloc";
#if !defined(_WIN32)
            // 2 MiB pages (transparent huge pages, madvise): a decode step reads ~700 experts of 3.8 MB each at random
            // places in 73 GB - with 4 KiB pages that is ~650,000 page walks per token; then registered with CUDA.
            // STRATA_NH_ARENA_4K=1: cudaHostAlloc's 4 KiB pages instead.
            if (std::getenv("STRATA_NH_ARENA_4K") == nullptr) {
                const size_t hp = (size_t) 2 << 20, len = (total + hp - 1) / hp * hp;
                void* q = mmap(nullptr, len + hp, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (q != MAP_FAILED) {
                    uint8_t* al = (uint8_t*) (((uintptr_t) q + hp - 1) & ~(uintptr_t) (hp - 1));
                    madvise(al, len, MADV_HUGEPAGE);
                    if (cudaHostRegister(al, len, cudaHostRegisterPortable) == cudaSuccess) {
                        base = al;
                        how = "huge pages, registered";
                    } else {
                        (void) cudaGetLastError();
                        munmap(q, len + hp);
                    }
                }
            }
#endif
            if (base == nullptr && cudaHostAlloc((void**) &base, total, cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                base = nullptr;
            }
            if (base != nullptr) {
                size_t off = 0;
                std::vector<std::pair<Moe*, int>> items;
                for (Moe* M : all) {
                    M->host = base + off;
                    off += (size_t) M->E * M->stride;
                    for (int e = 0; e < M->E; ++e) items.push_back({M, e});
                }
                // filled by 6 threads (from the page cache, or the disk on a cold start)
                std::atomic<int> next{0};
                const int n_items = (int) items.size();
                std::vector<std::thread> ts;
                for (int k = 0; k < 6; ++k)
                    ts.emplace_back([&] {
                        for (int i; (i = next.fetch_add(1)) < n_items;) {
                            Moe& M = *items[(size_t) i].first;
                            const int e = items[(size_t) i].second;
                            uint8_t* dst = M.host + (size_t) e * M.stride;
                            std::memcpy(dst, M.src_up + (size_t) e * M.up_bytes, M.up_bytes);
                            std::memcpy(dst + M.L.down_off, M.src_down + (size_t) e * M.d_bytes, M.d_bytes);
                        }
                    });
                for (auto& t : ts) t.join();
                have_arena = true;
            }
            std::fprintf(stderr, "strata-nh: expert arena: %s (%s, %.2f GiB pinned, %.1f s)\n",
                         have_arena ? "every routed expert in pinned RAM" : "pinning failed, the mapped file is read", how,
                         have_arena ? (double) total / (1 << 30) : 0.0, ms_since(t0) / 1000.0);
        } else {
            std::fprintf(stderr, "strata-nh: expert arena: not enough RAM (%.1f GiB available, %.1f needed), the mapped file "
                         "is read\n", (double) avail / (1 << 30), (double) (total + keep) / (1 << 30));
        }
    }
    if (!have_arena) E.xs.init_bounce();
    std::vector<std::vector<double>> prof;
    const bool have_prof = !profile.empty() && load_profile(profile, (int) all.size(), g.E, prof);
    if (!profile.empty())
        std::fprintf(stderr, "strata-nh: expert profile %s: %s\n", profile.c_str(), have_prof ? "loaded" : "not usable, slots start at 0 .. S-1");
    E.hy.resize(all.size());
    for (size_t mi = 0; mi < all.size(); ++mi) CK(cudaMalloc(&E.hy[mi].slot_of_d, (size_t) all[mi]->E * 4));
    E.pool = std::make_unique<strata::nemotron_h::CpuExperts>(threads);
    CK(cudaDeviceSynchronize());
    size_t fr = 0, tot = 0;
    vram_info(fr, tot);
    const size_t reserve = (size_t) std::max(256, reserve_mib) << 20;   // --vram-reserve-mib (the server's: 700)
    const size_t target_free = reserve + ((size_t) 32 << 20);           // headroom for runtime/desktop jitter
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30)) : (fr > target_free ? fr - target_free : 0);
    auto vram_for = [&](int s) { return slots_bytes(all, s); };
    int S_ = 0;
    if (slots > 0) {
        S_ = std::min(g.E, slots);
    } else {
        while (S_ < g.E && vram_for(S_ + 1) <= budget) ++S_;
    }
    size_t vram_slots = 0;
    auto alloc_slots = [&]() {
        vram_slots = 0;
        for (Moe* M : all) {
            const int ns = std::min(M->E, S_);
            M->n_slots = ns;
            vram_slots += (size_t) ns * M->stride;
            const size_t nb = M->stride * (size_t) std::max(ns, 1) + kTail;
            CK(cudaMalloc(&M->slots, nb));
            CK(cudaMemset(M->slots, 0, nb));
        }
        CK(cudaDeviceSynchronize());
        vram_info(fr, tot);
    };
    alloc_slots();
    while (slots <= 0 && vram_gib < 0 && fr < target_free && S_ > 1) {
        const size_t deficit = target_free - fr;
        size_t step = 0;
        for (Moe* M : all)
            if (M->E >= S_) step += M->stride;
        if (step == 0) step = per_slot;
        const int drop = std::max(1, (int) ((deficit + step - 1) / step));
        S_ = std::max(1, S_ - drop);
        for (Moe* M : all) {
            CK(cudaFree(M->slots));
            M->slots = nullptr;
        }
        alloc_slots();
    }
    for (size_t mi = 0; mi < all.size(); ++mi) {
        Moe& M = *all[mi];
        const int ne = M.E, ns = M.n_slots;
        std::vector<int> order((size_t) ne);
        for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
        if (have_prof)
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return prof[mi][(size_t) a] > prof[mi][(size_t) b]; });
        Engine::Hy& H = E.hy[mi];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) ns, 0);
        for (int s = 0; s < ns; ++s) {
            const int x = order[(size_t) s];
            H.slot_of[(size_t) x] = s;
            H.exp_in_slot[(size_t) s] = x;
            CK(cudaMemcpy(M.slots + (size_t) s * M.stride, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(M.slots + (size_t) s * M.stride + M.L.down_off, M.down_of(x), M.d_bytes, cudaMemcpyHostToDevice));
        }
        // the profile is also the adaptive tier's prior: its counts scaled to `prior` tokens' worth of routing, so a
        // request's first counts move only the experts it really prefers (they halve with every rebalance as usual).
        // STRATA_NH_PRIOR_TOKENS: that weight (default 1024).
        H.score.assign((size_t) ne, 0.0);
        if (have_prof) {
            static const double prior = std::getenv("STRATA_NH_PRIOR_TOKENS") ? std::atof(std::getenv("STRATA_NH_PRIOR_TOKENS")) : 1024.0;
            double sum = 0;
            for (int x = 0; x < ne; ++x) sum += prof[mi][(size_t) x];
            if (sum > 0)
                for (int x = 0; x < ne; ++x) H.score[(size_t) x] = prof[mi][(size_t) x] * prior * g.top_k / sum;
        }
        H.total.assign((size_t) ne, 0.0);
        if (have_prof)
            for (int x = 0; x < ne; ++x) H.total[(size_t) x] = prof[mi][(size_t) x];
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());
    vram_info(fr, tot);
    if (min_e == g.E)
        std::fprintf(stderr, "strata-nh: hybrid experts: %d of %d per layer in VRAM (%.2f GiB of %.2f GiB), a batch of <= %d rows "
                     "on the GPU + %d CPU threads (PCIe share %.2f, up to %d experts; prefetch %d), bigger chunks streamed through "
                     "%d buffers (%.2f GiB), %.2f GiB (%zu MiB) of VRAM free; %.1f s\n", S_, g.E, (double) vram_slots / (1 << 30),
                     (double) total / (1 << 30), E.hybrid_t, E.pool->threads(), E.pcie_frac, E.pcie_cap, E.pf_cap, E.xs.nr,
                     (double) E.xs.nr * max_stride / (1 << 30), (double) fr / (1 << 30), fr >> 20, ms_since(t0) / 1000.0);
    else
        std::fprintf(stderr, "strata-nh: hybrid experts: up to %d of %d..%d per layer in VRAM (%.2f GiB of %.2f GiB), a batch of <= %d rows "
                     "on the GPU + %d CPU threads (PCIe share %.2f, up to %d experts; prefetch %d), bigger chunks streamed through "
                     "%d buffers (%.2f GiB), %.2f GiB (%zu MiB) of VRAM free; %.1f s\n", S_, min_e, g.E, (double) vram_slots / (1 << 30),
                     (double) total / (1 << 30), E.hybrid_t, E.pool->threads(), E.pcie_frac, E.pcie_cap, E.pf_cap, E.xs.nr,
                     (double) E.xs.nr * max_stride / (1 << 30), (double) fr / (1 << 30), fr >> 20, ms_since(t0) / 1000.0);
    (void) ring;
    (void) err;
    E.vram_moe = all;
    E.slots_cap = std::max(S_, 1);
    E.slots_now = S_;
    E.vram_reserve_mib = std::max(256, reserve_mib);
    std::fprintf(stderr, "strata-nh: elastic VRAM: POST /v1/vram frees or refills expert slots "
                         "(dense weights, the KV cache and %d MiB stay)\n", E.vram_reserve_mib);
    return true;
}

/// Put `S` slots on every layer (capped by that layer's expert count).  The kept experts are the hottest by the
/// router statistics (the decayed score, then the lifetime count, then whoever is already resident).  The old
/// buffers are freed before the new ones are allocated: a shrink is asked for when both would not fit.
bool set_slots_once(Engine& E, int S, std::string& err) {
    auto fail = [&](cudaError_t e, const char* what) {
        if (e == cudaSuccess) return false;
        err = std::string("VRAM: ") + what + ": " + cudaGetErrorString(e);
        (void) cudaGetLastError();
        return true;
    };
    // A previous failed resize may have left a sticky error.  Drop it: this call rebuilds the slot buffers.
    (void) cudaDeviceSynchronize();
    (void) cudaGetLastError();
    E.xs.finish();
    for (Moe* M : E.vram_moe) {
        if (M->slots != nullptr) {
            cudaFree(M->slots);
            M->slots = nullptr;
        }
        M->n_slots = 0;
    }
    (void) cudaGetLastError();
    std::vector<uint8_t*> fresh(E.vram_moe.size(), nullptr);
    for (size_t mi = 0; mi < E.vram_moe.size(); ++mi) {
        Moe* M = E.vram_moe[mi];
        const int ns = std::min(M->E, S);
        const size_t nb = (size_t) std::max(ns, 1) * M->stride + kTail;
        const cudaError_t e = cudaMalloc((void**) &fresh[mi], nb);
        if (e != cudaSuccess) {
            err = std::string("VRAM: cudaMalloc: ") + cudaGetErrorString(e);
            (void) cudaGetLastError();
            for (uint8_t* p : fresh)
                if (p) cudaFree(p);
            return false;
        }
        if (fail(cudaMemset(fresh[mi], 0, nb), "cudaMemset")) {
            for (uint8_t* p : fresh)
                if (p) cudaFree(p);
            return false;
        }
    }
    for (size_t mi = 0; mi < E.vram_moe.size(); ++mi) {
        Moe& M = *E.vram_moe[mi];
        const int ne = M.E, ns = std::min(ne, S);
        M.slots = fresh[mi];
        M.n_slots = ns;
        Engine::Hy& H = E.hy[mi];
        if ((int) H.score.size() < ne || (int) H.total.size() < ne || H.slot_of_d == nullptr) {
            err = "VRAM: the router statistics are missing";
            return false;
        }
        std::vector<char> was((size_t) ne, 0);
        if ((int) H.slot_of.size() == ne)
            for (int x = 0; x < ne; ++x) was[(size_t) x] = H.slot_of[(size_t) x] >= 0 ? 1 : 0;
        std::vector<int> order((size_t) ne);
        for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const double sa = H.score[(size_t) a], sb = H.score[(size_t) b];
            if (sa != sb) return sa > sb;
            const double ta = H.total[(size_t) a], tb = H.total[(size_t) b];
            if (ta != tb) return ta > tb;
            if (was[(size_t) a] != was[(size_t) b]) return was[(size_t) a] > was[(size_t) b];
            return a < b;
        });
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) std::max(ns, 0), 0);
        for (int s = 0; s < ns; ++s) {
            const int x = order[(size_t) s];
            H.slot_of[(size_t) x] = s;
            H.exp_in_slot[(size_t) s] = x;
            uint8_t* dst = M.slots + (size_t) s * M.stride;
            if (fail(cudaMemcpyAsync(dst, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice, E.st), "expert upload") ||
                fail(cudaMemcpyAsync(dst + M.L.down_off, M.down_of(x), M.d_bytes, cudaMemcpyHostToDevice, E.st),
                     "expert upload"))
                return false;
        }
        if (fail(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, E.st),
                 "slot table"))
            return false;
    }
    if (fail(cudaStreamSynchronize(E.st), "cudaStreamSynchronize")) return false;
    E.slots_now = S;
    return true;
}

bool set_slots(Engine& E, int S, std::string& err) {
    S = std::max(1, std::min(S, std::max(E.slots_cap, 1)));
    if (set_slots_once(E, S, err)) return true;
    if (S == 1) return false;
    const std::string why = err;
    std::string kept;
    if (!set_slots_once(E, 1, kept)) {
        if (!kept.empty()) err = why + "; " + kept;
        return false;
    }
    std::fprintf(stderr, "strata-nh: VRAM resize to %d slots a layer failed (%s); kept 1\n", S, why.c_str());
    err.clear();
    return true;
}

/// `VRAM` or `VRAM <reserve_mib>` between requests.  Prints one `VRAM k=v ...` line on success (the server's
/// POST /v1/vram shape).  A bare command uses the reserve the engine started with.
bool vram_command(Engine& E, const std::string& cmd, std::string& err) {
    if (E.vram_moe.empty() || E.slots_cap <= 0) {
        err = "VRAM: this engine has no expert slots to resize";
        return false;
    }
    const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
    char* end = nullptr;
    const long long reserve = bare ? (long long) E.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
    if (!bare) {
        if (end == cmd.c_str() + 4 || reserve < 0) {
            err = "expected: VRAM [reserve_mib]";
            return false;
        }
        while (*end == ' ') ++end;
        if (*end != '\0') {
            err = "expected: VRAM [reserve_mib]";
            return false;
        }
    }
    const auto t0 = Clock::now();
    const int before = E.slots_now;
    // Below the startup margin the cache only refills to the size it was given at start (that size already
    // leaves the margin).  Above it, slots are dropped until the measured free VRAM covers the request.
    int S = E.slots_now;
    if (reserve <= (long long) E.vram_reserve_mib) {
        S = E.slots_cap;
    } else {
        size_t fr = 0, tot = 0;
        vram_info(fr, tot);
        const size_t cur = slots_bytes(E.vram_moe, E.slots_now);
        const size_t want = (size_t) reserve << 20;
        const size_t guard = (size_t) 256 << 20;
        const size_t budget = fr + cur > want + guard ? fr + cur - want - guard : 0;
        int t = 0;
        while (t < E.slots_cap && slots_bytes(E.vram_moe, t + 1) <= budget) ++t;
        S = std::max(1, t);
    }
    if (S != E.slots_now && !set_slots(E, S, err)) return false;
    size_t fr = 0, tot = 0;
    vram_info(fr, tot);
    const size_t target = (size_t) std::max(reserve, (long long) E.vram_reserve_mib) << 20;
    std::string note;
    for (int spins = 0; fr < target && E.slots_now > 1 && spins < 6; ++spins) {
        size_t step = 0;
        for (Moe* M : E.vram_moe)
            if (M->E >= E.slots_now) step += M->stride;
        if (step == 0) step = 1;
        const int drop = std::max(1, (int) (((target - fr) + step - 1) / step));
        const int next = std::max(1, E.slots_now - drop);
        if (next == E.slots_now) break;
        if (!set_slots(E, next, err)) return false;
        vram_info(fr, tot);
        note = " (trimmed until the reserve held)";
    }
    if (fr < ((size_t) reserve << 20)) note = " (the cache keeps its smallest size)";
    long long slots = 0, full = 0;
    for (Moe* M : E.vram_moe) {
        slots += M->n_slots;
        full += std::min(M->E, E.slots_cap);
    }
    const long long cache_mib = (long long) (slots_bytes(E.vram_moe, std::max(E.slots_now, 1)) >> 20);
    const long long full_mib = (long long) (slots_bytes(E.vram_moe, E.slots_cap) >> 20);
    std::fprintf(stderr, "strata-nh: VRAM %lld MiB kept free: expert slots %d -> %d of %d per layer "
                         "(%lld of %lld slots, %lld of %lld MiB), %zu MiB free now, in %.0f ms%s\n",
                 reserve, before, E.slots_now, E.slots_cap, slots, full, cache_mib, full_mib, fr >> 20,
                 ms_since(t0), note.c_str());
    std::printf("VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld expert_cache_mib=%lld "
                "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld\n",
                reserve, slots, full, cache_mib, full_mib, (long long) (fr >> 20), (long long) E.prompt_chunk);
    std::fflush(stdout);
    return true;
}

// ------------------------------------------------------------------------------------------------ sampling
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// The sampler's candidates and their weights (penalties, temperature, top-k, min-p, top-p applied; not
/// normalized).  Temperature 0: the argmax alone.
void candidates(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::vector<int>& idx,
                std::vector<double>& p) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) {
        idx.assign(1, (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()));
        p.assign(1, 1.0);
        return;
    }
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    idx.resize(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    p.assign(idx.size(), 0.0);
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    idx.resize(keep);
    p.resize(keep);
}

/// One draw from the candidates, `skip` (a rejected draft) left out.
int draw(const std::vector<int>& idx, const std::vector<double>& p, int skip, std::mt19937_64& rng) {
    double tot = 0;
    for (size_t i = 0; i < idx.size(); ++i)
        if (idx[i] != skip) tot += p[i];
    if (tot <= 0) return idx[0];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    int last = idx[0];
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] == skip) continue;
        last = idx[i];
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return last;
}

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    std::vector<int> idx;
    std::vector<double> p;
    candidates(lg, s, hist, idx, p);
    return draw(idx, p, -1, rng);
}

/// Speculative sampling with a deterministic drafter (exact: the tokens come out distributed as plain sampling's):
/// row i keeps draft i with the probability the sampler gives it there; a rejected draft's row samples again with
/// that draft left out; when all D are kept, the last row samples the next token.  `hist` (the history so far) gets
/// the kept drafts for the penalties and is restored.
int spec_judge(const std::vector<float>& rows, int V, const Sampling& s, std::vector<int32_t>& hist,
               std::mt19937_64& rng, int D, const int32_t* drafts, int32_t* picks) {
    std::vector<float> lg;
    std::vector<int> idx;
    std::vector<double> p;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const size_t h0 = hist.size();
    int a = 0;
    for (;; ++a) {
        lg.assign(rows.begin() + (size_t) a * V, rows.begin() + (size_t) (a + 1) * V);
        candidates(lg, s, hist, idx, p);
        if (a == D) {
            picks[a] = draw(idx, p, -1, rng);
            break;
        }
        double tot = 0, px = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            tot += p[i];
            if (idx[i] == drafts[a]) px = p[i];
        }
        if (u01(rng) * tot < px) {
            picks[a] = drafts[a];
            hist.push_back(drafts[a]);
            continue;
        }
        picks[a] = draw(idx, p, drafts[a], rng);
        break;
    }
    hist.resize(h0);
    return a;
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == kEos || t == kEos2; }

/// strata-vision's records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats), in prompt order.
bool read_embeddings(const std::string& path, int n_embd, std::vector<float>& rows, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open the embeddings file " + path; return false; }
    rows.clear();
    for (;;) {
        int32_t hd[5];
        f.read((char*) hd, sizeof hd);
        if (f.gcount() == 0) break;
        if (f.gcount() != (std::streamsize) sizeof hd || hd[0] != 0x31455653) { err = "not a strata-vision file"; return false; }
        if (hd[4] != n_embd) {
            err = "the image embeddings are " + std::to_string(hd[4]) + " wide, the model's are " + std::to_string(n_embd) +
                  " (the mmproj of another model?)";
            return false;
        }
        const size_t n = (size_t) hd[1] * (size_t) n_embd;
        const size_t at = rows.size();
        rows.resize(at + n);
        f.read((char*) (rows.data() + at), (std::streamsize) (n * 4));
        if ((size_t) f.gcount() != n * 4) { err = "a truncated embeddings file"; return false; }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ serve
int serve(Engine& E) {
    std::printf("READY %d stop\n", E.ctx);
    std::fflush(stdout);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    std::vector<int32_t> live;             // the tokens whose state the engine holds
    std::vector<std::string> live_keys;    // the pictures among them (GENI img_keys), in order
    std::vector<float> lg, img_rows;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        // Between requests only: a VRAM line that arrives during a GEN sits in this queue until DONE, so a shrink
        // cannot run against a generation in progress (the server also holds its FIFO until that generation ends).
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            std::string e;
            if (!vram_command(E, line, e)) {
                std::printf("ERR %s\n", e.c_str());
                std::fflush(stdout);
            }
            continue;
        }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...] <file> "
                        "<id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &p, 10);
        Sampling smp;
        std::vector<std::string> keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                std::string s = v;
                size_t a = 0;
                while (a <= s.size()) {
                    const size_t b = s.find(',', a);
                    const std::string k = s.substr(a, b == std::string::npos ? std::string::npos : b - a);
                    if (!k.empty()) keys.push_back(k);
                    if (b == std::string::npos) break;
                    a = b + 1;
                }
            }
        }
        std::string emb_file;
        if (geni) {
            while (*p == ' ') ++p;
            const char* s0 = p;
            while (*p && *p != ' ') ++p;
            emb_file.assign(s0, (size_t) (p - s0));
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.V;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        // the images: their rows at the prompt's <image> tokens, in order
        E.inject.clear();
        size_t n_pad = 0;
        for (int32_t t : ids) n_pad += t == kImagePad;
        if (geni) {
            std::string err;
            if (!read_embeddings(emb_file, E.m->g.N, img_rows, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
            if (img_rows.size() != n_pad * (size_t) E.m->g.N) {
                std::printf("ERR the prompt has %zu image tokens, the embeddings %zu rows\n", n_pad,
                            img_rows.size() / (size_t) E.m->g.N);
                std::fflush(stdout);
                continue;
            }
        }
        stop.store(false);
        // the held state continues when the request extends it (the recurrent state cannot step back); a picture
        // matches by its key
        size_t reuse = 0;
        const bool keys_ok = geni ? keys.size() >= live_keys.size() && std::equal(live_keys.begin(), live_keys.end(), keys.begin())
                                  : live_keys.empty();
        // the request's common prefix with the live sequence: all of it (the request extends it), or a part - then a
        // turn checkpoint at or below that part brings the recurrent state back there
        size_t common = 0;
        while (common < live.size() && common < ids.size() && live[common] == ids[common]) ++common;
        bool restored = false;
        if (keys_ok && live.size() < ids.size() && common == live.size()) reuse = live.size();
        else if (keys_ok && common > 0) {
            const int64_t at = E.ckpt_restore((int64_t) std::min(common, ids.size() - 1));
            if (at > 0) {
                reuse = (size_t) at;
                live.resize(reuse);
                E.ckpt_drop_past(at);
                restored = true;
            }
        }
        if (reuse == 0) {
            E.reset();
            live.clear();
            live_keys.clear();
            E.ckpt_drop_past(0);
        }
        if (restored)
            std::fprintf(stderr, "strata-nh: turn checkpoint: back to %zu tokens of %zu (the rest of the prompt: %zu)\n",
                         reuse, ids.size(), ids.size() - reuse);
        if (geni) {
            size_t r = 0;
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == kImagePad) {
                    if (i >= reuse) E.inject[(int64_t) i] = img_rows.data() + r * (size_t) E.m->g.N;
                    ++r;
                }
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        auto pp = [&](int64_t base) {
            return [&, base](int64_t done) {
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) (base + done), (long long) fresh, ms_since(t0),
                            (double) (base + done) / std::max(1e-9, ms_since(t0) / 1000.0));
                std::fflush(stdout);
            };
        };
        // the prompt but its last token, a turn checkpoint there, then the last token: a request that comes back to
        // this prompt (a regenerated answer) restores the checkpoint and reads one token
        if (fresh > 1) {
            E.process(ids.data() + reuse, fresh - 1, pp(0));
            E.ckpt_save();
            E.process(ids.data() + ids.size() - 1, 1, pp(fresh - 1));
        } else {
            E.process(ids.data() + reuse, fresh, pp(0));
        }
        E.inject.clear();
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        if (geni) live_keys = keys.empty() ? std::vector<std::string>{emb_file} : keys;
        const double prompt_ms = ms_since(t0);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        const bool spec = E.spec_d() > 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        std::vector<float> rows;
        Engine::Judge judge = [&](int D, const int32_t* dr, int32_t* pk) {
            E.host_logits_rows(D + 1, rows);
            return spec_judge(rows, E.m->g.V, smp, hist, rng, D, dr, pk);
        };
        const Engine::Judge* jp = smp.needs_logits() ? &judge : nullptr;
        int nx = smp.needs_logits() ? -1 : E.greedy();
        bool have_nx = !smp.needs_logits();
        bool done = false;
        while (!done) {
            if (!have_nx) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            auto emit = [&](int t) {
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (spec && E.pos + E.spec_d() + 1 <= E.ctx) {
                const int a = E.spec_any(nx, fix, drafts, picks, jp);
                offered += E.spec_d();
                accepted += a;
                live.push_back(nx);
                fix.assign(drafts.begin(), drafts.begin() + a);
                for (int i = 0; i < a; ++i) {
                    live.push_back(drafts[(size_t) i]);
                    if (!emit(drafts[(size_t) i])) { done = true; break; }
                }
                nx = picks[a];
                have_nx = true;
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            live.push_back(nx);
            fix.clear();
            have_nx = !smp.needs_logits();
            if (have_nx) nx = E.greedy();
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        static int served = 0;
        if (!E.profile_save.empty() && ++served % 8 == 0 && !save_profile(E.profile_save, E.hy, E.m->g.E))
            std::fprintf(stderr, "strata-nh: could not save the expert profile to %s\n", E.profile_save.c_str());
    }
}

}  // namespace

int strata::nemotron_h::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, dump_layers;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 64;
    std::string dflash;
    int dflash_d = 4;   // docs/NEMOTRON_H.md (Bolt): D = 4 best on chat, within 5% of D = 5 on code
    // the PCIe share and the prefetch are off by default: measured on the RTX 3090 + i7-7800X (DDR4-2133, quad
    // channel) the CPU experts already read RAM at its ceiling (~40 GB/s), and the uploads read the same RAM
    // (docs/NEMOTRON_H.md); a faster PCIe link or a slower CPU is where they pay
    int expert_slots = 0, ring = 64, reserve_mib = 1024, pcie_max = 48, prefetch = 0;
    float pcie_frac = -1.f;   // unset: 0.5 with DFlash (its 5-row verify shares many experts: 61 -> 72 tok/s), else 0
    bool arena = true;
    std::string profile, profile_save;
    int adapt_every = 32, adapt_swaps = 8;
    double expert_vram = -1.0;
    bool warm = true, serve_mode = false, pin = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-nh: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--pool-workers" || a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA_VISIBLE_DEVICES)
        else if (a == "--greedy" || a == "--no-graph" || a == "--no-token-graph") {}
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-nh: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N drafts)\n",
                              v.c_str());
        }
        else if (a == "--dflash") dflash = next();
        else if (a == "--dflash-d") dflash_d = std::atoi(next().c_str());
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--stream-ring") ring = std::atoi(next().c_str());
        else if (a == "--pin-experts") pin = true;   // (the arena is pinned already; kept for old command lines)
        else if (a == "--no-arena") arena = false;
        else if (a == "--pcie-frac") pcie_frac = std::strtof(next().c_str(), nullptr);
        else if (a == "--pcie-max") pcie_max = std::atoi(next().c_str());
        else if (a == "--prefetch") prefetch = std::atoi(next().c_str());
        else if (a == "--expert-profile") profile = next();
        else if (a == "--adapt-every") adapt_every = std::max(1, std::atoi(next().c_str()));
        else if (a == "--adapt-swaps") adapt_swaps = std::max(0, std::atoi(next().c_str()));
        else if (a == "--expert-profile-save") profile_save = next();
        else if (a == "--vram-reserve-mib") reserve_mib = std::atoi(next().c_str());
        else if (a == "--vram-elastic") {}   // the server's flag: slots resize on VRAM, there is no segment map
        else if (a == "--vram-segment-mib") (void) next();
        else if (a == "--vision") {}   // the server's: images come as GENI requests, which this engine always takes
        else if (serve_mode && a.rfind("--", 0) == 0) {
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-nh: ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "",
                         val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-nh --model GGUF [--max-context 32768] [--chunk 1024] [--mtp N] [--expert-vram GiB]\n"
                         "       [--dflash GGUF] [--dflash-d 4]\n"
                         "       [--expert-slots N] [--hybrid-rows 64] [--cpu-threads N] [--stream-ring 64] [--pin-experts]\n"
                         "       [--vram-reserve-mib 1024] [--pcie-frac 0, 0.5 with --dflash] [--pcie-max 48] [--prefetch 0] [--no-arena]\n"
                         "       [--expert-profile F] [--expert-profile-save F] [--adapt-every 32] [--adapt-swaps 8]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] [--dump-layers F] |\n"
                         "        --bench-pp N --bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-nh: --model is required\n"); return 2; }
#if defined(_WIN32)
    if (!std::getenv("CUDA_MODULE_LOADING")) {
        _putenv("CUDA_MODULE_LOADING=EAGER");
        SetEnvironmentVariableA("CUDA_MODULE_LOADING", "EAGER");
    }
#else
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-nh: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-nh: bad --chunk / --max-context\n"); return 2; }
    if (chunk == 0) chunk = 2048;   // a prompt chunk streams every non-resident expert once: bigger pays it less often
    Model m;
    std::string err;
    try {
        if (!load(model, m, err)) { std::fprintf(stderr, "strata-nh: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-nh: %s\n", e.what());
        return 1;
    }
    Engine E;
    E.hybrid_t = std::max(1, std::min(hybrid_rows, chunk));
    if (pcie_frac < 0.f) pcie_frac = dflash.empty() ? 0.f : 0.5f;
    E.pcie_frac = std::max(0.f, std::min(1.f, pcie_frac));
    E.pcie_cap = std::max(0, pcie_max);
    E.pf_cap = std::max(0, prefetch);
    E.profile_save = profile_save;
    E.adapt_every = adapt_every;
    E.adapt_swaps = adapt_swaps;
    (void) pin;
    E.init(m, ctx, chunk, ring);
    E.prompt_chunk = chunk;
    if (!dflash.empty()) {
        try {
            if (!E.load_dflash(dflash, dflash_d, err)) { std::fprintf(stderr, "strata-nh: --dflash: %s\n", err.c_str()); return 1; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-nh: --dflash: %s\n", e.what());
            return 1;
        }
        std::fprintf(stderr, "strata-nh: DFlash drafter %s, %d drafts a step\n", dflash.c_str(), E.df_d);
    }
    if (!place_experts(m, E, expert_vram, cpu_threads, expert_slots, ring, arena, reserve_mib, profile, err)) {
        std::fprintf(stderr, "strata-nh: %s\n", err.c_str());
        return 1;
    }
    struct Report {
        Engine& E;
        ~Report() {
            const long long n = E.hy_cpu + E.hy_gpu + E.tr_pcie + E.tr_pf_hit;
            auto pc = [&](long long v) { return n ? 100.0 * v / n : 0.0; };
            std::fprintf(stderr, "strata-nh: row entries %lld: resident %.1f%%, prefetched %.1f%% (%lld experts sent), PCIe share "
                         "%.1f%%, CPU %.1f%% (%.0f ms; route wait %.0f ms); %lld experts streamed, %lld slot replacements\n", n,
                         pc(E.hy_gpu), pc(E.tr_pf_hit), E.tr_pf_sent, pc(E.tr_pcie), pc(E.hy_cpu), E.hy_cpu_ms,
                         E.t_route_wait_ms, E.hy_streamed, E.hy_swaps);
            if (!E.profile_save.empty())
                std::fprintf(stderr, "strata-nh: expert profile %s %s\n", E.profile_save.c_str(),
                             save_profile(E.profile_save, E.hy, E.m->g.E) ? "saved" : "NOT saved");
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (1000 + rng() % 100000u);
        for (int rep = 0; rep < 2; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.2f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg, bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-nh: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    if (!dump_layers.empty()) E.dump_f = std::fopen(dump_layers.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.spec_d() > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_any(nx, fix, drafts, picks);
            offered += E.spec_d();
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-nh: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.2f tok/s), %s %d drafts: "
                     "%lld of %lld accepted (%.0f%%, %.2f tokens a step)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.df ? "DFlash" : "MTP", E.spec_d(), accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0, offered ? 1.0 + (double) accepted * E.spec_d() / offered : 0.0);
        if (E.df)
            std::fprintf(stderr, "strata-nh: DFlash drafting %.2f ms a step (%lld steps)\n", E.df_steps ? E.df_ms / E.df_steps : 0.0,
                         E.df_steps);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    if (E.dump_f) std::fclose(E.dump_f);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-nh: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.2f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}

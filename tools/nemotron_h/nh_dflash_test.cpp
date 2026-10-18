// tools/nemotron_h/nh_dflash_test.cpp
// Test harness for LiLiCorr DFlash speculative drafter.
// Validates parity against dumped llama.cpp reference data:
//  - Top-1 agreement > 0.98 for every draft position
//  - Mean KL divergence < 0.01 for every draft position
//  - Measures and reports GPU execution time per draft block

#include "nh_dflash.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

namespace {

struct DumpRecord {
    int32_t n_tok = 0;
    int32_t id_last = 0;
    int32_t pos0 = 0;
    int32_t n_draft = 0;
    int32_t n_vocab = 0;
    int32_t n_embd_enc = 0;
    std::vector<int32_t> prompt_tokens;
    std::vector<float> features;     // [n_tok, n_embd_enc]
    std::vector<float> ref_logits;   // [n_draft, n_vocab]
    std::vector<int32_t> ref_tokens; // [n_draft]
    std::vector<float> ref_lattice;  // [n_draft + 1, kLatticeWidth] when the dump has it (newer nh_dflash_dump)
};

constexpr int kLatticeWidth = 4096;   // n_embd of the drafter: each lattice row is padded to it
constexpr int kTopK = 8;

// Per lattice row 1..n_draft: candidate ids identical, max |score diff| and argmax agreement per predecessor
struct LatticeStats {
    std::vector<int> tests, cand_ok, arg_ok, arg_n;
    std::vector<double> max_diff, max_miss_gap;   // max_miss_gap: largest ref top-1 margin among argmax disagreements
    explicit LatticeStats(int n) : tests(n, 0), cand_ok(n, 0), arg_ok(n, 0), arg_n(n, 0), max_diff(n, 0.0), max_miss_gap(n, 0.0) {}

    void add(const float* ref, const float* mine, const char* tag) {
        for (int r = 1; r <= (int) tests.size(); ++r) {
            const float* a = ref + (size_t) r * kLatticeWidth;
            const float* b = mine + (size_t) r * kLatticeWidth;
            tests[r - 1]++;
            bool same = true;
            for (int k = 0; k < kTopK; ++k) same &= (int) a[k] == (int) b[k];
            if (!same) {
                std::cout << "[lattice " << tag << "] row " << r << " candidates differ: ref";
                for (int k = 0; k < kTopK; ++k) std::cout << " " << (int) a[k];
                std::cout << " / mine";
                for (int k = 0; k < kTopK; ++k) std::cout << " " << (int) b[k];
                std::cout << "\n";
                continue;
            }
            cand_ok[r - 1]++;
            const int n_pred = r == 1 ? 1 : kTopK;   // row 1 scores the candidates against the anchor only
            for (int p = 0; p < n_pred; ++p) {
                const float* sa = a + kTopK + p * kTopK;
                const float* sb = b + kTopK + p * kTopK;
                for (int k = 0; k < kTopK; ++k) max_diff[r - 1] = std::max(max_diff[r - 1], (double) std::fabs(sa[k] - sb[k]));
                arg_n[r - 1]++;
                const int ia = (int) (std::max_element(sa, sa + kTopK) - sa), ib = (int) (std::max_element(sb, sb + kTopK) - sb);
                if (ia == ib) arg_ok[r - 1]++;
                else max_miss_gap[r - 1] = std::max(max_miss_gap[r - 1], (double) (sa[ia] - sa[ib]));
            }
        }
    }

    // Prints the table; true when every row has identical candidates and scores within noise (see below)
    bool print(const char* title) const {
        bool ok = true;
        std::cout << "\n" << title << "\n" << std::left << std::setw(6) << "Row" << std::setw(14) << "Cands equal"
                  << std::setw(16) << "Max score diff" << std::setw(16) << "Argmax agree" << "Max ref margin of a miss\n";
        for (size_t r = 0; r < tests.size(); ++r) {
            const double cand_rate = (double) cand_ok[r] / tests[r];
            const double arg_rate = arg_n[r] ? (double) arg_ok[r] / arg_n[r] : 0.0;
            // llama.cpp's CUDA matmuls run on Q8_1/FP16 activations, ours on F32, so scores differ by ~0.1-0.2 (scale 8): pass
            // when no score is off by more than 0.5 and every argmax miss is a near tie (ref margin <= 0.5)
            if (cand_rate < 1.0 || max_diff[r] > 0.5 || max_miss_gap[r] > 0.5) ok = false;
            std::cout << std::left << std::setw(6) << (r + 1) << std::fixed << std::setprecision(4) << std::setw(14) << cand_rate
                      << std::scientific << std::setprecision(3) << std::setw(16) << max_diff[r]
                      << std::fixed << std::setprecision(4) << std::setw(16) << arg_rate << max_miss_gap[r] << "\n";
        }
        return ok;
    }
};

bool load_dump_file(const std::string& path, DumpRecord& rec) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return false;

    uint32_t magic = 0;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x44464c53) { // "DFLS"
        std::fprintf(stderr, "Invalid magic 0x%08x in %s\n", magic, path.c_str());
        return false;
    }

    fin.read(reinterpret_cast<char*>(&rec.n_tok), sizeof(rec.n_tok));
    fin.read(reinterpret_cast<char*>(&rec.id_last), sizeof(rec.id_last));
    fin.read(reinterpret_cast<char*>(&rec.pos0), sizeof(rec.pos0));
    fin.read(reinterpret_cast<char*>(&rec.n_draft), sizeof(rec.n_draft));
    fin.read(reinterpret_cast<char*>(&rec.n_vocab), sizeof(rec.n_vocab));
    fin.read(reinterpret_cast<char*>(&rec.n_embd_enc), sizeof(rec.n_embd_enc));

    rec.prompt_tokens.resize(rec.n_tok);
    fin.read(reinterpret_cast<char*>(rec.prompt_tokens.data()), rec.n_tok * sizeof(int32_t));

    size_t feat_count = (size_t) rec.n_tok * rec.n_embd_enc;
    rec.features.resize(feat_count);
    fin.read(reinterpret_cast<char*>(rec.features.data()), feat_count * sizeof(float));

    size_t logit_count = (size_t) rec.n_draft * rec.n_vocab;
    rec.ref_logits.resize(logit_count);
    fin.read(reinterpret_cast<char*>(rec.ref_logits.data()), logit_count * sizeof(float));

    rec.ref_tokens.resize(rec.n_draft);
    fin.read(reinterpret_cast<char*>(rec.ref_tokens.data()), rec.n_draft * sizeof(int32_t));

    std::vector<float> lat((size_t) (rec.n_draft + 1) * kLatticeWidth);
    fin.read(reinterpret_cast<char*>(lat.data()), lat.size() * sizeof(float));
    if (fin.gcount() == (std::streamsize) (lat.size() * sizeof(float))) rec.ref_lattice = std::move(lat);

    return !fin.bad();
}

void compute_top1_and_kl(const float* ref_logits,
                         const float* my_logits,
                         int n_vocab,
                         bool& top1_match,
                         double& kl_div,
                         int& ref_top1,
                         int& my_top1) {
    ref_top1 = 0;
    my_top1 = 0;
    double max_ref = ref_logits[0];
    double max_my = my_logits[0];

    for (int i = 1; i < n_vocab; ++i) {
        if (ref_logits[i] > max_ref) {
            max_ref = ref_logits[i];
            ref_top1 = i;
        }
        if (my_logits[i] > max_my) {
            max_my = my_logits[i];
            my_top1 = i;
        }
    }

    top1_match = (ref_top1 == my_top1);

    double sum_exp_ref = 0.0;
    double sum_exp_my = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp_ref += std::exp((double) ref_logits[i] - max_ref);
        sum_exp_my += std::exp((double) my_logits[i] - max_my);
    }

    double lse_ref = max_ref + std::log(sum_exp_ref);
    double lse_my = max_my + std::log(sum_exp_my);

    kl_div = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        double log_p = (double) ref_logits[i] - lse_ref;
        double log_q = (double) my_logits[i] - lse_my;
        double p = std::exp(log_p);
        if (p > 1e-12) {
            kl_div += p * (log_p - log_q);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string model_tgt;   // --model-tgt: the Bolt GGUF (required)
    std::string model_dft;   // --model-dft: its DFlash drafter GGUF (required)
    std::string dump_dir = "nh_dflash_dumps";

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--model-tgt") model_tgt = v;
        else if (a == "--model-dft") model_dft = v;
        else if (a == "--dump-dir") dump_dir = v;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (model_tgt.empty() || model_dft.empty()) {
        std::fprintf(stderr, "usage: %s --model-tgt <Bolt GGUF> --model-dft <DFlash drafter GGUF> [...]\n", argv[0]);
        return 1;
    }

    // Collect dump files
    std::vector<std::string> dump_files;
    if (fs::exists(dump_dir)) {
        for (const auto& entry : fs::directory_iterator(dump_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                dump_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(dump_files.begin(), dump_files.end());

    if (dump_files.empty()) {
        std::fprintf(stderr, "No parity dump files found in %s\n", dump_dir.c_str());
        return 1;
    }
    std::fprintf(stderr, "Found %zu dump files in %s\n", dump_files.size(), dump_dir.c_str());

    // Initialize DFlash drafter
    strata::nemotron_h::DFlashDrafter drafter;
    std::string err;
    std::fprintf(stderr, "Loading DFlash drafter weights...\n");
    if (!drafter.load(model_dft, model_tgt, err)) {
        std::fprintf(stderr, "Failed to load drafter: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "Drafter loaded successfully.\n");

    const int n_draft = drafter.config().n_draft; // 7
    const int n_vocab = drafter.config().n_vocab;

    // Allocate GPU buffer for output logits
    float* d_my_logits = nullptr;
    cudaMalloc(&d_my_logits, (size_t) n_draft * n_vocab * sizeof(float));

    float* d_features = nullptr;
    size_t current_feat_bytes = 0;

    std::vector<float> h_my_logits((size_t) n_draft * n_vocab);

    // Metrics accumulators per draft position (0..6)
    std::vector<int> total_tests(n_draft, 0);
    std::vector<int> top1_matches(n_draft, 0);
    std::vector<double> sum_kl(n_draft, 0.0);
    std::vector<double> max_kl(n_draft, 0.0);
    std::vector<int> token_matches(n_draft, 0);
    std::vector<float> draft_times_ms;
    // Lattice parity, twice: "own" = LiLiCorr on our drafter logits (near-tied candidates may swap ranks, and the
    // attention spreads one swap over the whole block, so this is informational); "fed" = LiLiCorr on llama.cpp's
    // logits, which isolates the LiLiCorr port itself and is gated.
    float* d_ref_logits = nullptr;
    cudaMalloc(&d_ref_logits, (size_t) n_draft * n_vocab * sizeof(float));
    LatticeStats lat_own(n_draft), lat_fed(n_draft);
    // Warmup GPU kernels and cuBLAS
    {
        DumpRecord warmup_rec;
        if (load_dump_file(dump_files[0], warmup_rec)) {
            size_t needed_bytes = (size_t) warmup_rec.n_tok * warmup_rec.n_embd_enc * sizeof(float);
            cudaMalloc(&d_features, needed_bytes);
            current_feat_bytes = needed_bytes;
            cudaMemcpy(d_features, warmup_rec.features.data(), needed_bytes, cudaMemcpyHostToDevice);
            drafter.reset();
            drafter.encode_and_inject(d_features, warmup_rec.n_tok, 0);
            drafter.draft(warmup_rec.id_last, warmup_rec.pos0, d_my_logits);
            cudaDeviceSynchronize();
        }
    }

    for (size_t f_idx = 0; f_idx < dump_files.size(); ++f_idx) {
        const std::string& fpath = dump_files[f_idx];
        DumpRecord rec;
        if (!load_dump_file(fpath, rec)) {
            std::fprintf(stderr, "Failed to load %s\n", fpath.c_str());
            continue;
        }

        std::cout << "\n=======================================================\n";
        std::cout << "Testing prompt dump [" << f_idx + 1 << "/" << dump_files.size() << "]: "
                  << fs::path(fpath).filename().string()
                  << " (n_tok=" << rec.n_tok << ", id_last=" << rec.id_last << ", pos0=" << rec.pos0 << ")\n";
        std::cout << "=======================================================\n" << std::flush;
        std::cerr << std::flush;

        // Upload features
        size_t needed_bytes = (size_t) rec.n_tok * rec.n_embd_enc * sizeof(float);
        if (needed_bytes > current_feat_bytes) {
            if (d_features) cudaFree(d_features);
            cudaMalloc(&d_features, needed_bytes);
            current_feat_bytes = needed_bytes;
        }
        cudaMemcpy(d_features, rec.features.data(), needed_bytes, cudaMemcpyHostToDevice);

        // 1. Reset drafter
        drafter.reset();

        // 2. Encode and inject features
        drafter.encode_and_inject(d_features, rec.n_tok, 0);

        // 3. Draft block
        std::vector<int32_t> my_tokens = drafter.draft(rec.id_last, rec.pos0, d_my_logits);
        float draft_ms = drafter.last_draft_ms();
        draft_times_ms.push_back(draft_ms);

        // Copy logits back
        cudaMemcpy(h_my_logits.data(), d_my_logits, (size_t) n_draft * n_vocab * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "GPU Draft time: " << std::fixed << std::setprecision(2) << draft_ms << " ms\n";
        std::cout << "Drafted tokens: ";
        for (int t : my_tokens) std::cout << t << " ";
        std::cout << "\nReference tokens: ";
        for (int t : rec.ref_tokens) std::cout << t << " ";
        std::cout << "\n";

        if (!rec.ref_lattice.empty()) {
            std::vector<float> my_lat((size_t) (n_draft + 1) * kLatticeWidth);
            drafter.get_lattice(my_lat.data());
            lat_own.add(rec.ref_lattice.data(), my_lat.data(), "own");

            cudaMemcpy(d_ref_logits, rec.ref_logits.data(), (size_t) n_draft * n_vocab * sizeof(float), cudaMemcpyHostToDevice);
            drafter.set_lilicorr_logits_override(d_ref_logits);
            drafter.draft(rec.id_last, rec.pos0);
            drafter.set_lilicorr_logits_override(nullptr);
            drafter.get_lattice(my_lat.data());
            lat_fed.add(rec.ref_lattice.data(), my_lat.data(), "fed");
        }
        std::cout << "\n";

        std::cout << std::left
                  << std::setw(6)  << "Pos"
                  << std::setw(12) << "Ref Top-1"
                  << std::setw(12) << "My Top-1"
                  << std::setw(10) << "Match"
                  << std::setw(14) << "KL Divergence"
                  << std::setw(12) << "Ref Logit"
                  << std::setw(12) << "My Logit"
                  << "\n";
        std::cout << std::string(78, '-') << "\n";

        for (int pos = 0; pos < n_draft; ++pos) {
            const float* ref_pos_logits = rec.ref_logits.data() + (size_t) pos * n_vocab;
            const float* my_pos_logits = h_my_logits.data() + (size_t) pos * n_vocab;

            bool top1_ok = false;
            double kl = 0.0;
            int ref_top1 = 0, my_top1 = 0;
            compute_top1_and_kl(ref_pos_logits, my_pos_logits, n_vocab, top1_ok, kl, ref_top1, my_top1);

            total_tests[pos]++;
            if (top1_ok) top1_matches[pos]++;
            sum_kl[pos] += kl;
            if (kl > max_kl[pos]) max_kl[pos] = kl;
            if (pos < (int) my_tokens.size() && pos < (int) rec.ref_tokens.size()) {
                if (my_tokens[pos] == rec.ref_tokens[pos]) token_matches[pos]++;
            }

            std::cout << std::left
                      << std::setw(6)  << (pos + 1)
                      << std::setw(12) << ref_top1
                      << std::setw(12) << my_top1
                      << std::setw(10) << (top1_ok ? "PASS" : "FAIL")
                      << std::setw(14) << std::scientific << std::setprecision(4) << kl
                      << std::fixed << std::setprecision(3)
                      << std::setw(12) << ref_pos_logits[ref_top1]
                      << std::setw(12) << my_pos_logits[my_top1]
                      << "\n";
        }
    }

    // Final summary
    std::cout << "\n=======================================================\n";
    std::cout << "                 SUMMARY PARITY REPORT                 \n";
    std::cout << "=======================================================\n";
    std::cout << std::left
              << std::setw(6)  << "Pos"
              << std::setw(14) << "Top-1 Agr"
              << std::setw(14) << "Mean KL"
              << std::setw(14) << "Max KL"
              << std::setw(16) << "Lattice Token Agr"
              << std::setw(10) << "Status"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    bool all_passed = true;
    for (int pos = 0; pos < n_draft; ++pos) {
        double top1_rate = (total_tests[pos] > 0) ? (double) top1_matches[pos] / total_tests[pos] : 0.0;
        double mean_kl = (total_tests[pos] > 0) ? sum_kl[pos] / total_tests[pos] : 0.0;
        double token_rate = (total_tests[pos] > 0) ? (double) token_matches[pos] / total_tests[pos] : 0.0;

        bool pos_passed = (top1_rate >= 0.98) && (mean_kl <= 0.01);
        if (!pos_passed) all_passed = false;

        std::cout << std::left
                  << std::setw(6)  << (pos + 1)
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << top1_rate
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << mean_kl
                  << std::setw(14) << max_kl[pos]
                  << std::fixed << std::setprecision(4)
                  << std::setw(16) << token_rate
                  << std::setw(10) << (pos_passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout << std::string(74, '-') << "\n";

    bool lattice_passed = true;
    if (lat_fed.tests[0] > 0) {
        lat_own.print("Lattice on our own drafter logits (informational: near-tied candidates may swap)");
        lattice_passed = lat_fed.print("Lattice on llama.cpp's logits (LiLiCorr port alone, gated)");
        std::cout << (lattice_passed ? "[LATTICE] PASS\n" : "[LATTICE] FAIL\n");
    } else {
        std::cout << "\n[LATTICE] the dumps carry no lattice; regenerate them with the current nh_dflash_dump\n";
    }

    float avg_time_ms = 0.0f;
    if (!draft_times_ms.empty()) {
        float sum = std::accumulate(draft_times_ms.begin(), draft_times_ms.end(), 0.0f);
        avg_time_ms = sum / (float) draft_times_ms.size();
    }
    std::cout << "\nAverage GPU Draft Time: " << std::fixed << std::setprecision(2)
              << avg_time_ms << " ms / draft block (7 tokens)\n";

    if (d_features) cudaFree(d_features);
    if (d_my_logits) cudaFree(d_my_logits);
    if (d_ref_logits) cudaFree(d_ref_logits);

    if (all_passed && lattice_passed) {
        std::cout << "\n[RESULT] ALL PARITY CHECKS PASSED (Top-1 >= 0.98, Mean KL <= 0.01)\n";
        return 0;
    } else {
        std::cout << "\n[RESULT] PARITY CHECKS FAILED\n";
        return 1;
    }
}

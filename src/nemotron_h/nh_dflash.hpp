// src/nemotron_h/nh_dflash.hpp
// Self-contained CUDA module for the LiLiCorr DFlash speculative drafter (Nemotron-H / Bolt)
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::nemotron_h {

struct DFlashConfig {
    int block_size = 8;             // 8 tokens total (1 committed token + 7 draft tokens)
    int n_draft = 7;                // 7 draft tokens
    int mask_token_id = 990;        // mask token
    int n_embd = 4096;              // hidden dimension
    int n_layers = 5;               // 5 transformer layers (blk.0..blk.4)
    int n_head = 32;                // 32 Q heads
    int n_head_kv = 2;              // 2 KV heads
    int head_dim = 128;             // 128 head dim
    int n_ff = 12288;               // intermediate size
    float eps = 1e-6f;              // RMS norm epsilon
    float rope_freq_base = 10000.0f;// RoPE base
    int sliding_window = 1024;      // attention sliding window
    std::vector<int> target_layers = {2, 19, 36, 52, 69, 86};
    int n_target_layers = 6;
    int n_embd_inp = 24576;         // 6 * 4096

    // LiLiCorr config
    int lilicorr_hidden = 1024;
    int lilicorr_top_k = 8;
    int lilicorr_heads = 8;
    int lilicorr_layers = 2;        // 2 transformer layers (blk.5, blk.6)
    int lilicorr_factor_dim = 1024;
    int lilicorr_mlp = 2048;
    float lilicorr_eps = 1e-4f;
    float lilicorr_logit_scale = 8.0f;
    int n_vocab = 131072;
};

class DFlashDrafter {
public:
    DFlashDrafter();
    ~DFlashDrafter();

    // Load weights from GGUF files.
    // If target_gguf_path is non-empty, loads token_embd.weight and output.weight from target GGUF.
    // Otherwise caller can supply them via set_target_weights().
    bool load(const std::string& dflash_gguf_path,
              const std::string& target_gguf_path,
              std::string& err);

    // Provide target weights if already resident in GPU memory (Q8_0 token_embd and output)
    void set_target_weights(const void* d_token_embd_q8, const void* d_output_q8, int vocab_size);

    // Reset KV cache and state for a new sequence
    void reset();

    // The sequence state (KV ring, anchor, fill counters) to/from host memory of state_bytes(): a turn checkpoint
    // keeps it beside the target's Mamba state, so a restored checkpoint drafts with its full context again
    size_t state_bytes() const;
    void copy_state(void* host, bool save, cudaStream_t stream = nullptr);

    // Encode target layer hidden states and inject into DFlash KV cache:
    // d_target_features: device pointer to [n_tokens, 24576] float32 array
    // pos0: starting position in sequence
    void encode_and_inject(const float* d_target_features,
                          int n_tokens,
                          int pos0,
                          cudaStream_t stream = nullptr);

    // The same from FP16 features [n_tokens, 24576] (the encoder reads FP16 either way)
    void encode_and_inject_f16(const __half* d_target_features,
                               int n_tokens,
                               int pos0,
                               cudaStream_t stream = nullptr);

    // Draft a block of tokens:
    // id_last: last committed/sampled token id
    // pos0: sequence position of id_last
    // d_logits_out: optional [7, 131072] float device buffer to retrieve drafter logits
    // returns 7 draft tokens
    std::vector<int32_t> draft(int32_t id_last,
                               int32_t pos0,
                               float* d_logits_out = nullptr,
                               cudaStream_t stream = nullptr);

    // Drafter GPU execution time of the last draft() call in milliseconds
    float last_draft_ms() const;

    // Retrieve computed lattice [8, 4096] floats
    void get_lattice(float* h_lattice_out) const;
    const float* lattice_device() const;

    // Test hook: LiLiCorr reads its candidates from these [7, n_vocab] device logits instead of the drafter's own
    // (nullptr restores normal drafting). Lets nh_dflash_test feed llama.cpp's logits to check LiLiCorr alone.
    void set_lilicorr_logits_override(const float* d_logits);

    const DFlashConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::nemotron_h

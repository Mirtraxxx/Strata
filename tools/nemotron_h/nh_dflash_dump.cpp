// tools/nemotron_h/nh_dflash_dump.cpp
// Dumps target hidden states (layers 2, 19, 36, 52, 69, 86) and llama.cpp's DFlash drafter logits
// Built against llama.cpp reference:
//   cl /O2 /std:c++17 tools\nemotron_h\nh_dflash_dump.cpp /I <llama.cpp>\include
//      /I <llama.cpp>\ggml\include /I <llama.cpp>\src
//      /I <llama.cpp>\common /link
//      /LIBPATH:<llama.cpp>\build\src
//      /LIBPATH:<llama.cpp>\build\ggml\src
//      /LIBPATH:<llama.cpp>\build\common
//      llama.lib ggml.lib ggml-base.lib ggml-cpu.lib llama-common.lib

#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace {

void make_directory(const std::string& path) {
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

std::vector<std::string> load_prompts_json(const std::string& path) {
    std::vector<std::string> prompts;
    std::ifstream f(path);
    if (!f.is_open()) return prompts;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    while ((pos = content.find('"', pos)) != std::string::npos) {
        size_t end = pos + 1;
        while (end < content.size() && content[end] != '"') {
            if (content[end] == '\\') end++;
            end++;
        }
        if (end < content.size()) {
            std::string s = content.substr(pos + 1, end - pos - 1);
            if (s.size() > 5) {
                // simple unescape for \n and \"
                std::string unesc;
                for (size_t i = 0; i < s.size(); ++i) {
                    if (s[i] == '\\' && i + 1 < s.size()) {
                        if (s[i+1] == 'n') { unesc += '\n'; i++; }
                        else if (s[i+1] == '"') { unesc += '"'; i++; }
                        else if (s[i+1] == '\\') { unesc += '\\'; i++; }
                        else unesc += s[i];
                    } else {
                        unesc += s[i];
                    }
                }
                prompts.push_back(unesc);
            }
            pos = end + 1;
        } else break;
    }
    return prompts;
}

static bool dflash_dump_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    std::string name = t->name ? t->name : "";
    if (name.find("noise_norm") != std::string::npos ||
        name.find("attn_conv_in") != std::string::npos ||
        name.find("Qcur") != std::string::npos ||
        name.find("Kcur") != std::string::npos ||
        name.find("Vcur") != std::string::npos ||
        name.find("kqv_out") != std::string::npos ||
        name.find("kqv") != std::string::npos ||
        name.find("attn_conv_out") != std::string::npos ||
        name.find("ffn_inp") != std::string::npos ||
        name.find("ffn_norm") != std::string::npos ||
        name.find("ffn_conv_in") != std::string::npos ||
        name.find("ffn_out") != std::string::npos ||
        name.find("ffn_conv_out") != std::string::npos) {

        size_t n_elems = ggml_nelements(t);
        size_t check_n = std::min((size_t) 1024, n_elems);
        std::vector<float> h(check_n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, h.data(), 0, check_n * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h16(check_n);
            ggml_backend_tensor_get(t, h16.data(), 0, check_n * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < check_n; ++i) h[i] = ggml_fp16_to_fp32(h16[i]);
        }
        float min_v = 1e30f, max_v = -1e30f;
        for (float v : h) {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        std::printf("[LLAMA-REF] %-20s (ne=[%lld, %lld, %lld]): min=%f, max=%f, val[0]=%f\n",
                    t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    min_v, max_v, h[0]);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_tgt;   // --model-tgt: the Bolt GGUF (required)
    std::string model_dft;   // --model-dft: its DFlash drafter GGUF (required)
    std::string prompts_file = "tools/nemotron_h/parity_prompts.json";
    std::string out_dir = "nh_dflash_dumps";
    int ngl = 99;
    int cmoe = 1;
    int max_prompts = 3;

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--model-tgt") model_tgt = v;
        else if (a == "--model-dft") model_dft = v;
        else if (a == "--prompts-file") prompts_file = v;
        else if (a == "--out-dir") out_dir = v;
        else if (a == "--ngl") ngl = std::atoi(v.c_str());
        else if (a == "--cmoe") cmoe = std::atoi(v.c_str());
        else if (a == "--max-prompts") max_prompts = std::atoi(v.c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (model_tgt.empty() || model_dft.empty()) {
        std::fprintf(stderr, "usage: %s --model-tgt <Bolt GGUF> --model-dft <DFlash drafter GGUF> [...]\n", argv[0]);
        return 1;
    }

    make_directory(out_dir);

    std::vector<std::string> prompts = load_prompts_json(prompts_file);
    if (prompts.empty()) {
        std::fprintf(stderr, "could not load prompts from %s\n", prompts_file.c_str());
        return 1;
    }
    if ((int) prompts.size() > max_prompts) prompts.resize(max_prompts);

    llama_backend_init();

    // 1. Load target model
    llama_model_params mp_tgt = llama_model_default_params();
    mp_tgt.n_gpu_layers = ngl;
    static llama_model_tensor_buft_override ov[2] = {{"_exps", nullptr}, {nullptr, nullptr}};
    if (cmoe && ngl > 0) {
        ov[0].buft = ggml_backend_cpu_buffer_type();
        mp_tgt.tensor_buft_overrides = ov;
    }
    std::fprintf(stderr, "Loading target model: %s ...\n", model_tgt.c_str());
    llama_model* mdl_tgt = llama_model_load_from_file(model_tgt.c_str(), mp_tgt);
    if (!mdl_tgt) {
        std::fprintf(stderr, "Failed to load target model\n");
        return 1;
    }

    // 2. Load draft model
    llama_model_params mp_dft = llama_model_default_params();
    mp_dft.n_gpu_layers = ngl;
    std::fprintf(stderr, "Loading drafter model: %s ...\n", model_dft.c_str());
    llama_model* mdl_dft = llama_model_load_from_file(model_dft.c_str(), mp_dft);
    if (!mdl_dft) {
        std::fprintf(stderr, "Failed to load draft model\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(mdl_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd_tgt = llama_model_n_embd(mdl_tgt);
    const int32_t* target_layer_ids = llama_model_target_layer_ids(mdl_dft);
    const uint32_t target_layer_ids_n = llama_model_target_layer_ids_n(mdl_dft);
    std::fprintf(stderr, "Target embd: %d, vocab: %d, extract layers count: %u\n",
                 n_embd_tgt, n_vocab, target_layer_ids_n);

    // 3. Init contexts
    const int nctx = 4096;
    llama_context_params cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx = nctx;
    cp_tgt.n_batch = nctx;
    cp_tgt.n_ubatch = nctx;
    cp_tgt.op_offload = ngl > 0;
    llama_context* ctx_tgt = llama_init_from_model(mdl_tgt, cp_tgt);

    llama_context_params cp_dft = llama_context_default_params();
    cp_dft.n_ctx = nctx;
    cp_dft.n_batch = nctx;
    cp_dft.n_ubatch = nctx;
    cp_dft.op_offload = ngl > 0;
    cp_dft.ctx_other = ctx_tgt; // DFlash borrows tok_embd and output from target
    cp_dft.cb_eval = dflash_dump_eval_cb;
    llama_context* ctx_dft = llama_init_from_model(mdl_dft, cp_dft);

    if (!ctx_tgt || !ctx_dft) {
        std::fprintf(stderr, "Failed to initialize contexts\n");
        return 1;
    }

    // Configure extraction and lattice
    for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
        llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
    }
    llama_set_embeddings_nextn(ctx_dft, true, false);
    llama_set_causal_attn(ctx_dft, false);

    const int32_t mask_token_id = llama_vocab_mask(llama_model_get_vocab(mdl_dft));
    char meta_buf[32] = {};
    int block_size = 8;
    if (llama_model_meta_val_str(mdl_dft, "dflash.block_size", meta_buf, sizeof(meta_buf)) >= 0) {
        block_size = std::atoi(meta_buf);
    }
    int lilicorr_top_k = 8;
    if (llama_model_meta_val_str(mdl_dft, "lilicorr.candidate_topk", meta_buf, sizeof(meta_buf)) >= 0) {
        lilicorr_top_k = std::atoi(meta_buf);
    }
    std::fprintf(stderr, "DFlash block_size=%d, mask_token_id=%d, lilicorr_top_k=%d\n",
                 block_size, mask_token_id, lilicorr_top_k);

    const int n_embd_enc = (int) target_layer_ids_n * n_embd_tgt;

    // Process each prompt
    for (size_t p_idx = 0; p_idx < prompts.size(); ++p_idx) {
        const std::string& prompt = prompts[p_idx];
        std::fprintf(stderr, "\n--- Processing Prompt %zu: %s ---\n", p_idx, prompt.c_str());

        // Tokenize
        std::vector<llama_token> ids(prompt.size() + 16);
        int n_tok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ids.data(), (int) ids.size(), false, true);
        if (n_tok < 0) {
            std::fprintf(stderr, "Tokenize failed\n");
            continue;
        }
        ids.resize(n_tok);
        std::fprintf(stderr, "Tokens: %d\n", n_tok);

        // Reset memory for fresh prompt
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);

        // Target prefill
        llama_batch b_tgt = llama_batch_get_one(ids.data(), n_tok);
        if (llama_decode(ctx_tgt, b_tgt) != 0) {
            std::fprintf(stderr, "Target decode failed\n");
            continue;
        }

        // Sample greedy next token
        const float* logits_tgt = llama_get_logits_ith(ctx_tgt, -1);
        int id_last = 0;
        for (int j = 1; j < n_vocab; ++j) {
            if (logits_tgt[j] > logits_tgt[id_last]) id_last = j;
        }
        std::fprintf(stderr, "Greedy next token: %d\n", id_last);

        // Gather target layer inputs
        std::vector<float> features_buf((size_t) n_tok * n_embd_enc);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float* layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                std::fprintf(stderr, "Failed to get target layer %d input\n", target_layer_ids[k]);
                return 1;
            }
            for (int i = 0; i < n_tok; ++i) {
                float* dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float* src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, n_embd_tgt * sizeof(float));
            }
        }

        // Inject into drafter
        common_batch batch_inject(ctx_dft);
        for (int i = 0; i < n_tok; ++i) {
            const llama_pos p = (llama_pos) i;
            const llama_pos pos_arr[4] = { p, p, p, 0 };
            batch_inject.add_embd({ features_buf.data() + (size_t) i * n_embd_enc, 1, (size_t) n_embd_enc },
                                  pos_arr, 0, false);
        }
        if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get()) != 0) {
            std::fprintf(stderr, "Draft injection failed\n");
            continue;
        }

        // Context embedding from projection (last token)
        const float* projected = llama_get_embeddings_nextn(ctx_dft);
        if (!projected) {
            std::fprintf(stderr, "Failed to get drafter context embeddings\n");
            continue;
        }
        const int n_embd_dec = llama_model_n_embd(mdl_dft);
        std::vector<float> anchor_embd(n_embd_dec);
        std::memcpy(anchor_embd.data(), projected + (size_t) (n_tok - 1) * n_embd_dec, n_embd_dec * sizeof(float));

        // Now run the draft block [id_last, mask * 7]
        common_batch batch_draft(ctx_dft);
        for (int i = 0; i < block_size; ++i) {
            const llama_token tok = (i == 0 ? id_last : mask_token_id);
            const int32_t idx = batch_draft.add(tok, n_tok + i, 0, true);
            batch_draft.set_embd_state(idx, { anchor_embd.data(), 1, (size_t) n_embd_dec });
        }

        if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_draft.get()) != 0) {
            std::fprintf(stderr, "Draft block decode failed\n");
            continue;
        }

        // Read drafter logits for draft positions 1..7 (7 tokens)
        const int n_draft = block_size - 1; // 7
        std::vector<float> dft_logits((size_t) n_draft * n_vocab);
        for (int i = 1; i <= n_draft; ++i) {
            const float* lg = llama_get_logits_ith(ctx_dft, i);
            if (!lg) {
                std::fprintf(stderr, "Failed to get logits for draft token %d\n", i);
                return 1;
            }
            std::memcpy(dft_logits.data() + (size_t) (i - 1) * n_vocab, lg, n_vocab * sizeof(float));
        }

        // Read lattice and walk to get drafted tokens
        const float* lattice = llama_get_embeddings_nextn(ctx_dft);
        std::vector<int32_t> draft_tokens;
        if (lattice) {
            int32_t predecessor = 0;
            for (int i = 1; i < block_size; ++i) {
                const float* row = lattice + (size_t) i * n_embd_dec;
                const float* scores = row + lilicorr_top_k + (size_t) predecessor * lilicorr_top_k;
                predecessor = 0;
                for (int k = 1; k < lilicorr_top_k; ++k) {
                    if (scores[k] > scores[predecessor]) predecessor = k;
                }
                draft_tokens.push_back((int32_t) row[predecessor]);
            }
        }
        std::fprintf(stderr, "Drafted tokens (%zu):", draft_tokens.size());
        for (int t : draft_tokens) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");

        // Write dump file
        std::string dump_path = out_dir + "/dump_prompt_" + std::to_string(p_idx) + ".bin";
        FILE* fout = std::fopen(dump_path.c_str(), "wb");
        if (!fout) {
            std::fprintf(stderr, "Cannot write to %s\n", dump_path.c_str());
            continue;
        }

        const uint32_t magic = 0x44464c53; // "DFLS"
        std::fwrite(&magic, sizeof(magic), 1, fout);
        std::fwrite(&n_tok, sizeof(n_tok), 1, fout);
        std::fwrite(&id_last, sizeof(id_last), 1, fout);
        const int pos0 = n_tok;
        std::fwrite(&pos0, sizeof(pos0), 1, fout);
        std::fwrite(&n_draft, sizeof(n_draft), 1, fout);
        std::fwrite(&n_vocab, sizeof(n_vocab), 1, fout);
        std::fwrite(&n_embd_enc, sizeof(n_embd_enc), 1, fout);

        // Prompt tokens
        std::fwrite(ids.data(), sizeof(int32_t), n_tok, fout);
        // Target features
        std::fwrite(features_buf.data(), sizeof(float), (size_t) n_tok * n_embd_enc, fout);
        // Llama drafter logits (7 * n_vocab)
        std::fwrite(dft_logits.data(), sizeof(float), (size_t) n_draft * n_vocab, fout);
        // Draft tokens (7)
        std::fwrite(draft_tokens.data(), sizeof(int32_t), draft_tokens.size(), fout);

        // LiLiCorr lattice (block_size rows of n_embd_dec floats), read by nh_dflash_test when present
        if (lattice) {
            std::fwrite(lattice, sizeof(float), (size_t) block_size * n_embd_dec, fout);
        }

        std::fclose(fout);
        std::fprintf(stderr, "Wrote dump to %s\n", dump_path.c_str());
    }

    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(mdl_dft);
    llama_model_free(mdl_tgt);
    llama_backend_free();

    std::fprintf(stderr, "Done dumping parity data.\n");
    return 0;
}

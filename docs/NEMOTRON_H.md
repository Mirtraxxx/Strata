# Nemotron-H MoE (Nemotron 3.5 Super VL 120B-A12B)

Strata runs NVIDIA's Nemotron 3.5 Super VL from unsloth's GGUFs (`unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF`,
architecture `nemotron_h_moe`) on an engine path of its own, `src/nemotron_h`. `strata` switches to it from the
file's architecture, the same way it switches to Qwen3.6's path; `strata-nh` runs it on its own. Flash-Next's and
Qwen3.6's paths are unchanged.

## What the model is

88 flat sublayers, each `x += op(rms(x) * attn_norm)`:

- 40 Mamba-2 mixers: `in_proj` (z | xBC | dt), a causal conv of 4 taps with a bias and SiLU, the selective scan
  (128 heads of 64, state 128, 8 groups, a scalar decay per head), `y + D x`, the SiLU(z) gate, an RMS norm per group
  of 1024 channels, `out_proj`;
- 8 attention mixers: 32 query heads over 2 KV heads, head 128, and no positional encoding at all;
- 40 MoE blocks: a sigmoid router with a selection bias (512 experts, top 22, the weights normalized and scaled by
  5), the routed experts in a 1024-wide latent space (`down(relu(up x)^2)`, no gate), and a shared expert
  `down(relu(up h)^2)` on the full 4096 width;
- one MTP block (`blk.88`): `eh_proj` over `[enorm(embedding) | hnorm(hidden)]`, an attention and a MoE sublayer, a
  LayerNorm head norm, the main output head.

Images go through the RADIO vision tower and projector in the repository's `mmproj-BF16.gguf`.

## How Strata runs it

The routed experts are 78 GB of the 87 GB `UD-Q4_K_XL` file, so this path is always in the hybrid mode:

- every dense weight (Mamba, attention, latent and shared-expert projections, router, embedding, head; about 8 GiB)
  is in VRAM;
- each MoE layer gets the same number of VRAM slots for experts from the VRAM that is left, and the slots follow the
  routing (the most used experts move in between requests and every 32 decoded tokens);
- a batch of up to `--hybrid-rows` rows (decode, MTP verify windows, short prompts) runs its resident experts on the
  GPU and the others on the CPU (ggml-cpu's dot products), side by side;
- a bigger prompt chunk (`--chunk`, 2048 by default) streams every non-resident expert through a ring of VRAM buffers
  (`--stream-ring`) and runs each expert's rows as two FP16 GEMMs.

What this path took over from Flash-Next's:

- **the expert arena**: every routed expert is copied once into pinned host memory at start (73.4 GiB, ~70-90 s), on
  2 MiB pages (transparent huge pages, then registered with CUDA; `STRATA_NH_ARENA_4K=1`: cudaHostAlloc's 4 KiB
  pages; `--no-arena`: the mapped file). The CPU experts read it and every upload is a DMA at full PCIe speed. It
  needs that much free RAM (checked; otherwise the mapped file is read);
- **the adaptive VRAM tier**: the slots follow the routing counts (every 32 decoded tokens, up to 512 swaps after a
  prompt);
- **the expert profile**: `--expert-profile F` starts every layer's slots with its most-routed experts,
  `--expert-profile-save F` writes the counts every 8 requests and at exit (they add up across restarts when both
  name the same file). Format `NHP1`: per MoE layer (the MTP block last) a float count per expert;
- **the PCIe share of the misses** (`--pcie-frac F`, up to `--pcie-max N` experts a layer): that part of a layer's
  missed experts uploaded and run on the GPU while the CPU computes the rest;
- **the prefetch** (`--prefetch N`): the next MoE layer's router applied to the hidden one sublayer early, its N most
  likely non-resident experts uploaded while that sublayer runs (92-96% of the uploaded ones were then routed, at
  N = 2-4);
- **turn checkpoints**: the Mamba state at the end of each request's prompt (but its last token), up to 4, in pinned
  RAM (173 MB each). The state cannot step back, the KV cache can, so a request that returns to one of them along
  the live sequence (a regenerated or edited last answer) reads only the rest of its prompt.

The PCIe share and the prefetch are off by default: on the PC below the CPU experts read RAM at its ceiling already
(~40 GB/s of DDR4-2133), and an upload reads the same RAM, so they measured neutral or slower there. A PC with a
faster PCIe link than RAM per core, or a slower CPU, is where they pay - the same reasoning as Flash-Next's
`--pcie-frac`.
Also:

- the KV cache is FP16, 9 attention layers (8 + the MTP block): 9 KiB per token of context;
- `--mtp N` drafts N tokens with the MTP block for greedy requests; the main model verifies them and the Mamba state
  is then committed for the accepted rows only, so the text is the plain greedy text.

The formulas are llama.cpp's for the same tensors (`src/models/nemotron-h.cpp`, `nemotron-h-moe.cpp`,
`mamba-base.cpp`, ggml's `ssm_conv` / `ssm_scan`); `tools/nemotron_h/parity.py` compares the two.

## Files

| File | What |
|---|---|
| `src/nemotron_h/nh_engine.cu` | the engine: loading, the sublayers, the hybrid experts, the stream, MTP, `--serve` |
| `src/nemotron_h/nh_kernels.cuh` | its kernels: Mamba-2 conv / scan / gated norm, NoPE attention, the router, combines |
| `src/nemotron_h/nh_cpu.cpp` | the CPU experts (ggml-cpu `vec_dot`, relu^2) |
| `src/kernels/cuda/iq_kernels.cu` | `native_expert_grouped_relu2`: the GPU experts without a gate (new, opt-in) |
| `tools/nemotron_h/nh_llama_dump.cpp` | llama.cpp's side of the parity check (`-cmoe`, sublayer dumps) |
| `tools/nemotron_h/parity.py` | KL / top-1 against llama.cpp, per-sublayer errors |
| `tools/strata_tokenizer.py` | the `tekken` pre-tokenizer (with `ignore_merges`) next to the Qwen one |
| `serve/server.py` | `<img> <image> x n </img>` for this model's pictures; stop ids that are single tokens only |

## Running it

```bash
# the tokenizer (once): vocab, merges, the chat template from the GGUF
STRATA_GGUF_PY=<llama.cpp>/gguf-py .venv/bin/python tools/strata_tokenizer.py --gguf <first shard> --out <pack dir>
# a quick greedy run
strata-nh --model <first shard> --tokens 10,1785,... --n 32 [--mtp 2]
# the server (strata-<name>.json: "exe" strata, "args" ["--native", "<first shard>", "--max-context", "65536",
# "--mtp", "2"], "tokenizer" <pack dir>/tokenizer, "arch" "nemotron_h_moe", "vision" {"exe": strata-vision, ...})
python serve/server.py --engine strata --config strata-<name>.json --port 8092
```

The image encoder needs llama.cpp's mtmd with `PROJECTOR_TYPE_NEMOTRON_OMNI` (upstream since PR #30520): build
`tools/vision` with `-DLLAMA_DIR=<that checkout>`.

## Measured

On one PC: RTX 3090 (24 GB, PCIe), i7-7800X (6 cores, AVX-512), 95 GiB RAM; `UD-Q4_K_XL` (87 GB), llama.cpp 0.7.0
(`620d2225c`) as the reference, 2026-10-15.

- **Parity** (`tools/nemotron_h/parity.py`, 5 chat prompts, thinking off, 32 teacher-forced steps each, 87 slots per
  layer): the first answer token matched on all 5; over the answers (122 steps up to `<|im_end|>`) mean KL 0.00106,
  median 0.000015, top-1 0.992 against llama.cpp CUDA (-cmoe).
- **A 981-token prompt** (the streamed prompt path): KL 0.00328 against llama.cpp CUDA, while llama.cpp's own CPU
  backend is at 0.00277 against the same; the per-sublayer error over the prompt's positions grows with depth exactly
  as llama.cpp CPU's does (layer 40: 0.176 vs 0.174, layer 80: 0.311 vs 0.302) - the text repeats one paragraph 12
  times, where both engines' rounding drifts the same way.
- **MTP**: with 2 and 3 drafts the greedy text is the same as without (96 tokens); 85% / 82% of the drafts accepted,
  but no faster on this PC (10.9 / 10.8 vs 11.2 tok/s): a verify window routes ~3x the distinct experts, and most of
  them are CPU misses. The config leaves `--mtp 0`.
- **Speed** (`strata-nh --bench-pp 2048 --bench-tg 32`, two runs): prompt 124-129 tok/s, decode 11.7-12.6 tok/s.
  Through the server: a 2795-token prompt in 27.2 s (103 tok/s), 256 tokens decoded at 10.7 tok/s.
- **The server**: text, thinking (reasoning separated), a tool call (`get_weather` with `{"city":"Kyiv"}`), a picture
  (a red square, a blue circle and the text "HELLO STRATA", described correctly; 300 image tokens, 59 ms in the
  encoder on the GPU), streaming, and the second turn of a conversation reusing the first (45 cached tokens).
- `--pin-experts`: page-locking the mapped GGUF failed on this PC (cudaHostRegister on the read-only mapping), the
  stream used its pinned buffers; same speed.
- The CPU experts against a dequantized f64 reference (`nh_cpu_test`, every MoE layer, 2 experts x 3 tokens): worst
  relative error 1.2% (the activations rounded to Q8_K / Q8_0, as llama.cpp's CPU backend does).

**With the Flash-Next mechanisms** (same PC, 2026-10-15 evening; 128-160 greedy tokens of one coding prompt unless
said, `--expert-slots 78` for the A/B rows so every row has the same VRAM tier):

| change | decode tok/s | note |
|---|---|---|
| expert arena, 4 KiB pages | 9.16 | the CPU experts at ~30 GB/s |
| expert arena, 2 MiB pages | **11.75** | +28%: ~0.094 ms an expert, ~40 GB/s = this RAM's measured read ceiling |
| + prefetch 2 / 4 / 8 | 11.62 / - / - | 4 KiB-page runs: +2.5% / +7% / +1%; 92-96% of the uploaded experts routed at 2-4 |
| + PCIe share 0.15 and prefetch 4 | 11.31 | the uploads read the same, already saturated RAM |
| + expert profile (10 mixed prompts), held-out prompt | 13.48 (11.83 without) | VRAM share of the routed experts 34% (16% without) |
| adaptive swaps 8 / 32 / 128 every 32, or 64 every 16 | 12.98 / 12.91 / 12.91 / 12.83 | 160 tokens; the rate does not matter, the profile does |

- `strata-nh --bench-pp 2048 --bench-tg 32` with the arena: prompt 337-350 tok/s (124-129 before), decode
  12.2-13.1 tok/s.
- Through the server (vision on the GPU, 63 slots a layer): a 2795-token prompt in 11.3 s (246 tok/s; 103 before), 256
  tokens decoded at 11.2 tok/s.
- A regenerated answer (the same prompt again) restored its turn checkpoint: the prompt in 88 ms instead of 2011 ms,
  the same text.
- Parity with the arena and the profile (2 chat prompts + the 981-token one): answers-only KL 0.00124, top-1 0.984;
  with the PCIe share 0.6 and prefetch 16 (82% of the routed entries on the GPU): KL 0.00279 over all steps, top-1
  0.979 - the GPU and CPU experts round their activations differently (q8_1 / Q8_K), the text stays the model's.
- The arena takes 70-130 s at start (from the page cache, or the disk when it was evicted) and leaves ~15 GiB of this
  PC's 95 GiB free.

Not done yet: MMQ for the prompt's experts (it takes a per-expert FP16 GEMM today), video input, HIP and SYCL builds,
a setup entry.

## Nemotron 3.5 Bolt 74B-A9B

The same path runs NVIDIA's Bolt (`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`, also `nemotron_h_moe`; the
abliterated files of the same layout load the same way). What differs from Super VL:

- the MoE layers do not share one geometry: 240-496 routed experts and a latent width of 1536-2560 per layer, top 9,
  a shared expert of 5376. The engine reads `E` and `FF` per layer (slots, arena, profile, CPU and GPU experts);
- the routed experts are IQ2_XXS (15.9 GiB of the 24 GB file) and the Mamba `in_proj` / `out_proj` are NVFP4
  (ggml type 40). Both go through the existing native kernels: IQ2_XXS is added to `STRATA_D_FMTS` and NVFP4 to
  `native_mmvq` (an Ampere card has no FP4 hardware, so it is dequantized in the kernel); the vector NVFP4 dequant is
  bitwise equal to the scalar one (`dequant_wide_parity`, 418 cases). Both edits are additive: no other format's
  instance changes;
- there is no MTP block. Speculative decoding uses NVIDIA's separate 1.13B LiLiCorr DFlash drafter instead (below);
- the tokenizer's `pre` is `pixtral`: llama.cpp's LLAMA3 split, merges ignored, BOS prepended.
  `tools/strata_tokenizer.py` gained that split (15/15 test strings identical to `llama-tokenize`), and the server
  prepends BOS once on the rendered prompt when the pack says `add_bos_token` (Qwen packs say false: unchanged).

The whole model is 24 GB, so on a 24 GB card most of the experts can be resident. Bolt-specific work:

- **prompt experts on MMQ**: a prompt chunk runs each group of experts as one MMQ launch that reads IQ2_XXS directly
  (not a per-expert FP16 GEMM), with the next group's expert copies double-buffered on their own stream.
  `STRATA_NH_EXPERT_MMQ=0` goes back to the FP16 GEMMs;
- **Mamba-2 for prompts and decode** (`nh_ssm_scan.cu`): a chunked SSD scan for prompt chunks, a tiled prompt conv,
  and a coalesced single-token state update (65 -> 12 us a layer). `nh_ssm_scan_test` checks the chunked scan
  against the sequential one from T = 1 to 8192, with no model needed;
- **small decode kernels**: warp-shuffle add+norm and router kernels (the old ones used a single block), about 2.5 ms
  less GPU time a token;
- **elastic VRAM**: `--vram-reserve-mib N` is what the slots leave free (default 1024). The server's `/v1/vram`
  shrinks the expert slots so another program can borrow the card, and grows them back afterwards. Dense weights
  and the KV cache never move. `--vram-elastic` is accepted for the server's sake (the slots are what resize);
- **DFlash** (`--dflash <LiLiCorr GGUF>`, `--dflash-d D`, default 4; `nh_dflash.cu`): the drafter reads the residual
  stream of its 6 target sublayers, drafts D tokens with its lattice corrector, and the engine verifies them in one
  window of D + 1 rows with the same commit + Mamba recompute as `--mtp`. The drafter's FFN stays packed NVFP4
  (1.53 -> 0.43 GB of VRAM; FP32 math, since an FP16-accumulating version overflowed on real activations). Sampled
  requests are verified by speculative sampling against the server's own sampler (the same penalty, top-k, min-p and
  top-p function), so their output distribution is exactly plain sampling's (checked with 4M Monte Carlo draws: every
  row within 0.0013). With `--dflash`, `--pcie-frac` defaults to 0.5, because a verify window's misses share many
  experts; without it the default stays 0, as for Super VL;
- `STRATA_NH_PROF=1`: a per-phase timer for prompt chunks (CUDA events; off by default).

| File | What |
|---|---|
| `src/nemotron_h/nh_dflash.cu`, `.hpp` | the LiLiCorr drafter: its 6-layer feature taps, attention + NVFP4 FFN, the lattice |
| `src/nemotron_h/nh_ssm_scan.cu`, `.cuh` | chunked SSD prompt scan, coalesced decode state update |
| `src/nemotron_h/nh_ssm_scan_test.cu` | the chunked scan against the sequential one (synthetic) |
| `tools/nemotron_h/nh_dflash_dump.cpp` | llama.cpp's side of the drafter check (draft logits + lattice dumps) |
| `tools/nemotron_h/nh_dflash_test.cpp` | the drafter's logits and lattice against those dumps |

```bash
strata-nh --model <Bolt GGUF> --max-context 131072 --vram-reserve-mib 512 --dflash <LiLiCorr GGUF> --bench-pp 2048 --bench-tg 128
# server config: "args" ["--model", <Bolt GGUF>, "--max-context", "131072", "--vram-reserve-mib", "512",
# "--dflash", <LiLiCorr GGUF>], "arch" "nemotron_h_moe", "vram_elastic" true; sampling temp 1.0, top-p 0.95
```

### Measured (Bolt)

On another PC: RTX 3090 (24 GB), Ryzen 7 5700X (8 cores, AVX2), 64 GB DDR4 at 2800 MT/s. NVIDIA's stock
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, 128K context, 512 MiB kept free (296-347 expert slots a layer, ~96% of
decode routes resident), llama.cpp `c0fd786` as the reference, 2026-10-18.

- **Parity** (`parity.py`, 5 chat prompts x 48 teacher-forced steps): first token 5/5, mean KL 0.00062 (median
  0.000010, p99 0.0079), top-1 0.9958 against llama.cpp CUDA. Long prompts (~1.5-1.9K tokens, 96 steps): mean KL
  0.00086, top-1 0.9896.
- **The drafter** (`nh_dflash_test`, 5 short + 2 long prompts): draft logits top-1 1.000 at all 7 positions, KL
  <= 0.00053. Fed llama.cpp's logits, the lattice gives the same candidates on every row, and every argmax miss is a
  near tie in llama.cpp too.
- **Speed** (`--bench-pp 2048 --bench-tg 128`, run 0 / 1): prompt 1,209 / 1,488 tok/s, decode 45.8 / 52.6 tok/s.
  llama.cpp on the same card (`llama-bench`, small context, no drafter): everything on the GPU 1,013 / 62.1; with
  50 blocks' experts in RAM (`-ncmoe 50`, which fits 256K context) 397 / 25.0.

| change (same card, plain decode unless said) | prompt pp2048 | decode tg |
|---|---|---|
| #2740 + per-layer geometry + NVFP4 (256K, 2 GiB free) | 758-763 | 28-30 |
| + prompt experts on MMQ | 1,038-1,178 | about the same |
| 128K, 512 MiB free (more slots) | 1,117-1,227 | 34.4-39.2 |
| + double-buffered copies, NVFP4 vector dequant, small decode kernels, SSD decode update | 1,209-1,488 | 45.8-52.6 |

- **DFlash in decode** (greedy, 200 tokens; D and the PCIe share swept): D = 4 with share 0.5 is 72.5 tok/s on chat
  and 78.0 on code against 36.7 / 37.4 plain (2.0x / 2.1x). D = 5 is 81.9 on code but slower on chat. A share of 1.0
  loses to 0.5 (the uploads and the CPU experts stop overlapping), and plain decode does not care about the share.
- **Through the server** (temp 1.0, top-p 0.95, top-k 64, DFlash on): a ~90-token paragraph at 53.3 tok/s, a
  300-token history answer at 50.3, a 300-token Python function at 73.6, short answers ~62. With `/v1/vram` asking
  for 12 GB, the slots shrank to 520 (from 10,608) and 12.3 GB was free (a short answer at 27.4 tok/s); growing back
  restored 10,608 slots (57.1 tok/s). The free VRAM never went under 616 MiB over 31 samples. Sampling costs DFlash
  about a third against greedy on chat (sampled tokens follow the draft less often).
- `--bench-tg` with `--dflash` loaded: 31.8 / 43.5 tok/s. The bench decodes without drafting, so this only shows
  the drafter's ~1.2 GB taken from the expert slots.

Not tested: Super VL 120B with these changes (it does not fit this PC's RAM). Its options keep their defaults
(`--pcie-frac` stays 0 without `--dflash`), and the new IQ2_XXS / NVFP4 instances are for types its file does not
use. But the add+norm, router and Mamba scan kernels are shared by both models, so Super VL runs the new ones too,
and they have been checked on Bolt only. HIP and SYCL were not built either.

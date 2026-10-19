# v0.2.0 vs the previous `mine` engine, RTX 3090 (2026-10-19)

Model: Swift 1.5 Flash-Next 180B IQ3_XXS uncensored, `strata-swift-iq3_xxs-uncensored.json` (`--spec 4`,
`--max-context 262144`, `--kv int8`, `--vram-reserve-mib 1500`, control-vector projection). RTX 3090 24 GB, 64 GB RAM.
Tool: `tools/strata_bench.py --long`, 5 runs per test; the new engine ran first and the old one last, the reverse of
the first pass (which agreed).

| Test (median) | old engine, 16K chunk | v0.2.0, 32K chunk |
|---|---|---|
| chat decode tok/s | 56.8 | 82.1 |
| agent decode tok/s | 53.9 | 83.5 |
| chat first token | 0.63 s (one 128.7 s outlier) | 0.41 s |
| 4K prompt tok/s | 1331 | 1350 |
| 16K prompt tok/s | 2014 | 2043 |
| 32K prompt tok/s | 2071 | 2386 |
| 32K first token | 15.6 s | 13.5 s |

## Prompt chunk (`--prefill`) on v0.2.0

First pass, 3 runs: 32K prompts 2417 tok/s with a 32K chunk vs 2091 with 16K; 16K prompts 2077 vs 2009. Idle VRAM is
the same. During a prompt the 32K chunk borrows 6631 of 7810 expert-cache slots (10.7 GiB) vs 3812 (6.1 GiB).

Decode right after the prompt (400 generated tokens, 3 runs each, median):

| Decode after | 16K chunk | 32K chunk |
|---|---|---|
| a short prompt | 90.1 tok/s | 99.2 |
| a 31.6K-token prompt | 83.5 | 87.4 |
| first token, 31.6K prompt | 13.6 s | 12.8 s |

The borrowed slots refill without slowing the answer, so the daily config now uses `--prefill 32768`.

# _honest_oj_fused_gpu: device winner pick of the fused optimal_joint honest floor

Box: GTX 1050 Ti (6 SMs, 4 GB), 100k rows x 100 features, all FE families, `nbins_strategy="optimal_joint"`.
Kernel inputs captured from a real fit: 3 folds x 166 re-binned columns x 4 candidates = 1992 cells, 33,334 validation rows per fold, binary target,
193 pool columns over 30,000 evaluation rows. Times per 64-permutation block, same process:

| route | block | winner pick |
|---|---|---|
| CPU `_fused_block` (numba prange) | 1.87 s | 1.93 s* |
| CPU `_fused_block_minority` (majority class by subtraction) | 1.63-1.83 s | |
| device, shared-atomic histograms (`oj_cell_mi`) | 0.50 s | 0.34-0.41 s (kernel 0.386 s) |
| device, per-thread 16-bit counters (`oj_cell_mi_private`) | 1.05 s | 0.92 s (rejected: 32 KB smem/block, 1 block/SM) |
| device, bitmask popcounts (binary target, default) | 0.275 s | 0.112 s |

\* winner step alone, measured separately; the pool step is 0.17 s of the CPU block.

Null maxima of every device route are bit-identical to the CPU kernel on the captured block (max diff 0.0) and in
`tests/feature_selection/filters/test_honest_oj_minority.py` (binary/multiclass, balanced/imbalanced).

Fit: optimal_joint 100k 221.0 s -> 160.8 s; permutation scoring per fit 68.0 s -> 9.4 s; floor total 80.0 s -> 21.7 s.
Remaining: ~12 s of floor setup outside the permutation loop (plans, count slots, pool), and the CPU pool step (could use the same popcount scheme
for a binary target, and overlap with the next block's device picks).

Dispatch: `scorer_device_choice("honest_oj_winners", n_rows, cells)` (work fallback until the KTC entry is swept), VRAM guard at half the free memory
(codes + bitmasks), `MLFRAME_HONEST_OJ_GPU=0` off switch, `MLFRAME_HONEST_OJ_GPU_BITMASK=0` forces the atomic route; any device fault -> CPU kernel.

## 2026-11-03: pool step on the device

The CPU pool step (largest corrected MI of 193 pool columns over 30,000 rows) was 0.146 s per 64-permutation block. `DevicePool`
reuses the bitmask popcounts for a binary target code: 0.05 s per block. Null maxima match the CPU pool to 1.1e-18 (device `log`);
30k floors unchanged (0.0006034131614497845). `MLFRAME_HONEST_OJ_GPU_POOL=0` keeps the pool on the CPU.

## 2026-11-03: fit after the device pool; KTC sweep

optimal_joint 100k fit 146.9 s (floor total 12.4 s, permutation scoring 5.8 s). The host/device choice now has its own KTC sweep
(`_sweep_winners`, entry `fe_scorer_device_honest_oj_winners`): one 64-permutation block, 30k rows x 300 cells: host 0.063 s,
device 0.016 s, same picks.

## 2026-11-04: candidate null from the floor

`honest_candidate_nfailed` re-binned each candidate column under its 99 permutations (0.13-0.15 s per call at 30k; 14.2 s per
100k fit). Its stream is the head of the floor stream (same `base_seed`), so the fused floor keeps the device pool step plain MI
(999 x pool) and a pool column count is a lookup (0.001 s). 24/24 calls at 30k equal the pooled path, including `None` for
engineered columns past the binner outputs. CPU-floor runs keep no matrix; candidates then take the pooled path.

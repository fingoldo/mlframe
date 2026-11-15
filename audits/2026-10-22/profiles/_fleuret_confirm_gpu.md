# `_fleuret_confirm_gpu` profile record (2026-11-15)

Device recheck of a window of screen candidates: kernel `fleuret_cmi_blocks`, one block per `(candidate x permutation, conditioning set)`. Consumer: `_confirm_predictor_helpers.fleuret_conditional_recheck` through `_ConfirmPrefetch.top_up` (the prefetch window of the confirmation walk). CPU twin: `get_fleuret_criteria_confidence` / `fleuret_confidence_prange` (njit). Host: GTX 1050 Ti (sm_61), Windows, nothing else running during the timed lines.

## What it replaces

Per candidate that passed the marginal bootstrap, `get_fleuret_criteria_confidence`: 3 permutations (the screen's default budget) of two LCG shuffles of 100000 codes, and for each selected variable the four joint entropies of the shuffled columns. In the 100000 x 100 all-families fit: `confirm_candidate` 178 calls, 25.0 ms a call, 4.45 s of thread time (`screen_calls.py`); the main thread of the stage waited 1.8 s for these futures.

## Kernel profile (nvprof, `fc_prof.py`: 4 candidates x 3 permutations, 5 selected variables, 100000 rows, 10 bins, 5 launches)

| item | first version (xyz, xz, yz, z tables: 4 shared atomics a row) | final (one xyz table, marginals by cell sums) |
|---|---|---|
| kernel time a launch | 436 us | 329 us |
| share of GPU time | 47.7 percent | 38.5 percent |
| `gld_efficiency` | 99.85 percent | 99.85 percent |
| `achieved_occupancy` | 0.760 | 0.737 |
| `sm_efficiency` | 97.6 percent | 86.8 percent |
| `shared_efficiency` | 0.27 percent | 1.34 percent |
| shared load / store transactions per request | 465 / 564 | 69 / 125 |
| `ipc` | 1.89 | 1.85 |
| `warp_execution_efficiency` | 99.9 percent | 99.5 percent |
| DRAM read | 12.4 GB/s | 13.4 GB/s |

The remaining limit of the kernel is the shared-memory atomic of a 300 to 3000 cell table (the same limit as the other histogram kernels of the package); the kernel is 0.33 ms of a 7.4 ms window, so it is not the cost any more.

## Window wall (min of 5, `fc_bench.py`; the counts equal the host kernel's)

| step | window of 4 |
|---|---|
| device, first version | 8.8 ms |
| device, generalised (nine statistics, multi-column x, orders), before the fast branches | 9.1 ms |
| device, final (single-column fast branches, no per-set int64 arithmetic for order 1) | 7.4 ms |
| host kernel, sequential | 53.2 ms |
| host kernel, 4 threads (as the prefetch runs it) | 46.7 ms |

cProfile of the call: the host LCG shuffles (njit prange over the jobs) are the largest piece, then the gather and the uploads. The shuffles are strictly sequential per job (an LCG Fisher-Yates), so they stay on the host; a device version would be one thread a job.

## In the 100000 x 100 all-families fit (`screen_calls.py`)

`confirm_candidate` 178 calls: 4.45 s of thread time before (25.0 ms a call), 0.92 s after (5.2 ms a call); `mi_direct` (20 calls) and the honest null (20 calls) are unchanged at 0.03 s and 0.11 s. Whole-fit wall 58.8 s against 58.1 s: inside the noise of the box (+-7 s), the thread time is what is resolved.

## Equivalence and scope

17 tests in `tests/feature_selection/gpu/test_fleuret_confirm_gpu.py`: counts equal the host kernel for the plug-in CMI, Miller-Madow, symmetric uncertainty and both JMIM variants, one- and two-column candidates, interactions order 1 and 2, the worker segments of the joblib route, over-cap conditioning sets, with and without the `x` shuffle, over gains that produce both outcomes; a real `MRMR.fit` selects the same features with the device on and off. Dispatch by the kernel-tuning entry `fleuret_confirm_backend` (host versus device on the row count, exact comparison of the counts), fallback mark 20000 rows. No environment switch: the tuner and the presence of a card decide, `use_gpu` of the constructor is not consulted.

# Why a 10 000-row fit costs almost as much as a 100 000-row fit (2026-10-11)

Question: a second `MRMR` pass on 10-20k rows costs 4-5 s, about the same as a whole fit at 100k. Where does the fixed part go? Measured with the F2 example in GPU strict-resident mode (the benchmark settings: `full_npermutations=10`, `baseline_npermutations=20`), warm process, novel data per fit, `cProfile` (inflates every call by about a microsecond, so absolute figures are upper bounds, the shares are reliable) and Nsight Systems 2022.4 on the CUDA side (`nvprof` runs too once `PATH` has the CUPTI directory, `export PATH="/d/CUDA/v12.8/extras/CUPTI/lib64:$PATH"`; it gives the same ranking: `radix_select_interp_f64_v3` 39.2% of GPU time, 821 ms). Scripts and raw numbers are in `D:/Temp/agent_brain/prof_cmp.py`, `prof_callees.py`, `nsys10k.nsys-rep` (not in the repository).

## 1. How much of a fit is fixed

| rows | wall (cProfile) | sum over functions of min(self time at 10k, self time at 100k) |
|---|---|---|
| 10 000 | 5.17 s | 3.74 s = **72% of the 10k fit is size-independent** |
| 100 000 | 7.18 s | the same 3.74 s is 52% of the 100k fit |

So going from 10k to 100k rows adds 2 s, the other 3.7 s is paid at any size.

## 2. Where the fixed 3.7 s is (cumulative time, 10k rows, share that does not grow to 100k)

| stage | cum at 10k | fixed share | what it is |
|---|---|---|---|
| `_assign_support` (post-selection passes) | 1.70 s | 100% | everything after the greedy: raw-redundancy drop, retention, support assembly |
| ... `retain_usable_pure_forms` -> `build_usability_candidate_pool` | 1.25 s / 1.11 s | 100% | the linear-wrapper retention builds a candidate pool twice |
| ... ... `score_pair_combos_table_resident` | 0.56 s | 100% | one GPU table of 1734 pair combinations, row-subsampled, so it does not depend on n |
| ... ... `_step3_pidx_n1_n2`, `_step1_ua_codes` | 0.45 s, 0.35 s | 100% | per-candidate code building (10 calls) |
| ... `drop_redundant_raw_operands` (`_excess_and_floor` 24 calls) | 0.32 s | 100% | one permutation-null evaluation per raw operand |
| `_run_fe_step` (pair search) | 1.47 s | 52% | 8 pairs, GPU chunks, partly size dependent |
| `_discretize_inputs` -> `per_feature_edges` | 0.48 s | 100% (0.04 s at 100k, a cache hit there) | edge search of the quantile binning |
| `_conditional_perm_null` (39 calls, 9 ms each) | 0.36 s | 100% | GPU permutation null of the CMI redundancy gate |
| numba dispatcher / disk-cache loads (48 first calls, 15 cache loads) | 0.30 s | per process | each process loads its cached kernels on first use |
| `gc.collect` (explicit) | 0.16 s | 100% | one call in the pipeline |

## 3. The CUDA side (Nsight Systems, one warm fit at 10k rows, `fe_max_steps=2`)

| item | count | time |
|---|---|---|
| kernel launches (`cuLaunchKernel` 28 121 + `cudaLaunchKernel` 7 102) | 35 223 | 0.72 s + 0.14 s = **0.86 s of CPU time spent only launching** (25 us per launch on this WDDM box) |
| `cudaMemcpyAsync` | 5 740 | 1.56 s inclusive (it carries the waits for queued kernels; one of them waits 0.50 s) |
| `cudaStreamSynchronize` + `cuStreamSynchronize` | 2 652 + 122 | 0.21 s |
| kernels executing | | about 2.1 s of GPU time, of which `radix_select_interp_f64_v3` **0.80 s (38%)**, 20 launches of 40 ms with 1734 blocks, i.e. the 1734-combination pair table above, **in float64** |
| `cudaMalloc` + `cuMemAlloc` | 47 + 69 | 0.21 s |

The 0.48 s that `cProfile` attributes to one `cupy.asnumpy` inside `score_pair_combos_table_resident` is the wait for that table's kernels, not copying.

Caveat from the profiling notes of this project: per-kernel GPU shares from `nvprof` and Nsight are serialisation-distorted (kernels that overlap CPU work in production look larger). The wall truth for the pair table is the `cProfile` figure of the blocking `cupy.asnumpy` that waits for it (0.48 s), and any saving from F-1 must be confirmed by a paired full-fit A/B before it counts.

## 4. Answer

The fixed cost is **not** one launch-overhead problem and not a single hot spot. It is four things, in order:

1. **The post-selection passes (about 1.7 s)**, above all the usability-aware pure-form retention (1.25 s), which runs a 1734-combination GPU pair table in float64 (0.8 s of kernel time, 38% of all GPU time of the fit) that does not depend on n, and then a Python-level per-candidate build; plus the raw-redundancy drop (0.32 s of permutation nulls).
2. **About 35 000 kernel launches at 25 us (0.86 s of pure CPU launch time)** and 5 700 copies / 2 700 synchronisations (the waits for GPU work surface there). This is the known launch-count problem; it is the same lever as the standing launch grind.
3. **Per-fit setup that repeats at any size**: quantile edge search (0.47 s on a cache miss), 39 separate GPU permutation-null calls (0.36 s), an explicit `gc.collect` (0.16 s), numba cache loading (0.3 s per process).
4. **Python orchestration**: about 820 000 small calls per fit (`isinstance` 91 000, `append` 67 000, `getattr` 25 000, two lambdas called 17 000 times each, `quantize_mi_tiebreak` 17 000 times), each cheap, 2.1 s together under `cProfile`.

## 5. What can be done, in the order of the expected saving at 10-100k rows

| id | change | expected saving | note |
|---|---|---|---|
| F-1 | the pair-combination table of the usability pool in float32 and on a smaller row subsample (it is already n-independent, but 40 ms per launch is a float64 radix select) | up to 0.4-0.6 s per fit | the project default dtype is float32; check where the f64 path is forced |
| F-2 | run the retention once instead of twice (`build_usability_candidate_pool` is called 2 times) and share the candidate codes | up to 0.5 s | needs a look at why two pools are built |
| F-3 | batch the 39 conditional permutation nulls into one launch set | up to 0.3 s | one batched kernel, the nulls are independent |
| F-4 | batch the 24 per-operand excess/floor evaluations of the raw-redundancy drop | up to 0.25 s | same pattern |
| F-5 | remove the explicit `gc.collect` from the pipeline (or run it once at the end) | 0.15 s | check why it is there |
| F-6 | cache the quantile edges per (data hash, nbins) across the passes of one fit | 0.4 s on a miss | the 100k run already hits the cache |
| F-7 | launch grind on the pair search and the usability pool (35 000 launches) | 0.3-0.6 s | continues the standing launch-count work |

Total of F-1 to F-6: about 1.9 s of the 5 s, without touching selection logic. For the target-mix pass this matters directly: a second pass is a whole fit, so each second saved in the fixed floor is saved per base.

## Update after the free-VRAM gate and gc fixes

* 1M-row bimodal spike (9.5 s vs 22 s) was the free-VRAM gate (`_gpu_has_free_vram`) reading the device free memory without the cupy pool's idle bytes: it flipped `fe_gpu_resident_codes_enabled()` off for the rest of the fit and the analytic noise gate ran on the CPU. Fixed; six consecutive 1M fits 9.0-10.0 s.
* `gc.collect` before discretisation now runs only when the frame is >= 1/512 of the free RAM: 10k fit 3.45 s -> 3.26 s (mix off).
* 10k fit after these (mix off): ~3.0 s. Flat profile: own Python 1.16 s (largest single function 0.11 s), cupy launches/syncs 1.08 s, numpy 0.58 s, numba 0.26 s, pandas 0.17 s. No function above 4%; the two stages that looked batchable (`_excess_and_floor` perm nulls, per-pair escalation proposers) have per-call varying conditioning supports (see the rejected-bench note in `_fe_raw_redundancy_helpers.py`) and are not batched.
* Under 2 s at 10k needs fewer launches overall (CUDA graphs or fused multi-stage kernels), not further per-stage trimming.

## Target-mix cost at 10k (fit_wall, 3 rounds, novel data)

* mix off: 2.94 s median (2.85-3.02)
* `fe_target_mix_max_bases=1`: 9.58 s median (rounds 5.24, 9.83, 9.58)
* `fe_target_mix_max_bases=2`: 10.63 s median (rounds 5.38, 10.63, 17.83)

The round-to-round spread (5 s to 18 s) is far above the mix-off spread, so these numbers come from a box with other GPU/CPU load (CI and nightly runs were in progress), not a quiet A/B. The direction is clear: the mix stage costs several seconds per fit at 10k, which dominates everything else on the table. Decision pending: default ON (as requested) is incompatible with a 1-2 s target unless the child fit is made much cheaper or the mix runs on a capped row subset with a cached scan.

## Where the target-mix child fit spends its time (10k parent, child on 5000 rows, cProfile)

* Child total 8.2 s under the profiler (about 2.7 s warm without it, the same as a full 5000-row fit).
* FE search: 4.4 s, of which pair scoring 3.4 s and the cascades 3.0 s (mid_a 1.3 s, conditional gate 0.8 s, early_a 0.7 s, mid_b 0.6 s).
* Support tail (`_assign_support`, residual rescue, drop sweep, friend graph): about 0.16 s each, so the lean flag already removed the expensive part of the tail.
* Conclusion: the child is the whole FE search on half the rows. Nothing is shared with the parent, because the target differs (T instead of y), so its pair scores, marginals and cascades must all be recomputed. A lossless cut does not exist; any cut (fewer cascades, fewer scan rows, a smaller pair budget) changes the candidate list and has to be judged on OOS error, not wall time alone.
* First fit in a process is 14.9 s against 2.7 s warm at 5000 rows (kernel compile and pool warm-up), so benchmarks must always discard a warm-up fit.

## Mix on, max_bases=1, 10k: reaching 5-6 s

* Rounds (fit_wall, 7): 5.50, 10.41, 10.06, 5.32, 5.42, 9.79, 5.30; median 5.50 s. Three of seven rounds sit at about 10 s, unexplained.
* Instrumented run (5 and 7 rounds, same settings): 4.9-5.8 s with no 10 s round. Split per round: mix stage 2.2-2.6 s, rest of the fit 3.0-3.3 s.
* The 10 s mode is not the VRAM gate (checked: vram_ok True, pool free 76-92 MB, every round) and not a stray process. Open item.
* Open quality question: max_bases=1 halves the mix cost against max_bases=2; the second base's OOS contribution has not been measured on the synthetic suite. Default left at 2 until that check is done.

## OOS check: fe_target_mix_max_bases 1 vs 2 (exp_target_mix_stage, 6 seeds, n_train=6000, 20000 test rows)

Paired ridge change against mix off (median, paired se in brackets); HGB in the same order.

| process | max_bases=1 ridge | max_bases=2 ridge | HGB mb1 / mb2 | fit time off / mb1 / mb2 (s) |
|---|---|---|---|---|
| masked_ratio | -15.8% (-1.8) | -25.2% (-4.6) | +0.5% / +0.7% | 3.8 / 8.5 / 12.3 |
| derived_base | -29.6% (-1.2) | 0.0% (0.0) | 0.0% / 0.0% | 2.1 / 6.2 / 9.5 |
| masked_interaction | -18.6% (-2.9) | -36.6% (-3.6) | -0.1% / -4.7% (-1.7) | 3.0 / 9.8 / 13.0 |
| two_dominant_weak_pair | -23.1% (-3.1) | -20.2% (-3.2) | +0.2% / -4.2% (-1.0) | 4.1 / 9.7 / 14.3 |
| heteroscedastic | 0.0% | 0.0% | 0.0% / 0.0% | 3.4 / 6.8 / 7.4 |
| regime | 0.0% | +0.3% | +1.5% / +0.2% | 4.0 / 9.8 / 20.2 |
| noise_control | 0.0% | 0.0% | 0.0% / 0.0% | 0.8 / 2.7 / 3.4 |
| monotone_control | 0.0% | +1.1% (+0.1) | 0.0% / +0.1% | 2.2 / 6.4 / 13.6 |

Reading:
* The two settings disagree on which processes benefit: max_bases=2 wins on masked_ratio and masked_interaction, max_bases=1 wins on derived_base and monotone_control. With 6 seeds the disagreements are within the noise, and the mean ridge change is about the same (roughly -11% for 1, -10% for 2 across the eight processes, by arithmetic mean of the rows).
* Neither setting harms the control processes (noise_control, heteroscedastic): zero extra columns, zero change.
* max_bases=2 costs about 1.5x the mix time (fit 12-14 s against 8-10 s on this loaded box).

Decision recorded: keep max_bases=1 as the cheaper setting with the same accuracy on this suite. The default is still 2 in code; changing it is a user decision, not made here.

## Rejected: batching or skipping the operand marginal MI (`_operand_marginal_mi`)

* Candidate: `_operand_marginal_mi` calls `mi_direct` per operand (48 calls, 0.49 s cumulative at 10k). Idea: drop the permutation test (`npermutations=0`) and take the plug-in MI, which costs microseconds.
* Measured (6 random columns, 10k rows, 10 bins): with `npermutations=10` the returned MI is 0.000 for all six (not significant at confidence 0.99); with `npermutations=0` it is the plug-in value 0.0007-0.0015. So the returned MI is the permutation-gated value, not the plug-in MI.
* Consequence: the baseline of the marginal-uplift gate changes, so selections change. Not a lossless change; rejected.
* A batched version has to reproduce the same gate per column (the batch noise gate also returns a gated MI). The seed stream differs from `mi_direct`, so borderline columns could flip. Needs a selection-parity check before it can ship. Left as an open item, not implemented.

## Attempted and reverted: batched operand marginal MI

* Implemented: one `batch_mi_with_noise_gate` call (through `_dispatch_batch_mi_with_noise_gate`) over all identity operands, filling the marginal cache before the pair loop. The batch kernel is documented bit-identical to the per-column `mi_direct` loop on this path.
* Parity: selected features identical in both modes on seeds 50 and 51 (10k rows).
* Timing: no gain. Precompute 0.117 s over 4 calls (2 fits); whole fit 3.41 s batched against 3.42 s with the lazy path. The per-operand loop was cheap on this workload, and the 0.49 s in the cProfile run was profiler inflation on operands that the cProfile run hit more often.
* Reverted, since no speedup and the code is not simpler (rule: skip an optimisation with no measured gain).

* Same batched marginal MI at 1M rows (seed 50): 10.39 s batched against 10.36 s lazy, identical selection. No gain at 1M either, so the change stays rejected.

## The 10 s mode in the target-mix stage: data-dependent, not an infrastructure fault

* 10 rounds, 10k rows, max_bases=1 (seeds 10-19): mix stage 2.2-2.6 s in the fast rounds and 6.4-15.5 s in the slow ones. Rest of the fit 3.0-3.4 s in every round.
* The VRAM gate never switched off in any round (`false 0`, minimum free VRAM 2.2-2.7 GB). The earlier 1M explanation does not apply here.
* The slow rounds are exactly the rounds in which the child fit returns engineered candidates (2-4 recipes); the fast rounds return none. The mix stage then grows with the candidates.
* Child profile (round 7): the FE steps take 13.2 s against 4.4 s in a fast round. The added time sits in `_run_fe_step` (second step over the pool enlarged by step-one admissions), not in the residual rescue or the support tail.
* Gate and cache growth: module-level caches grow by about 20 entries per round (`_RADIX_INTERP_CACHE`, `_FINITE`, `_CARD_MAX_CACHE`); this is small and does not explain the mode.
* So the mix cost is a property of the data: when the child finds structure, the second FE step runs over a bigger pool. A cap on the child's FE steps is the candidate fix; it changes the candidate list, so it is judged on OOS (below).

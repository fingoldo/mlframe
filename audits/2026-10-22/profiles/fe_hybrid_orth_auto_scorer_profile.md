# fe_hybrid_orth_auto_scorer_enable: profile (CPU, n=20000, p=35)

Measured 2026-10-28 with `D:/Temp/auto_prof.py` (`hybrid_orth_mi_auto_scorer_fe` on a standalone frame, njit warm) and `D:/Temp/auto_scorer_times.py` (one column, one call per scorer).

Whole call: 6.35 s. Most of the call is in `_bootstrap_lcb_tables`, which runs the per-column scorers on a thread pool (`ThreadPoolExecutor`, 5 bootstrap resamples). cProfile attributes that time to the pool wait, not to the scorers, so the per-scorer cost below comes from timing each scorer on one column:

| scorer | ms per column per call |
|---|---:|
| plug-in MI (batched elsewhere, bench CPX12) | 1.49 |
| KSG (`_ksg_mi_batch`, per column) | 7.11 |
| tail dependence (`_score_tail_dep`, 50 permutations, upper and lower) | 6.63 |
| HSIC (`_score_hsic`, n_sample 500) | 5.90 |
| copula MI | 3.99 |
| Chatterjee xi | 4.35 |
| distance correlation (n_sample 500) | 2.99 |

Reading: KSG, tail dependence, HSIC, xi and copula are called per column (one scalar call each), so the cost is the per-call overhead plus the kNN and permutation work repeated for every column and every bootstrap. KSG (`_ksg_mi_batch` takes a column matrix) and copula already accept a batch; the bench CPX12 batched only plug-in and copula. The next step is to batch KSG across all columns of a bootstrap in one call, then tail dependence and HSIC, keeping per-column results identical (the selection must not move). No change made yet.

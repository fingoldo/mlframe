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

## Attempt: batched KSG across columns (reverted, no wall gain)

Batching KSG over all columns of one bootstrap sample (`ksg_mi_discrete_target_columns`, each column drawing the same Kraskov noise it draws alone) gave a bit-identical output frame (`D:/Temp/ksg_batch_equiv.py`, 20000 x 39 frame equal). The batched KSG took 0.24 s and 0.43 s for the two table calls, against about 3.7 s of per-column KSG work in the old path, but the whole call did not get faster: 6.02 s before and 6.12 s after. The KSG share was already running on the thread pool in parallel, so the serial batch gave back the time it saved. The bottleneck is the sum of the other scorers inside the threaded tables. Change reverted; kept here so the next attempt starts from this result.

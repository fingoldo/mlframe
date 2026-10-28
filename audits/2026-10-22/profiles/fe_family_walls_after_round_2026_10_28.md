# FE family walls after this round of changes (all FE flags on, n=20000)

Re-measured 2026-10-28 with `D:/Temp/fe_modes_all.py` after the changes in this round (KSG batch in the ensemble, dCor batch in the ensemble, grouped-quantile sorted quartiles, conditional-routing correlation without the standardised copy). Same frame and method as `fe_family_walls_gpu_vs_cpu_n20000.md`; the second fit of each process, warm caches, default device mode on GPU and `MLFRAME_DISABLE_GPU=1` on CPU.

Whole fit: GPU 63.5 s (earlier pair 74.8 s, the second fit of each process), CPU 117.7 s (earlier 152.3 s). The change in the whole fit is within run-to-run noise of a single pair; the per-family numbers below are the signal.

| family | GPU now | GPU before | CPU now | CPU before |
|---|---:|---:|---:|---:|
| `fe_hybrid_orth_auto_scorer_enable` | 5.60 | 6.21 | 5.58 | 6.40 |
| `fe_composite_group_agg_enable` (exempt) | 4.81 | 5.57 | 4.86 | 5.70 |
| `fe_conditional_gate_enable` | 4.24 | 4.68 | 6.73 | 7.74 |
| `fe_binned_numeric_agg_enable` | 3.18 | 3.77 | 3.23 | 17.65 (first run) |
| `fe_grouped_quantile_enable` | 2.00 | 2.77 | 1.97 | 2.78 |
| `fe_group_distance_enable` | 1.63 | 1.86 | 1.65 | 1.88 |
| `fe_hybrid_orth_conditional_routing_enable` | 1.60 | 2.10 | 1.60 | 2.06 |
| `fe_hybrid_orth_triplet_enable` | 1.55 | 1.70 | 1.74 | 2.01 |
| `fe_hybrid_orth_ensemble_enable` | (below top 10) | 2.03 | 1.27 | 2.09 |

Notes:
- The CPU `fe_binned_numeric_agg_enable` of 17.65 s in the earlier table is a first-run number (cold kernel cache); this re-run gives 3.23 s on CPU as well. Treat the earlier 4.7x GPU speedup for this family as unreliable; the CPU and GPU walls are now about the same.
- `fe_hybrid_orth_ensemble_enable` drops from 2.09 s to 1.27 s on CPU (batched KSG and dCor, commits 86e1f1b5b and d09f6fe41), and leaves the GPU top ten.
- The conditional-routing and grouped-quantile gains are the commits 325a2b1d9 and 8a29718bb.
- The auto scorer and group distance are unchanged: their profiles (`fe_hybrid_orth_auto_scorer_profile.md`, `fe_group_distance_cpu_profile.md`) say where the cost is.

# FE family walls, GPU versus CPU, all FE flags on (n=20000)

Measured 2026-10-28 with `D:/Temp/fe_modes_all.py` (same frame as `fe_family_walls_cpu_n20000.md`, second fit in the process so the njit caches are warm). Two runs: default mode (GPU on, GTX 1050 Ti 4 GB) and `MLFRAME_DISABLE_GPU=1`. Whole fit: 121 s (rep 0) and 74 s (rep 1) on GPU; 167 s and 152 s on CPU. Kernel tuning cache was cold on this machine for 8 kernels, so the timings are the built-in defaults; treat the absolute numbers as indicative and the GPU/CPU ratios as the signal.

Heaviest families (wall seconds, rep 1 of each mode):

| family | GPU | CPU | GPU path measurable | contract status before this entry |
|---|---:|---:|---|---|
| `fe_hybrid_orth_auto_scorer_enable` | 6.21 | 6.40 | no (same on both) | pending |
| `fe_composite_group_agg_enable` | 5.57 | 5.70 | no (same on both) | exempt (blockwise CPU path, 73cz) |
| `fe_conditional_gate_enable` | 4.68 | 7.74 | yes (1.7x) | pending |
| `fe_binned_numeric_agg_enable` | 3.77 | 17.65 | yes (4.7x) | pending |
| `fe_grouped_quantile_enable` | 2.77 | 2.78 | no (same on both) | pending |
| `fe_hybrid_orth_conditional_routing_enable` | 2.10 | 2.06 | no | pending |
| `fe_hybrid_orth_ensemble_enable` | 2.03 | 2.09 | no | pending |
| `fe_group_distance_enable` | 1.86 | 1.88 | no | pending |
| `fe_hybrid_orth_triplet_enable` (`triplet`) | 1.70 | 2.01 | small | pending |

Reading: `fe_binned_numeric_agg_enable` is the largest CPU cost in the fit (17.65 s of 152 s on CPU) and already has a 4.7x GPU speedup; its CPU path is the one to tune first on hosts without a GPU. `fe_hybrid_orth_auto_scorer_enable` and `fe_grouped_quantile_enable` cost the same on both modes, so no GPU path reaches them: they are the GPU-coverage gap at this shape. `fe_composite_group_agg_enable` costs the same on both modes, which confirms its exemption (CPU blockwise path). Caveat: the GPU run is a single pair of fits on a cold kernel-tuning cache; repeat after `mlframe-tune-kernels ensure` before quoting a ratio.

Next: profile `fe_hybrid_orth_auto_scorer_enable` (largest GPU-uncovered family), then `fe_grouped_quantile_enable`, then `fe_binned_numeric_agg_enable` CPU path.

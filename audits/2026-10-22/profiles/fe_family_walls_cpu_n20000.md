# FE family wall times, CPU mode (MLFRAME_DISABLE_GPU=1)

Measured 2026-10-27 with `D:/Temp/fe_modes_all.py` (all FE flags on, synthetic frame n=20000 rows, 35 raw columns incl. 4 integer and 1 group column, second fit in the process so the njit caches are warm; whole fit 141 s). Wall seconds per `fe_family_timer` region.

69 regions sum to 50.4 s (36 percent of the fit). 23 regions of at least 0.5 s hold 43.8 s; the other 46 regions hold 6.6 s together.

Use: a family under 0.5 s costs under 0.4 percent of this fit, so its contract points 1 to 4 (device kernel, streaming bound, KTC dispatch) have no measurable payoff at this shape and it can be exempted with this table as the evidence; a family that grows past 0.5 s at another shape (larger n, more columns) leaves the exempt list and is profiled. The remaining 64 percent of the fit is the core pipeline (screening, the maxT pair-gate null, the honest MDLP null, LightGBM scorers), see audit rows 73cj.

| region | wall s | share of fit |
|---|---:|---:|
| `gate:fe_conditional_gate_enable` | 7.16 | 5.1% |
| `gate:fe_hybrid_orth_auto_scorer_enable` | 5.47 | 3.9% |
| `gate:fe_composite_group_agg_enable` | 5.02 | 3.6% |
| `gate:fe_binned_numeric_agg_enable` | 3.14 | 2.2% |
| `gate:fe_grouped_quantile_enable` | 2.43 | 1.7% |
| `gate:fe_hybrid_orth_triplet_enable` | 1.75 | 1.2% |
| `gate:fe_hybrid_orth_ensemble_enable` | 1.75 | 1.2% |
| `gate:fe_hybrid_orth_conditional_routing_enable` | 1.74 | 1.2% |
| `triplet` | 1.74 | 1.2% |
| `gate:fe_group_distance_enable` | 1.63 | 1.2% |
| `gate:fe_univariate_fourier_enable` | 1.27 | 0.9% |
| `gate:fe_cat_pair_enable` | 1.16 | 0.8% |
| `gate:fe_pairwise_modular_enable` | 1.06 | 0.8% |
| `gate:fe_hybrid_orth_three_gate_enable` | 1.06 | 0.8% |
| `gate:fe_mi_greedy_cmi_enable` | 1.00 | 0.7% |
| `gate:fe_hybrid_orth_quadruplet_enable` | 0.98 | 0.7% |
| `quadruplet` | 0.97 | 0.7% |
| `gate:fe_modular_enable` | 0.90 | 0.6% |
| `gate:fe_hybrid_orth_meta_enable` | 0.85 | 0.6% |
| `gate:fe_lof_enable` | 0.81 | 0.6% |
| `gate:fe_hybrid_orth_adaptive_degree_enable` | 0.75 | 0.5% |
| `gate:fe_hybrid_orth_cmim_enable` | 0.67 | 0.5% |
| `gate:fe_row_stat_enable` | 0.52 | 0.4% |
| `gate:fe_row_argmax_enable` | 0.46 | 0.3% |
| `gate:fe_ordinal_pattern_enable` | 0.41 | 0.3% |
| `gate:fe_hybrid_orth_hsic_enable` | 0.38 | 0.3% |
| `gate:fe_cat_triple_enable` | 0.38 | 0.3% |
| `gate:fe_hybrid_orth_adaptive_arity_enable` | 0.33 | 0.2% |
| `adaptive_arity` | 0.33 | 0.2% |
| `gate:fe_hybrid_orth_copula_enable` | 0.31 | 0.2% |
| `gate:fe_hybrid_orth_jmim_enable` | 0.31 | 0.2% |
| `gate:fe_univariate_basis_enable` | 0.30 | 0.2% |
| `orth_pair` | 0.30 | 0.2% |
| `gate:fe_conditional_quantile_rank_enable` | 0.28 | 0.2% |
| `gate:fe_grouped_agg_enable` | 0.27 | 0.2% |
| `gate:fe_hybrid_orth_ksg_enable` | 0.27 | 0.2% |
| `gate:fe_oof_warp_enable` | 0.23 | 0.2% |
| `gate:fe_numeric_decompose_enable` | 0.21 | 0.1% |
| `gate:fe_hybrid_orth_tc_enable` | 0.19 | 0.1% |
| `gate:fe_hybrid_orth_dcor_enable` | 0.18 | 0.1% |
| `gate:fe_random_fourier_enable` | 0.17 | 0.1% |
| `gate:fe_conditional_dispersion_enable` | 0.17 | 0.1% |
| `pair_maxt_null_floor` | 0.17 | 0.1% |
| `gate:fe_hybrid_orth_bootstrap_enable` | 0.14 | 0.1% |
| `gate:fe_mi_greedy_enable` | 0.13 | 0.1% |
| `gate:fe_hinge_enable` | 0.11 | 0.1% |
| `gate:fe_wavelet_enable` | 0.11 | 0.1% |
| `gate:fe_conditional_residual_enable` | 0.11 | 0.1% |
| `pairwise_mi_batch_precompute` | 0.06 | 0.0% |
| `gate:fe_rankgauss_enable` | 0.06 | 0.0% |
| `gate:fe_sir_direction_enable` | 0.06 | 0.0% |
| `gate:fe_offset_product_enable` | 0.05 | 0.0% |
| `gate:fe_integer_lattice_enable` | 0.04 | 0.0% |
| `gate:fe_mahalanobis_density_enable` | 0.04 | 0.0% |
| `gate:fe_hybrid_orth_diff_basis_enable` | 0.02 | 0.0% |
| `gate:fe_hybrid_orth_cluster_basis_enable` | 0.01 | 0.0% |
| `gate:fe_cat_num_interaction_enable` | 0.01 | 0.0% |
| `gate:fe_rare_category_enable` | 0.00 | 0.0% |
| `gate:fe_count_encoding_enable` | 0.00 | 0.0% |
| `gate:fe_kfold_te_enable` | 0.00 | 0.0% |
| `gate:fe_frequency_encoding_enable` | 0.00 | 0.0% |
| `gate:fe_discrete_structural_operators_enable` | 0.00 | 0.0% |
| `pairwise_mi_legacy_sweep` | 0.00 | 0.0% |
| `gate:fe_pairwise_ratio_enable` | 0.00 | 0.0% |
| `gate:fe_lagged_diff_enable` | 0.00 | 0.0% |
| `gate:fe_pairwise_log_ratio_enable` | 0.00 | 0.0% |
| `gate:fe_grouped_delta_enable` | 0.00 | 0.0% |
| `gate:fe_temporal_agg_enable` | 0.00 | 0.0% |
| `gate:fe_hybrid_orth_enable` | 0.00 | 0.0% |

# nbins_strategy: two criteria side by side (2026-11-23)

Truth recovery: `mrmr_binning_gt_sweep_2026-07-21.csv`, mean over five configs (rel2, rel4, rel8, rel16, wellbore50k). Model quality: `linear_oof` 20000 x 50, mean over two data seeds (0, 1), 3 folds, out of fold.

| Strategy | gt F1 | gt precision | gt FPR | AUC binary | R2 regression |
|---|---|---|---|---|---|
| mdlp (default) | 0.725 | 0.680 | 0.147 | 0.931 | 0.891 |
| mdlp_validated | 0.736 | 0.622 | 0.230 | 0.939 | 0.887 |
| mah | 0.732 | 0.654 | 0.150 | 0.950 | 0.889 |
| optimal_joint | 0.654 | 0.555 | 0.327 | 0.952 | 0.887 |
| freedman_diaconis | 0.590 | 0.553 | 0.291 | 0.951 | 0.885 |
| quantile_spacing | 0.641 | 0.575 | 0.303 | 0.951 | 0.888 |
| sturges | 0.670 | 0.556 | 0.342 | 0.950 | 0.886 |
| uniform | not in gt_sweep | - | - | 0.951 | 0.884 |

Mapping of gt_sweep labels: mdlp = fayyad_irani_insample_validated (the default), mdlp_validated = fayyad_irani_oos_validated, optimal_joint = optimal_joint_quantile, mah = mah_sci, sturges = sturges_quantile, freedman_diaconis = freedman_diaconis_quantile, quantile_spacing = qs_gupta.

The two criteria measure different things: gt counts selected raw columns against known roles; linear_oof scores a linear model on all selected features, raw or engineered.

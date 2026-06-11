# mlframe reporting chart gallery

Every chart / diagnostic in the mlframe reporting subsystem, rendered to PNG on synthetic
data chosen to make each chart meaningful. Regenerate with `python scripts/render_gallery.py`.

Total images: 40 across 19 categories.

## Contents

- [regression](#regression)
- [binary](#binary)
- [model_card](#model-card)
- [multiclass](#multiclass)
- [multilabel](#multilabel)
- [ltr](#ltr)
- [quantile](#quantile)
- [drift](#drift)
- [calibration_drift](#calibration-drift)
- [error_analysis](#error-analysis)
- [prediction_stability](#prediction-stability)
- [slice_finder](#slice-finder)
- [pdp_ice](#pdp-ice)
- [model_comparison](#model-comparison)
- [split_comparison](#split-comparison)
- [training_curve](#training-curve)
- [learning_curve](#learning-curve)
- [temporal](#temporal)
- [shap_panels](#shap-panels)

## regression

### regression_full

Pred-vs-actual scatter, residual hist, residual-vs-pred, error-by-decile, worm, residual ACF.

![regression_full](regression/regression_full.png)

### regression_hexbin_largen

Large-n (>50k) pred-vs-actual drawn as a log-density 2-D histogram instead of a point cloud.

![regression_hexbin_largen](regression/regression_hexbin_largen.png)

## binary

### binary_full

ROC, PR, score distribution, KS, threshold sweep, gain, PIT.

![binary_full](binary/binary_full.png)

### panel_emphasis_imbalanced

Data-aware panel emphasis on a rare-event target (base rate ~0.03): the adaptive order leads with PR + threshold-sweep and drops the optimistic-under-imbalance ROC, so the operator sees the diagnostics that actually matter under skew first.

![panel_emphasis_imbalanced](binary/panel_emphasis_imbalanced.png)

### panel_emphasis_balanced

Data-aware panel emphasis on a balanced target (base rate ~0.5): the adaptive order leads with ROC, which is informative when the classes are even, ahead of PR / score-dist / KS / threshold.

![panel_emphasis_balanced](binary/panel_emphasis_balanced.png)

### decile_table

Credit-scoring decile gain/lift table: per-decile response / cumulative-gain / lift / cumulative-KS (top deciles highlighted, TOTAL row).

![decile_table](binary/decile_table.png)

### decision_curve

Decision-curve analysis: model net-benefit vs treat-all / treat-none policies.

![decision_curve](binary/decision_curve.png)

### calibration_reliability

Reliability diagram with Wilson CI bands + binning-free smoothed (isotonic) overlay + bootstrap 95% band (significant-fraction annotation) + standard & debiased ECE annotation + population histogram.

![calibration_reliability](binary/calibration_reliability.png)

## model_card

### model_card_binary

One-glance executive model card: headline metrics + GREEN traffic-light verdict + mini ROC / score-dist / gain sparklines.

![model_card_binary](model_card/model_card_binary.png)

### model_card_regression

Regression model card: RMSE/MAE/R2/bias headline + verdict + mini residual-vs-pred / residual-hist / pred-vs-actual sparklines.

![model_card_regression](model_card/model_card_regression.png)

## multiclass

### multiclass_full

Normalized confusion, confused pairs, per-class P/R/F1, per-class ROC (DeLong CI), reliability, prob dist, top-k.

![multiclass_full](multiclass/multiclass_full.png)

### confusion_margins

Confusion heatmap flanked by class-support margins: right bar = per-true-class support, top bar = per-predicted-class volume. On an imbalanced + majority-biased synthetic the dominant class's support bar towers over the minorities and its predicted-volume bar exceeds its support, revealing imbalance + over-prediction at a glance.

![confusion_margins](multiclass/confusion_margins.png)

### multiclass_largeK

Large-K (K=40): per-class ROC / PR / reliability overlays auto-switch to the 8 worst-by-AUC classes + a macro-average instead of 40 spaghetti curves.

![multiclass_largeK](multiclass/multiclass_largeK.png)

## multilabel

### multilabel_full

Per-label P/R/F1, reliability, co-occurrence, cardinality, Jaccard dist, threshold-sweep heatmap.

![multilabel_full](multilabel/multilabel_full.png)

## ltr

### ltr_full

NDCG@k, per-query NDCG dist, NDCG by query size, lift, MRR dist, score-by-relevance, top-1 by query size.

![ltr_full](ltr/ltr_full.png)

## quantile

### quantile_full

Reliability, coverage, pinball-by-alpha, interval band, width dist, PIT, quantile reliability, pinball decomp, crossing, fan chart.

![quantile_full](quantile/quantile_full.png)

## drift

### psi_heatmap

Population Stability Index per feature per time bucket vs baseline (drifted features turn red).

![psi_heatmap](drift/psi_heatmap.png)

### residual_vs_time

Regression residual mean +/- std per time bin: bias drift + variance drift over time.

![residual_vs_time](drift/residual_vs_time.png)

### cusum_residual_drift

Two-sided tabular CUSUM of standardized residuals: a sustained mean shift trips the control limit (change-point marked).

![cusum_residual_drift](drift/cusum_residual_drift.png)

### metric_over_time

Rolling metric per time bucket with regime shading.

![metric_over_time](drift/metric_over_time.png)

### adversarial_validation

Train-vs-test LightGBM separability ROC + AUC + top drifting features.

![adversarial_validation](drift/adversarial_validation.png)

## calibration_drift

### calibration_drift

ECE-over-time line + small-multiple per-window reliability curves.

![calibration_drift](calibration_drift/calibration_drift.png)

## error_analysis

### weak_segment_heatmap

FreaAI-style weak-segment grid: mean error by feature slice (injected bad region shows as a hot cell).

![weak_segment_heatmap](error_analysis/weak_segment_heatmap.png)

### error_bias_per_feature

Evidently-style OVER/UNDER/MAJORITY feature-value distributions per feature.

![error_bias_per_feature](error_analysis/error_bias_per_feature.png)

### target_dist_overlay

Per-split overlaid density histograms of target and predictions (train p01/p99 envelope).

![target_dist_overlay](error_analysis/target_dist_overlay.png)

### segments_bar

Per-subgroup metric bars with a global-reference line (worst-first).

![segments_bar](error_analysis/segments_bar.png)

## prediction_stability

### prediction_stability

Ensemble member-disagreement: per-row spread histogram, spread-vs-mean scatter, uncertainty calibration (mean |error| rises with disagreement).

![prediction_stability](prediction_stability/prediction_stability.png)

## slice_finder

### slice_finder

Worst-K feature-value slices ranked by error-degradation x support.

![slice_finder](slice_finder/slice_finder.png)

## pdp_ice

### pdp_ice

1-D PDP+ICE for the top features and a 2-D PDP interaction heatmap (small sklearn model).

![pdp_ice](pdp_ice/pdp_ice.png)

## model_comparison

### model_comparison

ROC overlay + leaderboard bars + between-model prediction-correlation heatmap (3 synthetic models).

![model_comparison](model_comparison/model_comparison.png)

## split_comparison

### split_comparison

Cross-split overfitting view for ONE model: grouped headline-metric bars per train/val/test + delta table with a RED traffic-light verdict. Synthetic memorizes train (AUC ~0.99) but barely beats chance on test (AUC ~0.70).

![split_comparison](split_comparison/split_comparison.png)

## training_curve

### training_curve

Train/val metric vs iteration with the early-stopping marker + post-ES shading.

![training_curve](training_curve/training_curve.png)

## learning_curve

### learning_curve

Holdout score vs increasing train size (cheap sklearn estimator on synthetic).

![learning_curve](learning_curve/learning_curve.png)

## temporal

### target_acf_pacf

Target ACF + PACF by lag with Bartlett white-noise bounds (autocorrelated synthetic).

![target_acf_pacf](temporal/target_acf_pacf.png)

### target_temporal_audit

Target-rate-over-time audit: kept bins, sparse bins, segment means, change-points.

![target_temporal_audit](temporal/target_temporal_audit.png)

## shap_panels

### shap_shap_beeswarm

SHAP beeswarm + dependence plots for a small tree model.

![shap_shap_beeswarm](shap_panels/shap_shap_beeswarm.png)

### shap_shap_dependence_0_f0

SHAP beeswarm + dependence plots for a small tree model.

![shap_shap_dependence_0_f0](shap_panels/shap_shap_dependence_0_f0.png)

### shap_shap_dependence_1_f1

SHAP beeswarm + dependence plots for a small tree model.

![shap_shap_dependence_1_f1](shap_panels/shap_shap_dependence_1_f1.png)

### shap_shap_dependence_2_f2

SHAP beeswarm + dependence plots for a small tree model.

![shap_shap_dependence_2_f2](shap_panels/shap_shap_dependence_2_f2.png)

### shap_shap_dependence_3_f4

SHAP beeswarm + dependence plots for a small tree model.

![shap_shap_dependence_3_f4](shap_panels/shap_shap_dependence_3_f4.png)

# Changelog

All notable user-facing changes to **mlframe** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is intentionally lean and user-focused. For the full engineering
record (per-commit kernel tuning, profiling numbers, audit notes) see the git
history.

## [Unreleased]

### Added

- Recency-weighting primitives (`mlframe.core.recency_weights`): parametric poly/exp/power weight vectors over ordered histories, generalizing the exponential `ewma`. Identity parameters reproduce uniform weights.
- Per-entity recency-weighted feature primitives (`mlframe.feature_engineering`): `per_group_recency_weighted_mean` (weighted mean / event-rate), `per_group_recency_weighted_mode` (weighted Parzen-density mode), and `per_group_behavioral_stability` (density-peak-height predictability score). Opt-in; identity params match the plain unweighted mean. Parallel (`prange`) KDE kernel dispatched by entity count.
- Nadaraya-Watson kernel regression / smoothing (`mlframe.feature_engineering.nadaraya_watson`): `nadaraya_watson_smooth` (flat signal, gaussian/epanechnikov/boxcar/tricube kernels, optional per-sample weight composable with recency weights) and `per_group_nadaraya_watson_smooth` (per-entity denoising). Parallel query kernel dispatched by query count.
- Robust location estimators (`mlframe.core.robust_location`): `robust_mean_mestimator` (redescending Meshalkin/Huber/Tukey M-estimator with MAD scale, prange-dispatched at large n), `geometric_median` (Weiszfeld spatial median), and `trimmed_mean` / `winsorized_mean` (classic tail-robust location). Robust aggregators for contaminated data.
- Proportion statistics (`mlframe.core.proportion_stats`): `wilson_interval`, `required_n_for_proportion`, `proportions_significantly_different`, `z_for_confidence` — confidence intervals and sample-size planning for probability/rate estimates.
- Spectral matrix seriation (`mlframe.core.matrix_seriation`): `spectral_seriation` / `seriate` reorder a similarity/correlation matrix (Fiedler or leading-singular-vector) to surface block structure — for readable correlation heatmaps and feature-block detection.
- Benchmark-relative & threshold regression metrics (`mlframe.metrics.regression`): `fast_epsilon_band_accuracy` (fraction within ±ε, the dunnhumby acceptance functional), `fast_rel_mae` / `fast_mrae` / `fast_percent_better` (error relative to an arbitrary benchmark prediction), and `fast_logcosh_loss` (overflow-safe smooth loss).
- Optimal decision-threshold search (`mlframe.metrics.classification.optimal_threshold`): given scores and binary labels, find the threshold maximizing `f1` / `balanced_accuracy` / `mcc` / `youden` / `accuracy` via one O(n log n) confusion-count sweep — a per-functional operating-point picker (the F1-optimal and BA-optimal cuts differ under class imbalance).
- Single-population targeting curves (`mlframe.metrics.classification`): `cumulative_gains_curve` (CAP/Lorenz), `lift_curve`, `gains_table` (decile marketing table), and `exploss` (exponential proper scoring rule) — for how-deep-to-target model evaluation, distinct from the per-group LTR lift in `metrics.ranking`.
- Weighted / Quadratic Weighted Kappa (`mlframe.metrics.classification.quadratic_weighted_kappa`, `weighted_kappa`): distance-weighted agreement for ordinal multiclass targets (bit-matches `sklearn.cohen_kappa_score(weights=)`).
- Set-similarity coefficients (`mlframe.core.set_similarity`): `jaccard`, `dice`, `overlap`, `braun_blanquet`, `ochiai`, `kulczynski`, `tversky` over two boolean masks or Python sets — for set/interval/cluster-pair targets.
- RMSPE (`mlframe.metrics.regression.fast_rmspe`): root-mean-square percentage error (the Rossmann metric), scale-free squared-relative error excluding zero targets.
- Optimal ordinal cutpoints (`mlframe.metrics.classification.optimal_ordinal_cutpoints`, `apply_cutpoints`): tune the thresholds that digitize a continuous prediction into ordinal grades to maximize QWK/accuracy (the CrowdFlower direct-functional-tuning technique).
- Binning-smoothing (`mlframe.core.binning`): `fit_bin_smoother` / `apply_bin_smoother` / `bin_smooth` replace each value by its bin mean/median/boundary representative on the original scale (Han & Kamber smoothing-by-binning) — a leakage-safe, rank-preserving quantizer/denoiser.
- Categorical co-occurrence SVD embedding (`mlframe.feature_engineering`): `cat_cooccurrence_svd_fit` / `apply_cat_cooccurrence_svd` / `cat_cooccurrence_svd_with_recipes` encode a categorical column by the leading singular vectors of its co-occurrence matrix with another categorical (Dyakonov's `code_factor`) — a target-free structural encoding.
- ACE feature-significance filter (`mlframe.feature_selection.ace_select`): Artificial Contrasts with Ensembles (Tuv et al. 2009) — a parametric t-test of each feature's importance against permuted-contrast importances over replicates, plus a masking-removal loop; complements Boruta's binomial hit-count test with a continuous-margin test.
- Ensemble blending primitives (`mlframe.models.ensembling`): `caruana_greedy_selection` (metric-direct greedy forward selection with replacement, optimizing the actual competition metric) and `rank_average_blend` (scale-invariant AUC-oriented rank blend) over a base-model prediction matrix.
- Random-forest proximity metric (`mlframe.models.rf_proximity`): `rf_proximity_matrix` (Breiman leaf-co-occurrence similarity from any `.apply`-capable forest), `proximity_to_distance`, and `rf_outlier_measure` (Breiman within-class proximity outlier score) — a learned metric for RF-based clustering / MDS / anomaly detection, distinct from the FE proximity-weighted aggregate.
- Graph -> tabular feature generation (`mlframe.feature_engineering`): `graph_neighbor_aggregate` (leakage-safe homophily / social-influence: aggregate a label or feature over a node's graph neighbours) and `graph_structural_features` (per-node degree / weighted strength / local clustering coefficient / triangle count), plus tabular-to-graph constructors `knn_graph_edges` (k-NN similarity graph on float columns) and `shared_attribute_edges` (affiliation graph from a categorical / group column), each with an optional `timestamp=` for directed past-only (leakage-safe temporal) edges. Lets graph structural + homophily features be built for ordinary float/categorical tables.
- Pairwise link-prediction features (`mlframe.feature_engineering.link_prediction_features`): batched common-neighbours / Jaccard / Adamic-Adar / resource-allocation / preferential-attachment for candidate node pairs (numba two-pointer intersection over CSR adjacency) — the tabular feature block for link prediction as supervised classification. Per-node importance (PageRank / HITS / centralities) remains networkx; only these batched pairwise scores are provided here.

## [0.9.0]

### Added

- `train_mlframe_models_suite` orchestration for end-to-end multi-model training, calibration, feature selection, and reporting.
- Target types beyond regression/binary: multiclass, multilabel, quantile regression, and Learning-to-Rank.
- Composite-target discovery: auto-detect dominant-base composite targets, fit a transform, and predict on the original y-scale, with honest holdout/out-of-fold de-biasing of the winner selection.
- Feature-selection suite: MRMR (with categorical-interaction FE), RFECV (stability selection, knockoffs, permutation/SHAP importance, checkpoint resume), BorutaShap, ShapProxiedFS, and hybrid selectors — all reachable and configurable from the suite.
- Feature-engineering `feature_engineering.transformer` subpackage: row-attention aggregates, orthogonal-polynomial basis families (Hermite/Legendre/Chebyshev/Laguerre), univariate/pair/adaptive-arity FE, and an `fe_auto` meta knob that collapses the many FE flags into one.
- Feature-handling stack: text-vs-categorical detection, TF-IDF/hashing text encoding, embedding providers (incl. HuggingFace), leakage-safe target encoders, and a two-tier feature cache.
- Reporting backends (matplotlib and plotly), calibration and multi-target diagnostic panels, and pre-training dummy-baseline reports.
- Pre-training baseline diagnostics: dominant-feature discovery, init_score baselines, and composite-target recommendations.
- Feature/label drift detection and reporting, with opt-in auto-action.
- Typed (Pydantic) training/reporting/feature configuration surfaces.
- Optional GPU acceleration (cupy/CUDA) for feature-selection MI and SHAP kernels, auto-dispatched by hardware.
- Python 3.14 support.

### Changed

- **BREAKING:** repackaged from a flat layout into `src/mlframe/` sub-packages; update imports accordingly.
- Polars-native fast paths for CatBoost, XGBoost, and HistGradientBoosting (no dict→string round-trip), with unified categorical handling and auto-skip of redundant encoding.
- Corrective mechanisms (honest holdout selection, stratified FE subsampling, cluster-medoid RFECV pre-reduction, calibrated permutation-null gates) now default ON; opt out to restore legacy behaviour.
- Composite-target ensemble default changed to `nnls_stack`; discovery sub-phases parallelised across cores.
- BorutaShap is now sklearn-clone-able (works inside `GridSearchCV`/`Pipeline`).
- Extensive numba/GPU kernel performance tuning across feature selection, feature engineering, calibration, and metrics, dispatched per-host via a kernel-tuning cache (see git history).

### Fixed

- Corrected statistical methodology in several info-theory selectors: conditional (stratified) permutation null, add-one p-value estimator, KSG neighbour-count convention, and noise-floor quantile sample size.
- Numerous Polars fast-path robustness fixes: nullable/Categorical handling, category drift across train/val/test splits, global string-cache leaks, and stale `cat_features` state.
- MLP defaults no longer strangle linear signal; stabilised early stopping, eval-set normalisation, and clone-safety.
- Matplotlib figure-leak and redundant-plot fixes reducing wall time on plot-heavy fits.
- Metadata pickle failures with duplicate mlframe installs, and artifact leakage to CWD when `data_dir=""`.
- Many production bug fixes surfaced by combinatorial/metamorphic fuzzing and multi-agent audits (see git history).

### Removed

- `Normalizer_l2` preprocessing helper.
- `ensure_installed` runtime dependency-installation from library modules.

[Unreleased]: https://github.com/fingoldo/mlframe/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/fingoldo/mlframe/releases/tag/v0.9.0

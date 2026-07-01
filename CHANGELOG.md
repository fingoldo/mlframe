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

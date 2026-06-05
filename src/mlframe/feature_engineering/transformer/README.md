# `feature_engineering/transformer/` reachability audit

This subpackage holds **103 single-pass FE modules** (RFF, row-attention variants, SMOTE flavours, anchor / band / boundary attention, OOF KNN-target-encoders, etc). Every module exposes a top-level `compute_*_features(...)` (or `compute_*_attention(...)`) function and is re-exported via `transformer/__init__.py`.

## Reachability from `train_mlframe_models_suite`

**Direct grep:** `mlframe.training.*` does NOT import `mlframe.feature_engineering.transformer` (any submodule), and does NOT call any of the `compute_*` functions re-exported by it. Verified by:

```
grep -rE "from mlframe\.feature_engineering\.transformer|compute_(rff|row_attention|positional_encoding)" mlframe/src/mlframe/training
# -> 0 hits
```

The training suite consumes only:

- `mlframe.feature_engineering.basic` — `create_date_features`, `run_pysr_fe`
- `mlframe.feature_engineering.bruteforce` — `run_pysr_feature_engineering`
- `mlframe.feature_engineering.pysr_operators` — `get_preset_kwargs`, `VALID_PRESETS`

Everything under `transformer/` is **research-only**: users invoke these functions manually inside a custom `FeaturesAndTargetsExtractor.add_features` override, then let the suite consume the augmented frame. None of them are wired into `apply_preprocessing_extensions` or the model-loop.

## Buckets

### Reachable from `train_mlframe_models_suite` (0 modules)

None.

### Re-exported via `mlframe.feature_engineering` top-level (3 helpers)

These three round-trip through the package root and are the most common to wire-in via an extractor:

| Function | Module | Notes |
| --- | --- | --- |
| `compute_rff_features` | `random_features.py` | Random Fourier Features (Rahimi-Recht 2007). |
| `compute_positional_encoding` | `random_features.py` | Sinusoidal PE for within-group ordinal positions. |
| `compute_row_attention` | `row_attention.py` | Multi-head softmax-weighted kNN-target-encoding with OOF discipline. |

### Research-only, callable via `from mlframe.feature_engineering.transformer import ...` (100 modules)

The rest is research-grade FE not on the default suite path. They split into:

- **Attention family** (anchor / band / boundary / inducing / performer / spectral / stacked / multi-temp / boosted / class-conditional / hard-row / quantile-band / prediction-band / residual / row / band-conditional / cbhr / etc) — ~30 modules.
- **SMOTE / synthetic-row family** (adasyn / borderline / cluster / density-weighted / bgm-clustered / multiscale / pseudo / pure-pos / smote-distance) — ~10 modules.
- **OOF / KNN target-encoders** (quantile_neighbours, nn_oof_target_mean, target_kmeans_codebook, target_quantile, trust_score_oof, baseline_disagreement variants, y_quintile_baseline_knn) — ~10 modules.
- **Density / divergence / topology** (density_ratio, bgmm_*, pairwise_kl_divergence, persistence_diagram, fca_closed_concepts, mdl_binning_pairwise, ks_shift, distributional_moments) — ~12 modules.
- **Curvature / geometry / projection** (geodesic_kgraph, local_curvature, local_intrinsic_dim, local_density_gradient, local_linear, nca_projection, lda_projection, per_class_spectral, per_column_rff, random_features) — ~10 modules.
- **Conformal / robustness** (conformal_coverage_failure, conformal_locally_adaptive, jackknife_endpoint_stability, robustness_budget) — 4 modules.
- **Misc / experimental** (autoencoder, aux_mlp, focal_lgb, boosting_leaf, decision_region_depth, tree_path_boolean, cross_feature_reconstruction, gradient_direction_agreement, residual_stratified_distance, sign_residual_baseline, bidir_residual_band, fisher_weighted_residual, multi_aux_ensemble, multi_threshold_ordinal, predictive_info_delta, quantile_spread_fan, signed_residual_band, variance_baseline, rf_proximity, mixup_boundary, cutmix, diffusion_noise, adversarial_flip, counterfactual_substitution, apriori_itemsets, ib_baseline_codes, local_classifier, local_lift, anomaly_score_features, baseline_surprise, anchor_attention, multiscale_rate, multiscale_smote, active_virtual, adaptive_bandwidth) — ~25 modules.

Per-module status / OOS validation is tracked in `RESULTS.md` / `VALIDATION_TODO.md` already in this directory; treat those as the per-algorithm source of truth.

## Candidates to wire into the default suite

Flagged for the operator's consideration (high information density, OOF-disciplined, modest extra cost per the per-module headers / RESULTS.md):

1. **`compute_row_attention`** (`row_attention.py`) — already re-exported at package root; the "transformer FE" flagship. Multi-head softmax-weighted kNN-TE with strict OOF on train + train-only key-bank on val/test. Most impactful single addition for tree models on medium-to-wide tabular.
2. **`compute_rff_features`** (`random_features.py`) — Random Fourier Features approximate RBF kernel-maps; cheap (one matmul + cos/sin), helps linear stacking and bumps tree non-linearity.
3. **`compute_anchor_attention`** / **`compute_class_conditional_anchor_attention`** — class-prior-aware attention; classification head benefits.
4. **`compute_quantile_neighbours`** — quantile-aware k-NN TE; pairs with quantile regression / conformal.
5. **`compute_target_kmeans_codebook`** — discrete codebook over target distribution; lightweight and good for stacking inputs.
6. **`compute_trust_score_oof`** — OOF trust-score per row; orthogonal signal for ensembles.
7. **`compute_boosting_leaf_features`** / **`compute_tree_path_boolean`** — tree-derived leaf indices / path booleans; classic gradient-boosting-on-top-of-leaves trick.

Honest framing: wiring any of these requires either (a) adding a new `PreprocessingExtensionsConfig` stage that runs them with the same OOF / train-only-key-bank discipline `apply_preprocessing_extensions` already enforces for sklearn steps, or (b) a documented extractor recipe. Option (a) needs a per-algorithm "is this safe under k-fold CV without leakage" sign-off; option (b) is zero-suite-change and the current shipping path.

### Opt-in adapter (`ShortlistTransformerAdapter`)

`transformer.ShortlistTransformerAdapter` wraps any `(X_train, y_train, X_query, splitter, ...)` shortlist transformer (`compute_rff_features`, `compute_class_distance_features`, `compute_local_lift_features`, `compute_bgmm_*`, RSD-kNN, ...) as a leakage-safe sklearn `fit`/`transform` estimator. It can be passed into the suite via `custom_pre_pipelines`:

```python
from mlframe.feature_engineering.transformer import ShortlistTransformerAdapter, compute_rff_features

adapter = ShortlistTransformerAdapter(compute_rff_features, needs_y=False, compute_kwargs={"n_features": 64})
models, meta = train_mlframe_models_suite(..., custom_pre_pipelines={"rff": adapter})
```

`fit(X, y)` stashes the train fold; `transform(X)` runs the wrapped function in Mode B (`X_query=X`) so the internal scaler / bandwidth / class banks are fit on the train fold ONLY and applied to train/val/test/predict consistently. **Research-only remains the DEFAULT** — the adapter is an explicit opt-in, not an auto-wire, because stacking already subsumes most of these blocks.

## Out-of-scope research notes

Per the module-level docstring of `transformer/__init__.py`: "the 'transformer' name is structural, not algorithmic — no learnable attention weights are involved." These are random-projection + softmax-weighted kNN-TE blocks, not gradient-trained attention layers; don't expect transformer-scale wins on small tabular.

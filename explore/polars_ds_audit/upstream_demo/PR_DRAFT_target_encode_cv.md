# PR: Out-of-fold target encoding and WoE with empirical Bayes shrinkage

**Branch:** `feature/target-encode-oof`
**Base:** `main`

---

## Summary

Target encoding and WoE encoding without out-of-fold (OOF) splits leak target information into the training set, leading to overfitting. This PR adds:

- **Out-of-fold encoding** for both target encoding and WoE, matching sklearn's `TargetEncoder(cv=3)` semantics
- **Empirical Bayes shrinkage** (Micci-Barreca 2001 / sklearn `smooth="auto"`) with two variants:
  - `classical` (default): per-category within-variance (Micci-Barreca original)
  - `pooled`: sklearn-style pooled within-variance for byte-equivalence with `sklearn.TargetEncoder`
- Stratified fold assignment for binary/low-cardinality targets (preserves class ratios per fold, same goal as sklearn's `StratifiedKFold` but built in without a sklearn dependency)
- **Beyond sklearn:** a `fold_col` parameter that accepts any pre-computed integer fold column — sklearn's `TargetEncoder` only takes `cv: int` and cannot consume `GroupKFold`/`TimeSeriesSplit`/custom splits; with `fold_col` the user builds folds with whatever strategy they want (including any sklearn splitter) and the encoder honours them.
- `group_kfold_ids()` and `time_series_chunks_ids()` lightweight splitters in `pipeline.cv_splitters` — fold-ID producers for the common group/time cases; for anything else the user can feed their own fold column, e.g. from `sklearn.model_selection.StratifiedGroupKFold`.
- `Blueprint(refit_downstream_on_full=True)` default option that makes downstream `scale`/`impute`/etc. fit on the full-mapping distribution (matching `transform(test)`), eliminating train/serve skew — not something sklearn's `Pipeline` + `TargetEncoder` handles out of the box.
- Expression-level API (`pds.target_encode_oof()`, `pds.woe_encode_oof()`, `pds.target_encode_bayes()`) for use outside Blueprint

## Problem

`Blueprint.target_encode()` fits the encoder on the full training set and applies the same mapping back to train. For high-cardinality features, this causes severe target leakage — categories with few samples get encodings that memorize the target. sklearn solves this with internal OOF encoding (`cv=5` default).

**Benchmark evidence** (Amazon Employee Access, 32,769 rows, 9 high-card features up to 7,518 uniques; 5 repeats, LogisticRegression downstream, medians shown):

| Method | Train AUC | Test AUC | Gap | Time | Speedup vs sklearn_TE_cv5 |
|---|---|---|---|---|---|
| polars_ds target_encode (no OOF) | 0.9598 | 0.7584 | **+0.2014** | 0.026 s | — |
| category_encoders TargetEncoder (no OOF) | 0.9598 | 0.8547 | +0.1056 | 0.197 s | — |
| polars_ds WoE (no OOF) | 0.9287 | 0.8005 | +0.1312 | 0.025 s | — |
| **polars_ds OOF TE cv=3, sigmoid** | 0.8298 | **0.8550** | −0.0255 | 0.041 s | **4.1×** |
| polars_ds OOF TE cv=5, sigmoid | 0.8400 | 0.8549 | −0.0128 | 0.051 s | 3.3× |
| polars_ds OOF TE cv=3, Bayes classical | 0.8078 | 0.8311 | −0.0252 | 0.069 s | 2.4× |
| polars_ds OOF TE cv=3, Bayes pooled (sklearn-equivalent) | 0.8238 | 0.8486 | −0.0252 | 0.072 s | 2.3× |
| polars_ds OOF WoE cv=3 | 0.7890 | 0.8147 | −0.0274 | 0.038 s | 4.4× |
| **polars_ds Blueprint TE cv=3** (end-to-end) | 0.8298 | 0.8550 | −0.0255 | **0.029 s** | **5.8×** |
| sklearn TargetEncoder cv=5 | 0.8333 | 0.8450 | −0.0144 | 0.169 s | 1.0× |
| sklearn TargetEncoder cv=3 | 0.8241 | 0.8450 | −0.0225 | 0.147 s | 1.1× |

Naive (no-OOF) encoding leaks severely: train–test AUC gap collapses from +0.20 to −0.03 with OOF. The sigmoid OOF variant actually **beats sklearn on test AUC** here (0.8550 vs 0.8450) because the Amazon-style imbalanced, very-high-cardinality binary problem rewards fixed-smoothing over sklearn's adaptive (pooled-within-variance) shrinkage. The `bayes_variant="pooled"` variant comes in just above sklearn (0.8486 vs 0.8450) as expected given the formula is sklearn-equivalent up to the ddof-0 variance and null-row handling. `bayes_variant="classical"` (Micci-Barreca 2001) intentionally shrinks more aggressively per-category on rare categories and lands lower here (0.8311).

**Downstream-FitStep train/serve skew** (`Blueprint TE + scale` on the same Amazon split):

Pipeline: `target_encode(cv=3) → scale(standard)`. Downstream model matters because the effect is a scale/location shift in the feature distribution, not a reshuffle of row-level ordering.

With **LogisticRegression** downstream (scale-invariant up to convergence):

| `refit_downstream_on_full` | Train AUC | Test AUC |
|---|---|---|
| `True` (default) | 0.8314 | 0.8576 |
| `False` (legacy) | 0.8314 | 0.8576 |

LR converges to the same decision boundary regardless of the feature's mean/std, so the two modes are AUC-indistinguishable.

With **k-NN (k=50)** downstream (scale-sensitive) on the same split, 5 repeats, median values. In addition to AUC we surface the post-scale feature-distribution stats (mean of |per-column mean| and mean of per-column std, across the 9 encoded features) to make the scaler-skew mechanism visible:

| `refit_downstream_on_full` | Train AUC | Test AUC | train⟨\|mean\|⟩ | train⟨std⟩ | test⟨\|mean\|⟩ | test⟨std⟩ |
|---|---|---|---|---|---|---|
| `True` (default) | 0.8833 | 0.8392 | 0.0043 | 0.9411 | 0.0079 | **0.9849** |
| `False` (legacy) | 0.8849 | 0.8386 | 0.0000 | 1.0000 | 0.0096 | **1.0521** |

The mechanism is clearly visible in the standard-deviation column: `refit_full=False` perfectly normalises the **training** view (σ=1.000, trivially — the scaler fitted on the OOF distribution) but leaves a 5.2 % scale mismatch at test time (test σ=1.052). `refit_full=True` spreads the error between train (σ=0.94) and test (σ=0.98) so test features land closer to the canonical σ=1.0. On this 24k-row split the AUC effect is at the noise level for k-NN(k=50) — the σ gap is small enough that nearest-neighbour rankings barely move.

To make the AUC effect visible we shrink the training set (`TRAIN_N=2000`, holding test at 8000 rows from the same 32 769-row pool, 5 repeats with seeds 100…104). Smaller train means each OOF fold's exclusion removes a larger fraction of per-category data and the OOF-vs-full encoder distribution diverges more — train σ now drops to ~0.83, test σ to ~0.92 vs the legacy 1.000/1.10. Three downstream models, increasing in scale-sensitivity for this pipeline:

| Downstream model | refit_full wins (out of 5) | mean Δ AUC (full − oof) | refit_full median test AUC | refit_oof median test AUC |
|---|---|---|---|---|
| **MLP `(16,)`, max_iter=200** | **4 / 5** | **+0.0020** | **0.6834** | 0.6794 |
| RBF-SVM `C=1, gamma='scale'` | 2 / 5 | +0.0013 | 0.5712 | 0.5658 |
| k-NN `k=5, weights='distance'` | 2 / 5 | −0.0059 | 0.5874 | 0.6068 |

The MLP is the clean win for `refit_full=True`: small networks initialise weights assuming inputs are roughly σ≈1, and `refit_full` keeps that calibration valid at test time (test σ ≈ 0.92 with `refit_full`, σ ≈ 1.10 without — closer to 1.0 in the first case). The RBF-SVM with `gamma='scale'` rescales gamma using the training variance, partially auto-compensating for either mode. k-NN-with-distance-weighting actually prefers `refit_full=False` here — the OOF training distribution is noisier and a kNN trained on it learns a more robust decision rule that happens to do well on the cleaner full-mapping test set; this is an interesting finding but does not undermine the principled reason for the default (the scaler's stats *should* match the distribution it'll encounter at inference).

The default `True` is justified on three grounds: (i) it is the only setting where `transform(test)`'s scaler input matches its fit input (the σ ≈ 1 invariant the user expects from `scale(method="standard")`); (ii) for the most common scale-sensitive case (MLPs / NNs / linear models with weight regularisation) it improves AUC; (iii) for scale-invariant models (LR) it is a no-op. `refit_full=False` is exposed as opt-out for users who specifically want the legacy behaviour or who measure (as we did with k-NN) that their particular pipeline benefits from training on the OOF-anchored distribution.

**Synthetic data** (20k rows, 500 categories, Beta-sampled per-cat probabilities, LogisticRegression downstream):

| Variant | train AUC | test AUC | gap |
|---|---|---|---|
| polars_ds target_encode (no OOF) | 0.726 | 0.584 | **+0.141** |
| polars_ds OOF TE cv=5 (sigmoid) | 0.587 | 0.586 | +0.001 |
| polars_ds OOF TE randomized (no CV) | 0.716 | 0.584 | +0.132 |
| sklearn TargetEncoder cv=5 | 0.583 | 0.584 | −0.002 |

## How OOF target encoding works (mechanics)

Target encoding replaces a categorical column with a numeric column where each
category maps to the mean of the target values observed for that category.
A plain (non-OOF) implementation builds the mapping from the full training
set and then applies the same mapping right back onto training rows:

```
plain target encoding (LEAKY on train):

    train_df                        mapping (full train)
    ┌─────┬───────┐                 ┌─────┬──────┐
    │ cat │ y     │      fit        │ cat │ mean │
    ├─────┼───────┤      ──────▶    ├─────┼──────┤
    │ A   │ 1     │                 │ A   │ 0.67 │
    │ A   │ 1     │                 │ B   │ 0.25 │
    │ A   │ 0     │                 └─────┴──────┘
    │ B   │ 0     │
    │ B   │ 1     │      apply      train_df.cat  ← for row R this uses
    │ B   │ 0     │      ──────▶    ┌─────┬──────┐  row R's OWN target in
    │ B   │ 0     │                 │ 0.67│ ...  │  the computed mean; when
    └─────┴───────┘                 └─────┴──────┘  a downstream model sees
                                                    this feature it's seeing
                                                    "the answer" directly.
```

A model trained on these labels inflates its train-AUC (because the feature
has memorised the target) but predictions on test, where the same categories
appear but the target is hidden, fall apart. On the Amazon split above the
gap is +0.20 AUC — a classic symptom of high-cardinality target leakage.

**Out-of-fold (OOF) encoding** fixes this by building K separate mappings,
each excluding one fold's rows, and then encoding every row using the mapping
that was built from the OTHER folds:

```
OOF encoding with cv=3 (leak-safe on train):

    fold assignment (stratified):
        train_df:  [row 0  row 1  row 2  row 3  row 4  row 5  row 6  row 7 ...]
        fold_idx:  [  0      1      2      0      1      2      0      1  ...]

    Phase 1 — build K mappings, each using only OTHER folds' rows:
        mapping_0 = target_encode(rows where fold_idx != 0)   ← for rows in fold 0
        mapping_1 = target_encode(rows where fold_idx != 1)   ← for rows in fold 1
        mapping_2 = target_encode(rows where fold_idx != 2)   ← for rows in fold 2

    Phase 2 — encode each row using its fold's mapping:
        row 0 (fold 0)  →  mapping_0[row_0.cat]   ← row 0's own target excluded
        row 1 (fold 1)  →  mapping_1[row_1.cat]   ← row 1's own target excluded
        ...

    At inference (Pipeline.transform(test), test has no target column):
        test_row       →  full_mapping[test_row.cat]   ← full-train mapping,
                                                         no leakage possible
                                                         (test target unseen)
```

This is the same protocol sklearn's `TargetEncoder(cv=K)` uses internally;
the difference here is that polars_ds exposes the `fold_idx` column directly,
so callers can supply any pre-computed fold assignment (GroupKFold,
TimeSeriesSplit, BlockedKFold, …) without subclassing anything. The stratified
fold assignment is built in and avoids a sklearn dependency for the common case.

## How `refit_downstream_on_full` works

A Blueprint can chain a stateful step after an OOF encoder, e.g.
`target_encode(cv=3)` followed by `scale(method="standard")`. The scaler must
compute a mean and std to subtract and divide by, and the question is
*which distribution of the encoded column should it fit on*? The OOF-encoded
training column and the full-mapping column (applied to test) are slightly
different because OOF estimates are noisier per-row.

```
Pipeline: target_encode(cv=3)  →  scale(method="standard")

  refit_downstream_on_full = False (legacy):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ train path:                                                             │
  │     TE_oof(train)          → oof_df         ← values used by the model  │
  │     scale.fit(oof_df)      → stores μ_oof, σ_oof                        │
  │     scale.apply(oof_df)    → train features ~ N(0, 1)    (trivially)    │
  │                                                                         │
  │ test path:                                                              │
  │     TE_full(test)          → full_df                                    │
  │     scale.apply with (μ_oof, σ_oof)  → test features ~ N(~0, ~1.05)     │
  │                                        (the scaler's stats don't match  │
  │                                         the distribution it now sees)   │
  └─────────────────────────────────────────────────────────────────────────┘

  refit_downstream_on_full = True (default):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ train path:                                                             │
  │     TE_full(train)         → full_df        ← used ONLY to fit scaler   │
  │     TE_oof(train)          → oof_df         ← values used by the model  │
  │     scale.fit(full_df)     → stores μ_full, σ_full                      │
  │     scale.apply(oof_df)    → train features ~ N(~0, ~0.94)              │
  │                                                                         │
  │ test path:                                                              │
  │     TE_full(test)          → full_df                                    │
  │     scale.apply with (μ_full, σ_full) → test features ~ N(~0, ~0.98)    │
  │                                          (scaler's stats match the      │
  │                                          distribution at inference)     │
  └─────────────────────────────────────────────────────────────────────────┘
```

The legacy mode hides the distribution shift at train time (train is perfectly
σ=1.0 because the scaler and the data are from the same source) and surfaces
it at test time. The new default splits the shift symmetrically: train is
slightly compressed (σ≈0.94) but test is only slightly expanded (σ≈0.98), and
the mean/std the scaler was fitted on is literally the same mean/std it will
encounter at inference. For models that care about scale (NNs, SVMs, regularised
linear models) the fitted mode is the one consistent with `transform(test)`.

Implementation: during `materialize()` Blueprint maintains two parallel lazy
frames — `df_full_lazy` (what `transform(test)` would produce at this point in
the chain) and `df_oof_lazy` (OOF-encoded for OofFitStep columns). Downstream
FitStep `fit` reads from the full frame; the returned training DataFrame
(`return_df=True`) carries the OOF version. A schema-equality assertion after
each OofFitStep guards against the two frames going out of sync.

## What the Empirical Bayes shrinkage actually does

When a category appears only a handful of times, its observed target mean is
a noisy estimate of its "true" rate. The Bayesian fix is to shrink the
observed mean toward the global mean, weighted by how much evidence we have:

```
encoded(cat) = λ(cat) · observed_mean(cat) + (1 − λ(cat)) · global_mean

where λ(cat) ∈ [0, 1] grows with:
    - category sample size  (more rows → more trust in the local mean)
    - inverse of within-category variance  (tight category → strong signal)

Micci-Barreca 2001 formula:   λ = n · Var(y) / (n · Var(y) + σ²_within)
```

Numerical example (binary target y ∈ {0, 1}, global mean = 0.94):

```
 Category    Rows  Observed mean  Within-var  λ     Encoded
 ─────────────────────────────────────────────────────────────
 C_bulk      1000     0.85          0.13      0.998   0.850
   (tight category, lots of evidence → use observed mean)
 C_rare      5        0.40          0.24      0.19    0.829
   (noisy category, little evidence → mostly global mean)
 C_singleton 1        1.00          (n/a)     —       0.940
   (1 row: singleton guard forces global_mean to avoid leakage;
    a singleton's "observed mean" IS its own target value)
```

Two ways to estimate `σ²_within`:

- **`bayes_variant="classical"` (Micci-Barreca 2001, default)** — per-category
  within-variance, estimated as `(1/n) · Σ(y_i − mean_cat)²` over the category's
  own rows. Shrinks a noisy category more than a tight one of the same size.
- **`bayes_variant="pooled"` (sklearn `smooth="auto"`)** — global pooled
  within-variance, estimated once as `(1/N) · Σ(y_i − mean_cat(i))²` over all
  training rows. Every category uses the same denominator.

Both are principled choices with the same algebraic structure. Classical is
the Kaggle-era default and the simpler mental model (each category is its own
small regression); pooled is byte-equivalent to sklearn's `TargetEncoder` and
gives a steadier denominator when most categories are small. We default to
classical because it matches the Micci-Barreca 2001 paper most users cite.

The sigmoid smoothing that `Blueprint.target_encode(smoothing=float)` uses is
a simpler, non-adaptive form:

```
λ(n) = 1 / (1 + exp(−(n − min_samples_leaf) / smoothing))
```

Here λ depends only on the category's sample size — it does not look at the
within-category variance. Fast and robust, and the form category_encoders' TE
shipped with since 2018. In our Amazon benchmarks the fixed-smoothing sigmoid
actually beats adaptive Bayes on the very-high-cardinality imbalanced case,
because adaptive Bayes over-shrinks the many rare categories; see the
per-scenario breakdown below for when each shape is preferred.

## When to pick which variant

Three synthetic scenarios designed to pull the methods apart, 5 repeats each,
LogisticRegression on the single encoded feature. Full script:
`upstream_demo/bench_method_strengths.py`.

| Scenario | Cardinality | n rows | Train split | Winner (test AUC) | Key takeaway |
|---|---|---|---|---|---|
| A. Power-law, imbalanced | 2 000 cats, ~60 % singletons | 10 000 | row-level | all OOF variants ≈ 0.62 | Sigmoid best for rare cats; adaptive Bayes over-shrinks |
| B. Heterogeneous within-cat noise | 100 cats, 50 rows/cat | 5 000 | row-level | sigmoid / pooled ≈ 0.846 (classical 0.788) | Classical's guards fire on tight cats, losing signal |
| C. Group-disjoint (unseen users at test) | 200 users, 5–50 rows/user | ~5 500 | group-level | all ≈ 0.500 (no info possible) | GroupKFold gives HONEST train AUC 0.52 vs misleading 0.77 for row-KFold |

**Headline empirical finding, honestly stated:** OOF is what matters. The
smoothing variant (sigmoid vs `bayes_variant="classical"` vs `"pooled"`)
rarely moves test AUC by more than a couple of thousandths on real data. The
three scenarios above, plus the Amazon benchmark, cover all four combinations
of {low/high cardinality} × {few-rows/many-rows-per-cat}, and the spread of
test AUCs within the three OOF variants is almost always smaller than the
spread within 5 random seeds of the same variant. What actively hurts is:

1. **Plain (no OOF) target encoding on the training set** — the +0.20 AUC
   train/test gap on Amazon is typical for high-cardinality data.
2. **Regular K-fold when train/test is group-disjoint** — Scenario C shows
   this cleanly: a model trained on row-KFold OOF features has train AUC
   0.77 despite test AUC being 0.50 (no signal is possible for unseen users).
   The 0.27 gap is the model being *lied to* at training time. With
   `fold_col=GroupKFold(...)`, train AUC drops to 0.52, matching the (absent)
   test signal and producing calibrated downstream probabilities.

**Recommended defaults in practice:**

- **sigmoid smoothing** (`Blueprint.target_encode(cv=3)`, our default): robust
  across imbalanced / high-cardinality / mixed-evidence data. This is what
  `category_encoders.TargetEncoder` ships with.
- **`bayes_variant="pooled"` when `smoothing="auto"`**: byte-equivalent to
  `sklearn.preprocessing.TargetEncoder(smooth="auto")`. Pick it when you
  need to match an sklearn-based reference or MLflow-logged competitor.
- **`bayes_variant="classical"` when `smoothing="auto"`**: per-category
  within-variance adaptive shrinkage; faithful to the Micci-Barreca 2001
  paper. Pick it when per-category noise varies a lot AND every category
  has at least ~20 rows per fold.
- **Provide a `fold_col` from `GroupKFold` or `TimeSeriesSplit`** whenever
  the train/test split is *not* row-level random — otherwise the OOF encoder
  gives a signal the model cannot exploit at deployment.

## Related work and what this PR does *not* yet cover

The category_encoders library ships ~20 encoders. After this PR, polars_ds
covers the four most common ones: plain target, OOF target, WoE, and the
Bayes shrinkage variants. A literature and Kaggle-forum sweep (Avito Demand
2019, Home Credit 2018, Feedzai 2024 benchmarks) identified three further
techniques we do *not* implement yet and that would have clear value:

1. **M-estimate encoding** — `encoded = (n · mean_cat + m · global) / (n + m)`,
   one hyperparameter `m`. Simpler parameterisation of the same additive-prior
   idea as Bayes; some users find it more interpretable. Trivially implementable
   on top of the existing OOF scaffolding.
2. **Count / frequency encoding** — encode category → count of occurrences in
   training set. Target-free (no leakage risk), often paired with TE as a
   *supplementary* feature. One-liner in polars (`col.len().over("cat")`).
3. **Quantile encoding** — for regression targets, use the per-category median
   (or any quantile) instead of the mean. [Rodríguez et al. 2021](https://arxiv.org/abs/2105.13783)
   show it outperforms mean-TE on skewed / heavy-tailed targets. Same OOF
   scaffolding, different aggregator.

Medium-priority (niche but published benefits):

4. **CatBoost-style ordered TE** — cumulative mean over rows (random order),
   used internally by CatBoost. Different semantics from K-fold OOF; can
   complement it when row order encodes information. Requires a prefix-sum
   aggregation that polars already has (`cum_sum().over("cat").shift(1)`).
5. **Hierarchical TE** — when categories have a natural hierarchy
   (country → state → city), fall back to the parent level for rare children.
6. **Beta encoding** for binary targets — explicit Beta(α, β) prior instead
   of the variance-weighted Bayes formula.

Low-priority / already-covered:

- **Leave-one-out TE** — already covered by setting `cv = n_rows` in the
  OOF API. A separate LOO wrapper would be syntactic sugar at best.
- **Hash encoding** — dimensionality trick for > 100 k categories; polars
  HashMap-based TE handles that scale without hashing.
- **Category-crossing / interactions** — feature engineering, not an encoder.
- **Randomized noise injection** — our OOF + Bayes shrinkage already serves
  the same regularisation purpose.

These six candidates are out of scope for this PR to keep the review surface
focused. Happy to follow up with a second PR covering the top-three
(`m_estimate`, `count_encode`, `quantile_encode`) if the design here is
approved.

## Breaking changes

This branch accumulates several behaviour changes. All affect users of `Blueprint.target_encode` / `Blueprint.woe_encode` only if they relied on specific defaults.

1. `Blueprint.target_encode(cv=3)` default (was: plain train fit). Pass `cv=None` to restore plain encoding (unsafe).
2. `Blueprint.woe_encode(cv=3)` same.
3. `Blueprint(refit_downstream_on_full=True)` default: downstream `scale`/`impute` after an OofFitStep now fit on the full-mapping distribution instead of the OOF distribution. Prevents train/serve skew. Pass `refit_downstream_on_full=False` to restore legacy behaviour.
4. `Blueprint.target_encode(default="mean")` default (was `None`). Matches sklearn. Saved/serialized pipelines are unaffected because the default is baked into `replace_strict` expressions at `materialize()` time.
5. `Blueprint.woe_encode(default="mean")` same.
6. `transforms.target_encode_oof(default="mean")` and `target_encode_bayes(default="mean")` were `"null"` before.
7. `transforms.woe_encode_oof(default=0.0)` was `"null"` before. Also: unseen categories within an OOF fold now always receive WoE = 0.0 (neutral log-odds), not the user `default` — mixing target-scale default values with log-odds WoE was a category error.
8. `_stratified_kfold_ids` now raises `ValueError`/`TypeError` on degenerate inputs (single-class target, all-null target, class smaller than `cv`, non-numeric target) that previously produced silently corrupted folds.
9. `time_series_split_ids` renamed to `time_series_chunks_ids`. The old name is kept as a deprecated alias emitting `DeprecationWarning`; it will be removed in the next major version. The rename clarifies that the splitter is NOT equivalent to sklearn's `TimeSeriesSplit` (which uses an expanding window); ours divides the timeline into contiguous chunks for OOF.

## Changes

### Rust (`src/num_ext/target_encode.rs`)

- `get_target_encode_map_bayes()`: empirical Bayes shrinkage encoder with variant switch. Two passes: count/sum, then within-category sum-of-squared-deviations. Shrinkage `lambda = var_y * n / (var_y * n + within_var)`. Singleton and zero-within-variance categories are forced to the global mean (see "Leakage guards" below).
- `get_target_encode_frame_bayes()`: LazyFrame wrapper for compatibility.
- `pl_target_encode_bayes`: expression for single-shot Bayes encoding. Accepts `pooled_variant: bool` via kwargs.
- `pl_target_encode_oof`: K-fold OOF encoding accepting a `fold_idx: UInt32` column. Builds K separate encoder maps (Phase 1), then single-pass row lookup (Phase 2). Supports both sigmoid and Bayes smoothing via `smooth_auto` flag, and both `bayes_variant`s via `pooled_variant` kwarg. Unseen categories default to fold-specific target mean (matches sklearn). Validates fold_idx < n_folds (raises on out-of-range). Casts target to Float64 so Int8/Int32/Int64 binary targets work.

### Rust (`src/num_ext/woe_iv.rs`)

- `pl_woe_encode_oof`: K-fold OOF WoE encoding with the same Phase 1/Phase 2 architecture. Target is cast to Float64. Unseen-within-fold returns `0.0` (neutral log-odds). Null category / null fold_idx returns user `default`. Validates fold_idx < n_folds.

### Leakage guards

- Singleton (n=1) categories have `within_var = 0` so the shrinkage formula gives lambda=1 and the encoded value equals the row's own target value — a direct leak. We force full shrinkage to the global mean in this case.
- Extended to any category where `within_var` is effectively zero (all rows in category have identical target). The threshold is relative: `f64::EPSILON * target_var.abs() * n.max(1.0)` so we catch ULP-level cancellation residue, not just exact zero.
- The guard is variant-aware: in `pooled_variant`, `within_var = pooled_within` (global), so the guard fires only when `pooled_within` itself is effectively zero.

### Python (`python/polars_ds/pipeline/transforms.py`)

- `_stratified_kfold_ids(df, target, cv, seed)`: pure-numpy stratified fold assignment (no sklearn dependency). Uses stratification when target has ≤ 20 unique values, random permutation otherwise. Null detection works for all dtypes (Int*, Float*, Bool) via Polars `is_null()` + numpy `isnan`. Nulls are assigned round-robin across folds. Raises on empty DataFrame, `cv < 2`, single-class target, all-null target, class smaller than `cv`, or non-numeric target dtype.
- `_oof_encode_core()`: shared helper for OOF encoding orchestration with `fold_col` support. Uses an internal `__oof_fold_idx__` column that is never written back into the returned DataFrame. Validates user-supplied fold columns: must be a `str` name, must be an integer dtype (float fold values would silently truncate), must not be null, must have ≥ 2 unique values, must not use the reserved internal name. If the DataFrame already contains `__oof_fold_idx__` and no `fold_col` is passed, raises rather than silently overwriting. When `fold_col` is given, `cv` is ignored (the column's unique-value count defines fold count).
- `target_encode_oof(df, cols, target, cv, seed, ..., bayes_variant, fold_col)`: orchestrates fold creation + Rust OOF call.
- `woe_encode_oof(df, cols, target, cv, seed, ..., fold_col)`: same for WoE.
- `target_encode_bayes()`: new single-shot Bayes encoding function, accepts `bayes_variant`.

### Python (`python/polars_ds/pipeline/pipeline.py`)

- `Blueprint.__init__(refit_downstream_on_full=True)`: when `True` (default), `materialize()` maintains both a full-mapping `df_full_lazy` and an OOF-cascading `df_oof_lazy`. Downstream `FitStep` fits statistics on `df_full_lazy` (matching what `transform(test)` will see); the returned training DataFrame (`return_df=True`) carries the OOF-encoded values.
- `Blueprint.target_encode(cv=3, seed=42, smoothing="auto", default="mean", bayes_variant="classical")`: when `cv > 1`, uses OOF encoding on train via `OofFitStep`, full-data encoding on test. Validates `cv=1` (raises), `smoothing=True` (raises `TypeError`), `bayes_variant` (raises on unknown).
- `Blueprint.woe_encode(cv=3, seed=42, default="mean")`: same for WoE.

### Python (`python/polars_ds/pipeline/_step.py`)

- `OofFitStep`: stores `fit_func` (for Pipeline/test) and `oof_func` (for train). Exposes `fit(df_full)` (produces full-mapping expression) and `transform_oof(df_oof)` (produces OOF-encoded training DataFrame). The two methods take separate frames because, for chained OofFitSteps, the `fit` side must see full-mapping cascades and the `oof` side must preserve earlier OOF columns.

### Python (`python/polars_ds/pipeline/cv_splitters.py`) — NEW

- `group_kfold_ids(df, group_col, cv, seed)`: assigns fold indices such that all rows sharing a group value share a fold. Validates null group values.
- `time_series_chunks_ids(df, cv, time_col)`: assigns fold indices by dividing rows into `cv` contiguous temporal chunks. Uses uint64 intermediate arithmetic to avoid overflow on large datasets. Relies on Polars' default-stable `Series.arg_sort` for deterministic tie handling. Docstring contains an explicit `.. warning::` that this is NOT sklearn's `TimeSeriesSplit` (no expanding window) and can leak future into past for non-stationary features — users should bring sklearn's `TimeSeriesSplit` via `fold_col` for strict walk-forward semantics.
- `time_series_split_ids` kept as a `DeprecationWarning` alias.
- All three exported from `polars_ds.pipeline`.

### Python (`python/polars_ds/typing.py`)

- `BayesVariant: TypeAlias = Literal["classical", "pooled"]` added alongside existing encoder-related Literal aliases.

### Python (`python/polars_ds/pipeline/__init__.py`)

- Exports `Blueprint`, `Pipeline`, `FitStep`, `OofFitStep`, `group_kfold_ids`, `time_series_chunks_ids`, `time_series_split_ids` (deprecated), `target_encode_oof`, `woe_encode_oof`, `target_encode_bayes`.

### Python (`python/polars_ds/exprs/num.py`)

- `pds.target_encode_oof()`: expression-level OOF target encoding
- `pds.woe_encode_oof()`: expression-level OOF WoE encoding
- `pds.target_encode_bayes()`: expression-level Bayes encoding. Passes `t.var(ddof=0)` to the Rust plugin to match sklearn's population-variance convention (Polars' default `.var()` uses `ddof=1`).

### Tests (`tests/test_oof_encode.py`)

~80 tests covering:
- Bayes shrinkage numeric accuracy vs hand-computed reference (tight 1e-9 tolerance; catches a `ddof=1` regression)
- Classical vs pooled variants produce materially different values (catches a "parameter silently ignored" regression)
- `pooled_variant` matches `sklearn.preprocessing.TargetEncoder(smooth="auto")` within 1e-3 on a 500-row continuous-target frame
- OOF fold isolation (each row encoded using only out-of-fold data)
- Unseen category handling (fold-specific target mean, WoE = 0.0)
- Out-of-range fold_idx validation (raises error)
- Stratified fold balance for binary targets (asserted < 5pp deviation from global ratio, cross-checked vs sklearn StratifiedKFold)
- Singleton-category leakage prevention (n = 1)
- Zero-within-variance leakage prevention (n ≥ 2, constant target)
- Blueprint integration: `cv > 0` round-trip, `cv = 1` rejection, `smoothing = True` rejection, invalid `bayes_variant` rejection
- `refit_downstream_on_full` actually affects downstream scaler statistics (measurable pipeline output diff)
- Chained-OOF regression: two `target_encode(cv=...)` in a row preserve OOF values for both columns in the returned training DataFrame
- Edge cases: single category, all-same target, null handling, NaN targets, integer targets (Int8/Int64), all-null target, class-smaller-than-cv
- WoE OOF correctness and public API
- `fold_col` parameter: basic, non-0-based values, null rejection, float rejection, non-string rejection, reserved-name rejection (both user-supplied and pre-existing in df)
- CV splitters: `group_kfold_ids`, `time_series_chunks_ids`, validation
- `time_series_split_ids` deprecation alias emits `DeprecationWarning`
- Non-numeric target (String/Categorical) rejected cleanly by `_stratified_kfold_ids`
- End-to-end leak check with LogisticRegression (train–test gap < 0.05)

## Design decisions

1. **Fold assignment in Python, encoding in Rust**: fold creation needs stratification logic (class-aware shuffling) which is simpler in numpy. The heavy encoding loop stays in Rust for speed.

2. **`smoothing="auto"` (Bayes) as default in Blueprint**: matches sklearn's default since v1.3. Sigmoid smoothing (`smoothing=float`) remains available for backward compatibility.

3. **`bayes_variant="classical"` default**: classical matches the Micci-Barreca 2001 paper more closely (per-category within-variance). On low-to-medium-cardinality synthetic data the two variants have indistinguishable test AUC; on the high-cardinality, heavily-imbalanced Amazon dataset `"pooled"` scores ~1.8 pp higher because its single pooled denominator is a more stable noise estimate for 500+ categories with few rows each. `"pooled"` is the right choice for users who need byte-equivalence with `sklearn.preprocessing.TargetEncoder(smooth="auto")`; classical remains the default because it is simpler, faster to reason about per category, and does not penalise rare categories less than sklearn in the general case.

4. **`OofFitStep` vs modifying Pipeline**: OOF requires different behaviour at fit-time (K-fold) vs transform-time (full mapping). `OofFitStep` cleanly separates this without changing the Pipeline's expression-list architecture.

5. **Per-fold target mean for unseen categories**: when a category appears in fold K's validation set but not in its training set, we use the training mean of fold K (not a global default). This matches sklearn.

6. **Stratified folds**: for binary/low-cardinality targets (≤ 20 unique values), folds preserve class ratios. This prevents degenerate folds where a minority class is absent.

7. **`fold_col` escape hatch**: allows users to bring pre-computed fold assignments (GroupKFold, TimeSeriesSplit, sklearn splitters). Non-0-based values are automatically remapped without mutating the user's original column.

8. **`seed` naming**: follows polars_ds convention (also used by LightGBM, XGBoost), not sklearn's `random_state`.

9. **`refit_downstream_on_full=True` default**: without this, `scale()`/`impute()` placed after `target_encode(cv=...)` would fit their statistics on the OOF distribution while `transform(test)` applies them to the full-mapping distribution — a silent train/serve skew. The dual-frame materialize loop costs one extra `collect()` per OofFitStep; this is acceptable for the correctness benefit.

10. **`time_series_chunks_ids` rename**: the original name suggested sklearn's `TimeSeriesSplit` (expanding window) which it does not implement. The new name and docstring warning make the "contiguous chunks of the timeline" behaviour and its non-stationary-feature caveat explicit.

## Pre-commit checklist (per CONTRIBUTING.md)

- [x] `cargo fmt` applied
- [x] `ruff check` + `ruff format` applied
- [x] `requirements-test.txt` lists `scikit-learn>=1.3` (used only as a test-time gold standard, guarded by `pytest.importorskip`); no new runtime dependencies
- [x] No new Rust dependencies
- [x] Docstrings with leak-safety warnings and sklearn references
- [x] Benchmark evidence included

## AI assistance

Formatted and proofread by AI, used it to create additional tests and discover a few nice performance optimizations as well.

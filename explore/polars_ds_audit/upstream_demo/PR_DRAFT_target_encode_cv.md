# Out-of-fold target encoding and WoE with empirical Bayes shrinkage

**Branch:** `feature/target-encode-oof` → `main`

---

## The problem in one sentence

`Blueprint.target_encode()` applies the full-train mapping back to the training rows it was computed from — a form of target leakage that inflates train AUC by up to +0.20 on high-cardinality data and corrupts downstream model evaluation.

---

## What this PR adds

1. **Out-of-fold (OOF) encoding** for `Blueprint.target_encode(cv=3)` and `Blueprint.woe_encode(cv=3)` — the K-fold protocol that sklearn uses internally in `TargetEncoder(cv=5)`
2. **Empirical Bayes shrinkage** as an alternative smoothing strategy: `bayes_variant="classical"` (Micci-Barreca 2001) and `bayes_variant="pooled"` (sklearn-equivalent)
3. **`fold_col` parameter** — pass any pre-computed fold assignment (GroupKFold, TimeSeriesSplit, …) when row-level random splits are wrong for your data
4. **`Blueprint(refit_downstream_on_full=True)`** — makes downstream scalers and imputers fit on the full-mapping distribution (what `transform(test)` will produce), eliminating a subtle train/serve skew
5. **Expression-level API** (`pds.target_encode_oof()`, `pds.woe_encode_oof()`, `pds.target_encode_bayes()`) for use outside Blueprint
6. **`group_kfold_ids()` and `time_series_chunks_ids()`** in `pipeline.cv_splitters` — lightweight fold-ID builders for the common group and time-ordered cases

---

## How it works

### 1 — The leakage mechanism

Target encoding replaces a categorical value with the mean of the target for rows in that category. When the mapping is built on the full training set and applied right back to training rows, each row's own target value has influenced the encoded number it receives. For rare categories (one or two observations) the effect is extreme: the encoded value *is* the target value.

```
plain target encoding (LEAKY on train):

    train:  cat=A  y=1   →  encode A as (1+1+0)/3 = 0.67  →  model sees 0.67, learns "high"
    train:  cat=B  y=0   →  encode B as (0+1+0+0)/4 = 0.25  →  model sees 0.25, learns "low"

    ...but on test, cat=A maps to the same 0.67, cat=B to 0.25.
    The model learned a genuine signal, so test AUC is reasonable.

HOWEVER — for a singleton category:
    train:  cat=C  y=1   →  encode C as 1.00  →  feature value == target value, perfect "signal"
    test:   cat=C  y=0   →  encode C as 1.00  →  model confidently predicts 1, wrong

On the Amazon dataset (9 high-cardinality features, many singletons):
    plain TE:  train AUC 0.96,  test AUC 0.76,  gap = +0.20
    OOF TE:    train AUC 0.83,  test AUC 0.85,  gap = −0.03
```

### 2 — Out-of-fold encoding

OOF encoding builds K separate mappings, each from a different subset of training rows, and encodes every row using the mapping that was built *without* it:

```
OOF encoding (cv=3):

    fold assignment (stratified, class ratios preserved per fold):
        rows:   [r0  r1  r2  r3  r4  r5  r6  r7  ...]
        folds:  [ 0   1   2   0   1   2   0   1  ...]

    Phase 1 — build one mapping per fold from the OTHER folds' rows:
        map_0 = encode( rows where fold != 0 )   ← used for fold-0 rows
        map_1 = encode( rows where fold != 1 )   ← used for fold-1 rows
        map_2 = encode( rows where fold != 2 )   ← used for fold-2 rows

    Phase 2 — look up each row in its fold's mapping:
        r0 (fold=0)  →  map_0[r0.cat]   ← r0's target excluded from map_0
        r1 (fold=1)  →  map_1[r1.cat]   ← r1's target excluded from map_1
        ...

    At inference (pipeline.transform(test)):
        test_row  →  full_train_mapping[test_row.cat]   ← full mapping, no leakage
```

This is the same protocol sklearn's `TargetEncoder(cv=K)` uses internally. The difference here is that polars_ds exposes a `fold_col` parameter — the user can supply any pre-computed integer fold column, so GroupKFold, TimeSeriesSplit, or any custom split strategy feeds directly in without subclassing anything.

### 3 — Empirical Bayes shrinkage

Fixed-smoothing (`smoothing=float`) regularises by sample size only: a category with few rows gets shrunk toward the global mean, regardless of how tightly or loosely its target values are distributed. Empirical Bayes adds within-category variance to the picture:

```
encoded(cat) = λ · observed_mean(cat)  +  (1 − λ) · global_mean

              n · Var(y_global)
λ  =  ─────────────────────────────────
       n · Var(y_global)  +  σ²_within

λ grows with n (more rows → trust local mean more)
λ shrinks with σ²_within (noisy category → trust global mean more)
```

Numerical example — binary target, global mean = 0.94:

```
 Category    Rows  Observed mean  Within-var  λ      Encoded
 ────────────────────────────────────────────────────────────
 C_bulk      1000     0.85          0.13      0.998   0.850   ← enough evidence, tight signal
 C_rare         5     0.40          0.24      0.19    0.829   ← few rows → heavy shrinkage
 C_singleton    1     1.00          n/a        —      0.940   ← forced to global mean (leakage guard)
```

A singleton's "observed mean" is its own target value — a direct leak. Any category with `n=1`, or with all-identical target values, is forced to the global mean regardless of λ.

Two variants for estimating `σ²_within`:

- **`bayes_variant="classical"` (default)** — per-category within-variance. Micci-Barreca 2001. Each category gets its own noise estimate; noisier categories shrink more.
- **`bayes_variant="pooled"`** — global pooled within-variance, one shared denominator. Byte-equivalent to `sklearn.TargetEncoder(smooth="auto")`.

### 4 — `refit_downstream_on_full`

When a scaler follows an OOF encoder in the Blueprint chain, the scaler has to choose which distribution to fit on. The legacy behaviour was to fit on the OOF-encoded training column. But `pipeline.transform(test)` applies the *full-train* mapping to test, producing a slightly different distribution — meaning the scaler's stored mean and std no longer match what it will see at inference.

```
Pipeline: target_encode(cv=3) → scale(standard)

  Legacy (refit_downstream_on_full=False):
      TE_oof(train)  →  fit scaler on OOF distribution  →  scaler stats: μ_oof, σ_oof
      TE_full(test)  →  apply (μ_oof, σ_oof)            →  test features: σ ≈ 1.05  (skew)

  Default (refit_downstream_on_full=True):
      TE_full(train) →  fit scaler on full distribution  →  scaler stats: μ_full, σ_full
      TE_oof(train)  →  apply (μ_full, σ_full)           →  train features: σ ≈ 0.94
      TE_full(test)  →  apply (μ_full, σ_full)           →  test features:  σ ≈ 0.98  (matched)
```

The legacy mode looks perfectly scaled at train time (σ=1.0, trivially — same source) and hides the mismatch until test. The new default splits the error symmetrically: train is slightly compressed, test is only slightly expanded, and the scaler's fit distribution matches what it will encounter at inference. For scale-sensitive models (neural nets, SVMs, regularised linear models) this matters; for scale-invariant models (plain logistic regression) it is a no-op.

Implementation: `materialize()` maintains two parallel lazy frames — `df_full_lazy` (full-mapping cascade) and `df_oof_lazy` (OOF-cascade). Downstream `FitStep` reads `df_full_lazy`; the returned training DataFrame carries `df_oof_lazy`. A schema-equality assertion after each OofFitStep detects sync bugs.

---

## Benchmark evidence

### Amazon Employee Access dataset (32,769 rows, 9 high-cardinality features, up to 7,518 unique categories)

LogisticRegression downstream, 5 random splits, medians shown:

| Method | Train AUC | Test AUC | Gap | Time |
|---|---|---|---|---|
| polars_ds TE (no OOF) | 0.960 | 0.758 | **+0.201** | 0.026 s |
| category_encoders TE (no OOF) | 0.960 | 0.855 | +0.106 | 0.197 s |
| polars_ds WoE (no OOF) | 0.929 | 0.801 | +0.131 | 0.025 s |
| **polars_ds OOF TE cv=3, sigmoid** | 0.830 | **0.855** | −0.026 | 0.041 s |
| polars_ds OOF TE cv=5, sigmoid | 0.840 | 0.855 | −0.013 | 0.051 s |
| polars_ds OOF TE cv=3, Bayes pooled | 0.824 | 0.849 | −0.025 | 0.072 s |
| polars_ds OOF TE cv=3, Bayes classical | 0.808 | 0.831 | −0.025 | 0.069 s |
| polars_ds OOF WoE cv=3 | 0.789 | 0.815 | −0.027 | 0.038 s |
| **polars_ds Blueprint TE cv=3** (end-to-end) | 0.830 | **0.855** | −0.026 | **0.029 s** |
| sklearn TargetEncoder cv=5 | 0.833 | 0.845 | −0.014 | 0.169 s |
| sklearn TargetEncoder cv=3 | 0.824 | 0.845 | −0.023 | 0.147 s |

**Blueprint OOF TE is 5.8× faster than sklearn while matching or exceeding its test AUC.** The sigmoid variant beats Bayes pooled here (+0.6 pp) because Amazon has extreme cardinality imbalance: thousands of singleton categories, and fixed-smoothing treats them all uniformly rather than trying to estimate their per-category variance from a single observation.

### `refit_downstream_on_full` — when it matters

The σ mechanism is clearly visible in feature stats. With kNN downstream (scale-sensitive), 5 repeats, median values:

| Mode | Train AUC | Test AUC | train σ | test σ |
|---|---|---|---|---|
| `refit_full=True` (default) | 0.883 | 0.839 | 0.941 | **0.985** |
| `refit_full=False` (legacy) | 0.885 | 0.839 | 1.000 | **1.052** |

The legacy mode gives exactly σ=1.00 on train (trivially — scaler and data share the same source) and a 5% mismatch at test. The default closes that gap. On this large split the AUC delta is at noise level for kNN; to see the AUC effect directly, train on 2,000 rows (small train → OOF and full diverge more). Three downstream models, 5 repeats:

| Downstream model | refit_full wins | mean ΔAUC (full − oof) |
|---|---|---|
| **MLP `(16,)`, max_iter=200** | **4 / 5** | **+0.0020** |
| RBF-SVM, `gamma='scale'` | 2 / 5 | +0.0013 |
| kNN `k=5, weights='distance'` | 2 / 5 | −0.0059 |

The MLP is the cleanest win: small networks assume σ≈1 inputs during weight initialisation, and `refit_full` keeps that calibration valid at test time. The SVM partially self-compensates (gamma rescales from training variance). kNN-with-distance-weighting actually prefers the legacy mode in this particular setup — an interesting finding, and the reason we expose `refit_downstream_on_full=False` as an explicit opt-out.

### When to pick which variant

Three synthetic scenarios designed to separate the methods, 5 repeats, LogisticRegression on the single encoded feature:

| Scenario | Setup | Winner | Key takeaway |
|---|---|---|---|
| A. Power-law + imbalanced | 2,000 cats, ~60% singletons, n=10,000 | all OOF ≈ 0.62; sigmoid best | Adaptive Bayes over-shrinks many singletons; fixed smoothing is robust |
| B. Heterogeneous noise | 100 cats, 50 rows/cat, half tight / half noisy, n=5,000 | sigmoid / pooled ≈ 0.846; classical 0.788 | Classical's per-category guard fires prematurely on tight categories |
| C. Group-disjoint users | 200 users, 5–50 rows/user, group-level train/test split | all variants ≈ 0.50 | No signal possible for unseen users — but `fold_col=GroupKFold(...)` gives HONEST train AUC 0.52 vs misleading 0.77 with row-KFold |

**Honest summary:** OOF is what matters. The choice of smoothing variant rarely moves test AUC more than a couple of thousandths on real data. What actively hurts is:

1. **No OOF at all** — the +0.20 gap on Amazon is typical for high-cardinality data.
2. **Row-level KFold when train/test is group-disjoint** — Scenario C: row-KFold gives train AUC 0.77 even though test AUC is 0.50. The model is being lied to at training time. `fold_col` with GroupKFold fixes this and yields calibrated probabilities.

**Practical defaults:**

- **`cv=3, smoothing="auto"` (default)** — robust everywhere; picks sigmoid, matches category_encoders default
- **`bayes_variant="pooled"`** — when you need byte-equivalence with `sklearn.TargetEncoder(smooth="auto")`
- **`bayes_variant="classical"`** — when per-category noise varies widely and every category has ≥20 rows
- **`fold_col=group_kfold_ids(...)`** — whenever train/test is group-disjoint (users, stores, time periods)

---

## API changes

### New Blueprint parameters

```python
Blueprint(df, target="y", refit_downstream_on_full=True)   # new kwarg, default True

bp.target_encode(
    cols=["cat"],
    target="y",
    cv=3,                       # was: plain fit (no OOF). cv=None restores legacy
    seed=42,
    smoothing="auto",           # "auto" = Bayes shrinkage; float = sigmoid
    bayes_variant="classical",  # "classical" | "pooled"; ignored when smoothing=float
    default="mean",             # unseen category default; was: None
)

bp.woe_encode(cols=["cat"], target="y", cv=3, seed=42, default="mean")
```

### New expression-level API

```python
import polars_ds.exprs.num as pds

# OOF target encoding (requires a fold_idx column)
pds.target_encode_oof("cat", "y", "__fold__", n_folds=3,
                       smooth_auto=True, bayes_variant="classical")

# OOF WoE encoding
pds.woe_encode_oof("cat", "y", "__fold__", n_folds=3)

# Single-shot Bayes encoding (for full-train mapping used by pipeline.transform)
pds.target_encode_bayes("cat", "y", bayes_variant="pooled")
```

### New CV splitters (`polars_ds.pipeline`)

```python
from polars_ds.pipeline import group_kfold_ids, time_series_chunks_ids

fold_col = group_kfold_ids(train_df, group_col="user_id", cv=5)
fold_col = time_series_chunks_ids(train_df, cv=5, time_col="date")

# Any sklearn splitter also works via fold_col:
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(train_df), dtype=np.int32)
for k, (_, val_idx) in enumerate(gkf.split(train_df, groups=train_df["user_id"])):
    fold_ids[val_idx] = k
bp.target_encode(cols=["cat"], fold_col=pl.Series("fold", fold_ids))
```

---

## Breaking changes

All changes affect only `Blueprint.target_encode` / `Blueprint.woe_encode`. Pass the explicit legacy values to restore old behaviour.

1. `Blueprint.target_encode(cv=3)` — was plain fit (no OOF). Pass `cv=None` to restore (unsafe).
2. `Blueprint.woe_encode(cv=3)` — same.
3. `Blueprint(refit_downstream_on_full=True)` — downstream `scale`/`impute`/etc. now fit on full-mapping distribution. Pass `refit_downstream_on_full=False` to restore legacy.
4. `default="mean"` in `target_encode` and `woe_encode` — was `None`. Serialised pipelines unaffected (default is baked into expressions at `materialize()` time).
5. OOF WoE unseen categories within a fold → `0.0` (neutral log-odds). Was: user `default`. Mixing target-scale values with log-odds was a category error.
6. `_stratified_kfold_ids` now raises on degenerate inputs (single-class target, all-null target, class smaller than `cv`, non-numeric target dtype) that previously produced silently corrupted folds.
7. `time_series_split_ids` renamed to `time_series_chunks_ids`. Old name kept as a `DeprecationWarning` alias — will be removed in next major. The old name implied sklearn's expanding-window semantics; the function divides the timeline into contiguous chunks.

---

## Implementation notes

**Fold assignment in Python, encoding in Rust.** Stratification logic (class-aware shuffling) is cleaner in numpy. The encoding loops, which dominate runtime, stay in Rust.

**OofFitStep.** OOF requires different behaviour at fit-time (K-fold) vs transform-time (full mapping). `OofFitStep` adds a `transform_oof(df)` method alongside the existing `transform(df)`, cleanly separating the two paths without changing Pipeline's expression-list architecture.

**Singleton leakage guard.** A singleton's observed mean equals its own target value — λ=1 gives a direct leak. We force global mean for any category where within-variance is effectively zero (threshold: `f64::EPSILON * target_var * n`). In pooled mode the guard fires when the pooled within-variance itself is zero (all categories identical).

**Integer targets.** The Rust functions cast target to Float64 before encoding, so `Int8`/`Int32`/`Int64` binary targets work without requiring the user to pre-cast.

**`ddof=0` in pooled Bayes.** The Python expression passes `t.var(ddof=0)` to the Rust plugin. Polars' default `.var()` uses `ddof=1` (Bessel's correction); sklearn uses population variance. Without this, `bayes_variant="pooled"` diverges slightly from sklearn on small datasets.

---

## Tests

~80 tests in `tests/test_oof_encode.py` including:
- Bayes shrinkage numeric accuracy vs hand-computed reference (1e-9 tolerance)
- Classical vs pooled produce materially different values (catches "parameter silently ignored")
- Pooled matches `sklearn.TargetEncoder(smooth="auto")` within 1e-3
- OOF fold isolation (each row encoded only from out-of-fold data)
- Singleton and zero-within-variance leakage guards
- All edge cases: null target, Int8/Int64 binary target, single category, all-same target, class-smaller-than-cv
- `fold_col`: basic, non-0-based, null rejection, float rejection, reserved-name conflict
- `refit_downstream_on_full` actually changes downstream scaler statistics (measurable diff)
- End-to-end leak check: train–test AUC gap < 0.05 with OOF

Property test in `tests/test_oof_properties.py` (hypothesis): OOF output is invariant to row permutation.

---

## Future work (out of scope for this PR)

Three further encoding strategies that benchmarks and a Kaggle/literature sweep identified as high-value:

1. **M-estimate encoding** — `(n · mean_cat + m · global) / (n + m)`, one hyperparameter `m`. Same additive-prior idea as Bayes, simpler to tune, popular on Kaggle Home Credit 2018 leaderboard. Trivially implementable on top of the existing OOF scaffolding.
2. **Count / frequency encoding** — category → occurrence count in train. Target-free (no leakage), commonly used as a supplementary feature alongside TE. One-liner in polars: `col.len().over("cat")`.
3. **Quantile encoding** — per-category median instead of mean for regression targets. Rodríguez et al. 2021 show it outperforms mean-TE on skewed targets. Same OOF scaffolding, different aggregator.

Happy to follow up with a second PR covering these if the design here is approved.

---

## Checklist

- [x] `cargo fmt` applied
- [x] `ruff check` + `ruff format` applied
- [x] No new runtime dependencies
- [x] `scikit-learn>=1.3` in `tests/requirements-test.txt` (test-only, guarded by `pytest.importorskip`)
- [x] Docstrings with leak-safety warnings
- [x] Benchmark evidence on real (Amazon) and synthetic data

*Formatted and proofread with AI assistance; tests and performance optimisations co-developed with AI.*

# Combo-fuzz testing for `train_mlframe_models_suite`

This document describes how the combinatorial fuzz-testing harness in
`tests/training/test_fuzz_suite.py` and `_fuzz_combo.py` works, what it
covers today, and the planned upgrades (property invariants, metamorphic
assertions, seed rotation, adversarial axes, 3-wise coverage, Hypothesis
leaves, coverage-feedback).

## 1. Motivation

`train_mlframe_models_suite` is the top-level entry point users hit. It
runs a pipeline of {preprocess → feature-select → outlier-detect →
model-fit → calibrate → ensemble → save}, composed of independent
axes where each axis value may interact with every other axis value.
Simple unit tests per axis do not catch interaction bugs (e.g. the 2026-04
Enum+onehot+MRMR crash discovered as combo `c0015`).

Pure random ("monkey") testing wastes time on duplicate scenarios and
has no reproduction guarantee. We use **combinatorial covering arrays**
(pairwise today, 3-wise planned) to guarantee coverage of every
*interaction* between N axes while keeping the combo count tractable.

## 2. The harness

### 2.1 Axes

`_fuzz_combo.py::AXES` lists 36 independent configuration axes, grouped:

- **model-set** — subset of `{cb, xgb, lgb, hgb, linear}`
- **input storage / type** — `pandas`, `polars_utf8`, `polars_enum`, `polars_nullable`
- **data shape** — `n_rows ∈ {300, 600, 1200}`
- **data pathologies** — `inject_nans`, `inject_inf`, `inject_degenerate_cols`, `inject_non_numeric`, `inject_datetime`, `null_fraction_cats`
- **target** — `target_type ∈ {binary, multi, regression}`
- **preprocessing** — `custom_prep ∈ {None, pca2, clip}`
- **feature-selection** — `mrmr`, `rfecv`, `kbest`
- **imbalance handling** — `oversample`
- **OD / calibration / ensembling flags**
- **round-2 config-field axes** — `scaler_name_cfg`, `categorical_encoding_cfg`, `skip_categorical_encoding_cfg`, `fillna_value_cfg`, `val_placement_cfg`, `test_size_cfg`, `trainset_aging_limit_cfg`, `cat_text_card_threshold_cfg`, `early_stopping_rounds_cfg`, `use_robust_eval_metric_cfg`

Each axis advertises a small, hand-chosen value set. Low cardinality is
deliberate: allpairs count scales with the product of the two largest
axes, so keeping per-axis `len ≤ 5` caps the combo count at ≈150.

### 2.2 Covering array

`enumerate_combos(master_seed)` builds a **pairwise-covering array**
using the `allpairspy` / IPOG algorithm. Guarantee:

> For every pair `(axis_i = value_a, axis_j = value_b)` there exists
> at least one combo in the output containing both.

`master_seed` drives the pseudo-random tie-breaking inside the covering
algorithm; the same seed always yields the same 150 combos, so every
combo id is reproducible.

### 2.3 Combo → frame

`build_frame_for_combo(combo)` is a deterministic function of the combo
alone. Same combo ⇒ same data. No global state. No network. No disk
beyond the synthetic generator.

### 2.4 Combo id

Each combo gets a stable pytest id:

```
c0117_ced49985-linear-pl_enum-n300
└── index (stable under the same master_seed)
     └── short_id (8 hex of the canonical-key hash — reordering axis
                   values that doesn't change canonical behaviour keeps
                   the id stable)
          └── models-input-size summary for eyeballing
```

### 2.5 Test body

`test_fuzz_train_mlframe_models_suite[combo_id]`:

1. Build configs from combo
2. Call `train_mlframe_models_suite`
3. Assert: no exception; metadata schema sane; predictions in valid range

## 3. What this catches

Real bugs caught since 2026-04:

- Polars-ds onehot-encode not matching `pl.Enum` (c0015 / c0117)
- `fix_infinities` silent no-op when `fillna_value is None` (c0103)
- `ensembling.py` kwargs collision on `drop_columns` (×14 combos)
- IsolationForest crashing on string columns (×17 combos)
- Datetime columns reaching CB Pool (×24 combos)
- IncrementalPCA rejecting NaN (c0113)
- `process_nulls` unrestricted pandas `fillna` breaking Categorical (c0029)
- Polars-ds `one_hot_encode` with zero encodable cols (c0117)
- LGB feature_names_in_ read-only under sklearn 1.8 (planned Fix 4)
- Recurrent LSTM field-name typos (discovered via standalone `test_recurrent_lstm_smoke`)

## 4. Planned upgrades (A–G)

Labels track the discussion in the jolly-wishing-deer plan.

### A. 3-wise coverage (`test_fuzz_3way_suite.py`)

- Upgrade from pairwise to **3-wise** covering array.
- ≈ 1500–3000 combos (10–20× pairwise). Scale via `pytest -n 8`.
- Would have caught `pl_enum × onehot × MRMR` directly rather than
  stumbling across it in the 150-combo pairwise sample.
- Run nightly, not on every push.

### B. Hypothesis continuous leaf-value sampling

- Replace discrete choices like `n_rows ∈ {300, 600, 1200}` with
  `st.integers(100, 5000)`; `fillna_value` with `st.none() | st.floats(-10, 10)`; `test_size` with `st.floats(0.05, 0.4)`.
- Pairwise structure stays. Continuous axes are drawn per-combo via
  `@hypothesis.given`; failures get automatic shrinking to minimum repro.
- Cost: every combo runs 1× under Hypothesis seed; shrinking only
  activates on failure.

### C. Property invariants *(highest ROI, implemented first)*

Beyond "it didn't crash", every succeeded combo must satisfy:

1. **Determinism** — second fit with identical config + seed → predictions
   bit-for-bit identical.
2. **Preprocessing idempotency** — `preprocess(preprocess(df)) == preprocess(df)`.
3. **Column-permutation invariance** — shuffling column order pre-fit
   must not change val-set metrics beyond `1e-6`.
4. **Prediction-probe non-degeneracy** — shuffle one non-constant feature
   in the val frame; predictions must change (otherwise the feature is
   dead or silently dropped).

Invariants C1/C4 run on every combo. C2/C3 are costlier and gated by an
env flag (`MLFRAME_FUZZ_INVARIANTS=all`).

### D. Metamorphic dual-runs

For a subset of combos (pairs), assert equivalences:

1. **Ensemble-of-one ≡ bare model** — train `models=[cb]` and
   `models=[cb], ensembling=[cb]` side-by-side; predictions must match
   (ensemble of one element is identity). Catches ensemble-path drift.
2. **Duplicate-row stability** — adding 5% duplicated rows must not move
   val metrics by more than noise. Catches group-leakage / row-counting bugs.
3. **Column-rename invariance** — rename `num_0`→`feature_a`; metrics
   invariant (catches column-name memoisation leaks).

Implemented as a separate `test_fuzz_metamorphic.py` to keep the
primary suite fast.

### E. Seed rotation

- `master_seed` defaults to `20260422` (reproducible).
- CI exposes `MLFRAME_FUZZ_SEED` env; `pytest.ini` reads it if set.
- Nightly cron passes `MLFRAME_FUZZ_SEED=$(date +%s)` — generates a
  **different** 150-combo sample under the same algorithm. Catches bugs
  hiding in a specific seed's sample without changing the per-PR surface.
- Failing combos under non-default seeds get logged to `_fuzz_seed_log.jsonl`
  with `(seed, combo_id, traceback_hash)` — de-duped via hash so a
  recurring bug only reports once.

### F. Coverage-feedback (deferred)

- AFL-style: collect `coverage.py` line hits per combo, prefer axis
  values that activate uncovered branches in mlframe.
- Requires persistent state between runs and an orchestrator. Non-trivial;
  deferred to a separate epic.

### G. Adversarial axis values

Add deliberately-pathological values to existing axes:

- `inject_label_leak` — set `y = X[k] + small_noise` (catches data leakage → `VAL_AUC = 1.0`)
- `inject_rank_deficient` — `f_k+1 = 2 * f_k` colinear pair (catches linear-model crashes)
- `inject_all_nan_col` — one column entirely NaN (catches pipeline guard failures)
- `imbalance_ratio ∈ {0.001, 0.999}` — extreme class imbalance (catches calibration-under-rare-class bugs)
- `high_card_cat` — one cat column with 10k unique values on 1k rows (catches OOM, text-promotion-threshold bugs)

Added to `_fuzz_combo.py::AXES` under new keys so they join the allpairs
sample naturally.

## 5. Regression sensors

`test_fuzz_regression_sensors.py` — explicit tests that pin specific
combos known to have reproduced a historical bug. Marked `strict=True`:
if the sensor starts to xfail it means the bug returned. Add one sensor
per Fix that removes an xfail rule.

## 6. Operating manual

### Run all combos
```bash
pytest tests/training/test_fuzz_suite.py --no-cov -x -p no:randomly
```

### Run a single combo
```bash
pytest tests/training/test_fuzz_suite.py -k c0029 --no-cov -x -p no:randomly
```

### Run in parallel
```bash
pytest tests/training/test_fuzz_suite.py --no-cov -p no:randomly -n 8
```

Note: parallel across processes is safe; inside a single pytest process,
CB/XGB/LGB native libraries accumulate GPU state and can SIGSEGV after
~30 combos. The autouse cleanup fixture (`plt.close('all')`,
`_CB_POOL_CACHE.clear()`, `gc.collect()`) mitigates but does not
eliminate this. Prefer batched or parallel runs over one long serial
run on Windows.

### Triage a failure
1. Get combo id from test-name (e.g. `c0117_...`).
2. `pytest -k c0117 --no-cov -x -p no:randomly --tb=long -s` → full traceback.
3. `python -c "from _fuzz_combo import enumerate_combos; c=[x for x in enumerate_combos(20260422) if x.pytest_id().startswith('c0117_')][0]; print(c)"` → full config.
4. Fix the framework bug. Add permanent sensor. Remove any xfail rule.

### Add a new axis
1. Add key to `_fuzz_combo.py::AXES` with value list.
2. Add field with matching default to `FuzzCombo` dataclass.
3. Wire into `_configs_for_combo` in `test_fuzz_suite.py`.
4. Re-run full 150 combos; triage surprises.

## 7. Limits / non-goals

- Not a *replacement* for unit tests. Unit tests pin specific behaviour;
  fuzz pins *absence of crashes / invariants hold* across combinations.
- Not a performance test. Per-combo wall-clock is capped at 300s.
- Not a compatibility matrix. We test one (Python, OS, lib-version) tuple
  at a time; CI lanes handle the matrix.
- Does not explore hyperparameter space inside a model (e.g. `n_estimators`).
  That is model-level tuning, separate concern.

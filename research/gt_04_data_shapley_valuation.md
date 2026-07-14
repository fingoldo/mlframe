# gt_04: Data Shapley / KNN-Shapley — per-row data valuation for mlframe

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

Cooperative-game valuation along the ROW axis: training examples are players, v(S) = validation
performance of a model trained on subset S, and a row's Shapley value measures its contribution —
negative for mislabeled/harmful rows, near-zero for redundant ones, high for informative ones.
Uses: (a) label-noise detection and cleaning, (b) per-row `sample_weight` for training
(down-weight harmful rows instead of deleting), (c) data-acquisition prioritization.

References the implementer must follow:
- Ghorbani & Zou, ICML 2019, "Data Shapley: Equitable Valuation of Data for Machine Learning" —
  TMC-Shapley (Truncated Monte Carlo permutation sampling).
- Jia et al., VLDB 2019, "Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms"
  — **exact closed-form KNN-Shapley in O(n log n) per validation point**: for a KNN surrogate
  classifier the Shapley value of every training point has a recursive closed form (sort train
  points by distance to the val point; s_{(n)} = 1[y_{(n)}=y_val]/n; then
  s_{(i)} = s_{(i+1)} + (1[y_{(i)}=y_val] − 1[y_{(i+1)}=y_val])/K · min(K, i)/i, iterating from
  farthest to nearest). This is the recommended DEFAULT engine — exact, fast, no retraining.
- Wang & Jia 2023 (Data Banzhaf, see gt_03) — Banzhaf variant for noise robustness; share the
  permutation/mask sampling infrastructure with TMC.

## 2. Integration verdict

**NOT a feature selector — a new standalone component `src/mlframe/data_valuation/`** (greenfield;
no existing data-valuation code in the repo). The APPLICATION plumbing already exists end-to-end:
- Weight injection choke point: `_setup_sample_weight(sample_weight, train_idx, model_obj,
  fit_params)` at `src/mlframe/training/_data_helpers.py:200` — checks the model's fit signature,
  slices by train_idx, injects into fit_params. Any (n,) weight vector fed to the training config
  flows through here.
- Weight-scheme loop: `training/core/_phase_train_one_target_weight_iteration.py`
  (`_run_one_weight_iteration`, driven by `_phase_train_one_target_body.py`) — iterates named
  weight schemas (`weight_name`, `weight_values`); a data-valuation weight registers as one more
  schema.
- Existing sibling/template: `src/mlframe/feature_engineering/magnitude_sample_weight.py:16`
  (`magnitude_sample_weight(y_multi, norm="mean_abs", robust=False, winsor_quantile=0.99)`) —
  a function producing a sample_weight vector; mirror its shape/contract.

## 3. Package design

```
src/mlframe/data_valuation/
    __init__.py            # facade: knn_shapley, tmc_shapley, data_banzhaf, valuation_sample_weight
    _knn_shapley.py        # exact closed-form engine (default)
    _mc_sampling.py        # TMC-Shapley + MSR-Banzhaf (shared sampling loop, pluggable utility fn)
    _weights.py            # valuation -> sample_weight transforms
    _benchmarks/profile_data_valuation.py
```

### 3.1 `_knn_shapley.py`
```python
def knn_shapley(
    X_train, y_train, X_val, y_val, *,
    k: int = 5, metric: str = "euclidean", standardize: bool = True,
    n_jobs: int = -1, batch_val: int = 256,
) -> np.ndarray:
    """Exact KNN-Shapley values, shape (n_train,). Average of per-val-point closed-form values
    (Jia et al. 2019 recursion, stated in §1). Distances via sklearn.neighbors.NearestNeighbors
    (need the FULL sorted order per val point, not just top-k: use kneighbors with
    n_neighbors=n_train on batches, or argsort of pairwise chunks -- batch_val bounds memory to
    batch_val*n_train floats). standardize=True z-scores X on train stats (distance sanity).
    Classification only in v1 (the closed form is for classification agreement); regression:
    raise NotImplementedError with a message pointing to tmc_shapley."""
```
Numba note (repo ladder): the per-val-point recursion over sorted indices is a tight scalar loop —
implement as `@njit` kernel taking (sorted_label_match (n_train,) uint8, k) → (n_train,) values;
measure vs numpy; keep both per the dispatcher convention if the win is real.

### 3.2 `_mc_sampling.py`
```python
def tmc_shapley(utility_fn, n_rows, *, n_permutations=200, truncation_tol=1e-3,
                rng, n_jobs=1) -> tuple[np.ndarray, dict]
def data_banzhaf(utility_fn, n_rows, *, n_coalitions=2048, rng, n_jobs=1)
    -> tuple[np.ndarray, dict]
```
`utility_fn(idx_array) -> float` is caller-supplied (e.g. "fit xgboost on rows idx, return val
AUC"). TMC: per permutation walk prefixes, marginal = u(prefix∪{next})−u(prefix), truncate the
permutation when |u(prefix) − u(full)| < truncation_tol (the T in TMC). These engines are for
small/medium n (each marginal = one retrain) — document cost honestly: n_permutations × n
truncated retrains; provide a `subsample_rows` helper for applying to a stratified subsample and
propagating values by nearest-neighbor imputation to the rest.

### 3.3 `_weights.py`
```python
def valuation_sample_weight(values, *, mode="clip_negative",  # "clip_negative"|"rank"|"softmax"
                            floor=0.0, temperature=1.0) -> np.ndarray
```
`clip_negative`: w = max(values, 0) normalized to mean 1 (harmful rows → ~0, everyone else ~1) —
the recommended default: least aggressive, preserves sample-size semantics. `rank`/`softmax` for
experimentation. Output contract: (n,) float64, mean≈1, no NaN — verified by the function.

### 3.4 Training integration
A thin adapter producing a named weight schema: given a fitted valuation vector, register
`("data_shapley", weights)` into the weight-iteration inputs. Follow how existing named weights
enter `_run_one_weight_iteration` (read `_phase_train_one_target_body.py` to locate the schema
list construction; add via config, not hardcode). Config surface: this is a training-config
concern — add `data_valuation_weighting: bool = False` + kwargs dict to the relevant training
config (find the config class feeding `_phase_train_one_target_body`; keep the flag default OFF).
v1 may ship WITHOUT the config wiring (facade + engines + weights fn + tests), with integration as
a documented follow-up — decide by effort; the facade functions alone are already usable manually.

## 4. biz_val tests

File: `tests/data_valuation/test_biz_val_data_valuation.py` (new test dir mirrors the package).

1. `test_biz_val_knn_shapley_flags_label_noise` — synthetic 2-class blobs n=2000, flip 10% of
   train labels. Threshold: ≥90% of flipped rows have value below the CLEAN-row median; AUROC of
   (−value) as a noise detector ≥ 0.85. Runtime target <5s (closed form).
2. `test_biz_val_valuation_weights_improve_downstream_auc` — same bed: xgboost with
   `sample_weight=valuation_sample_weight(values)` vs unweighted, 30% holdout:
   AUC(weighted) ≥ AUC(unweighted) + 0.01 (measure first; floor 5-15% below measurement).
3. `test_biz_val_tmc_matches_knn_on_small_n` — n=200, logistic-regression utility_fn: Spearman
   between TMC (200 permutations) and KNN-Shapley values ≥ 0.6 (they're different games — the
   correlation just guards against sign/scale bugs; calibrate the floor on first run).
4. `test_biz_val_duplicate_rows_share_credit` — inject 5 exact copies of one high-value row:
   each copy's value ≈ original_solo_value/6 within 50% tolerance (KNN-Shapley dilution property).

Unit tests: closed-form recursion vs brute-force Shapley on n_train=8 (exact enumeration of the
KNN game — feasible, PROVES the formula implementation); values sum ≈ v(N)−v(∅) (efficiency);
weight-transform contracts (mean≈1, nonneg, NaN-free); njit-vs-numpy kernel bit-identity.

## 5. Acceptance criteria
- Package importable, mypy-clean; brute-force-vs-closed-form unit test green (scientific core).
- All 4 biz_val green locally (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`).
- cProfile harness committed: knn_shapley wall at n_train∈{2k, 20k, 100k} × n_val∈{200, 1000};
  numba-vs-numpy verdict recorded (keep both if win, per kernel ladder).
- README of the package (module docstring) states the classification-only v1 limitation and the
  TMC cost model explicitly.

## 6. Known risks / rejected alternatives
- Full-matrix distances at n_train=100k × n_val=1000 = 8·10¹⁰ bytes if materialized — the
  batch_val chunking is mandatory, and document the O(n_train·n_val) floor honestly.
- Influence functions / TracIn as alternatives: out of scope (gradient-based, model-specific);
  mention in module docstring as related work only.
- Do NOT wire valuation weights ON by default anywhere — weighting training data changes model
  behaviour globally; strictly opt-in with the biz_val evidence attached.

# gt_05: Shapley model weighting & ensemble pruning for votenrank / composite ensembles

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

Ensemble member weighting in mlframe today: Caruana hill-climb (frequency-of-selection weights,
`src/mlframe/votenrank/hill_climb.py:84`), NNLS OOF gate (`stacking_aware_gate`,
`src/mlframe/training/composite/ensemble/stacking.py:143`, drops members below `min_weight=0.05`),
constrained/geometric/adversarial blends (various `votenrank/*_blend`), and a leave-one-out
diversity ablation (`votenrank/correlation_diversity_ablation.py`). None of these gives a
principled per-model CONTRIBUTION accounting:
- Hill-climb weights confound "good" with "selected early"; duplicates of a strong model can crowd
  the selection.
- NNLS weights are regression coefficients — collinear (near-duplicate) models get arbitrary
  weight splits (one can get everything, its twin zero), so pruning by NNLS weight is unstable.
- Leave-one-out ablation misses redundancy: dropping ONE of two duplicates shows ~zero loss, so
  BOTH look useless, yet dropping both hurts.

The Shapley value fixes exactly these: models are players, v(C) = ensemble score of blending
coalition C on OOF predictions. Duplicates SHARE credit (symmetry axiom: two identical models get
equal values summing to their joint contribution); useless models get ~0 (dummy axiom); the
leave-one-out trap disappears because all coalition sizes are averaged. Pruning by Shapley≈0 is
therefore stable under redundancy, and the values themselves are defensible blend weights.

## 2. Integration verdict

Two seams, one new engine, NOT a feature selector:
- (a) **`src/mlframe/votenrank/shapley_blend.py`** — a new `*_blend`-family entry (prediction-level:
  takes OOF prediction matrix → weights + blended prediction), exported from
  `votenrank/__init__.py` like its siblings. Template neighbors: `hill_climb.py` (API shape:
  returns dict with `weights`, `ensemble_pred`, `score`, `selected`) and
  `correlation_diversity_ablation.py` (contribution-analysis reporting style).
- (b) **Shapley pruning variant beside `stacking_aware_gate`** in
  `training/composite/ensemble/stacking.py` — same call shape
  `(transform_predictions, y_train, min_weight)` → member weights with sub-threshold members
  dropped, so the composite pipeline can switch gates via config.

## 3. Design

### 3.1 Core engine (shared by both seams) — put in `votenrank/shapley_blend.py`
```python
def shapley_model_values(
    preds: np.ndarray,            # (n_models, n_rows) OOF predictions (proba or margin)
    y: np.ndarray, *,
    score_fn=None,                # (y, blended) -> float, higher better; default roc_auc for
                                  # binary y, negative RMSE otherwise (detect via y cardinality)
    coalition_blend: str = "mean",   # how a coalition C predicts: "mean" | "rank_mean"
    estimator: str = "permutation",  # "permutation" | "msr_banzhaf" (share gt_03 sampling ideas)
    n_permutations: int = 200, rng=None, n_jobs: int = 1,
) -> tuple[np.ndarray, dict]:
    """Shapley values (n_models,) of v(C) = score_fn(y, blend(preds[C])). v(∅) := score of the
    constant mean(y) prediction. Permutation estimator: for each of n_permutations random orders,
    walk prefixes accumulating marginals; INCREMENTAL MEAN trick: maintain running sum of
    coalition predictions so each marginal is O(n_rows), not O(|C|·n_rows). info: stderr per
    model, n_evals, v_full, v_empty."""
```
```python
def shapley_blend(preds, y, *, prune_below: float = 0.0, renormalize: bool = True, **kwargs)
    -> dict:
    """weights = clip(values, 0); prune members with value <= prune_below * values.sum();
    blended = weighted mean of survivors. Returns the hill_climb-compatible dict
    (weights, ensemble_pred, score, selected, values, info) so downstream consumers/tests reuse
    the same access pattern."""
```
Cost model (state in docstring): n_permutations × n_models marginal evaluations × O(n_rows)
score_fn. AUC is the expensive part (sort-based) — offer `score_subsample: int | None = 20000`
to score on a fixed row subsample for large n, with the standard caveat.

### 3.2 Stacking-gate variant — `training/composite/ensemble/stacking.py`
```python
def shapley_aware_gate(transform_predictions, y_train, min_weight: float = 0.05, **engine_kwargs):
    """Same contract as stacking_aware_gate (read it first and mirror the input/output types
    EXACTLY, including how members are named/keyed), but weights = normalized clipped Shapley
    values instead of NNLS coefficients. Import the engine from votenrank.shapley_blend (check
    for import-cycle risk: training.composite must not be imported by votenrank -- it isn't;
    votenrank is a leaf package -- verify with a grep before wiring)."""
```
Config exposure: wherever `stacking_aware_gate` is selected (grep call sites), add a
`gate_kind: str = "nnls"` switch with `"shapley"` option — read the call-site config plumbing
first; keep default `"nnls"` until benched.

## 4. biz_val tests

Template: `tests/votenrank/test_biz_val_hill_climb_ensemble.py` (read for fixture style/conventions).
New file: `tests/votenrank/test_biz_val_shapley_blend.py`.

Standard bed (build once as a module-level helper): n=4000 rows binary y; model pool of 7
synthetic predictors built from y + controlled noise: 2 STRONG (corr with y-margin ~0.6,
independent noise), 3 DUPLICATES of strong-1 (same predictions + tiny jitter eps~N(0,0.01)),
2 PURE NOISE predictors.

1. `test_biz_val_shapley_blend_noise_models_pruned` — noise models' normalized weights < 0.05
   and both excluded from `selected` at `prune_below=0.02`.
2. `test_biz_val_shapley_blend_duplicates_share_credit` — sum of values of {strong-1, dup-1..3}
   ≈ value strong-1 would get if duplicates were absent (run engine on the pool minus dups to get
   the reference), within 25% relative tolerance; AND each duplicate's value ≈ that sum/4 within
   50% (symmetry).
3. `test_biz_val_shapley_blend_score_competitive_with_hill_climb` — blended OOF score ≥
   hill_climb_ensemble score − 0.002 on the same pool (Shapley is an attribution, not an
   optimizer — near-parity is the bar, superiority is not claimed).
4. `test_biz_val_shapley_gate_prunes_where_nnls_flips` — stacking-gate seam: on the duplicate-heavy
   pool, run both gates across 4 seeds (jitter reshuffled): NNLS's kept-set Jaccard across seeds
   vs Shapley's — require Shapley ≥ NNLS + 0.1 (the instability of NNLS under collinearity is the
   motivating claim; measure first, floor below measurement).

Unit tests: dummy model (constant prediction) value ≈ 0; two identical models get equal values
(exact permutation symmetry within stderr); efficiency Σvalues ≈ v(full)−v(∅); incremental-mean
marginal path bit-identical to naive recompute on a tiny pool; hill_climb-compatible return dict
keys present.

## 5. Acceptance criteria
- Engine + both seams implemented, exported (`votenrank/__init__.py` import line added),
  mypy-clean; no import cycle (votenrank stays leaf).
- All biz_val + unit tests green locally (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`).
- cProfile harness `votenrank/_benchmarks/profile_shapley_blend.py`: wall at
  (n_models, n_rows) ∈ {(7, 4k), (20, 50k)} with and without score_subsample; committed.
- `gate_kind` default stays `"nnls"`; flip only after a bench across ≥3 real-ish pools shows
  majority-win on kept-set stability at non-inferior blend score (record verdict either way).

## 6. Known risks / rejected alternatives
- Weighted-blend-aware Shapley (v(C) uses OPTIMAL weights within C, e.g. NNLS per coalition):
  rejected for v1 — n_permutations × n_models NNLS solves; the mean-blend game is the standard,
  cheap, and sufficient for pruning/attribution. Note as future work.
- Using Shapley values directly as blend weights vs values-as-pruning + equal/NNLS weights on
  survivors: benchmark both compositions in the bench file (cheap add) — values are attribution,
  not optimal weights; the honest framing matters.
- n_models > ~30: permutation estimator cost grows linearly but stderr per model grows too;
  document n_permutations scaling guidance (≥10×n_models) in the docstring.

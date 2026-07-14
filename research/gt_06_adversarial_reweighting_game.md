# gt_06: Non-cooperative games in mlframe — DRO reweighting as a practical minimax instantiation

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Scope honesty (read before implementing)

"Nash equilibria / mechanism design in ML" is a research AREA, not a feature. GANs, multi-agent RL
and federated mechanism design do not land in a tabular-ML framework. This document deliberately
narrows to the ONE non-cooperative-game construction that concretely fits mlframe's plumbing:

**Distributionally Robust Optimization (DRO) as a two-player zero-sum game**: the model (player 1)
minimizes weighted loss; an adversary (player 2) chooses row weights within an uncertainty set to
MAXIMIZE that loss. The equilibrium model is robust to the worst reweighting the set allows —
practically: robust to subpopulation shift, minority-group degradation, and mild covariate shift.
Reference formulation: chi-square-ball DRO (Namkoong & Duchi, NeurIPS 2016/2017); group-free
practical variant. Everything reuses the sample_weight plumbing mapped in gt_04.

A companion diagnostic (cheap, independently useful): **adversarial validation** — train a
classifier to distinguish train rows from test rows; its AUC quantifies train/test shift and its
per-row train probabilities identify "test-like" rows. This is a folk-standard Kaggle technique
with a clean game reading (discriminator of a domain game) and near-zero implementation cost.

## 2. Integration verdict

Extension of the (gt_04-created) `src/mlframe/data_valuation/` package — two new modules. If gt_04
has NOT been implemented yet, create the package skeleton per gt_04 §3 first (facade +
`_benchmarks/`); the two plans share it deliberately.
- `data_valuation/_adversarial_reweighting.py` — the DRO game loop.
- `data_valuation/_adversarial_validation.py` — the shift diagnostic.
Both produce/consume (n,) weight vectors compatible with the `_setup_sample_weight` choke point
(`src/mlframe/training/_data_helpers.py:200`). Strictly opt-in; no default behaviour changes
anywhere.

## 3. Design

### 3.1 `_adversarial_validation.py` (implement FIRST — small, standalone value)
```python
def adversarial_validation(
    X_train, X_test, *, model=None,            # default: xgboost 200 trees depth 4
    n_splits: int = 5, rng=None,
) -> dict:
    """Label train=0/test=1, fit OOF classifier on the concatenation, return dict:
    auc (0.5 = no shift), train_test_proba (n_train,) = per-train-row P(test-like),
    top_shift_features (feature importances of the discriminator, first 20 names),
    suggested_weights (n_train,) = p/(1-p) density-ratio weights, clipped to
    [0.1, 10] and normalized to mean 1 (importance weighting under covariate shift)."""
```

### 3.2 `_adversarial_reweighting.py` — the chi²-ball DRO game
```python
def dro_reweight_fit(
    fit_fn,                     # (X, y, sample_weight) -> fitted model with predict/predict_proba
    loss_fn,                    # (y, pred) -> (n,) per-row losses
    X, y, *,
    rho: float = 0.5,           # chi2-ball radius; 0 -> ERM, larger -> more adversarial
    n_rounds: int = 8,
    step_mix: float = 0.5,      # weight-update smoothing between rounds
    rng=None,
) -> tuple[object, np.ndarray, dict]:
    """Alternating best-response minimax:
      round t: model_t = fit_fn(X, y, w_t)  (player 1 best response)
               losses_t = loss_fn(y, model_t.predict(X))   [OOF variant: see below]
               w_raw = project_chi2_ball(losses_t, rho)    (player 2 best response: the chi2-ball
                       inner max has a CLOSED FORM: w ∝ max(0, losses - eta) for the eta solving
                       the ball constraint; implement the 1-D bisection for eta -- standard
                       Namkoong-Duchi projection, ~10 lines)
               w_{t+1} = (1-step_mix)*w_t + step_mix*w_raw   (smoothing = fictitious-play flavor,
                       prevents oscillation of pure best-response dynamics)
      Returns (final model, final weights, info(history of worst-case & average loss per round,
      converged flag: max |w_{t+1}-w_t| < 1e-3)).
    OVERFITTING GUARD (mandatory): losses for the adversary MUST be out-of-fold (KFold predict
    inside each round), else the adversary just upweights noise/outliers the model memorized.
    n_rounds * n_splits fits total -- document the cost."""
```

### 3.3 Facade exports & training integration
Export both from `data_valuation/__init__.py`. Training-config integration is OPTIONAL v2 (same
posture as gt_04 §3.4): the functions are directly usable; wiring a `dro_weighting` flag into the
training config follows the gt_04 adapter pattern if/when wanted.

## 4. biz_val tests

File: `tests/data_valuation/test_biz_val_adversarial_reweighting.py`.

Bed A (subpopulation): binary classification, two latent groups — majority (85% of rows, easy
signal) and minority (15%, DIFFERENT coefficient vector); a plain model sacrifices the minority.
1. `test_biz_val_dro_improves_worst_group_auc` — xgboost fit_fn, logloss loss_fn, rho=0.5,
   n_rounds=8: worst-group AUC(dro) ≥ worst-group AUC(erm) + 0.02, overall AUC ≥ erm − 0.01.
   `@pytest.mark.slow @pytest.mark.timeout(600)`.
2. `test_biz_val_dro_rho_zero_matches_erm` — rho→0 recovers uniform weights (max|w−1| < 0.05)
   and model score within noise of plain fit (the game degenerates correctly).

Bed B (covariate shift): train/test with shifted feature means on 3 of 20 features.
3. `test_biz_val_adversarial_validation_detects_shift` — auc ≥ 0.75 on shifted bed AND
   auc ≤ 0.55 on an unshifted control split of the same data; top_shift_features contains ≥2 of
   the 3 truly shifted feature names.
4. `test_biz_val_shift_weights_improve_test_score` — model trained with suggested_weights vs
   unweighted, scored on the shifted test: AUC(weighted) ≥ AUC(unweighted) + 0.01 (measure first,
   floor below measurement per convention).

Unit tests: chi²-projection satisfies the ball constraint to 1e-6 and reduces to uniform at
rho=0 (property test over random loss vectors); weights mean≈1, nonneg; convergence flag fires on
a trivially-separable bed; adversarial_validation returns all documented keys.

## 5. Acceptance criteria
- Both modules + facade exports, mypy-clean; all tests green locally
  (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`).
- The chi²-projection property test proves the game's inner step (scientific core).
- cProfile harness in `data_valuation/_benchmarks/`: dro_reweight_fit wall vs n_rounds at
  n∈{5k, 50k}; committed.
- Module docstrings explicitly state what this is NOT (no GAN/MARL/mechanism-design claims) and
  cite Namkoong-Duchi + the game framing — the honest-scoping is part of the deliverable.

## 6. Known risks / rejected alternatives
- Group-DRO (known group labels) is easier and stronger when groups exist — mention in docstring;
  the group-free chi² variant is chosen because mlframe data has no group annotations in general.
- Pure best-response without smoothing oscillates (classic matching-pennies pathology) — the
  step_mix smoothing is mandatory, not cosmetic; the unit convergence test pins it.
- In-sample adversary losses (skipping OOF) look better in benches and are silently wrong —
  the OOF guard is non-negotiable; add a comment referencing this section.
- Mechanism-design / federated-incentive ideas: explicitly out of scope; do not scaffold for them.

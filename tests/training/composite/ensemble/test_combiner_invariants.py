"""Stacked cross-target ensembles: the fallback gate can fire, and capping keeps the blend on its level.

NNLS scored on the matrix it was fitted on never loses to its best single component (every unit vector is feasible), so
the gate that falls back to the best single component could not fire for ``nnls_stack``; and capping a non-convex stack
kept the top components' raw weights, so the served blend lost the dropped weight mass on every row.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble as Ens
from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import cross_fitted_stack_rmse, refit_capped_stack


def _pool(seed: int, n: int = 60, n_noise: int = 20):
    """One good component (y + noise) and ``n_noise`` pure-noise components around the mean."""
    rng = np.random.default_rng(seed)
    y = rng.normal(100.0, 10.0, n)
    good = y + rng.normal(0.0, 5.0, n)
    P = np.column_stack([good] + [np.full(n, y.mean()) + rng.normal(0.0, 10.0, n) for _ in range(n_noise)])
    comps = [types.SimpleNamespace() for _ in range(P.shape[1])]
    return y, P, comps, [f"c{i}" for i in range(P.shape[1])]


@pytest.mark.parametrize("seed", [1, 2])
def test_the_gate_can_fire_on_a_noise_padded_pool(seed: int):
    """Scored in-sample NNLS always beats the good component; cross-fitted, the noise it fitted costs it the comparison."""
    y, P, comps, names = _pool(seed)
    best_single = float(np.sqrt(np.mean((P[:, 0] - y) ** 2)))
    ens = Ens.from_nnls_stack(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    in_sample = float(np.sqrt(np.mean((P @ ens.weights - y) ** 2)))
    assert in_sample < best_single, "the in-sample stack should beat its best single by construction (the old gate's blind spot)"
    assert cross_fitted_stack_rmse(Ens, "nnls_stack", comps, names, P, y) > best_single, "cross-fitted, the noise-padded stack must lose"


def test_a_capped_stack_is_refit_on_the_columns_it_kept():
    """Capping a non-convex stack to its top components refits the weights, so the blend stays on the target's level."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(100.0, 10.0, n)
    P = np.column_stack([y + rng.normal(0.0, s, n) for s in (3.0, 4.0, 5.0, 6.0)])
    comps = [types.SimpleNamespace() for _ in range(4)]
    names = [f"c{i}" for i in range(4)]
    full = Ens.from_nnls_stack(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    capped = full.cap_inference_components(2)
    kept = [names.index(nm) for nm in capped.component_names]
    raw_mean = float(np.mean(P[:, kept] @ np.asarray(capped.weights)))
    refit = refit_capped_stack(Ens, "nnls_stack", capped, names, P, y)
    refit_mean = float(np.mean(P[:, kept] @ np.asarray(refit.weights)))
    assert abs(raw_mean - y.mean()) / y.mean() > 0.02, "the fixture must show the raw-weight cap losing mass"
    assert abs(refit_mean - y.mean()) / y.mean() < 0.02


def test_oof_rows_align_the_suite_weights_with_the_matrix():
    """``compute_oof_holdout_predictions(return_rows=True)`` gives each OOF row's train position, so weights line up with it."""
    from mlframe.training.composite import compute_oof_holdout_predictions
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X @ np.array([1.0, -1.0]) + rng.normal(0.0, 0.1, n)
    comps = [LinearRegression().fit(X, y), LinearRegression().fit(X[:, :1], y)]
    comps[1].predict = lambda Z, m=comps[1]: LinearRegression.predict(m, np.asarray(Z)[:, :1])
    _P, y_h, names, rows = compute_oof_holdout_predictions(
        component_models=comps, component_names=["a", "b"], component_specs=[None, None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, kfold=3, return_rows=True,
    )
    assert names == ["a", "b"] and rows is not None and np.array_equal(y[rows], y_h)
    three = compute_oof_holdout_predictions(component_models=comps, component_names=["a", "b"], component_specs=[None, None], train_X=X,
                                            y_train_full=y, base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, kfold=3)
    assert len(three) == 3


def test_the_stack_and_its_rmses_follow_the_sample_weights():
    """Two components each good on one half; weights on the second half move the NNLS blend and the RMSEs toward its expert."""
    from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import column_rmses

    rng = np.random.default_rng(1)
    n = 400
    y = rng.normal(0.0, 5.0, n)
    half = np.arange(n) >= n // 2
    good_first = np.where(half, y + rng.normal(0.0, 4.0, n), y + rng.normal(0.0, 0.3, n))
    good_second = np.where(half, y + rng.normal(0.0, 0.3, n), y + rng.normal(0.0, 4.0, n))
    P = np.column_stack([good_first, good_second])
    comps = [types.SimpleNamespace(), types.SimpleNamespace()]
    w = np.where(half, 10.0, 1.0)
    plain = Ens.from_nnls_stack(component_models=comps, component_names=["f", "s"], component_predictions=P, y_train=y)
    weighted = Ens.from_nnls_stack(component_models=comps, component_names=["f", "s"], component_predictions=P, y_train=y, sample_weight=w)
    assert weighted.weights[1] > plain.weights[1] + 0.1, (plain.weights, weighted.weights)
    r_plain, r_w = column_rmses(P, y), column_rmses(P, y, w)
    assert r_w[1] < r_plain[1] and r_w[0] > r_plain[0]


def _blend(ens, P_cols: np.ndarray) -> np.ndarray:
    """What a stack serves on the given component columns: its weights, plus the intercept for a linear stack."""
    w = np.asarray(ens.weights, dtype=np.float64)
    return P_cols @ w + float(getattr(ens, "_linear_stack_intercept", 0.0) or 0.0)


@pytest.mark.parametrize("strategy", ["nnls_stack", "linear_stack"])
@pytest.mark.parametrize("seed", [1, 2])
def test_the_gate_value_itself_falls_back_on_a_noise_padded_pool(strategy: str, seed: int):
    """``gate_stack_rmse`` - the number the fallback compares - loses to the best single on 1 good + 20 noise components.

    The in-sample stack beats the good component by construction; the gate has to read the cross-fitted value, or the
    fallback it exists for can never fire.
    """
    from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import gate_stack_rmse

    y, P, comps, names = _pool(seed)
    best_single = float(np.sqrt(np.mean((P[:, 0] - y) ** 2)))
    build = Ens.from_linear_stack if strategy == "linear_stack" else Ens.from_nnls_stack
    ens = build(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    in_sample = _blend(ens, P)
    assert float(np.sqrt(np.mean((in_sample - y) ** 2))) < best_single
    assert gate_stack_rmse(Ens, strategy, comps, names, P, y, in_sample) > best_single


@pytest.mark.parametrize("strategy", ["nnls_stack", "linear_stack"])
def test_a_capped_stack_serves_predictions_on_the_uncapped_level(strategy: str):
    """After capping and refitting, the served blend's mean stays within 2% of the full stack's on the same rows.

    Checking only which component names survive the cap said nothing about what the capped ensemble predicts.
    """
    rng = np.random.default_rng(3)
    n = 400
    y = rng.normal(100.0, 10.0, n)
    P = np.column_stack([y + rng.normal(0.0, s, n) for s in (3.0, 4.0, 5.0, 6.0, 7.0)])
    comps = [types.SimpleNamespace() for _ in range(P.shape[1])]
    names = [f"c{i}" for i in range(P.shape[1])]
    build = Ens.from_linear_stack if strategy == "linear_stack" else Ens.from_nnls_stack
    full = build(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    capped = refit_capped_stack(Ens, strategy, full.cap_inference_components(2), names, P, y)
    kept = [names.index(nm) for nm in capped.component_names]
    assert len(kept) == 2
    full_mean = float(np.mean(_blend(full, P)))
    capped_mean = float(np.mean(_blend(capped, P[:, kept])))
    assert abs(capped_mean - full_mean) / abs(full_mean) < 0.02, (strategy, capped_mean, full_mean)


# ---------------------------------------------------------------------------
# The same invariants for every combining strategy.
# ---------------------------------------------------------------------------


class _Column:
    """A component that serves one fixed prediction column, or raises when ``fail`` is set."""

    def __init__(self, col: np.ndarray, fail: bool = False) -> None:
        self.col = np.asarray(col, dtype=np.float64)
        self.fail = fail

    def predict(self, X):
        """The stored column (``X`` only sets the row count), or a RuntimeError."""
        if self.fail:
            raise RuntimeError("component down")
        return self.col[: len(X)]


def _build(strategy: str, comps, names, P, y):
    """The ensemble a strategy builds from an OOF matrix ``P`` and its targets ``y``."""
    from mlframe.training.composite.ensemble._stackers import build_meta_stack_ensemble

    if strategy == "mean":
        return Ens.from_uniform_weights(component_models=comps, component_names=names)
    if strategy == "oof_weighted":
        rmses = [float(np.sqrt(np.mean((P[:, j] - y) ** 2))) for j in range(P.shape[1])]
        return Ens.from_train_metrics(component_models=comps, component_names=names, component_oof_rmse=rmses,
                                      baseline_oof_rmse=float(np.std(y)))
    if strategy == "nnls_stack":
        return Ens.from_nnls_stack(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    if strategy == "linear_stack":
        return Ens.from_linear_stack(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    if strategy.startswith("meta_"):
        return build_meta_stack_ensemble(Ens, comps, names, P, y, stacker=strategy[len("meta_"):])
    raise KeyError(strategy)


# Every way a cross-target ensemble is built. The suite's strategies are the config's; the meta-stackers are the
# build_meta_stack_ensemble learners.
STRATEGIES = ("mean", "oof_weighted", "nnls_stack", "linear_stack", "meta_nnls", "meta_ridge", "meta_lasso", "meta_elasticnet", "meta_gbm")
# The meta-stackers' input is the full K-column design the meta-model was fit on, so dropping columns to cap latency would
# change its input space: capping is defined for the weight-based strategies only.
_CAPPABLE = ("mean", "oof_weighted", "nnls_stack", "linear_stack")
# Only the stacks fit weights on the OOF rows they are gated on; mean has no fitted weights, and oof_weighted's are K
# per-column RMSE statistics, not a solve over the rows.
_ROW_FITTED = ("nnls_stack", "linear_stack")


def test_every_ensemble_constructor_and_suite_strategy_is_in_the_registry():
    """A new ``from_*`` constructor, suite strategy or meta-stacker must join the parametrised invariants below."""
    import inspect

    from mlframe.training.composite.ensemble._stackers import META_STACKER_KINDS
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    constructors = {n for n, _ in inspect.getmembers(Ens) if n.startswith("from_")}
    covered = {"from_uniform_weights", "from_train_metrics", "from_nnls_stack", "from_linear_stack"}
    assert constructors == covered, sorted(constructors ^ covered)
    for s in ("mean", "oof_weighted", "nnls_stack", "linear_stack"):
        CompositeTargetDiscoveryConfig(cross_target_ensemble_strategy=s)
    with pytest.raises(ValueError, match="must be one of"):
        CompositeTargetDiscoveryConfig(cross_target_ensemble_strategy="some_new_strategy")
    assert {f"meta_{x}" for x in META_STACKER_KINDS} == {s for s in STRATEGIES if s.startswith("meta_")}, META_STACKER_KINDS


def _fixture(n: int = 300, seed: int = 0):
    """Five decent components of y on one scale."""
    rng = np.random.default_rng(seed)
    y = rng.normal(100.0, 10.0, n)
    P = np.column_stack([y + rng.normal(0.0, s, n) for s in (3.0, 4.0, 5.0, 6.0, 7.0)])
    return y, P, [f"c{i}" for i in range(P.shape[1])]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_failing_component_keeps_the_blend_on_its_level(strategy: str):
    """One component raising at predict leaves the served mean within 5% of the full blend's (EST-03: a 0.5-weight
    dropout halved the forecast, 449 against 899)."""
    y, P, names = _fixture()
    ens = _build(strategy, [_Column(P[:, j]) for j in range(P.shape[1])], names, P, y)
    X = np.zeros((len(y), 1))
    full = float(np.mean(ens.predict(X)))
    heavy = int(np.argmax(np.abs(np.asarray(ens.weights, dtype=float)))) if len(np.asarray(ens.weights)) else 0
    ens.component_models[heavy] = _Column(P[:, heavy], fail=True)
    degraded = float(np.mean(ens.predict(X)))
    assert abs(degraded - full) / abs(full) < 0.05, (strategy, full, degraded)


@pytest.mark.parametrize("strategy", _CAPPABLE)
def test_capping_keeps_the_served_blend_on_its_level(strategy: str):
    """Capped to two components (and a stack refit on them, as the suite does), the served mean stays within 2% (EST-06)."""
    y, P, names = _fixture(seed=3)
    comps = [_Column(P[:, j]) for j in range(P.shape[1])]
    full = _build(strategy, comps, names, P, y)
    capped = refit_capped_stack(Ens, strategy, full.cap_inference_components(2), names, P, y)
    assert len(capped.component_names) == 2
    X = np.zeros((len(y), 1))
    a, b = float(np.mean(full.predict(X))), float(np.mean(capped.predict(X)))
    assert abs(b - a) / abs(a) < 0.02, (strategy, a, b)


@pytest.mark.parametrize("strategy", _CAPPABLE)
@pytest.mark.parametrize("seed", [1, 2])
def test_the_gate_can_fire_for_every_strategy(strategy: str, seed: int):
    """On 1 good + 20 noise components the number the fallback gate compares is never better than the best single component
    (EST-05: in-sample NNLS always was). oof_weighted already falls back to that single component here, so it ties."""
    from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import gate_stack_rmse

    y, P, comps, names = _pool(seed)
    ens = _build(strategy, comps, names, P, y)
    w = np.asarray(ens.weights, dtype=float)
    in_sample = _blend(ens, P) if not getattr(ens, "is_convex", True) else P @ (w / w.sum())
    best_single = float(np.sqrt(np.mean((P[:, 0] - y) ** 2)))
    assert gate_stack_rmse(Ens, strategy, comps, names, P, y, in_sample) >= best_single - 1e-12


@pytest.mark.parametrize("strategy", _ROW_FITTED)
def test_the_gate_scores_rows_its_weights_were_not_fitted_on(strategy: str):
    """Through the row-role ledger: every cross-fitted fold reports on OOF rows disjoint from the rows its weights saw, and
    every row is reported once."""
    from mlframe.training.composite import _row_roles
    from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import gate_stack_rmse

    y, P, names = _fixture(n=200, seed=5)
    comps = [_Column(P[:, j]) for j in range(P.shape[1])]
    ens = _build(strategy, comps, names, P, y)
    prev = _row_roles._FORCED
    _row_roles._FORCED = True
    _row_roles._LOG.clear()
    try:
        gate_stack_rmse(Ens, strategy, comps, names, P, y, _blend(ens, P))
        log = list(_row_roles._LOG)
    finally:
        _row_roles._FORCED = prev
    folds = {r.consumer for r in log}
    assert folds and all(c.startswith("xt_stack_gate[fold") for c in folds), folds
    for c in folds:
        fit = np.concatenate([r.rows for r in log if r.consumer == c and r.role == "fit"])
        rep = np.concatenate([r.rows for r in log if r.consumer == c and r.role == "report"])
        assert np.intersect1d(fit, rep).size == 0, c
    reported = np.concatenate([r.rows for r in log if r.role == "report"])
    assert np.array_equal(np.sort(reported), np.arange(len(y)))

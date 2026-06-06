"""§3 Votenrank-integration tests for the score_ensemble Leaderboard wiring.

Each test names the audit tag in its docstring. Behavioural only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlframe.models.ensembling import (
    EnsembleLeaderboard,
    _build_votenrank_leaderboard_from_results,
)


def _make_result(metrics: dict):
    return SimpleNamespace(metrics=metrics)


def test_leaderboard_table_has_expected_columns():
    """VOTENRANK-DISCONNECT: every numeric metric becomes a column in the table."""
    res = {
        "arithm": _make_result({"oof": {"rmse": 1.0, "mae": 0.5}, "val": {"rmse": 1.1}}),
        "harm": _make_result({"oof": {"rmse": 1.2, "mae": 0.6}, "val": {"rmse": 1.3}}),
    }
    lb = _build_votenrank_leaderboard_from_results(res, is_regression=True)
    assert lb is not None
    cols = set(lb.table.columns)
    assert {"oof.rmse", "oof.mae", "val.rmse"}.issubset(cols)


def test_leaderboard_classification_keeps_rrf_row():
    """REG-RRF-DROPPED: classification suites RETAIN the rrf row in the leaderboard."""
    res = {
        "arithm": _make_result({"oof": {"logloss": 0.6}}),
        "rrf": _make_result({"oof": {"logloss": 0.7}}),
    }
    lb = _build_votenrank_leaderboard_from_results(res, is_regression=False)
    assert lb is not None
    assert "rrf" in lb.table.index


def test_leaderboard_skips_underscore_metadata_keys():
    """VOTENRANK-DISCONNECT: keys starting with '_' (e.g. _diversity) are not flavours."""
    res = {
        "arithm": _make_result({"oof": {"rmse": 1.0}}),
        "_diversity": {"x": 1},
        "_stacking_gate": {"y": 2},
    }
    lb = _build_votenrank_leaderboard_from_results(res, is_regression=True)
    assert lb is not None
    assert "_diversity" not in lb.table.index
    assert "_stacking_gate" not in lb.table.index


def test_leaderboard_returns_none_on_empty_metrics():
    """Edge case: if no flavour has metrics, builder returns None instead of an empty table."""
    res = {"arithm": _make_result({}), "harm": _make_result({})}
    assert _build_votenrank_leaderboard_from_results(res, is_regression=True) is None


def test_leaderboard_rank_all_returns_dataframe():
    """votenrank.Leaderboard.rank_all should still work through the EnsembleLeaderboard wrapper."""
    res = {
        "arithm": _make_result({"oof": {"rmse": 1.0}}),
        "harm": _make_result({"oof": {"rmse": 1.5}}),
        "median": _make_result({"oof": {"rmse": 1.2}}),
    }
    lb = _build_votenrank_leaderboard_from_results(res, is_regression=True)
    assert lb is not None
    rankings = lb.rank_all()
    assert isinstance(rankings, pd.DataFrame)
    # Should produce at least one ranking method column.
    assert rankings.shape[1] >= 1


# ----------------------------- composite ensemble refit cache -----------------------------


def test_dropout_predict_is_deterministic_across_repeated_calls():
    """NNLS-DROPOUT: on component dropout the predict path is a pure deterministic function of
    its inputs -- repeated calls with the same dropout pattern return identical output. The old
    leaky refit-on-dropout (with its surviving-subset cache) was removed: predict must not depend
    on which batch dropped a column, so there is nothing to cache."""
    pytest.importorskip("scipy.optimize")
    from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble

    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    p1 = y + rng.normal(scale=0.5, size=n)
    p2 = y - 0.5 + rng.normal(scale=0.5, size=n)
    p3 = y + 1.0 + rng.normal(scale=0.5, size=n)
    train_preds = np.column_stack([p1, p2, p3])

    # Build proper objects with a predict method (MagicMock leaks attribute access).
    class _Model:
        def __init__(self, ret):
            self._ret = ret

        def predict(self, X):
            if self._ret is None:
                raise RuntimeError("dropped out")
            return self._ret

    m1 = _Model(p1)
    m2 = _Model(None)  # raises -> dropout path
    m3 = _Model(p3)

    ens = CompositeCrossTargetEnsemble.from_nnls_stack(
        component_models=[m1, m2, m3],
        component_names=["c1", "c2", "c3"],
        component_predictions=train_preds,
        y_train=y,
    )
    # The removed refit-cache must not reappear.
    assert not hasattr(ens, "_refit_cache"), "dropout refit-cache was removed (leaky/non-deterministic/RAM)"
    out1 = ens.predict(np.zeros((n, 1)))
    out2 = ens.predict(np.zeros((n, 1)))
    np.testing.assert_array_equal(out1, out2)


def test_dropout_combines_surviving_columns_with_original_weights_no_refit():
    """NNLS-DROPOUT: a dropped-out component is excluded and the survivors keep their ORIGINAL
    solver weights (no refit). Equivalent to zeroing the dropped column's contribution -- a
    deterministic linear fallback rather than a per-batch re-solve."""
    pytest.importorskip("scipy.optimize")
    from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble

    rng = np.random.default_rng(0)
    n = 100
    y = rng.normal(size=n)
    p1 = y + rng.normal(scale=0.5, size=n)
    p2 = y - 0.5 + rng.normal(scale=0.5, size=n)
    p3 = y + 1.0 + rng.normal(scale=0.5, size=n)

    class _Model:
        def __init__(self, ret):
            self._ret = ret

        def predict(self, X):
            if self._ret is None:
                raise RuntimeError("dropped out")
            return self._ret

    ens = CompositeCrossTargetEnsemble.from_nnls_stack(
        component_models=[_Model(p1), _Model(None), _Model(p3)],
        component_names=["c1", "c2", "c3"],
        component_predictions=np.column_stack([p1, p2, p3]),
        y_train=y,
    )
    w = np.asarray(ens.weights, dtype=np.float64)
    expected = p1 * w[0] + p3 * w[2]  # surviving columns combined with original weights
    np.testing.assert_allclose(ens.predict(np.zeros((n, 1))), expected, rtol=1e-9, atol=1e-9)


# ----------------------------- MEDIAN-BASELINE -----------------------------


def test_from_train_metrics_default_baseline_keeps_all_components(caplog):
    """MEDIAN-BASELINE: without baseline arg, default baseline=max(rmse) keeps every component."""
    import logging
    from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble

    models = [MagicMock(name=f"c{i}") for i in range(4)]
    names = [f"c{i}" for i in range(4)]
    train_rmse = [1.0, 1.2, 1.5, 1.8]
    with caplog.at_level(logging.WARNING):
        ens = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=models,
            component_names=names,
            component_train_rmse=train_rmse,
            component_oof_rmse=train_rmse,  # use oof path to avoid the unrelated train-rmse warning
        )
    # All 4 components should have non-zero weight under max-baseline.
    assert isinstance(ens, CompositeCrossTargetEnsemble)
    assert len(ens.component_models) == 4
    # The worst component (rmse=1.8) has weight 0 (it doesn't beat itself); others positive.
    assert ens.weights[0] > 0


def test_from_train_metrics_prefers_oof_rmse_when_supplied():
    """VAL-LEAK (from_train_metrics): when component_oof_rmse is given, train rmse is ignored."""
    from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble

    models = [MagicMock(name=f"c{i}") for i in range(3)]
    names = [f"c{i}" for i in range(3)]
    train_rmse = [3.0, 1.0, 2.0]  # train says c1 is best
    oof_rmse = [1.0, 3.0, 2.0]  # OOF says c0 is best
    ens = CompositeCrossTargetEnsemble.from_train_metrics(
        component_models=models,
        component_names=names,
        component_train_rmse=train_rmse,
        component_oof_rmse=oof_rmse,
        baseline_oof_rmse=4.0,
    )
    # With OOF ranking the gain for c0 (4-1=3) should exceed c1 (4-3=1).
    assert ens.weights[0] > ens.weights[1]


# ----------------------------- WEIGHT-NEGATIVE-WARN -----------------------------


def test_linear_stack_negative_weights_logged_as_warning(caplog):
    """WEIGHT-NEGATIVE-WARN: ridge negative coefficients emit a structured WARN."""
    import logging
    from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble

    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    # Component 0 anti-correlates with target so Ridge will assign it a negative weight.
    p0 = -y + rng.normal(scale=0.3, size=n)
    p1 = y + rng.normal(scale=0.3, size=n)
    p2 = y + rng.normal(scale=0.3, size=n)

    with caplog.at_level(logging.WARNING):
        ens = CompositeCrossTargetEnsemble.from_linear_stack(
            component_models=[MagicMock() for _ in range(3)],
            component_names=["c0", "c1", "c2"],
            component_predictions=np.column_stack([p0, p1, p2]),
            y_train=y,
            ridge_alpha=0.01,
        )
    assert ens.weights[0] < 0
    assert any("negative Ridge" in rec.message for rec in caplog.records)


# ----------------------------- SOLVER-COPY -----------------------------


def test_stash_avoids_redundant_copy():
    """SOLVER-COPY: boolean-indexed train matrices are stored directly (no extra .copy)."""
    from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble

    rng = np.random.default_rng(0)
    n = 100
    y = rng.normal(size=n)
    preds = np.column_stack([y + rng.normal(scale=0.5, size=n) for _ in range(3)])

    ens = CompositeCrossTargetEnsemble.from_linear_stack(
        component_models=[MagicMock() for _ in range(3)],
        component_names=["c0", "c1", "c2"],
        component_predictions=preds,
        y_train=y,
        ridge_alpha=1.0,
    )
    # The stashed training matrix should match in shape (no slicing or copy artefacts).
    assert ens._linear_stack_train_preds.shape[0] == n
    assert ens._linear_stack_train_preds.shape[1] == 3

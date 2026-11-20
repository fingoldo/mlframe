"""Diff-basis fit/replay fidelity: NaN fill, float64 diff axis and the visible freeze-failure warning."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_diff_basis_fe as mod


def _fit_and_replay(X: pd.DataFrame):
    """Fit the single (a, b) cell at degree 2 and replay it on the same frame; returns (fit values, replay values)."""
    cand_cols, cand_values, cand_meta = mod._enumerate_diff_cells(X, [("a", "b")], (2,), "hermite", np.float64)
    assert cand_cols
    recipe = SimpleNamespace(
        name=cand_cols[0], src_names=("a", "b"), extra={"basis": "hermite", "degree": 2, "preprocess_params": cand_meta[0][3]},
    )
    return cand_values[0], mod._apply_orth_diff_basis(recipe, X)


def test_diff_basis_replay_matches_fit_when_nans_sit_in_different_rows():
    """NaNs in different rows of a and b: the fit-time fill equals the per-column fill the replay applies."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(5.0, 2.0, n)
    b = a + rng.normal(0, 0.3, n) - 3.0
    a[:40] = np.nan
    b[200:240] = np.nan
    fit_vals, replay_vals = _fit_and_replay(pd.DataFrame({"a": a, "b": b}))
    np.testing.assert_allclose(replay_vals, fit_vals, atol=1e-9)


def test_diff_basis_replay_keeps_float64_diff_axis_for_offset_pairs():
    """Two near-identical columns at a 1e5 offset: a float32 replay diff would quantise to 0.0078 steps."""
    rng = np.random.default_rng(1)
    n = 400
    a = 1e5 + rng.normal(0, 0.05, n)
    b = a + rng.normal(0, 0.05, n)
    fit_vals, replay_vals = _fit_and_replay(pd.DataFrame({"a": a, "b": b}))
    np.testing.assert_allclose(replay_vals, fit_vals, atol=1e-9)


def test_diff_basis_failed_param_freeze_logs_warning(monkeypatch, caplog):
    """A legacy score row without basis_params and a raising basis evaluation must warn, not stay silent."""
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) ** 1.1})
    X_aug = X.copy()
    X_aug["diff_a_b__He2"] = 1.0
    scores = pd.DataFrame([{"engineered_col": "diff_a_b__He2", "col_a": "a", "col_b": "b", "basis": "hermite", "degree": 2}])
    monkeypatch.setattr(mod, "hybrid_orth_mi_diff_basis_fe", lambda *a, **k: (X_aug, scores))

    def boom(*a, **k):
        """Raise like a broken basis evaluation."""
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_evaluate_basis_column", boom)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod.hybrid_orth_mi_diff_basis_fe_with_recipes(X, np.zeros(50, dtype=int))
    assert any("could not freeze preprocess params" in r.getMessage() for r in caplog.records)

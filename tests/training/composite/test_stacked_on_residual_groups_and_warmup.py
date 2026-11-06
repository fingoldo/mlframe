"""Residual-target pass 2 passes the group ids to its OOF and survives the NaN warm-up block of a forward-chaining split."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import _stacked
from mlframe.training.composite.ensemble import feature_stacking


def _fake(n=40):
    """Discovery stand-in whose pass 1 yields one spec and records the pass-2 fit call."""
    spec = SimpleNamespace(name="s1")
    calls = {}

    def fit(df, target, feature_cols, train_idx, *a, **k):
        """Record the pass-2 fit's train rows (pass 1 has already filled specs_)."""
        calls.setdefault("fits", []).append((target, np.asarray(train_idx)))

    obj = SimpleNamespace(
        fit=fit, specs_=[spec], tiny_rerank_scores_={"s1": 0.1}, config=SimpleNamespace(random_state=0), _group_ids_for_rerank=np.repeat(np.arange(8), 5)
    )
    return obj, calls


def test_oof_gets_groups_in_the_residual_pass(monkeypatch):
    """Without the ids the OOF folds spread each group across train and validation."""
    seen = {}

    def fake_oof(factory, X, y, **kw):
        """Record groups and stop."""
        seen["groups"] = kw.get("groups")
        raise RuntimeError("stop")

    monkeypatch.setattr(feature_stacking, "composite_oof_predictions", fake_oof)
    n = 40
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
    obj, _ = _fake()
    train_idx = np.arange(10, 30)
    with contextlib.suppress(Exception):
        _stacked.fit_stacked_on_residual(obj, df, "y", ["a"], train_idx)
    np.testing.assert_array_equal(seen["groups"], np.repeat(np.arange(8), 5)[train_idx])


def test_nan_warmup_rows_do_not_cancel_pass_two(monkeypatch):
    """The first 5 train rows are NaN in the OOF (never tested); pass 2 still runs, on the other 15 rows."""

    def fake_oof(factory, X, y, **kw):
        """OOF = y except for a NaN warm-up block."""
        out = np.asarray(y, dtype=float).copy()
        out[:5] = np.nan
        return out

    monkeypatch.setattr(feature_stacking, "composite_oof_predictions", fake_oof)
    n = 40
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
    obj, calls = _fake()
    monkeypatch.setattr(_stacked, "_annotate_residual_specs", lambda self, specs: [])
    train_idx = np.arange(10, 30)
    with contextlib.suppress(Exception):
        _stacked.fit_stacked_on_residual(obj, df, "y", ["a"], train_idx)
    residual_fits = [t for tgt, t in calls.get("fits", []) if tgt.startswith("__y_residual__")]
    assert residual_fits and len(residual_fits[0]) == 15

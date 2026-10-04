"""The shared real-datasets matrix helper must be able to fail: erroring, non-finite and collapsed transformer arms are rejected."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("lightgbm")

_PATH = pathlib.Path(__file__).with_name("test_biz_val_real_datasets.py")
_NAME = "_real_datasets_helper_under_test"


def _load_helper_module():
    """Import the big real-datasets module under a private name so its helpers can be called directly."""
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _rec(boost: str, feat: str, score: float, error=None) -> dict:
    """One matrix record with just the fields the discrimination helper reads."""
    return {"dataset": "d", "boosting": boost, "features": feat, "score": score, "error": error}


def test_helper_accepts_healthy_matrix():
    """An FE arm at or slightly below raw is a modelling outcome, not a failure."""
    mod = _load_helper_module()
    mod._assert_matrix_discriminates([_rec("lgb", "raw", 0.80), _rec("lgb", "+rff", 0.78), _rec("lgb", "+rowattn", 0.85)], "d")


def test_helper_rejects_collapsed_arm():
    """An arm scoring 0.5 below raw (a zero/garbage feature block) must fail the helper."""
    mod = _load_helper_module()
    with pytest.raises(AssertionError, match="collapsed"):
        mod._assert_matrix_discriminates([_rec("lgb", "raw", 0.80), _rec("lgb", "+rff", 0.30)], "d")


def test_helper_rejects_collapse_exactly_beyond_bound_and_keeps_boundary():
    """The collapse bound is inclusive of the threshold: a drop of 0.29 passes, 0.31 fails."""
    mod = _load_helper_module()
    mod._assert_matrix_discriminates([_rec("lgb", "raw", 0.80), _rec("lgb", "+rff", 0.51)], "d")
    with pytest.raises(AssertionError, match="collapsed"):
        mod._assert_matrix_discriminates([_rec("lgb", "raw", 0.80), _rec("lgb", "+rff", 0.49)], "d")


def test_helper_rejects_arm_that_raised_even_with_finite_score():
    """A recorded arm exception fails the helper regardless of the score attached to it."""
    mod = _load_helper_module()
    with pytest.raises(AssertionError, match="raised"):
        mod._assert_matrix_discriminates([_rec("lgb", "raw", 0.80), _rec("lgb", "+rff", 0.80, error="ValueError: boom")], "d")


def test_helper_rejects_non_finite_score_and_empty_matrix():
    """NaN scores and an empty record list are failures, not silent passes."""
    mod = _load_helper_module()
    with pytest.raises(AssertionError, match="non-finite"):
        mod._assert_matrix_discriminates([_rec("lgb", "raw", 0.80), _rec("lgb", "+rff", float("nan"))], "d")
    with pytest.raises(AssertionError, match="no records"):
        mod._assert_matrix_discriminates([], "d")


def _zero_block_builder(X_tr, X_te, y_tr, task):
    """Deliberately broken transformer: replaces the whole feature block by zeros."""
    return np.zeros_like(X_tr), np.zeros_like(X_te)


def _raising_builder(X_tr, X_te, y_tr, task):
    """Deliberately broken transformer that raises inside the arm."""
    raise RuntimeError("transformer exploded")


def _run_small_matrix(mod, monkeypatch, builders):
    """Run the real _run_matrix with a single lightgbm booster on a small friedman1 draw."""
    from sklearn.datasets import make_friedman1

    X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=0)
    monkeypatch.setattr(mod, "BOOSTING_FACTORIES", {"lgb": lambda task: __import__("lightgbm").LGBMRegressor(n_estimators=60, random_state=0, verbose=-1, n_jobs=2)})
    return mod._run_matrix(X.astype(np.float32), y.astype(np.float32), "regression", "friedman_small", builders)


def test_real_matrix_with_zero_block_transformer_fails_helper(monkeypatch):
    """End to end: a transformer returning zeros scores near R2=0 against a raw arm near 0.9 and must trip the helper."""
    mod = _load_helper_module()
    records = _run_small_matrix(mod, monkeypatch, {"raw": mod._features_raw, "+broken": _zero_block_builder})
    raw = next(r["score"] for r in records if r["features"] == "raw")
    assert raw > 0.7
    with pytest.raises(AssertionError, match="collapsed"):
        mod._assert_matrix_discriminates(records, "friedman_small")


def test_real_matrix_transformer_exception_propagates(monkeypatch):
    """A transformer builder that raises is not swallowed by the matrix runner."""
    mod = _load_helper_module()
    with pytest.raises(RuntimeError, match="transformer exploded"):
        _run_small_matrix(mod, monkeypatch, {"raw": mod._features_raw, "+boom": _raising_builder})


def test_real_matrix_booster_failure_is_recorded_and_fails_helper(monkeypatch):
    """A booster fit that raises is recorded on the arm and fails the helper instead of only printing a skip line."""
    mod = _load_helper_module()
    from sklearn.datasets import make_friedman1

    X, y = make_friedman1(n_samples=300, n_features=10, noise=1.0, random_state=0)

    def _failing_factory(task):
        """Booster factory whose model cannot be built."""
        raise ValueError("cannot build booster")

    monkeypatch.setattr(mod, "BOOSTING_FACTORIES", {"lgb": _failing_factory})
    records = mod._run_matrix(X.astype(np.float32), y.astype(np.float32), "regression", "tiny", {"raw": mod._features_raw})
    assert records[0]["error"] == "ValueError: cannot build booster"
    with pytest.raises(AssertionError, match="raised"):
        mod._assert_matrix_discriminates(records, "tiny")

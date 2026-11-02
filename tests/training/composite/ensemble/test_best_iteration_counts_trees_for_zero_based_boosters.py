"""The round count handed to the OOF clone is a tree count: index + 1 for XGBoost / CatBoost, as-is for LightGBM."""

from __future__ import annotations

from mlframe.training.composite.ensemble._oof_split import _best_iteration_of


def _fake(module: str, **attrs):
    """Object whose class claims to come from ``module``."""
    cls = type("Fake", (), {"__module__": module + ".sklearn", **attrs})
    return cls()


def test_xgboost_index_zero_is_one_tree():
    """best_iteration=0 means one tree; it used to be read as 'did not early stop'."""
    assert _best_iteration_of(_fake("xgboost", best_iteration=0)) == 1


def test_xgboost_index_is_incremented():
    """best_iteration=99 is 100 trees."""
    assert _best_iteration_of(_fake("xgboost", best_iteration=99)) == 100


def test_catboost_attribute_is_incremented():
    """CatBoost's best_iteration_ is 0-based too."""
    assert _best_iteration_of(_fake("catboost", best_iteration_=49)) == 50


def test_lightgbm_count_is_returned_unchanged():
    """LightGBM's best_iteration_ is already a count (negative control)."""
    assert _best_iteration_of(_fake("lightgbm", best_iteration_=37)) == 37

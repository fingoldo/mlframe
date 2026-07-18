"""Regression: model-facing feature names must be GBM-compatible.

The MRMR interaction-FE namer (``get_new_feature_name``) builds names like
``mul(log(f2),sin(f3))`` -- a parseable internal contract (``_mi_greedy_fe``
splits on ``(``) whose embedded comma is a JSON-structural character. The moment
such an engineered frame reaches LightGBM it raises *"Do not support special
JSON characters in feature name"* (and XGBoost rejects ``[ ] <``). Surfaced by
fuzz combos ``c0026`` / ``c0143`` (``use_mrmr_fs`` + ``mrmr_cat_fe``).

``mlframe.training._feature_name_sanitize`` renames only the model-facing column
labels via a pure, idempotent, deterministic map (so train/test frames, produced
at different points by one fitted pipeline, map identically without stored
state), leaving the internal engineered name untouched. The trainer applies it
to the train/val/test frames right after the pre-pipeline transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._feature_name_sanitize import (
    build_safe_mapping,
    has_hostile_name,
    safe_feature_name,
    sanitize_frame_columns,
    sanitize_name_list,
)

_HOSTILE = ',[]{}":<>'
_ENGINEERED = "mul(log(f2),sin(f3))"  # real shape produced by get_new_feature_name


def test_safe_feature_name_strips_all_hostile_chars():
    """Safe feature name strips all hostile chars."""
    out = safe_feature_name(_ENGINEERED)
    assert not any(ch in out for ch in _HOSTILE), out
    # Parens are JSON/GBM-safe and kept -> structure (hence uniqueness) preserved.
    assert out == "mul(log(f2)_sin(f3))"
    # Idempotent.
    assert safe_feature_name(out) == out


def test_safe_feature_name_is_noop_for_clean_names():
    """Safe feature name is noop for clean names."""
    for clean in ("f0", "cat_1", "cross_cat_0_cat_1", "rbf_3", "log(f2)"):
        assert safe_feature_name(clean) == clean


def test_build_safe_mapping_only_hostile_and_dedupes():
    """Build safe mapping only hostile and dedupes."""
    cols = ["f0", _ENGINEERED, "a,b", "a_b"]  # "a,b" sanitizes to "a_b" -> collides
    mapping = build_safe_mapping(cols)
    assert "f0" not in mapping and "a_b" not in mapping  # clean names untouched
    assert mapping[_ENGINEERED] == "mul(log(f2)_sin(f3))"
    # Collision with the pre-existing clean "a_b" must be disambiguated.
    assert mapping["a,b"] != "a_b"
    assert mapping["a,b"].startswith("a_b")
    # No mapped target collides with a kept or other mapped name.
    finals = set(c for c in cols if c not in mapping) | set(mapping.values())
    assert len(finals) == len(cols)


def test_sanitize_frame_columns_renames_and_is_strict_noop_when_clean():
    """Sanitize frame columns renames and is strict noop when clean."""
    clean = pd.DataFrame({"f0": [1, 2], "f1": [3, 4]})
    assert sanitize_frame_columns(clean) is clean  # same object -> zero overhead

    dirty = pd.DataFrame({_ENGINEERED: [1.0, 2.0], "f0": [3, 4]})
    out = sanitize_frame_columns(dirty)
    assert list(out.columns) == ["mul(log(f2)_sin(f3))", "f0"]
    assert not has_hostile_name(out.columns)


def test_sanitize_name_list_passes_through_indices_and_clean():
    """Sanitize name list passes through indices and clean."""
    assert sanitize_name_list(None) is None
    assert sanitize_name_list([0, 1, 2]) == [0, 1, 2]
    clean = ["cat_0", "cat_1"]
    assert sanitize_name_list(clean) is clean
    assert sanitize_name_list(["x", "a,b"]) == ["x", "a_b"]


def _toy_xy(seed=0):
    """Toy xy."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(120, 3)),
        columns=[_ENGINEERED, "f0", "f1"],  # first col carries a hostile name
    )
    y = (X["f0"] + X["f1"] > 0).astype(int)
    return X, y


def test_lightgbm_rejects_hostile_name_then_accepts_sanitized():
    """The actual crash + the fix: a comma-named feature trips LightGBM, and
    sanitizing the frame columns lets the identical data fit cleanly."""
    lgb = pytest.importorskip("lightgbm")

    X, y = _toy_xy()
    # Pre-sanitize: LightGBM refuses the JSON-special feature name.
    with pytest.raises(Exception) as exc:
        lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
    assert "JSON" in str(exc.value) or "special" in str(exc.value).lower()

    # Post-sanitize: same values, safe labels -> fits, and fit/predict are
    # consistent because the map is a pure function of the names.
    Xs = sanitize_frame_columns(X)
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(Xs, y)
    preds = model.predict(sanitize_frame_columns(X.copy()))
    assert len(preds) == len(y)


def test_xgboost_rejects_bracket_name_then_accepts_sanitized():
    """Xgboost rejects bracket name then accepts sanitized."""
    xgb = pytest.importorskip("xgboost")

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["emb[0]", "f0"])
    y = (X["f0"] > 0).astype(int)
    with pytest.raises(ValueError):
        xgb.XGBClassifier(n_estimators=5, verbosity=0).fit(X, y)
    Xs = sanitize_frame_columns(X)
    assert not has_hostile_name(Xs.columns)
    xgb.XGBClassifier(n_estimators=5, verbosity=0).fit(Xs, y)


def test_polars_frame_columns_sanitized_if_available():
    """Polars frame columns sanitized if available."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({_ENGINEERED: [1.0, 2.0], "f0": [3, 4]})
    out = sanitize_frame_columns(df)
    assert not has_hostile_name(out.columns)
    assert "mul(log(f2)_sin(f3))" in out.columns

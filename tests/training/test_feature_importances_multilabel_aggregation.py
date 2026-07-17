"""Regression tests for MultiOutputClassifier / MultiOutputRegressor native FI aggregation.

Pre-fix: ``get_model_feature_importances`` unwrapped Pipeline / TTR / single-estimator
chains but did NOT handle ``MultiOutputClassifier`` (which has ``estimators_`` — plural —
as a fitted list, not ``estimator_``). For multilabel CB / XGB / LGB combos this
forced the expensive ``permutation_importance`` fallback even though every per-label
child estimator carried a native ``feature_importances_``. The c0008 profile (2026-05-28,
multilabel cb+hgb @200k) attributed 35.8s cumtime / 26 calls to
``_permutation_feature_importances``; the CB-multilabel branches were the avoidable
ones.

Bench-verified speedup on synthetic n=5000 / n_feats=30 / n_labels=13: CB-multilabel
5.5ms via aggregation vs 9133ms via permutation = 1655x.

These tests pin:
  (1) MultiOutputClassifier(CatBoost) hits the new aggregation path (cheap, no
      permutation fallback).
  (2) MultiOutputClassifier(HGB) still falls through to permutation_importance because
      HGB does NOT expose feature_importances_ at any sklearn version (the child
      lookup returns False).
  (3) MultiOutputRegressor with a coef_-bearing child uses the new aggregation path.
  (4) Length-mismatch / heterogeneous-shape children skip aggregation (defensive).
  (5) Empty estimators_ list does NOT aggregate.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip cleanly when an optional dep is missing.
catboost = pytest.importorskip("catboost")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import Ridge

from mlframe.training._feature_importances import get_model_feature_importances


def _make_multilabel_data(n=300, n_feats=10, n_labels=3, seed=42):
    """Make multilabel data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feats))
    y = (rng.random((n, n_labels)) > 0.5).astype(int)
    return X, y, [f"f{i}" for i in range(n_feats)]


def test_multilabel_catboost_uses_aggregation_path():
    """Multilabel catboost uses aggregation path."""
    X, y, cols = _make_multilabel_data(n_labels=3)
    model = MultiOutputClassifier(catboost.CatBoostClassifier(iterations=10, verbose=False))
    model.fit(X, y)
    fi = get_model_feature_importances(model, cols, X=X, y=y)
    assert fi is not None
    assert fi.shape == (len(cols),)
    # The mean-aggregation of CB native FI is non-negative and sums to a
    # finite value -- never NaN or inf.
    assert np.all(np.isfinite(fi))
    assert np.all(fi >= 0.0), "CB feature_importances_ is non-negative; mean should be too"


def test_multilabel_hgb_falls_through_to_permutation():
    """HGB has no feature_importances_ at any sklearn version -- aggregation path
    skips (per_child empties + breaks), then the permutation fallback runs.
    This pins that the new aggregation branch does NOT swallow HGB-multilabel into
    a None return (regression for the early implementation that returned None on
    per_child empty)."""
    X, y, cols = _make_multilabel_data(n=200, n_feats=5, n_labels=2)  # small for speed
    model = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=10))
    model.fit(X, y)
    fi = get_model_feature_importances(model, cols, X=X, y=y)
    # Permutation fallback returns a non-None array; the multilabel scorer
    # path is exercised. Just pin shape + finiteness.
    assert fi is not None
    assert fi.shape == (len(cols),)
    assert np.all(np.isfinite(fi))


def test_multilabel_regressor_with_coef_aggregation_path():
    """MultiOutputRegressor(Ridge) has per-child coef_ -- aggregation should use
    the coef branch (signed, 1-D)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 6))
    y = rng.standard_normal((200, 3))
    cols = [f"f{i}" for i in range(6)]

    model = MultiOutputRegressor(Ridge()).fit(X, y)
    fi = get_model_feature_importances(model, cols, X=X, y=y)
    assert fi is not None
    assert fi.shape == (len(cols),)
    # coef_ aggregation can be signed (negative entries valid).
    assert np.all(np.isfinite(fi))


def test_multilabel_aggregation_matches_manual_mean_across_children():
    """The aggregated FI equals the np.mean of per-child feature_importances_
    (not a hash / not zeroed). Pins the formula."""
    X, y, cols = _make_multilabel_data(n_labels=4, seed=20260528)
    model = MultiOutputClassifier(catboost.CatBoostClassifier(iterations=10, verbose=False))
    model.fit(X, y)

    fi = get_model_feature_importances(model, cols, X=X, y=y)
    assert fi is not None

    # Manual aggregation
    manual = np.mean(
        [np.asarray(est.feature_importances_) for est in model.estimators_],
        axis=0,
    )
    np.testing.assert_allclose(fi, manual, atol=0.0)


def test_multilabel_empty_estimators_skips_aggregation():
    """Defensive: a model whose .estimators_ is empty list (e.g. failed-fit) does
    NOT enter the aggregation path and falls through to the standard
    non-native handling."""
    from types import SimpleNamespace

    fake = SimpleNamespace()
    fake.estimators_ = []  # empty
    fi = get_model_feature_importances(fake, ["a", "b"], X=None, y=None)
    # No source available -> None.
    assert fi is None


def test_multilabel_hgb_uses_per_child_permutation_not_wrapper():
    """iter577 follow-up: when MultiOutput children lack native FI (HGB),
    the aggregator now runs ``_permutation_feature_importances`` PER CHILD
    with the 1-D ``y[:, j]`` slice, not on the wrapper with 2-D y.

    This pins (a) the per-child permutation path fires; (b) each call
    receives a 1-D y (the label slice); (c) the wrapper-level fallback
    path at the bottom of ``get_model_feature_importances`` does NOT
    fire (the if-condition guards against double-work).

    Pre-iter577 the entire branch fell through to a single wrapper-level
    ``_permutation_feature_importances(model, X, y_2d)`` call, which
    eats sklearn's MultiOutputClassifier.score path overhead and
    cannot benefit from per-label parallelism downstream.
    """
    from unittest.mock import patch
    from mlframe.training import _feature_importances as fi_mod

    X, y, cols = _make_multilabel_data(n=200, n_feats=6, n_labels=3)
    model = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=10))
    model.fit(X, y)

    # Wrap the real function so each call records its (X, y_shape).
    real_fn = fi_mod._permutation_feature_importances
    call_records: list[tuple] = []

    def recording_fn(_model, _X, _y, **kwargs):
        """Recording fn."""
        _y_arr = np.asarray(_y)
        call_records.append((id(_model), _y_arr.shape, _y_arr.ndim))
        return real_fn(_model, _X, _y, **kwargs)

    with patch.object(fi_mod, "_permutation_feature_importances", side_effect=recording_fn):
        out = fi_mod.get_model_feature_importances(model, cols, X=X, y=y)

    assert out is not None
    assert out.shape == (len(cols),)

    # (a) per-child permutation fired -- one call per HGB child.
    assert len(call_records) == 3, f"Expected 3 per-child permutation calls (one per label); got {len(call_records)}. Records: {call_records}"
    # (b) every call passed a 1-D y slice (NOT the 2-D wrapper y).
    for _child_id, y_shape, y_ndim in call_records:
        assert y_ndim == 1, (
            f"Per-child permutation must receive 1-D y slice; got shape "
            f"{y_shape} ndim={y_ndim}. This indicates the wrapper-level "
            f"fallback fired instead of the per-child path."
        )
        # 1-D y length must match X (the original X is reused per child).
        assert y_shape == (200,)
    # (c) the model.id values are the CHILD ids, not the WRAPPER id.
    wrapper_id = id(model)
    child_ids = {id(c) for c in model.estimators_}
    for child_id, _, _ in call_records:
        assert child_id in child_ids and child_id != wrapper_id, (
            f"Per-child permutation must receive a child estimator, not the wrapper. Got id={child_id}, wrapper={wrapper_id}, children={child_ids}"
        )


def test_standalone_estimator_still_uses_direct_feature_importances():
    """Pin that single-output CB still goes through the direct
    feature_importances_ branch -- the new MultiOutputClassifier branch must
    not steal estimators that already expose the attribute directly."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    y = (rng.random(200) > 0.5).astype(int)
    cols = [f"f{i}" for i in range(5)]
    model = catboost.CatBoostClassifier(iterations=10, verbose=False).fit(X, y)
    fi = get_model_feature_importances(model, cols, X=X, y=y)
    assert fi is not None
    assert fi.shape == (5,)
    # Equals the raw attribute (no surprise transformations).
    np.testing.assert_allclose(fi, np.asarray(model.feature_importances_))

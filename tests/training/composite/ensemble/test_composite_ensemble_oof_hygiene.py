"""Regression sensors for the 2026-06-10 composite cross-target OOF audit.

- N1: sample_weight was prefix-sliced (sw[:len(y_fit)]) after a group-aware
  eval carve that returns mask-SCATTERED fit rows -> every weight aligned to
  the wrong row (silent: lengths matched).
- N2: the kfold OOF composite branch fit the inner with NO eval_set, so
  early-stopping boosters raised and the per-component except silently dropped
  every ES composite component each fold (default oof source IS kfold).
- N3: the OUTER OOF split was group-blind (shuffled KFold / row permutation),
  so same-group rows spanned refit-train and holdout and leaked into the OOF
  surface the NNLS weights consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import (
    CompositeTargetEstimator,
    compute_oof_holdout_predictions,
)
from mlframe.training.composite.ensemble import (
    _align_fit_sw,
    _carve_inner_eval_split,
)


class TestAlignFitSampleWeight:
    def test_group_mask_alignment_not_prefix(self) -> None:
        """N1: on a scattered fit_mask the aligned weights must be sw[mask],
        NOT the (wrong) prefix sw[:n_fit]."""
        sw = np.arange(10, dtype=np.float64)
        fit_mask = np.array([True, False, True, True, False, True, False, True, True, False])
        aligned = _align_fit_sw(sw, fit_mask, int(fit_mask.sum()))
        np.testing.assert_array_equal(aligned, sw[fit_mask])
        # The old prefix slice would have been a different (wrong) vector.
        assert not np.array_equal(aligned, sw[: int(fit_mask.sum())])

    def test_prefix_path_when_no_mask(self) -> None:
        sw = np.arange(10, dtype=np.float64)
        aligned = _align_fit_sw(sw, None, 6)
        np.testing.assert_array_equal(aligned, sw[:6])

    def test_carve_group_path_returns_scattered_mask(self) -> None:
        n = 2000
        X = np.arange(n).reshape(-1, 1).astype(np.float64)
        y = np.arange(n).astype(np.float64)
        g = np.repeat(np.arange(20), 100)
        out = _carve_inner_eval_split(X, y, group_ids=g, return_fit_mask=True)
        assert len(out) == 5
        fit_mask = out[4]
        assert fit_mask is not None
        # Scattered (not a contiguous prefix): the fit rows are not 0..k-1.
        assert not np.array_equal(np.nonzero(fit_mask)[0], np.arange(int(fit_mask.sum())))


class _ESComposite(BaseEstimator, RegressorMixin):
    """Inner that REQUIRES an eval_set (mimics an early-stopping booster)."""

    def fit(self, X, y, eval_set=None, sample_weight=None, **kw):
        if eval_set is None:
            raise ValueError("early stopping requires an eval_set")
        self.n_features_in_ = X.shape[1]
        self._t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._t, dtype=np.float64)


def _make_composite_component(X, y, base):
    inner = _ESComposite().fit(
        X,
        y - base,
        eval_set=(X.iloc[:10], (y - base)[:10]),
    )
    spec = {
        "transform_name": "diff",
        "base_column": "b",
        "fitted_params": {},
        "extra_base_columns": (),
    }
    wrapped = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="diff",
        base_column="b",
        transform_fitted_params={},
        y_train=y,
    )
    return wrapped, spec


class TestKfoldCompositeESCarve:
    def test_es_composite_component_survives_kfold_oof(self) -> None:
        """N2: an early-stopping composite component must survive the kfold OOF
        (it gets a carved eval_set). Pre-fix it raised every fold and was
        silently dropped, yielding an empty OOF."""
        rng = np.random.default_rng(0)
        n = 3000
        b = rng.normal(0.0, 1.0, size=n)
        feat = rng.normal(0.0, 1.0, size=n)
        y = b + 0.5 * feat + rng.normal(0.0, 0.1, size=n)
        X = pd.DataFrame({"b": b, "feat": feat})
        comp, spec = _make_composite_component(X, y, b)
        oof, y_oof, names = compute_oof_holdout_predictions(
            component_models=[comp, comp],
            component_names=["c0", "c1"],
            component_specs=[spec, spec],
            train_X=X,
            y_train_full=y,
            base_train_full_per_spec={"b": b},
            holdout_frac=0.2,
            random_state=0,
            kfold=2,
        )
        assert "c0" in names, "ES composite component dropped from kfold OOF (no eval_set carve)"
        assert oof.shape[0] > 0 and np.isfinite(oof).any()


class _GidRecordingRaw(BaseEstimator, RegressorMixin):
    """Raw component that records the group ids it was fit on vs predicted on
    (via a 'gid' feature column) so a group-blind outer split is detectable."""

    fit_gids: set = set()
    pred_gids: set = set()

    def fit(self, X, y, eval_set=None, sample_weight=None, **kw):
        type(self).fit_gids |= set(np.asarray(X["gid"]).tolist())
        self.n_features_in_ = X.shape[1]
        self._t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        type(self).pred_gids |= set(np.asarray(X["gid"]).tolist())
        return np.full(X.shape[0], self._t, dtype=np.float64)


class TestOuterSplitGroupAware:
    def test_no_group_spans_fit_and_holdout(self) -> None:
        """N3: with group_ids, no group may appear in both the refit-train and
        the OOF holdout of the outer split."""
        _GidRecordingRaw.fit_gids = set()
        _GidRecordingRaw.pred_gids = set()
        rng = np.random.default_rng(1)
        n = 4000
        gid = np.repeat(np.arange(40), 100)
        feat = rng.normal(0.0, 1.0, size=n)
        y = gid.astype(np.float64) + rng.normal(0.0, 0.1, size=n)
        X = pd.DataFrame({"gid": gid.astype(np.float64), "feat": feat})
        comp = _GidRecordingRaw()
        compute_oof_holdout_predictions(
            component_models=[comp],
            component_names=["raw0"],
            component_specs=[None],
            train_X=X,
            y_train_full=y,
            base_train_full_per_spec={},
            holdout_frac=0.25,
            random_state=0,
            kfold=1,
            group_ids=gid,
        )
        overlap = _GidRecordingRaw.fit_gids & _GidRecordingRaw.pred_gids
        assert not overlap, f"group(s) {sorted(overlap)[:5]} span outer fit and holdout (group-blind split leak)"

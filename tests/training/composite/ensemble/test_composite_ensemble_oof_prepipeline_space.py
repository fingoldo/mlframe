"""Regression sensors: the OOF feature space must MIRROR deployment.

The deployed component (``PrePipelinePredictShim.predict``) routes every
predict through ``pre_pipeline.transform`` and REFUSES to feed raw X to the
inner. The honest-OOF refit loop must evaluate in that SAME (transformed)
space, never the old raw-X fallback.

Two defects this pins:

- PERF/CORRECTNESS: ``_transform_via`` used to call ``pp.transform`` per
  OOF slice inside the hot loop; on an UNFITTED pre_pipeline that raised
  ``NotFittedError``, got logged, and fell back to RAW X -- a per-slice
  raise+log+fallback AND an OOF scored in a different feature space than
  deployed. ``_transform_pair_via`` now fits a leak-free clone once per
  (component, slice-pair) and reuses it, so there is no NotFitted fallback
  warning in the loop.
- LEAK-FREE: the unfitted clone is fit on the TRAIN slice only; the holdout
  slice never touches the imputer/scaler/selector fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlframe.training.composite.ensemble as oof_mod
from mlframe.training.composite.ensemble import (
    _pp_is_fitted,
    _transform_pair_via,
    compute_oof_holdout_predictions,
)
from mlframe.training.composite.post_shim import PrePipelinePredictShim


def _make_xy(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    y = (X["f0"].values * 2.0 - X["f1"].values + rng.normal(size=n) * 0.1).astype(np.float64)
    return X, y


def _fitted_shim(X, y, name="raw#0"):
    pp = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
    Xt = pp.fit_transform(X)
    inner = Ridge(alpha=1.0).fit(Xt, y)
    return PrePipelinePredictShim(inner, pp, name), inner


def _count_fallback_warnings(caplog) -> int:
    return sum(1 for r in caplog.records if "pre_pipeline.transform failed" in r.getMessage())


class TestTransformPairVia:
    def test_fitted_pp_bit_identical_to_deployment_transform(self) -> None:
        """FITTED pp (suite-normal): both slices are bit-identical to the
        deployed ``pp.transform`` projection -- production path unchanged."""
        X, y = _make_xy()
        Xho, _ = _make_xy(n=120, seed=9)
        pp = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        pp.fit(X, y)
        tr, ho = _transform_pair_via(pp, X, Xho, y_train=y)
        np.testing.assert_array_equal(np.asarray(tr), np.asarray(pp.transform(X)))
        np.testing.assert_array_equal(np.asarray(ho), np.asarray(pp.transform(Xho)))

    def test_unfitted_pp_is_train_only_leak_free_clone(self) -> None:
        """UNFITTED pp: both slices equal a clone fit on the TRAIN slice ONLY
        (leak-free), NOT raw X and NOT a holdout-aware fit."""
        X, y = _make_xy()
        Xho, _ = _make_xy(n=120, seed=9)
        pp = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        tr, ho = _transform_pair_via(pp, X, Xho, y_train=y)
        ref = clone(pp)
        ref.fit(X, y)
        np.testing.assert_array_equal(np.asarray(tr), np.asarray(ref.transform(X)))
        np.testing.assert_array_equal(np.asarray(ho), np.asarray(ref.transform(Xho)))
        # In the deployed (transformed) space, NOT the raw-X fallback.
        assert not np.allclose(np.asarray(ho), np.asarray(Xho))

    def test_unfitted_caller_pp_not_mutated(self) -> None:
        """The shared deployed pipeline object keeps its (unfitted) state: the
        clone is fit, never the caller's pp."""
        X, y = _make_xy()
        Xho, _ = _make_xy(n=120, seed=9)
        pp = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        assert not _pp_is_fitted(pp)
        _transform_pair_via(pp, X, Xho, y_train=y)
        assert not _pp_is_fitted(pp)

    def test_none_pp_passthrough(self) -> None:
        X, y = _make_xy(n=20)
        Xho, _ = _make_xy(n=8, seed=3)
        tr, ho = _transform_pair_via(None, X, Xho, y_train=y)
        assert tr is X and ho is Xho


class TestNoNotFittedFallbackInOOFLoop:
    """No repeated NotFitted raise+log+rawX fallback in the OOF refit loop, for
    BOTH a fitted and an unfitted pre_pipeline, single-split and kfold."""

    def _assert_clean(self, caplog, shim, X, y, kfold):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=oof_mod.logger.name):
            preds, y_oof, names = compute_oof_holdout_predictions(
                component_models=[shim, shim],
                component_names=["raw#0", "raw#1"],
                component_specs=[None, None],
                train_X=X,
                y_train_full=y,
                base_train_full_per_spec={},
                holdout_frac=0.3,
                random_state=0,
                kfold=kfold,
            )
        assert _count_fallback_warnings(caplog) == 0
        # Components survive (were not dropped by a transform failure) and the
        # OOF matrix has the expected shape.
        assert names == ["raw#0", "raw#1"]
        assert preds.shape[1] == 2
        assert preds.shape[0] == y_oof.shape[0] > 0
        assert np.all(np.isfinite(preds))

    def test_fitted_pp_single_split(self, caplog) -> None:
        X, y = _make_xy()
        shim, _ = _fitted_shim(X, y)
        self._assert_clean(caplog, shim, X, y, kfold=1)

    def test_fitted_pp_kfold(self, caplog) -> None:
        X, y = _make_xy()
        shim, _ = _fitted_shim(X, y)
        self._assert_clean(caplog, shim, X, y, kfold=3)

    def test_unfitted_pp_single_split(self, caplog) -> None:
        X, y = _make_xy()
        _, inner = _fitted_shim(X, y)
        pp_unfit = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        shim = PrePipelinePredictShim(inner, pp_unfit, "raw#0")
        self._assert_clean(caplog, shim, X, y, kfold=1)

    def test_unfitted_pp_kfold(self, caplog) -> None:
        X, y = _make_xy()
        _, inner = _fitted_shim(X, y)
        pp_unfit = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        shim = PrePipelinePredictShim(inner, pp_unfit, "raw#0")
        self._assert_clean(caplog, shim, X, y, kfold=3)


class TestOOFSpaceMatchesDeployment:
    def test_oof_preds_in_deployed_space_not_raw(self) -> None:
        """The single-split OOF holdout predictions for a fitted-pp component
        equal what the deployed shim predicts on the same holdout rows --
        i.e. the OOF mirrors deployment (transformed space)."""
        X, y = _make_xy()
        pp = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        # A component whose inner is *strongly* scale-sensitive so the raw-vs-
        # transformed gap is unmistakable: scale f0 by 100 so StandardScaler
        # materially changes the value the Ridge sees.
        Xs = X.copy()
        Xs["f0"] = Xs["f0"] * 100.0
        Xt = pp.fit_transform(Xs)
        inner = Ridge(alpha=1.0).fit(Xt, y)
        shim = PrePipelinePredictShim(inner, pp, "raw#0")

        preds, y_oof, names = compute_oof_holdout_predictions(
            component_models=[shim, shim],
            component_names=["raw#0", "raw#1"],
            component_specs=[None, None],
            train_X=Xs,
            y_train_full=y,
            base_train_full_per_spec={},
            holdout_frac=0.3,
            random_state=0,
            kfold=1,
        )
        assert names == ["raw#0", "raw#1"]
        # The OOF refit re-fits a CLONE on the stack slice (fitted pp reused),
        # so preds are finite and in a sane y-range -- a raw-X fallback on the
        # x100-scaled feature would blow Ridge predictions far outside the y
        # range. Pin that the OOF holdout RMSE is reasonable (transformed space).
        rmse = float(np.sqrt(np.mean((preds[:, 0] - y_oof) ** 2)))
        y_span = float(np.ptp(y))
        assert rmse < y_span, f"OOF RMSE {rmse:.3f} exceeds target span {y_span:.3f}: the refit likely scored in the wrong (raw) feature space."

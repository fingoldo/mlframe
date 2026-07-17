"""Regression + biz_value sensors for the recurrent-forward FUTURE findings
landed 2026-06-11.

Covered items:

- T13 (``src/mlframe/training/composite/estimator/_estimator.py``): for
  time-recurrent transforms (``ewma_residual`` / ``rolling_quantile_ratio`` /
  ``frac_diff``) the fit-time domain filter used to COMPACT the row sequence
  BEFORE the recurrent forward, so the T value at a train row near a filtered
  gap differed from the predict-time T (which is always computed on the
  uncompacted frame). The fix runs the forward over the FULL (uncompacted)
  y / base sequence and masks to the valid rows AFTER, exactly mirroring
  predict. A new ``Transform.recurrent`` flag marks the three sequence
  transforms; pointwise transforms (default ``recurrent=False``) are unaffected
  and stay bit-identical to the compacted order.

- E7 (``src/mlframe/training/composite/post_shim.py``): the
  ``PrePipelinePredictShim.fit`` ``sample_weight`` pass-through is now
  SIGNATURE-GATED (``has_fit_parameter`` / ``inspect.signature``) instead of the
  old catch-all ``except TypeError`` retry, which mis-attributed a TypeError
  raised DEEP inside a weight-AWARE inner fit to "no sample_weight support" and
  silently re-fit UNWEIGHTED -- dropping the weighting AND hiding the real error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.post_shim import PrePipelinePredictShim
from mlframe.training.composite.transforms import get_transform


# ----------------------------------------------------------------------
# Inner that captures the T it was trained on (so the test can read back the
# wrapper's fit-time t_train and compare it against the predict-time forward).
# ----------------------------------------------------------------------


class _CaptureTInner(BaseEstimator, RegressorMixin):
    """Stores the (T-scale) target the wrapper passed to ``fit`` so a test can
    assert the inner was trained on the same T values predict reconstructs."""

    def fit(self, X, y, **kw):
        self.t_train_ = np.asarray(y, dtype=np.float64).copy()
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(np.mean(self.t_train_)) if self.t_train_.size else 0.0
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._mean_t, dtype=np.float64)


def _smooth_series(n=140, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.0, 1.0, size=n)) + 50.0
    y = base + rng.normal(0.0, 0.05, size=n)
    return base, y


def _carry_forward_fill(arr):
    """Mirror of the wrapper's fit-time gap-fill: NON-FINITE rows take the last
    preceding finite value (leading non-finite back-fill). Used to build the
    full-sequence predict-consistent reference T."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1).copy()
    n = a.size
    keep = np.isfinite(a)
    if keep.all():
        return a
    idx = np.where(keep, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    first_kept = int(np.argmax(keep)) if keep.any() else 0
    idx[idx < 0] = first_kept
    return a[idx]


# ----------------------------------------------------------------------
# T13: recurrent forward runs full-sequence-then-mask, matching predict
# ----------------------------------------------------------------------


class TestT13RecurrentForwardFullThenMask:
    """For each recurrent transform, the inner's fit-time T at valid rows must
    equal the predict-time forward (full sequence) masked to those rows -- and
    must NOT equal the old compacted-forward T (which the pre-fix code used).
    """

    def _fit_capture(self, transform_name, X, y):
        est = CompositeTargetEstimator(
            base_estimator=_CaptureTInner(),
            transform_name=transform_name,
            base_column="" if not get_transform(transform_name).requires_base else "base",
            base_columns=None,
        )
        est.fit(X, y)
        return est

    def test_rolling_quantile_ratio_train_T_matches_predict_forward(self) -> None:
        """rolling_quantile_ratio: centred window shrinks across a filtered gap
        when the sequence is compacted; full-then-mask keeps each window intact.
        Drop rows by non-finite BASE (the natural domain violation)."""
        base, y = _smooth_series(seed=1)
        bad = [50, 51, 95]
        base2 = base.copy()
        base2[bad] = np.nan
        valid = np.isfinite(base2)
        X = pd.DataFrame({"base": base2})

        est = self._fit_capture("rolling_quantile_ratio", X, y)
        tr = get_transform("rolling_quantile_ratio")
        params = est.fitted_params_

        # Reference: forward over the FULL (gap-filled) frame, masked to valid --
        # row positions preserved, so each centred window stays intact.
        base_seq = _carry_forward_fill(base2)
        t_predict_ref = np.asarray(
            tr.forward(y, base_seq, params),
            dtype=np.float64,
        ).reshape(-1)[valid]
        # Old (pre-fix) reference: forward over the COMPACTED sequence.
        t_compact = np.asarray(
            tr.forward(y[valid], base2[valid], params),
            dtype=np.float64,
        ).reshape(-1)

        captured = est.estimator_.t_train_
        assert captured.shape == t_predict_ref.shape
        # POST-FIX: the inner trained on the full-then-mask T (== predict-time).
        np.testing.assert_allclose(captured, t_predict_ref, rtol=0, atol=1e-9)
        # DISCRIMINATOR: the compacted T differs near the gaps, so a regression
        # back to compact-then-forward would break this test.
        assert np.max(np.abs(t_predict_ref - t_compact)) > 1e-3, "fixture must produce a measurable compact-vs-full divergence"
        assert np.max(np.abs(captured - t_compact)) > 1e-3, "captured T must NOT equal the pre-fix compacted T"

    def test_frac_diff_train_T_matches_predict_forward(self) -> None:
        """frac_diff: the weight tail convolves each row with its lagged
        predecessors; compacting re-aligns the tail onto the wrong rows. This
        transform is requires_base=False (y-only), so the domain filter drops by
        NON-FINITE y -- the recurrent y-convolution then diverges between the
        compacted and the full (gap-filled) sequence."""
        base, y = _smooth_series(seed=2)
        bad = [40, 41, 90, 91]
        base2 = base.copy()
        y2 = y.copy()
        y2[bad] = np.nan
        valid = np.isfinite(y2)
        X = pd.DataFrame({"base": base2})

        est = CompositeTargetEstimator(
            base_estimator=_CaptureTInner(),
            transform_name="frac_diff",
        )
        est.fit(X, y2)
        tr = get_transform("frac_diff")
        params = est.fitted_params_
        zeros = np.zeros_like(y2)
        # frac_diff convolves over y; the dropped (NaN-y) rows carry-forward so
        # the convolution stays finite and position-preserving.
        y_seq = _carry_forward_fill(y2)
        t_predict_ref = np.asarray(
            tr.forward(y_seq, zeros, params),
            dtype=np.float64,
        ).reshape(-1)[valid]
        t_compact = np.asarray(
            tr.forward(y2[valid], zeros[valid], params),
            dtype=np.float64,
        ).reshape(-1)

        captured = est.estimator_.t_train_
        assert captured.shape == t_predict_ref.shape
        np.testing.assert_allclose(captured, t_predict_ref, rtol=0, atol=1e-9)
        assert np.max(np.abs(t_predict_ref - t_compact)) > 1e-2, "fixture must produce a measurable compact-vs-full divergence"
        assert np.max(np.abs(captured - t_compact)) > 1e-2

    def test_ewma_residual_train_T_matches_predict_forward(self) -> None:
        """ewma_residual: the EWMA carry-forward over a NON-FINITE base makes a
        base-NaN drop equivalent to compaction, so to surface the divergence we
        drop rows by NON-FINITE y (finite base) -- the EWMA full-sequence still
        consumes those base values, the compacted sequence does not."""
        base, y = _smooth_series(seed=3)
        bad = [45, 46, 100]
        y2 = y.copy()
        y2[bad] = np.nan
        valid = np.isfinite(y2) & np.isfinite(base)
        X = pd.DataFrame({"base": base})

        est = CompositeTargetEstimator(
            base_estimator=_CaptureTInner(),
            transform_name="ewma_residual",
            base_column="base",
        )
        est.fit(X, y2)
        tr = get_transform("ewma_residual")
        params = est.fitted_params_
        t_predict_ref = np.asarray(
            tr.forward(y2, base, params),
            dtype=np.float64,
        ).reshape(-1)[valid]
        t_compact = np.asarray(
            tr.forward(y2[valid], base[valid], params),
            dtype=np.float64,
        ).reshape(-1)

        captured = est.estimator_.t_train_
        assert captured.shape == t_predict_ref.shape
        np.testing.assert_allclose(captured, t_predict_ref, rtol=0, atol=1e-9)
        assert np.max(np.abs(t_predict_ref - t_compact)) > 1e-2, "fixture must produce a measurable compact-vs-full divergence"
        assert np.max(np.abs(captured - t_compact)) > 1e-2

    def test_pointwise_transform_unaffected_by_full_then_mask(self) -> None:
        """Behaviour-preservation: a pointwise transform (diff, recurrent=False)
        gets the forward applied to the already-filtered rows -- bit-identical to
        the full-then-mask order. Captured T must equal BOTH references."""
        base, y = _smooth_series(seed=4)
        bad = [30, 31, 70]
        base2 = base.copy()
        base2[bad] = np.nan
        valid = np.isfinite(base2)
        X = pd.DataFrame({"base": base2})

        est = self._fit_capture("diff", X, y)
        tr = get_transform("diff")
        params = est.fitted_params_
        t_full_masked = np.asarray(
            tr.forward(y, base2, params),
            dtype=np.float64,
        ).reshape(-1)[valid]
        t_compact = np.asarray(
            tr.forward(y[valid], base2[valid], params),
            dtype=np.float64,
        ).reshape(-1)
        # For a pointwise transform the two orders agree exactly.
        np.testing.assert_allclose(t_full_masked, t_compact, rtol=0, atol=0)
        np.testing.assert_allclose(
            est.estimator_.t_train_,
            t_compact,
            rtol=0,
            atol=0,
        )

    def test_biz_value_recurrent_predict_consistency_improves(self) -> None:
        """biz_value (T13): with the full-then-mask fix, the inner is trained on
        T values that are *consistent* with the predict-time forward, so an
        identity-style inner reproduces y on the train rows. We measure the gap
        between (a) the wrapper's stored train-T and (b) the predict-time forward
        T. Post-fix that gap is ~0; the pre-fix compacted path left a gap > 1e-2
        on the rolling transform near every filtered boundary.
        """
        base, y = _smooth_series(n=200, seed=9)
        # Several scattered gaps so multiple windows straddle a boundary.
        bad = [20, 21, 60, 61, 120, 160, 161]
        base2 = base.copy()
        base2[bad] = np.nan
        valid = np.isfinite(base2)
        X = pd.DataFrame({"base": base2})

        est = self._fit_capture("rolling_quantile_ratio", X, y)
        tr = get_transform("rolling_quantile_ratio")
        params = est.fitted_params_
        base_seq = _carry_forward_fill(base2)
        t_predict_ref = np.asarray(
            tr.forward(y, base_seq, params),
            dtype=np.float64,
        ).reshape(-1)[valid]
        post_fix_gap = float(np.max(np.abs(est.estimator_.t_train_ - t_predict_ref)))

        t_compact = np.asarray(
            tr.forward(y[valid], base2[valid], params),
            dtype=np.float64,
        ).reshape(-1)
        pre_fix_gap = float(np.max(np.abs(t_compact - t_predict_ref)))

        assert post_fix_gap <= 1e-9, f"post-fix train/predict T gap {post_fix_gap}"
        # The win: the gap collapsed by at least 1e7x relative to pre-fix.
        assert pre_fix_gap > 1e-2, f"pre-fix gap should be material, got {pre_fix_gap}"
        assert post_fix_gap < pre_fix_gap / 1e6


# ----------------------------------------------------------------------
# E7 (post_shim): signature-gated sample_weight
# ----------------------------------------------------------------------


class _ShimWeightAwareButBuggyInner(BaseEstimator, RegressorMixin):
    """Declares ``sample_weight`` (IS weight-aware) but raises a deep TypeError
    inside fit. E7: that TypeError must PROPAGATE on the FIRST (weighted) call --
    the shim must NOT trigger an unweighted retry."""

    fit_call_count = 0

    def fit(self, X, y, sample_weight=None, **kw):
        type(self).fit_call_count += 1
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        raise TypeError("deep boom: simulated downstream dtype error inside fit")

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float64)


class _ShimWeightAwareInner(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None, **kw):
        self.got_sample_weight_ = sample_weight is not None
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self._m = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        return np.full(len(X), self._m, dtype=np.float64)


class _ShimNoWeightInner(BaseEstimator, RegressorMixin):
    def fit(self, X, y, **kw):  # no sample_weight, no **kwargs swallow
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self._m = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        return np.full(len(X), self._m, dtype=np.float64)


class TestE7ShimSampleWeightSignatureGate:
    def _xy(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = rng.normal(size=n)
        sw = np.abs(rng.normal(1.0, 0.1, size=n))
        return X, y, sw

    def test_deep_typeerror_in_weight_aware_inner_propagates_no_retry(self) -> None:
        """E7: a deep TypeError from a weight-AWARE inner.fit must propagate on
        the FIRST call -- pre-fix the ``except TypeError`` retry re-fit UNWEIGHTED
        (2 fit calls, weighting silently dropped); post-fix the gate keeps it to
        exactly ONE call and re-raises."""
        X, y, sw = self._xy()
        _ShimWeightAwareButBuggyInner.fit_call_count = 0
        shim = PrePipelinePredictShim(
            model=_ShimWeightAwareButBuggyInner(),
            pre_pipeline=None,
            name="t",
        )
        with pytest.raises(TypeError, match="deep boom"):
            shim.fit(X, y, sample_weight=sw)
        assert _ShimWeightAwareButBuggyInner.fit_call_count == 1, (
            f"expected 1 inner.fit call (no retry); got {_ShimWeightAwareButBuggyInner.fit_call_count} (pre-fix retry bug)"
        )

    def test_weight_aware_inner_receives_sample_weight(self) -> None:
        X, y, sw = self._xy(seed=1)
        inner = _ShimWeightAwareInner()
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=None, name="t")
        shim.fit(X, y, sample_weight=sw)
        assert inner.got_sample_weight_ is True

    def test_weight_unaware_inner_falls_back_cleanly(self) -> None:
        X, y, sw = self._xy(seed=2)
        shim = PrePipelinePredictShim(
            model=_ShimNoWeightInner(),
            pre_pipeline=None,
            name="t",
        )
        shim.fit(X, y, sample_weight=sw)  # gated OUT, no exception
        y_hat = shim.predict(X)
        assert y_hat.shape == (len(X),)
        assert np.all(np.isfinite(y_hat))

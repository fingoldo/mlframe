"""Regression + biz_value sensors for the deferred (FUTURE) composite-estimator
audit findings landed 2026-06-10/11.

Covered items (all in ``src/mlframe/training/composite/estimator/_estimator.py``
plus its sibling ``_utils.py``):

- E7  : ``sample_weight`` pass-through is now SIGNATURE-GATED (inspect.signature /
        sklearn ``has_fit_parameter``) instead of the old catch-all
        ``except TypeError`` retry. The old retry mis-attributed a TypeError
        raised DEEP inside a weight-AWARE inner fit to "no sample_weight
        support" and silently re-fit UNWEIGHTED. Pinned both ways: a deep
        TypeError must now PROPAGATE (not be swallowed), and a genuinely
        weight-unaware inner must still fall back cleanly.
- E10 : ``from_fitted_inner`` reconstructs the EXACT T-train envelope for unary
        transforms (``requires_base=False``) via ``transform.forward(y, 0, params)``
        and applies the MAD clip, instead of a symmetric ``+/-10*std(y)`` envelope
        centered at 0 that mis-centers offset unary transforms (log_y / cbrt_y /
        yeo_johnson_y) and clips in-distribution T_hat flat. biz_value: an
        in-distribution prediction round-trips to ~y instead of being clipped
        ~60x off.
- E11 : grouped transforms keep the sklearn invariant
        ``n_features_in_ == len(feature_names_in_)``. Pre-fix the wrapper exposed
        ``feature_names_in_`` WITH ``group_column`` (F cols) but delegated
        ``n_features_in_`` to the inner fit on F-1 cols.
- DX15: the runtime-bound public surface (update / get_buffer_state /
        predict_pre_clip / get_booster + the 5 delegated properties) is now
        defined in-body so it is discoverable to mypy / IDE / help().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform


# ----------------------------------------------------------------------
# Inner mocks
# ----------------------------------------------------------------------


class _ConstInner(BaseEstimator, RegressorMixin):
    """Predicts a fixed T-scale value (defaults to mean of training T)."""

    def __init__(self, t_value: float | None = None):
        self.t_value = t_value

    def fit(self, X, y, **kw):
        """Fit."""
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(self.t_value) if self.t_value is not None else float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Predict."""
        return np.full(X.shape[0], self._mean_t, dtype=np.float64)


class _WeightAwareButBuggyInner(BaseEstimator, RegressorMixin):
    """Inner whose ``fit`` DECLARES ``sample_weight`` (so it IS weight-aware)
    but raises a TypeError DEEP inside fit (simulating a downstream dtype/shape
    bug). E7: this TypeError must PROPAGATE on the FIRST (weighted) call -- the
    gate must NOT trigger an unweighted retry.

    ``fit_call_count`` is a CLASS-level counter (the wrapper clones the
    prototype, so per-instance state is lost; the class counter survives the
    clone). Pre-fix the ``except TypeError`` retry called fit TWICE (silent
    unweighted re-fit); post-fix the signature gate keeps fit to ONE call.
    """

    fit_call_count = 0

    def fit(self, X, y, sample_weight=None, **kw):
        """Fit."""
        type(self).fit_call_count += 1
        self.n_features_in_ = X.shape[1]
        # Simulate a deep TypeError unrelated to sample_weight acceptance.
        raise TypeError("deep boom: simulated downstream dtype error inside fit")

    def predict(self, X):
        """Predict."""
        return np.zeros(X.shape[0], dtype=np.float64)


class _WeightAwareInner(BaseEstimator, RegressorMixin):
    """Records whether it received sample_weight (and that it was non-None)."""

    def fit(self, X, y, sample_weight=None, **kw):
        """Fit."""
        self.n_features_in_ = X.shape[1]
        self.got_sample_weight_ = sample_weight is not None
        self._mean_t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Predict."""
        return np.full(X.shape[0], self._mean_t, dtype=np.float64)


class _NoWeightInner(BaseEstimator, RegressorMixin):
    """``fit`` does NOT declare sample_weight and would TypeError if passed one
    (the genuine 'unweighted estimator' case that must still fall back cleanly).
    """

    def fit(self, X, y, **kw):  # no sample_weight, no **kwargs swallow of it
        """Fit."""
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Predict."""
        return np.full(X.shape[0], self._mean_t, dtype=np.float64)


def _diff_frame(n=200, seed=0):
    """Diff frame."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    y = base + 0.5 * feat + rng.normal(0.0, 0.1, size=n)
    X = pd.DataFrame({"base": base, "feat": feat})
    return X, y


# ----------------------------------------------------------------------
# E7: signature-gated sample_weight
# ----------------------------------------------------------------------


class TestE7SampleWeightSignatureGate:
    """Groups tests covering e7 sample weight signature gate."""
    def test_deep_typeerror_in_weight_aware_inner_propagates_no_retry(self) -> None:
        """E7: a TypeError raised DEEP inside a weight-AWARE inner.fit must
        propagate on the FIRST call -- no silent unweighted retry. Pre-fix the
        ``except TypeError`` retry called inner.fit TWICE (silently dropping the
        weighting); post-fix the signature gate keeps it to exactly ONE call.
        Asserting the call count makes this discriminate pre-fix vs post-fix
        (both raise TypeError, but only pre-fix retries)."""
        X, y = _diff_frame()
        sw = np.abs(np.random.default_rng(1).normal(1.0, 0.1, size=len(y)))
        _WeightAwareButBuggyInner.fit_call_count = 0
        est = CompositeTargetEstimator(
            base_estimator=_WeightAwareButBuggyInner(),
            transform_name="diff",
            base_column="base",
        )
        with pytest.raises(TypeError, match="deep boom"):
            est.fit(X, y, sample_weight=sw)
        # Exactly one fit attempt -- the gate did NOT trigger an unweighted retry.
        assert _WeightAwareButBuggyInner.fit_call_count == 1, (
            f"expected 1 inner.fit call (no retry); got {_WeightAwareButBuggyInner.fit_call_count} (pre-fix retry bug)"
        )

    def test_weight_aware_inner_receives_sample_weight(self) -> None:
        """A weight-aware inner gets the sample_weight kwarg threaded through."""
        X, y = _diff_frame()
        sw = np.abs(np.random.default_rng(2).normal(1.0, 0.1, size=len(y)))
        est = CompositeTargetEstimator(
            base_estimator=_WeightAwareInner(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X, y, sample_weight=sw)
        assert est.estimator_.got_sample_weight_ is True

    def test_weight_unaware_inner_falls_back_cleanly(self) -> None:
        """The genuine 'inner has no sample_weight param' case still works
        (gated OUT, no exception)."""
        X, y = _diff_frame()
        sw = np.abs(np.random.default_rng(3).normal(1.0, 0.1, size=len(y)))
        est = CompositeTargetEstimator(
            base_estimator=_NoWeightInner(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X, y, sample_weight=sw)  # pre-fix: also worked (via retry); now via gate
        y_hat = est.predict(X)
        assert y_hat.shape == (len(X),)
        assert np.all(np.isfinite(y_hat))


# ----------------------------------------------------------------------
# E10: from_fitted_inner unary T-clip centering
# ----------------------------------------------------------------------


class TestE10FromFittedInnerUnaryTClip:
    """Groups tests covering e10 from fitted inner unary t clip."""
    def _fit_log_y_params(self, y):
        """Fit log y params."""
        tr = get_transform("log_y")
        return dict(tr.fit(y, np.zeros_like(y)))

    def test_unary_t_clip_centered_on_true_T_not_zero(self) -> None:
        """E10: for a unary offset transform (log_y), the reconstructed T-clip
        envelope must bracket the TRUE T-train distribution (median of T), NOT
        be a symmetric +/-10*std(y) band centered at 0."""
        rng = np.random.default_rng(7)
        y = 1000.0 + rng.normal(0.0, 0.3, size=500)  # tight; log(y) ~ 6.9, far from 0
        params = self._fit_log_y_params(y)
        tr = get_transform("log_y")
        T = tr.forward(y, np.zeros_like(y), params)
        t_med = float(np.median(T))

        inner = _ConstInner(t_value=t_med).fit(
            pd.DataFrame({"f": np.zeros(3)}),
            np.zeros(3),
        )
        w = CompositeTargetEstimator.from_fitted_inner(
            inner,
            "log_y",
            "",
            params,
            y,
        )
        lo = w.fitted_params_["t_clip_low"]
        hi = w.fitted_params_["t_clip_high"]
        # Envelope brackets the true median (centered on ~6.9, not 0).
        assert lo <= t_med <= hi
        # And it is NOT the old symmetric +/-10*std(y) band (which here is
        # +/- ~2.8, clipping the true T=6.9 OUT). The fixed low bound must be
        # well above the old -2.8 (i.e. positive, near the true T).
        old_symmetric_low = -10.0 * float(np.std(y))
        assert lo > old_symmetric_low + 1.0, f"t_clip_low={lo} looks like the old symmetric envelope ({old_symmetric_low})"

    def test_biz_value_in_distribution_roundtrip_not_clipped_flat(self) -> None:
        """biz_value (E10): an in-distribution unary prediction must round-trip
        to ~y. The PRE-FIX symmetric +/-10*std(y) band (~+/-2.8 here) clips the
        in-distribution T_hat=6.9 down to 2.8 -> y_hat=exp(2.8)-offset ~ 16,
        catastrophically wrong (~60x off). Measured post-fix y_hat ~= 1000."""
        rng = np.random.default_rng(11)
        y = 1000.0 + rng.normal(0.0, 0.3, size=500)
        params = self._fit_log_y_params(y)
        tr = get_transform("log_y")
        T = tr.forward(y, np.zeros_like(y), params)
        t_med = float(np.median(T))

        inner = _ConstInner(t_value=t_med).fit(
            pd.DataFrame({"f": np.zeros(3)}),
            np.zeros(3),
        )
        w = CompositeTargetEstimator.from_fitted_inner(
            inner,
            "log_y",
            "",
            params,
            y,
        )
        X_pred = pd.DataFrame({"f": np.zeros(10)})
        y_hat = w.predict(X_pred)
        # Post-fix: round-trips to ~1000 (median of y). Floor well above the
        # ~16 the pre-fix symmetric clip produced.
        assert np.all(y_hat > 500.0), f"in-distribution unary prediction clipped flat: y_hat={y_hat[:3]} (pre-fix symmetric T-clip bug)"
        assert np.all(np.abs(y_hat - 1000.0) < 50.0), f"y_hat should round-trip to ~1000; got {y_hat[:3]}"

    def test_base_dependent_transform_keeps_symmetric_proxy(self) -> None:
        """Behaviour-preservation: a base-dependent transform (diff) on the
        from_fitted_inner route keeps the conservative symmetric +/-10*std(y)
        envelope centered at 0 (T = y - base has mean ~0 by construction). The
        E10 fix must NOT alter this path."""
        rng = np.random.default_rng(5)
        y = rng.normal(0.0, 2.0, size=300)
        inner = _ConstInner(t_value=0.0).fit(
            pd.DataFrame({"f": np.zeros(3)}),
            np.zeros(3),
        )
        w = CompositeTargetEstimator.from_fitted_inner(
            inner,
            "diff",
            "base",
            {},
            y,
        )
        lo = w.fitted_params_["t_clip_low"]
        hi = w.fitted_params_["t_clip_high"]
        expected = 10.0 * float(np.std(y))
        assert lo == pytest.approx(-expected)
        assert hi == pytest.approx(+expected)


# ----------------------------------------------------------------------
# E11: grouped n_features_in_ invariant
# ----------------------------------------------------------------------


class TestE11GroupedNFeaturesInvariant:
    """Groups tests covering e11 grouped n features invariant."""
    def _grouped_frame(self, n=400, seed=0):
        """Grouped frame."""
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 5, size=n).astype(str)
        base = rng.normal(0.0, 1.0, size=n)
        feat = rng.normal(0.0, 1.0, size=n)
        y = 2.0 * base + 0.3 * feat + rng.normal(0.0, 0.1, size=n)
        X = pd.DataFrame({"base": base, "feat": feat, "grp": g})
        return X, y

    def test_grouped_n_features_in_matches_feature_names_in(self) -> None:
        """E11: ``n_features_in_`` must equal ``len(feature_names_in_)`` for a
        grouped transform. Pre-fix the wrapper exposed 3 feature names (incl.
        group_column) but n_features_in_ delegated to the inner (fit on 2)."""
        X, y = self._grouped_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstInner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
        )
        est.fit(X, y)
        assert len(est.feature_names_in_) == 3
        # Inner was fit on F-1 (group col dropped).
        assert est.estimator_.n_features_in_ == 2
        # Wrapper invariant restored.
        assert est.n_features_in_ == len(est.feature_names_in_) == 3

    def test_ungrouped_n_features_in_unchanged(self) -> None:
        """Behaviour-preservation: ungrouped transform keeps the inner's count
        (== feature_names_in_ length, no drop)."""
        X, y = _diff_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstInner(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X, y)
        assert est.n_features_in_ == len(est.feature_names_in_) == 2

    def test_n_features_in_none_pre_fit(self) -> None:
        """Pre-fit n_features_in_ stays None (long-standing mlframe convention)."""
        est = CompositeTargetEstimator(
            base_estimator=_ConstInner(),
            transform_name="diff",
            base_column="base",
        )
        assert est.n_features_in_ is None


# ----------------------------------------------------------------------
# DX15: in-body discoverability of the public surface
# ----------------------------------------------------------------------


class TestDX15InBodySurface:
    """Groups tests covering d x15 in body surface."""
    @pytest.mark.parametrize(
        "name",
        ["update", "get_buffer_state", "predict_pre_clip", "get_booster", "predict", "predict_quantile"],
    )
    def test_method_defined_in_class_body(self, name) -> None:
        """The public methods are defined in the class body (DX15) so they are
        discoverable to mypy / IDE / help(), not only runtime-bound."""
        assert name in CompositeTargetEstimator.__dict__, f"{name} not defined in class body (runtime-bound only?)"
        assert callable(CompositeTargetEstimator.__dict__[name])

    @pytest.mark.parametrize(
        "name",
        ["feature_importances_", "coef_", "intercept_", "booster_", "n_features_in_"],
    )
    def test_property_defined_in_class_body(self, name) -> None:
        """The delegated sklearn-convention properties are defined in the class
        body as ``property`` objects (DX15)."""
        attr = CompositeTargetEstimator.__dict__.get(name)
        assert isinstance(attr, property), f"{name} should be an in-body property, got {type(attr)!r}"

    def test_in_body_property_still_delegates(self) -> None:
        """The in-body property bodies must still delegate to the carved
        implementation (behaviour preserved)."""
        from sklearn.linear_model import LinearRegression

        X, y = _diff_frame()
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X, y)
        # LinearRegression exposes coef_ / intercept_; the wrapper delegates.
        assert est.coef_ is not None
        assert est.intercept_ is not None

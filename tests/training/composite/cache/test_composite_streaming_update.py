"""Tests for ``CompositeTargetEstimator.update`` (R10c follow-up OPEN-4; rolling-buffer streaming alpha refit).

The wrapper carries a rolling buffer of last-N (y, base) observations across ``update()`` calls. When the buffer fills + drift z-score crosses threshold, the wrapper's ``fitted_params_["alpha"]`` / ``["beta"]`` get updated in-place. Default OFF (sklearn.clone() semantics: stateful estimator clones lose buffer; must be explicit opt-in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from sklearn.base import clone

from mlframe.training.composite import (
    CompositeTargetEstimator,
    _linear_residual_fit,
)


def _fit_basic_wrapper(online: bool = False, **kwargs) -> CompositeTargetEstimator:
    """Helper: fit a CompositeTargetEstimator on a stable AR(1)-like DGP."""
    rng = np.random.default_rng(0)
    n = 500
    base = rng.normal(loc=10.0, scale=2.0, size=n)
    y = 0.95 * base + 1.0 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"b1": base, "x_other": rng.normal(size=n)})
    inner = lgb.LGBMRegressor(n_estimators=20, num_leaves=7, verbose=-1)
    wrap = CompositeTargetEstimator(
        base_estimator=inner,
        transform_name="linear_residual",
        base_column="b1",
        online_refit_enabled=online,
        **kwargs,
    )
    wrap.fit(df, y)
    return wrap


# ===========================================================================
# Default behavior: update() raises when disabled
# ===========================================================================


class TestDefaultDisabled:
    def test_update_raises_when_online_disabled(self) -> None:
        wrap = _fit_basic_wrapper(online=False)
        with pytest.raises(RuntimeError, match="online_refit_enabled is False"):
            wrap.update(y_recent=np.array([1.0, 2.0]), base_recent=np.array([1.0, 2.0]))


# ===========================================================================
# Non-linear_residual transforms reject update()
# ===========================================================================


class TestUnsupportedTransform:
    def test_update_raises_for_non_linear_residual(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"b1": rng.uniform(1, 10, size=300)})
        y = df["b1"].values + rng.normal(scale=0.1, size=300)
        inner = lgb.LGBMRegressor(n_estimators=10, num_leaves=5, verbose=-1)
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="diff",
            base_column="b1",
            online_refit_enabled=True,
        )
        wrap.fit(df, y)
        with pytest.raises(NotImplementedError, match="only supported for 'linear_residual'"):
            wrap.update(y_recent=np.array([1.0]), base_recent=np.array([1.0]))


# ===========================================================================
# Buffer behavior
# ===========================================================================


class TestBuffer:
    def test_buffer_grows_across_updates(self) -> None:
        wrap = _fit_basic_wrapper(online=True, online_refit_min_buffer_n=500, online_refit_buffer_n=1000)
        # Empty buffer initially.
        state = wrap.get_buffer_state()
        assert state["buffer_n"] == 0
        # After 100 updates of 5 rows each = 500 rows in buffer.
        rng = np.random.default_rng(1)
        for _ in range(100):
            wrap.update(
                y_recent=rng.normal(loc=10, scale=2, size=5),
                base_recent=rng.normal(loc=10, scale=2, size=5),
            )
        state = wrap.get_buffer_state()
        assert state["buffer_n"] == 500

    def test_buffer_evicts_oldest_at_capacity(self) -> None:
        wrap = _fit_basic_wrapper(online=True, online_refit_buffer_n=100)
        rng = np.random.default_rng(2)
        # Append 250 rows -> should cap at 100.
        wrap.update(
            y_recent=rng.normal(size=150),
            base_recent=rng.normal(size=150),
        )
        wrap.update(
            y_recent=rng.normal(size=100),
            base_recent=rng.normal(size=100),
        )
        state = wrap.get_buffer_state()
        assert state["buffer_n"] == 100  # FIFO eviction worked

    def test_length_mismatch_raises(self) -> None:
        wrap = _fit_basic_wrapper(online=True)
        with pytest.raises(ValueError, match="equal length"):
            wrap.update(y_recent=np.array([1.0, 2.0]), base_recent=np.array([1.0]))


# ===========================================================================
# Drift trigger + in-place refit
# ===========================================================================


class TestDriftTrigger:
    def test_no_drift_keeps_alpha_unchanged(self) -> None:
        """Same-distribution updates -> no drift -> fitted_params not modified."""
        wrap = _fit_basic_wrapper(online=True, online_refit_min_buffer_n=200)
        alpha_before = float(wrap.fitted_params_["alpha"])
        # Stream same-distribution data (alpha ~ 0.95, sigma ~ 0.3).
        rng = np.random.default_rng(3)
        for _ in range(50):
            base_new = rng.normal(loc=10, scale=2, size=10)
            y_new = 0.95 * base_new + 1.0 + rng.normal(scale=0.3, size=10)
            info = wrap.update(y_recent=y_new, base_recent=base_new)
        alpha_after = float(wrap.fitted_params_["alpha"])
        # Z-score should be small -> no refit; alpha unchanged.
        assert info["refit"] is False
        assert alpha_after == alpha_before

    def test_drift_triggers_refit(self) -> None:
        """Updates from a SHIFTED distribution (alpha changed) trigger refit AT LEAST ONCE; fitted_params updates in-place. Note: once the first refit fires, subsequent calls see consistent alpha (no second drift) so we track ANY refit across iterations, not just the LAST."""
        wrap = _fit_basic_wrapper(online=True, online_refit_min_buffer_n=200, online_refit_buffer_n=500)
        alpha_before = float(wrap.fitted_params_["alpha"])
        # Stream from a DIFFERENT regime: alpha=2.0 instead of 0.95.
        rng = np.random.default_rng(4)
        any_refit = False
        last_info = None
        for _ in range(30):
            base_new = rng.normal(loc=10, scale=2, size=20)
            y_new = 2.0 * base_new + rng.normal(scale=0.3, size=20)
            last_info = wrap.update(y_recent=y_new, base_recent=base_new)
            if last_info["refit"]:
                any_refit = True
        alpha_after = float(wrap.fitted_params_["alpha"])
        assert any_refit, f"expected drift to trigger refit at least once across 30 iterations; last_info={last_info}"
        # New alpha close to 2.0 (the new regime), not 0.95 (the old one).
        assert abs(alpha_after - 2.0) < 0.2
        assert abs(alpha_after - alpha_before) > 0.5

    def test_returned_info_has_buffer_n_total(self) -> None:
        wrap = _fit_basic_wrapper(online=True)
        rng = np.random.default_rng(5)
        info = wrap.update(
            y_recent=rng.normal(size=10),
            base_recent=rng.normal(size=10),
        )
        assert "buffer_n_total" in info
        assert info["buffer_n_total"] == 10


# ===========================================================================
# sklearn.clone() semantics (cloned wrapper starts with empty buffer)
# ===========================================================================


class TestSklearnCloneSafety:
    def test_clone_starts_with_empty_buffer(self) -> None:
        """sklearn.clone() must produce a wrapper with empty rolling buffer (no shared state with original)."""
        wrap = _fit_basic_wrapper(online=True)
        # Push some rows into the original.
        rng = np.random.default_rng(6)
        wrap.update(y_recent=rng.normal(size=50), base_recent=rng.normal(size=50))
        assert wrap.get_buffer_state()["buffer_n"] == 50
        # Clone -- the buffer (underscore-suffixed runtime state) should NOT carry over.
        cloned = clone(wrap)
        # The cloned wrapper carries the SAME init kwargs (online_refit_enabled=True etc.).
        assert cloned.online_refit_enabled is True
        # But fresh state: no buffer attr until update is called.
        assert not hasattr(cloned, "_buffer_y_")

    def test_clone_preserves_init_kwargs(self) -> None:
        """The constructor kwargs (online_refit_enabled, buffer_n, z_threshold) survive sklearn.clone()."""
        wrap = _fit_basic_wrapper(
            online=True,
            online_refit_buffer_n=777,
            online_refit_z_threshold=2.5,
        )
        cloned = clone(wrap)
        assert cloned.online_refit_enabled is True
        assert cloned.online_refit_buffer_n == 777
        assert cloned.online_refit_z_threshold == 2.5

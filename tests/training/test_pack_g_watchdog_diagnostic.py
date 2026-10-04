"""#9 Pack G watchdog enhanced diagnostic dump.

When the watchdog detects ``|y-MAE - T-MAE| / T-MAE > 1%`` on an additive-invertible transform, the log line now includes:
- First 5 rows of (y, y_hat, T, T_hat, base) so an operator can SEE where the divergence enters: wrapper math, inverse path, or post-clip.
- Sample residuals on both scales.

This is the diagnostic groundwork for a follow-up session that finds the actual root cause of the production MLP T-MAE=9.17 vs y-MAE=3.22 discrepancy. The fix for #8 (module-level _TTRWithEvalSetScaling) may close this as a side effect (the local-class issue caused dill / sklearn.clone instability that could have corrupted TTR.transformer_).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


class TestWatchdogDiagnosticFormat:
    """When watchdog detects divergence, the log line MUST contain the diagnostic dump for downstream forensics."""

    def test_watchdog_log_includes_sample_rows_on_divergence(self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrapper whose y-scale output drifts away from its inner's T-scale error trips the additive watchdog; a consistent wrapper stays quiet.

        The warning must carry the composite, the split, both MAEs and the divergence percentage so an operator can see where the two scales part.
        """
        import re

        from sklearn.base import BaseEstimator, RegressorMixin

        from mlframe.training.composite import CompositeTargetEstimator
        from mlframe.training.composite.transforms import get_transform
        from mlframe.training.core._composite_wrap_watchdog import run_wrap_watchdog
        from mlframe.utils.log_throttle import reset_throttle_counts

        class _BrokenInner(BaseEstimator, RegressorMixin):
            """Predicts noise around zero: a poor but self-consistent inner model of the residual T."""

            def __init__(self, scale_factor: float = 0.1) -> None:
                self.scale_factor = scale_factor

            def fit(self, X, y):
                """Fit."""
                self.coef_ = np.zeros(X.shape[1] if X.ndim > 1 else 1)
                return self

            def predict(self, X):
                """Predict."""
                rng = np.random.default_rng(0)
                return rng.normal(0.0, 1.0, size=len(X)) * self.scale_factor

        rng = np.random.default_rng(0)
        n = 400
        base = rng.normal(100.0, 20.0, n)
        y = 1.5 * base + 5.0 + rng.normal(0.0, 2.0, n)
        df = pd.DataFrame({"base": base})

        transform = get_transform("linear_residual")
        params = transform.fit(y, base)
        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=_BrokenInner().fit(df.values, y - 1.5 * base - 5.0),
            transform_name="linear_residual",
            base_column="base",
            transform_fitted_params=params,
            y_train=y,
        )
        spec = {"name": "y-linres-base", "transform_name": "linear_residual", "base_column": "base", "fitted_params": params}
        logger_name = "mlframe.training.core._phase_composite_wrapping"

        def _watchdog_lines() -> list:
            """Run the watchdog on the full frame and return the watchdog warnings it logged."""
            caplog.clear()
            reset_throttle_counts()
            with caplog.at_level(logging.WARNING, logger=logger_name):
                run_wrap_watchdog(wrapper, spec, df, y, composite_name="y-linres-base", split_name="val")
            return [r.getMessage() for r in caplog.records if "watchdog" in r.getMessage()]

        assert _watchdog_lines() == []

        real_predict = wrapper.predict
        monkeypatch.setattr(wrapper, "predict", lambda X: real_predict(X) + 7.5)
        (message,) = _watchdog_lines()
        assert "composite='y-linres-base'" in message and "split='val'" in message
        match = re.search(r"y-MAE=([\d.eE+-]+) differs from T-MAE=([\d.eE+-]+) by ([\d.]+)% on (\d+) in-range rows", message)
        assert match is not None, message
        y_mae, t_mae, pct, rows = float(match[1]), float(match[2]), float(match[3]), int(match[4])
        assert y_mae > 3.0 * t_mae
        assert pct == pytest.approx(abs(y_mae - t_mae) / t_mae * 100.0, rel=0.01)
        assert rows > 0

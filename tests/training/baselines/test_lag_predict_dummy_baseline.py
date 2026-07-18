"""Regression test for the ``lag_predict`` dummy baseline.

For strongly auto-regressive regression targets (e.g. TVT-style data
where ``lag1_corr ~ 0.999`` within groups), the dumbest possible
prediction -- ``y_hat = lag_target_value`` in the same row -- can
dramatically outperform mean / median baselines AND sometimes the
user's trained models. Production TVT 2026-05-23: BaselineDiagnostics
measured ``init_score(TVT_prev)`` RMSE=8.06 vs Ridge raw RMSE=11.63 --
a 31% improvement available for free, but the framework didn't expose
it as a dummy baseline.

The new baseline scans for a column matching ``{target_name}_prev``,
``{target_name}_lag_1``, ``{target_name}_lag1``, or
``{target_name}_lag`` in ``train_X`` / ``val_X`` / ``test_X``. If
found, ``y_hat`` for each row equals that column's value. The
existing dummy-strongest-pick logic then surfaces this baseline when
it beats mean / median / per_group_mean. The cross-target verdict
formatter flags ``MODELS_BARELY_BEAT_TRIVIAL`` when the user's best
model RMSE is within 1.5x of lag_predict.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def _ar_synthetic_frame(n: int = 2000, seed: int = 0):
    """AR(1) target with autocorr 0.999 and a *_prev column."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    y[0] = rng.normal(0, 10)
    for i in range(1, n):
        y[i] = 0.999 * y[i - 1] + rng.normal(scale=1.0)
    y_prev = np.r_[y[0], y[:-1]].astype(np.float64)
    x_noise = rng.normal(size=n).astype(np.float64)
    df = pd.DataFrame(
        {
            "TVT_prev": y_prev,
            "x_noise": x_noise,
        }
    )
    return df, y


def _dummy_config():
    """Dummy config."""
    return SimpleNamespace(
        per_group_max_cardinality_ratio=0.5,
        per_group_min_val_coverage_pct=50.0,
        per_group_high_overlap_threshold=0.95,
        ts_extra_periods=[7, 30],
    )


class TestLagPredictBaseline:
    """Groups tests covering lag predict baseline."""
    def test_lag_predict_added_for_ar_target(self) -> None:
        """Lag predict added for ar target."""
        from mlframe.training.baselines._dummy_baseline_regression import (
            _compute_regression_baselines,
        )

        df, y = _ar_synthetic_frame()
        train_X = df.iloc[:1500]
        val_X = df.iloc[1500:1750]
        test_X = df.iloc[1750:]
        train_y = y[:1500]
        val_y = y[1500:1750]
        test_y = y[1750:]
        val_preds, test_preds, extras = _compute_regression_baselines(
            target_name="TVT",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            timestamps_train=None,
            timestamps_val=None,
            timestamps_test=None,
            cat_features=None,
            config=_dummy_config(),
            target_type="regression",
        )
        assert "lag_predict" in val_preds, f"lag_predict baseline missing from val_preds; got keys: {list(val_preds.keys())}"
        assert "lag_predict" in test_preds
        assert extras["lag_predict"]["feature_used"] == "TVT_prev"

    def test_lag_predict_beats_mean_baseline_on_ar1(self) -> None:
        """The whole point: on AR(1) data the lag baseline should
        crush the mean baseline. Verifies the RMSE delta exists."""
        from mlframe.training.baselines._dummy_baseline_regression import (
            _compute_regression_baselines,
        )

        df, y = _ar_synthetic_frame(n=4000, seed=1)
        train_X = df.iloc[:3000]
        val_X = df.iloc[3000:3500]
        test_X = df.iloc[3500:]
        train_y = y[:3000]
        val_y = y[3000:3500]
        test_y = y[3500:]
        val_preds, _test_preds, _extras = _compute_regression_baselines(
            target_name="TVT",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            timestamps_train=None,
            timestamps_val=None,
            timestamps_test=None,
            cat_features=None,
            config=_dummy_config(),
            target_type="regression",
        )
        rmse_lag = float(np.sqrt(np.mean((val_preds["lag_predict"] - val_y) ** 2)))
        rmse_mean = float(np.sqrt(np.mean((val_preds["mean"] - val_y) ** 2)))
        assert (
            rmse_lag < 0.1 * rmse_mean
        ), f"lag_predict ({rmse_lag:.3f}) should crush mean ({rmse_mean:.3f}) on AR(1) data with autocorr=0.999; expected >=10x improvement"

    def test_lag_predict_skipped_when_no_lag_column(self) -> None:
        """Lag predict skipped when no lag column."""
        from mlframe.training.baselines._dummy_baseline_regression import (
            _compute_regression_baselines,
        )

        rng = np.random.default_rng(0)
        n = 1000
        y = rng.normal(size=n)
        df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        train_X = df.iloc[:800]
        val_X = df.iloc[800:900]
        test_X = df.iloc[900:]
        val_y = y[800:900]
        test_y = y[900:]
        val_preds, _test_preds, _extras = _compute_regression_baselines(
            target_name="SomeOtherTarget",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=y[:800],
            val_y=val_y,
            test_y=test_y,
            timestamps_train=None,
            timestamps_val=None,
            timestamps_test=None,
            cat_features=None,
            config=_dummy_config(),
            target_type="regression",
        )
        assert "lag_predict" not in val_preds

    def test_lag_predict_picks_lag_1_suffix(self) -> None:
        """Verifies the ``_lag_1`` suffix variant is also detected."""
        from mlframe.training.baselines._dummy_baseline_regression import (
            _compute_regression_baselines,
        )

        df, y = _ar_synthetic_frame()
        df = df.rename(columns={"TVT_prev": "TVT_lag_1"})
        train_X = df.iloc[:1500]
        val_X = df.iloc[1500:1750]
        test_X = df.iloc[1750:]
        val_preds, _test_preds, extras = _compute_regression_baselines(
            target_name="TVT",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=y[:1500],
            val_y=y[1500:1750],
            test_y=y[1750:],
            timestamps_train=None,
            timestamps_val=None,
            timestamps_test=None,
            cat_features=None,
            config=_dummy_config(),
            target_type="regression",
        )
        assert "lag_predict" in val_preds
        assert extras["lag_predict"]["feature_used"] == "TVT_lag_1"

"""Regression tests for three audit-confirmed bugs (batch 1).

#6  XGB seed: ``get_training_configs`` must key the RNG on ``random_state``
    (XGBoost silently ignores ``random_seed``), else a requested seed is
    dropped and XGB trains non-reproducibly while CB/LGB/HGB are seeded.
#5  ``save_split_artifacts`` must subset timestamps/group_ids POSITIONALLY
    (``.iloc``), not via label-based ``pandas.Series[idx]`` -- on a
    non-RangeIndex frame the bare ``[]`` selects the WRONG rows (or raises),
    saving artifacts misaligned with the split rows the model trained on.
#10 ``CompositeTargetEstimator.from_fitted_inner`` must gate the T-clip
    envelope on the COUNT of finite y (``finite.sum()``), not the array
    LENGTH (``finite.size``); a mostly-NaN y must not get a tight envelope
    estimated from a handful of points.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest


def test_xgb_config_uses_random_state_not_random_seed():
    """Xgb config uses random state not random seed."""
    from mlframe.training._helpers_training_configs import get_training_configs

    cfg = get_training_configs(random_seed=42)
    xgb = cfg.XGB_GENERAL_PARAMS
    assert xgb.get("random_state") == 42, f"XGB params must carry the seed under 'random_state'; got {xgb.get('random_state')!r}"
    # ``random_seed`` is the key XGBoost ignores; it must NOT be the seed carrier.
    assert "random_seed" not in xgb, "'random_seed' is silently ignored by XGBoost -- it must not be the seed carrier"
    # The key we now emit is actually honored by XGBoost's sklearn API.
    xgboost = pytest.importorskip("xgboost")
    m = xgboost.XGBClassifier(random_state=xgb["random_state"])
    assert m.get_params()["random_state"] == 42


def test_save_split_artifacts_positional_on_non_rangeindex(tmp_path):
    """Save split artifacts positional on non rangeindex."""
    from mlframe.training.preprocessing import save_split_artifacts

    n = 10
    # NON-RangeIndex pandas Series: label-based [] would select different rows
    # than positional .iloc for the same positional idx.
    shuffled_index = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    ts = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"), index=shuffled_index)
    gid = pd.Series(np.arange(100, 100 + n), index=shuffled_index)
    train_idx = np.array([0, 1, 2])  # positional rows 0,1,2

    save_split_artifacts(
        train_idx=train_idx,
        val_idx=None,
        test_idx=None,
        timestamps=ts,
        group_ids_raw=gid,
        artifacts=None,
        data_dir=str(tmp_path),
        models_dir="models",
        target_name="t",
        model_name="m",
    )

    ts_files = glob.glob(os.path.join(str(tmp_path), "**", "train_timestamps.parquet"), recursive=True)
    assert ts_files, "train_timestamps.parquet was not written"
    saved = pd.read_parquet(ts_files[0])
    saved_vals = pd.to_datetime(saved.iloc[:, 0]).to_numpy()

    expected = pd.to_datetime(ts.iloc[train_idx].to_numpy()).to_numpy()  # positional (correct)
    label_based = pd.to_datetime(ts.loc[train_idx].to_numpy()).to_numpy()  # what the bug produced
    np.testing.assert_array_equal(saved_vals, expected)
    # Discriminating guard: positional and label-based selections genuinely differ here,
    # so this test would FAIL on the pre-fix label-based path.
    assert not np.array_equal(expected, label_based), "fixture must distinguish positional vs label"


def test_composite_from_fitted_inner_envelope_gates_on_finite_count():
    """Composite from fitted inner envelope gates on finite count."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite.estimator._estimator import CompositeTargetEstimator

    inner = LinearRegression().fit(np.arange(20, dtype=float).reshape(-1, 1), np.arange(20, dtype=float))
    # 100 rows, only 3 finite -> finite.sum()=3 < 10 -> must NOT build a tight
    # envelope (pre-fix finite.size=100 >= 10 fired and used std of 3 points).
    y = np.full(100, np.nan)
    y[:3] = [10.0, 11.0, 12.0]

    est = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="cbrt_y",
        base_column="y",
        transform_fitted_params={},
        y_train=y,
    )
    fp = est.fitted_params_
    assert fp["t_clip_low"] == float("-inf") and fp["t_clip_high"] == float(
        "inf"
    ), f"mostly-NaN y (3 finite < 10) must leave the T-clip envelope open; got [{fp['t_clip_low']}, {fp['t_clip_high']}]"

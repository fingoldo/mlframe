"""Ensembles of a composite target get y-scale metrics (they only carry T-scale preds and had none before)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.core._phase_composite_wrapping import _record_ensemble_y_scale_metrics


def test_ensemble_gets_y_scale_row():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = np.exp(1 + 0.5 * X["x"].to_numpy() + rng.normal(scale=0.5, size=n))
    member = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="log_y").fit(X.iloc[:3000], y[:3000])
    val_idx = np.arange(3000, n)
    X_val = X.iloc[val_idx]
    t_val = member.estimator_.predict(X_val)  # the member's own T-scale predictions
    meta: dict = {}
    entries = [SimpleNamespace(model=member, model_name="cb"), SimpleNamespace(model=None, model_name="EnsARITHM", val_preds=t_val)]
    _record_ensemble_y_scale_metrics(
        entries=entries, y_full=y, metadata=meta, target_type="regression", composite_name="y-logY",
        splits=(("val", val_idx, X_val), ("test", None, None)),
    )
    rows = meta["composite_target_y_scale_metrics"]["regression"]["y-logY"]
    assert [r["model_name"] for r in rows] == ["EnsARITHM"]
    expected = float(np.sqrt(np.mean((member.predict(X_val) - y[val_idx]) ** 2)))
    assert abs(rows[0]["metrics"]["val"]["RMSE"] - expected) < 1e-9

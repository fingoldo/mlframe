"""Regression test: ``_build_learning_curve`` must store its result under the caller-supplied
``metadata_target_name`` key, not a hardcoded ``"learning_curve"`` literal.

Bug (vulture unused-variable audit): the function accepted ``metadata_target_name`` but ignored it,
always writing to ``metrics["learning_curve"]``. A caller passing a custom name (e.g. to log two
learning curves for two different targets in one ``metrics`` dict) would have both silently collide
under the same key.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.diagnostics.learning_curve import LearningCurveConfig
from mlframe.training.reporting._reporting_diagnostics import _build_learning_curve


def test_build_learning_curve_uses_custom_metadata_target_name():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = df["x1"] * 2.0 + rng.normal(scale=0.1, size=n)

    lc_cfg = LearningCurveConfig(enabled=True, sizes=(0.5, 1.0), n_jobs=1, score_repeats=1)
    metrics: dict = {}
    panel = _build_learning_curve(
        LinearRegression(), df, y, ["x1", "x2"], "regression", lc_cfg, metrics,
        metadata_target_name="my_custom_target",
    )

    assert panel is not None
    assert "my_custom_target" in metrics, "result must be stored under the caller-supplied metadata_target_name key"
    assert "learning_curve" not in metrics, "must not fall back to the hardcoded literal key when a custom name is given"

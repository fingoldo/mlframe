"""Unit tests for the public TTA uncertainty-evaluation helper (Workstream B).

``evaluate_tta_quality`` is the supported way to assess TTA predictive uncertainty on val/test: call it
with your fitted model + the held-out (X, y). (In-suite auto-stamping was investigated and reverted --
the trained model and the model-ready transformed frame do not co-exist at any clean point in the
per-target body, so it cannot be wired without destabilising the hot training path.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._uncertainty_eval import _narrow_numeric_frame, evaluate_tta_quality


def test_evaluate_tta_quality_metrics():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3))
    y = X[:, 0] * 2 - X[:, 1]

    def f(Z):
        return Z[:, 0] * 2 - Z[:, 1] + 0.5 * np.sin(11 * Z[:, 2])

    rep = evaluate_tta_quality(f, X, y, n=24, sigma_scale=0.2)
    assert set(rep) == {"rmse_point", "rmse_tta", "tta_rmse_gain", "spread_error_corr", "mean_spread"}
    assert rep["mean_spread"] > 0
    assert rep["rmse_tta"] <= rep["rmse_point"] + 1e-9  # averaging should not hurt here


def test_narrow_numeric_frame_pandas_and_reject_non_numeric():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
    arr = _narrow_numeric_frame(df, ["a", "b"])
    assert arr.shape == (2, 2)
    assert _narrow_numeric_frame(df, ["a", "c"]) is None  # non-numeric column -> None


def test_narrow_numeric_frame_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    arr = _narrow_numeric_frame(df, ["a", "b"])
    assert arr.shape == (3, 2)


def test_evaluate_tta_quality_is_public():
    import mlframe.training as training_mod

    assert hasattr(training_mod, "evaluate_tta_quality")
    assert "evaluate_tta_quality" in training_mod.__all__

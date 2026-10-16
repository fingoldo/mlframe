"""An explicit empty ``feature_names`` must restrict the drift report to zero columns, not widen it to all columns."""
import pandas as pd

from mlframe.training.feature_drift_report import compute_categorical_drift_psi


def _frames():
    """Build train/val/test frames with one categorical column."""
    df = pd.DataFrame({"c": pd.Categorical(["a", "b"] * 50)})
    return df, df.copy(), df.copy()


def test_empty_feature_names_selects_no_columns():
    """feature_names=[] yields no per-feature entries; None still auto-detects the categorical column."""
    tr, va, te = _frames()
    res_empty = compute_categorical_drift_psi(tr, va, te, feature_names=[])
    assert res_empty["per_feature"] == {}
    res_auto = compute_categorical_drift_psi(tr, va, te, feature_names=None)
    assert "c" in res_auto["per_feature"]

"""Regression: _compute_split_metrics must skip a 0-feature frame instead of crashing in the model backend.

Fuzz combo c0012 (regression, hgb+xgb, MRMR + xgb_rfecv) had feature selection remove every feature, leaving a
0-column test frame. The eval path still called model.predict on it, which crashed with
``IndexError: list index out of range`` deep in xgboost building a DMatrix from 0 columns. The skip now also fires on a
non-None model when the feature frame has 0 columns (nothing to predict from).
"""

import numpy as np
import pandas as pd

from mlframe.training._eval_helpers import _compute_split_metrics


class _ExplodingModel:
    """Predict raises IndexError on any input -- stands in for the xgboost 0-column DMatrix crash."""

    def predict(self, X, *a, **k):
        raise IndexError("list index out of range")


def test_compute_split_metrics_skips_zero_feature_frame():
    df = pd.DataFrame(index=range(20))  # 20 rows, 0 columns
    assert df.shape == (20, 0)
    target = np.arange(20.0)
    preds, probs, columns = _compute_split_metrics(
        split_name="test",
        df=df,
        target=target,
        idx=None,
        model=_ExplodingModel(),
        model_type_name="XGBRegressor",
        model_name="zerofeat",
        metrics_dict={},
        print_report=False,
        show_perf_chart=False,
    )
    assert preds is None and probs is None
    assert columns == []

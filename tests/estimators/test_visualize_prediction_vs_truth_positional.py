"""visualize_prediction_vs_truth picks rows by position, not by index label (tracker 2026-11-02 wave 10, ALIGN10-6)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from mlframe.estimators.pipelines import visualize_prediction_vs_truth


def test_series_truth_with_offset_index_and_frame_predictions():
    """A Series truth indexed 100.. and a DataFrame of predictions raised KeyError because [] looked up labels and columns."""
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.normal(size=20), index=np.arange(100, 120))
    y_preds = pd.DataFrame(rng.normal(size=(20, 3)), index=np.arange(100, 120))
    fig = visualize_prediction_vs_truth(y_true, y_preds, samples=(1, 5))
    assert len(fig.axes) == 2

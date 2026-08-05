"""CORE_INFRA_MISC-4: visualize_prediction_vs_truth must handle a single-sample plot.

plt.subplots(1, len(samples)) returns a bare Axes (not an array) when len(samples)==1; axs[i] indexing
must not crash in that case."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from mlframe.estimators.pipelines import visualize_prediction_vs_truth


def test_visualize_prediction_vs_truth_single_sample_does_not_crash():
    """A single-element samples tuple must not raise TypeError from bare-Axes indexing."""
    rng = np.random.default_rng(0)
    y_true = pd.DataFrame(rng.standard_normal((5, 4)))
    y_preds = rng.standard_normal((5, 4))
    visualize_prediction_vs_truth(y_true, y_preds, samples=(0,))


def test_visualize_prediction_vs_truth_multi_sample_still_works():
    """Multi-sample plotting (the already-covered path) keeps working unchanged."""
    rng = np.random.default_rng(0)
    y_true = pd.DataFrame(rng.standard_normal((5, 4)))
    y_preds = rng.standard_normal((5, 4))
    visualize_prediction_vs_truth(y_true, y_preds, samples=(0, 1, 2))

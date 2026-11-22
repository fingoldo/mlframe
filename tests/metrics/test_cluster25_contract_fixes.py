"""Audit 2026-11-20 findings 25-F38 and 25-F45 in the metrics, and 25-F24 and 25-F54 in feature selection and tuning.

25-F38: compute_fairness_metrics returns an empty DataFrame, not None, when no subgroup produced a row.
25-F45: fast_log_loss_binary takes plain lists, as its sklearn drop-in role promises.
25-F24: diversification refuses data whose width does not match the ranked features, instead of silently turning off.
25-F54: prepare_trials_dataset is annotated with the tuple it returns.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from mlframe.feature_selection.varying_size_top_k_subsets import _cluster_anchors
from mlframe.metrics._fairness_metrics import compute_fairness_metrics
from mlframe.metrics._log_loss_and_separation import fast_log_loss_binary
from mlframe.models.tuning_rules import prepare_trials_dataset


def test_fairness_with_no_subgroups_returns_an_empty_dataframe():
    """No subgroups gives an empty DataFrame with the documented index, not None."""
    out = compute_fairness_metrics({}, {}, {}, np.arange(4), np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_log_loss_accepts_plain_lists():
    """A list pair gives the same value sklearn gives, not an AttributeError."""
    y, p = [0, 1, 1, 0], [0.2, 0.8, 0.7, 0.1]
    assert fast_log_loss_binary(y, p) == pytest.approx(log_loss(y, p))


def test_width_mismatch_refuses_rather_than_disabling_diversification():
    """Two columns for three ranked features raises, instead of making every feature its own cluster silently."""
    with pytest.raises(ValueError, match="must line up"):
        _cluster_anchors(["a", "b", "c"], np.zeros((5, 2)), 0.9)


def test_prepare_trials_dataset_is_annotated_as_the_tuple_it_returns():
    """The return annotation names the (DataFrame, list) pair that the function returns."""
    annotation = str(inspect.signature(prepare_trials_dataset).return_annotation)
    assert "tuple" in annotation and "DataFrame" in annotation

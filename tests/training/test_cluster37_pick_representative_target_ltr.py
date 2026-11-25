"""Cluster 37 regression: a learning-to-rank target is not labelled classification by the representative-target fallback (audit 2026-11-20, finding 37-F46).

The fallback used to take the first non-empty bag of any type and call it "classification" unless the key name ended in REGRESSION. It now only
takes the supervised types it understands; an LTR-only input yields no picked target, and a regression target is still picked as regression.
"""

import numpy as np

from mlframe.training._configs_base import TargetTypes
from mlframe.training.core._main_train_suite_target_distribution import _pick_representative_target


def test_learning_to_rank_only_is_not_picked_as_classification():
    """An LTR-only target map yields no picked target instead of a mislabelled classification one (37-F46)."""
    target, type_name, _name = _pick_representative_target({TargetTypes.LEARNING_TO_RANK: {"rel": np.arange(5)}})
    assert target is None
    assert type_name != "classification"


def test_regression_is_still_picked_as_regression():
    """Control: a regression target is picked and labelled regression (37-F46)."""
    target, type_name, name = _pick_representative_target({TargetTypes.REGRESSION: {"y": np.arange(5.0)}})
    assert type_name == "regression"
    assert name == "y"
    assert target is not None

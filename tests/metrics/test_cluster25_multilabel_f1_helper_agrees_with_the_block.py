"""Cluster 25 regression: the standalone multilabel F1 functions and the fused metrics block cannot drift apart (audit 2026-11-20, finding 25-F52)."""

import numpy as np
import pytest

from mlframe.metrics._multilabel_extras import (
    fast_multilabel_classification_metrics_block,
    multilabel_f1_macro,
    multilabel_f1_micro,
    multilabel_f1_weighted,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_macro_micro_weighted_f1_equal_the_block_values_including_empty_labels(seed):
    """On random indicators with an all-zero label (zero support) the three standalone F1s match the block's f1_macro / f1_micro / f1_weighted exactly."""
    rng = np.random.default_rng(seed)
    y_true = (rng.uniform(size=(300, 5)) < 0.3).astype(np.int64)
    y_pred = (rng.uniform(size=(300, 5)) < 0.35).astype(np.int64)
    y_true[:, 4] = 0
    y_pred[:, 4] = 0
    block = fast_multilabel_classification_metrics_block(y_true, y_pred)
    assert multilabel_f1_macro(y_true, y_pred) == pytest.approx(block["f1_macro"], abs=1e-15)
    assert multilabel_f1_micro(y_true, y_pred) == pytest.approx(block["f1_micro"], abs=1e-15)
    assert multilabel_f1_weighted(y_true, y_pred) == pytest.approx(block["f1_weighted"], abs=1e-15)

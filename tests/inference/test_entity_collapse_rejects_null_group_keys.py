"""collapse_predictions_by_group refuses null group keys in both the weighted and unweighted paths."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.inference.entity_prediction_collapse import collapse_predictions_by_group


@pytest.mark.parametrize("weights", [None, np.array([1.0, 1.0, 1.0])])
def test_null_group_key_raises_in_both_paths(weights):
    """group = ['a', 'a', None]: the unweighted path used to return NaN for the last row silently, the weighted one crashed in bincount."""
    with pytest.raises(ValueError, match="null"):
        collapse_predictions_by_group(np.array([0.1, 0.2, 0.3]), ["a", "a", None], weights=weights)


def test_clean_groups_still_collapse():
    """Negative control: no nulls, plain mean per group."""
    out = collapse_predictions_by_group(np.array([0.0, 1.0, 5.0]), ["a", "a", "b"])
    np.testing.assert_allclose(out, [0.5, 0.5, 5.0])

"""The binary entropy helper agrees with the multi-class entropy it is a reduced form of, to float round-off."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.mdl_binning_pairwise import _entropy_binary, _entropy_from_counts


@pytest.mark.parametrize("total", [10, 100, 1000])
def test_binary_entropy_matches_two_class_counts(total: int) -> None:
    """For every class split of ``total`` rows, the binary helper equals the count-based entropy in bits."""
    for k in range(0, total + 1, max(1, total // 10)):
        p = k / total
        counts = np.array([total - k, k], dtype=np.int64)
        assert _entropy_binary(p) == pytest.approx(_entropy_from_counts(counts, total), abs=1e-12)


def test_degenerate_proportions_have_zero_entropy() -> None:
    """A pure split (p=0 or p=1) carries no entropy."""
    assert _entropy_binary(0.0) == 0.0
    assert _entropy_binary(1.0) == 0.0

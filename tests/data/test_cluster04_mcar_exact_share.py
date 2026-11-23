"""Cluster 04 regression: MCAR hides an exact share of rows, matching the documented contract (audit 2026-11-20, finding 04-F33)."""

import numpy as np
import pytest

from mlframe.data.datasets._missing import missing_mask
from mlframe.data.datasets.spec import MissingnessSpec


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mcar_hides_exactly_the_requested_count(seed):
    """At rate 0.3 on 100 rows, exactly 30 rows are hidden for every seed, not a Bernoulli 25..35 (04-F33)."""
    values = np.arange(100.0)
    spec = MissingnessSpec(column="x", mechanism="mcar", rate=0.3)
    mask = missing_mask(spec, {"x": values}, 0.3, np.random.default_rng(seed))
    assert int(mask.sum()) == 30


def test_mcar_zero_rate_hides_nothing():
    """Control: a zero share hides no rows (04-F33)."""
    spec = MissingnessSpec(column="x", mechanism="mcar", rate=0.0)
    mask = missing_mask(spec, {"x": np.arange(50.0)}, 0.0, np.random.default_rng(0))
    assert not mask.any()

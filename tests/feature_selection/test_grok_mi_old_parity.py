"""The superseded naive MI and the live log-space MI agree on the same joint histogram, to float round-off."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.mi import grok_mutual_information, grok_mutual_information_old


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_old_and_live_mi_agree(seed: int) -> None:
    """On random integer-binned columns, the naive per-cell ratio and the log-space form give the same MI."""
    rng = np.random.default_rng(seed)
    n = 2_000
    a = rng.integers(0, 15, n).astype(np.int64)
    b = (a + rng.integers(0, 4, n)) % 15
    old = grok_mutual_information_old(a, b, n_bins=15)
    live = grok_mutual_information(a, b, 1.0 / n, float(np.log(n)), n_bins=15)
    assert old == pytest.approx(live, rel=1e-9, abs=1e-12)

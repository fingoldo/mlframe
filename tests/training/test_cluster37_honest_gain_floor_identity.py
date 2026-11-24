"""Cluster 37 regression: the honest-gain floor keeps specs by identity, not dict equality (audit 2026-11-20, finding 37-F30).

Specs carry ndarray values. Filtering with ``p not in _below`` compared dicts by value and could raise on array truth. The filter now uses object
identity, and the kept/dropped split is unchanged.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.core._phase_composite_discovery_gates import _drop_below_honest_gain_floor


def test_filter_keeps_and_drops_by_identity_with_array_values():
    """Specs whose first key is an ndarray are filtered without an array-truth error (37-F30)."""
    cfg = SimpleNamespace(min_honest_gain_to_train=0.001, min_honest_gain_z=0.0)
    below = {"values": np.arange(3), "name": "weak", "rmse_gain": True, "gain": 0.0005}
    above = {"values": np.arange(3), "name": "strong", "rmse_gain": True, "gain": 0.5}
    kept = _drop_below_honest_gain_floor([below, above], cfg)
    assert kept == [above] or (len(kept) == 1 and kept[0] is above)

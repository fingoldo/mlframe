"""The sequential seed early-stop is disabled unless the cross-family consensus is `union`."""

from __future__ import annotations

import math
import types

from mlframe.training.composite.discovery._tiny_rerank import _multiseed_early_stop_threshold


def _self(consensus):
    """A discovery stand-in whose config enables the early stop with the given consensus."""
    cfg = types.SimpleNamespace(enable_multiseed_early_stop=True, raw_baseline_tolerance=1.02, tiny_consensus=consensus)
    return types.SimpleNamespace(config=cfg)


def test_union_consensus_keeps_the_threshold():
    """Under `union` (the best family) the early stop stays active."""
    assert _multiseed_early_stop_threshold(_self("union"), 10.0, False, False) == 10.0 * 1.02


def test_borda_consensus_disables_the_early_stop():
    """Under `borda` the mean across families would carry a stopped family's inflated median, so every seed runs."""
    assert math.isinf(_multiseed_early_stop_threshold(_self("borda"), 10.0, False, False))

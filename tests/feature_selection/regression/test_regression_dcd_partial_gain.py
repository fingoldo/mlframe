"""Regression unit test for the 2026-06-02 DCD ``find_best_partial_gain`` fix
(commit c3595d70).

BUG: ``find_best_partial_gain`` had no view of the DCD prune mask.
``partial_gains`` persists across the confirmation ``while``-retries within one
interactions-order; when ``discover_cluster_members`` prunes a candidate AFTER
it was scored (a same-cluster member got selected -> SU > tau), the candidate is
skipped from RE-scoring by ``should_skip_candidate`` but its now-STALE high
partial gain stays in ``partial_gains``. Pre-fix, ``find_best_partial_gain``
returned that pruned candidate as "the best other option", so the confirmation
loop redirected to it forever (it can never be confirmed -- it is skipped) and
the genuinely-good candidate that DID confirm was never committed -> the screen
stopped early and dropped real signal (sensor-mesh: 6 features -> 2, -4% AUC).

These are the unit-level companions to the end-to-end pins in
``test_biz_value_mrmr_layer49.py::TestLayer49_ScenarioA_SensorMesh`` (S3/S4).
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.evaluation import find_best_partial_gain


class _StubDCDState:
    """Minimal stand-in carrying just the ``pool_pruned_mask`` that
    ``should_be_pruned`` reads (column-indexed bool array)."""

    def __init__(self, pruned_cols, n_cols):
        mask = np.zeros(int(n_cols), dtype=bool)
        for c in pruned_cols:
            mask[int(c)] = True
        self.pool_pruned_mask = mask


def _setup():
    # candidates: single-feature tuples, position == column index.
    #   idx 0 -> col 0 (a low-gain unpruned candidate)
    #   idx 1 -> col 1 (the HIGH-gain candidate that gets DCD-pruned)
    #   idx 2 -> col 2 (a mid-gain unpruned candidate)
    candidates = [(0,), (1,), (2,)]
    partial_gains = {0: (0.05, 0), 1: (0.133, 0), 2: (0.072, 0)}
    return candidates, partial_gains


def test_find_best_partial_gain_legacy_returns_highest_when_no_dcd():
    """Bit-stable legacy path (dcd_state=None): the highest partial gain wins,
    including the (here col-1) candidate -- this is the PRE-FIX behaviour and
    must be preserved for the DCD-off default."""
    candidates, partial_gains = _setup()
    best_gain, best_key = find_best_partial_gain(
        partial_gains=partial_gains,
        failed_candidates=set(),
        added_candidates=set(),
        candidates=candidates,
        selected_vars=[],
        dcd_state=None,
    )
    assert best_key == 1 and abs(best_gain - 0.133) < 1e-12, (
        f"legacy find_best_partial_gain must return the global max (key=1, 0.133); got key={best_key}, gain={best_gain}"
    )


def test_find_best_partial_gain_skips_dcd_pruned_candidate():
    """THE FIX: when col 1 is DCD-pruned, its stale 0.133 partial must be
    skipped, so the next-best UNPRUNED candidate (col 2, 0.072) wins. On the
    pre-fix code this returned key=1 (the pruned candidate) and the confirmation
    loop redirected to it forever."""
    candidates, partial_gains = _setup()
    dcd_state = _StubDCDState(pruned_cols=[1], n_cols=3)
    best_gain, best_key = find_best_partial_gain(
        partial_gains=partial_gains,
        failed_candidates=set(),
        added_candidates=set(),
        candidates=candidates,
        selected_vars=[],
        dcd_state=dcd_state,
    )
    assert best_key == 2 and abs(best_gain - 0.072) < 1e-12, (
        f"DCD-pruned candidate (key=1, stale 0.133) must be skipped; expected the next-best unpruned (key=2, 0.072), got key={best_key}, gain={best_gain}"
    )


def test_find_best_partial_gain_all_pruned_returns_sentinel():
    """When every scored candidate is DCD-pruned, no redirect target exists:
    best_key is None and best_gain is the large-negative sentinel (the loop
    then commits the candidate that DID confirm instead of redirecting)."""
    candidates, partial_gains = _setup()
    dcd_state = _StubDCDState(pruned_cols=[0, 1, 2], n_cols=3)
    best_gain, best_key = find_best_partial_gain(
        partial_gains=partial_gains,
        failed_candidates=set(),
        added_candidates=set(),
        candidates=candidates,
        selected_vars=[],
        dcd_state=dcd_state,
    )
    assert best_key is None and best_gain < 0, f"all-pruned must yield no redirect target (None, negative sentinel); got key={best_key}, gain={best_gain}"

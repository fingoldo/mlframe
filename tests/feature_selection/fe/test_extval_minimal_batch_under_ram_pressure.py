"""When the free-RAM budget refuses every buffer, the external-validation step still scores one external factor against all ops as a batch (a few MiB) instead of one candidate at a time.

On a host with less free RAM than the reserve the budget refused a 5 MiB buffer, so a default 100000 x 100 fit scored 9540 candidates through single-column ``discretize_array`` + ``mi_direct`` calls. The minimal batch is taken
under ``_EXTVAL_MIN_BATCH_BYTES``; a bigger one (millions of rows) keeps the loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_emit_extval as ev


def _call(n_rows, n_ext, n_ops, budget_says):
    """``(can_batch, chunk)`` that part 2 settles on when the budget helper always answers ``budget_says``."""
    rng = np.random.default_rng(0)
    cols = {f"e{k}": rng.standard_normal(4).astype(np.float64) for k in range(n_ext)}
    binary = {f"op{k}": (lambda a, b: a + b) for k in range(n_ops)}
    out = ev._emit_pair_features_step2_read_here_once_part2(
        np.zeros((4, 1), dtype=np.float32), 0, False, lambda: False, None, None, SimpleNamespace(), None, None, binary, ("a", "b"), "op0",
        list(cols), 0, rng, lambda name: cols[name], np.empty(n_rows, dtype=np.int8), lambda nbytes, n_workers=1: (budget_says, nbytes, 0), 1,
    )
    return out[7], out[8]


def test_a_refusing_budget_still_batches_the_minimal_chunk():
    """Under pressure the factors are halved down to one and that single-factor batch is taken (the old code returned False)."""
    can, chunk = _call(100_000, 12, 6, False)
    assert can is True and chunk == 1


def test_a_budget_that_accepts_keeps_the_whole_set():
    """No pressure, no change: all factors in one batch."""
    can, chunk = _call(100_000, 12, 6, True)
    assert can is True and chunk == 12


def test_a_minimal_batch_over_the_cap_keeps_the_loop(monkeypatch):
    """A one-factor batch above the cap stays refused: the per-candidate loop is the memory-safe path there."""
    monkeypatch.setattr(ev, "_EXTVAL_MIN_BATCH_BYTES", 1000)
    can, chunk = _call(100_000, 12, 6, False)
    assert can is False and chunk == 1

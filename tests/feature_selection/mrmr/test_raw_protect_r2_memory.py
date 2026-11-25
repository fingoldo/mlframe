"""``heldout_r2_scorer`` builds its train block once, column by column, in column-major order and lets the QR consume it: the peak is a small multiple of one block, not the list of copies, the stack and the LAPACK copy."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy import _raw_protect_r2 as rp


def _problem(n: int = 120_000, p: int = 40, seed: int = 0):
    """A sequence of float32 columns, a target and a two-thirds / one-third split."""
    rng = np.random.default_rng(seed)
    cols = [rng.standard_normal(n).astype(np.float32) for _ in range(p)]
    y = (cols[0] + 0.5 * cols[1] + rng.standard_normal(n)).astype(np.float64)
    tr = np.zeros(n, dtype=bool)
    tr[: int(n * 2 / 3)] = True
    return cols, y, tr, ~tr


def _heldout_r2_lstsq(cols, y, tr, va, extra=None) -> float:
    """The probe computed directly: least squares on the train rows, R^2 on the validation rows."""
    parts = [np.asarray(c, dtype=np.float64) for c in cols] + ([np.asarray(extra, dtype=np.float64)] if extra is not None else [])
    design = np.column_stack(parts)
    coef = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
    resid = y[va] - design[va] @ coef
    return 1.0 - float(np.sum(resid**2)) / float(np.sum((y[va] - y[va].mean()) ** 2))


def test_the_scores_equal_the_direct_least_squares_fit():
    """The base and the base with an extra column score as ``lstsq`` does: the memory change is not a change of value."""
    cols, y, tr, va = _problem()
    scorer = rp.heldout_r2_scorer(cols, y, tr, va)
    extra = np.random.default_rng(3).standard_normal(len(y))
    assert scorer() == pytest.approx(_heldout_r2_lstsq(cols, y, tr, va), abs=1e-9)
    assert scorer(extra) == pytest.approx(_heldout_r2_lstsq(cols, y, tr, va, extra), abs=1e-9)


def test_the_build_peaks_at_a_small_multiple_of_one_train_block():
    """The train block of a sequence base is gathered column by column into one column-major array and factorised in place: about one block for the block, one for Q and a half for the validation rows."""
    cols, y, tr, va = _problem()
    block = int(tr.sum()) * len(cols) * 8
    tracemalloc.start()
    try:
        scorer = rp.heldout_r2_scorer(cols, y, tr, va)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert scorer() is not None
    assert peak < 2.2 * block, peak / block


def test_the_fallback_that_needs_the_train_block_rebuilds_it():
    """A base with a column twice (rank deficient: the QR path is refused) still scores through ``lstsq``, which needs the train block the factorisation consumed."""
    cols, y, tr, va = _problem(n=6000, p=4)
    cols = [*cols, cols[0].copy()]
    scorer = rp.heldout_r2_scorer(cols, y, tr, va)
    assert scorer() == pytest.approx(_heldout_r2_lstsq(cols[:4], y, tr, va), abs=1e-6)
    extra = np.random.default_rng(5).standard_normal(len(y))
    assert scorer(extra) == pytest.approx(_heldout_r2_lstsq(cols[:4], y, tr, va, extra), abs=1e-6)

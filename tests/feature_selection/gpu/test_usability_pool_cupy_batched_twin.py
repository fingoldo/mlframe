"""The batched cupy twin of the usability pair-combo MI kernel agrees with the CPU kernel and is not launch-bound.

The earlier twin launched ~15 cupy kernels and synced several times per combo (~4 ms each), so it lost to the CPU at every size and made the tuning sweep cost minutes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_njit_pool as pool

pytestmark = pytest.mark.skipif(not pool._CUPY_AVAIL, reason="cupy device unavailable")


@pytest.mark.parametrize("n_rows,n_combos", [(2_000, 64), (10_000, 289), (30_000, 578)])
def test_batched_twin_matches_the_cpu_kernel(n_rows, n_combos):
    """Per-combo MI agrees to fp64 round-off, and the same combos are flagged as constant (-1 sentinel)."""
    args = pool._make_usability_inputs({"n_rows": n_rows, "n_combos": n_combos})
    ref = pool._pair_combo_mi_njit(*args)
    got = pool._pair_combo_mi_cupy(*args)
    assert np.abs(ref - got).max() < 1e-9
    assert np.array_equal(ref < 0, got < 0)


def test_batched_twin_splits_into_batches_without_changing_the_result(monkeypatch):
    """A batch budget that forces many small batches gives the same answer as one batch."""
    args = pool._make_usability_inputs({"n_rows": 5_000, "n_combos": 200})
    whole = pool._pair_combo_mi_cupy(*args)
    monkeypatch.setattr(pool, "_GPU_BATCH_BYTES", 8 * 5_000 * 7)
    split = pool._pair_combo_mi_cupy(*args)
    assert np.array_equal(whole, split)


def test_batched_twin_is_much_cheaper_per_combo_than_the_loop_twin():
    """The loop twin costs milliseconds per combo; the batched one must be well below that (an order of magnitude with margin)."""
    args = pool._make_usability_inputs({"n_rows": 10_000, "n_combos": 289})
    for fn in (pool._pair_combo_mi_cupy, pool._pair_combo_mi_cupy_loop):
        fn(*args)
    cp.cuda.Device().synchronize()
    t = time.perf_counter()
    pool._pair_combo_mi_cupy(*args)
    batched = time.perf_counter() - t
    t = time.perf_counter()
    pool._pair_combo_mi_cupy_loop(*args)
    loop = time.perf_counter() - t
    assert batched * 5 < loop, (batched, loop)

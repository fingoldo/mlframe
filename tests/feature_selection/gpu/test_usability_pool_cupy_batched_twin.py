"""The batched cupy twin of the usability pair-combo MI kernel agrees with the CPU kernel and is not launch-bound.

The earlier twin launched ~15 cupy kernels and synced several times per combo (~4 ms each), so it lost to the CPU at every size and made the tuning sweep cost minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_njit_pool as pool
from tests._perf_paired import assert_paired_speedup

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


def test_a_refused_cushion_raises_without_allocating(monkeypatch):
    """GPU11-13/14 class fix: when the VRAM cushion is refused, the function raises before the (n, m) candidate block allocation, instead of attempting and failing it on every remaining batch."""
    import mlframe.feature_selection.filters._fe_gpu_vram as vram_mod

    called = {"n": 0}

    def _declining(*a, **k):
        """Stand-in for fe_gpu_has_vram_cushion that always declines, counting its own calls."""
        called["n"] += 1
        return False

    monkeypatch.setattr(vram_mod, "fe_gpu_has_vram_cushion", _declining)
    args = pool._make_usability_inputs({"n_rows": 2_000, "n_combos": 16})
    with pytest.raises(RuntimeError, match="VRAM cushion"):
        pool._pair_combo_mi_cupy(*args)
    assert called["n"] == 1, "the cushion check must run exactly once, before any allocation"


def test_an_oom_in_the_gpu_branch_notes_the_shared_cooldown(monkeypatch):
    """GPU11-13 follow-up: score_pair_combos's "gpu" branch catches any _pair_combo_mi_cupy failure and falls
    through to CPU; when that failure is an OutOfMemoryError (not the cushion-refusal RuntimeError, nor an
    unrelated error), it must arm the shared cooldown for other call sites."""
    pytest.importorskip("cupy")
    import cupy as cp
    import mlframe.feature_selection.filters._fe_gpu_vram as vram_mod

    vram_mod._reset_gpu_oom_cooldown()
    monkeypatch.setattr(pool, "_usability_backend_choice", lambda *a, **k: "gpu")

    def _boom(*a, **k):
        """Simulate the resident candidate build hitting the pool's own OutOfMemoryError."""
        raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)

    monkeypatch.setattr(pool, "_pair_combo_mi_cupy", _boom)
    rng = np.random.default_rng(3)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.exponential(1.2, n)
    y_codes = rng.integers(0, 3, n).astype(np.int64)
    cy = np.bincount(y_codes).astype(np.float64) / n
    h_y = float(-(cy[cy > 0] * np.log(cy[cy > 0])).sum())
    k_y = int(y_codes.max()) + 1
    out = pool.score_pair_combos(x1, x2, y_codes, (y_codes, h_y, k_y), 10, [0, 1], [0, 1], [0, 1])
    assert out is not None  # falls through to the CPU twin, never raises
    assert vram_mod._in_oom_cooldown() is True
    vram_mod._reset_gpu_oom_cooldown()


def test_batched_twin_is_much_cheaper_per_combo_than_the_loop_twin():
    """The loop twin costs milliseconds per combo; the batched one must beat it by a wide margin on paired interleaved trials, with identical output."""
    args = pool._make_usability_inputs({"n_rows": 10_000, "n_combos": 289})

    def run(fn):
        """Run one twin to completion (device synchronised) and return its host result."""
        out = fn(*args)
        cp.cuda.Device().synchronize()
        return out

    loop_out, batched_out = assert_paired_speedup(
        lambda: run(pool._pair_combo_mi_cupy_loop),
        lambda: run(pool._pair_combo_mi_cupy),
        base_ratio=5.0,
        what="the batched cupy twin",
    )
    np.testing.assert_allclose(batched_out, loop_out, rtol=0, atol=1e-9)

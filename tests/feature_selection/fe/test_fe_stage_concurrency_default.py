"""The concurrent FE stage cascade is on by default and ``MLFRAME_FE_STAGE_CONCURRENCY=0`` turns it off."""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency import concurrency_enabled


def test_default_on_and_off_switch(monkeypatch):
    """Unset or ``1`` -> on; ``0`` -> off."""
    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY", raising=False)
    assert concurrency_enabled()
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "1")
    assert concurrency_enabled()
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "0")
    assert not concurrency_enabled()


def test_vram_weights_split_memory_not_walls():
    """Placement by wall weights, memory by VRAM weights: the shortest stage with the largest memory need gets the largest budget."""
    from mlframe.feature_selection.filters._fe_device_scheduler import plan_stage_slots

    facts = [(0, 3 << 30, 6)]  # one device, 3 GiB free, 6 SMs
    by_wall = plan_stage_slots([14.0, 34.0, 18.0], facts=facts, cushion_bytes=1 << 30, min_budget_bytes=1)
    by_vram = plan_stage_slots([14.0, 34.0, 18.0], facts=facts, cushion_bytes=1 << 30, min_budget_bytes=1, vram_weights=[461.0, 274.0, 110.0])
    assert by_wall[0].budget_bytes < by_wall[1].budget_bytes
    assert by_vram[0].budget_bytes > by_vram[1].budget_bytes > by_vram[2].budget_bytes
    assert sum(s.budget_bytes for s in by_vram) <= 2 << 30


def test_concurrency_declines_when_the_scaled_peaks_do_not_fit(monkeypatch):
    """At 500000 rows the stage peaks measured at 50000 rows, scaled up, exceed a 4 GB card: the cascades run one by one; at 50000 rows they fit."""
    import mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency as C

    class _Pool:
        """A pool with no cached blocks."""

        def free_bytes(self):
            """No cached free bytes."""
            return 0

        def get_limit(self):
            """No cap."""
            return 0

        def used_bytes(self):
            """Nothing in use."""
            return 0

    class _Dev:
        """A 4 GB card with 3 GB free."""

        mem_info = (3 * 2**30, 4 * 2**30)

    import cupy as cp

    monkeypatch.setattr(cp.cuda, "Device", lambda *a, **k: _Dev())
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _Pool())
    assert C._stages_fit_device(C.STAGE_VRAM_WEIGHTS, 50_000) is True
    assert C._stages_fit_device(C.STAGE_VRAM_WEIGHTS, 500_000) is False
    ran = []
    assert C.run_stages_concurrently(object(), [lambda e: ran.append(1)] * 3, n_rows=500_000) is None
    assert not ran


def test_concurrency_respects_the_cap_of_the_memory_pool(monkeypatch):
    """The fit caps the pool (here 1.5 GB): four stages whose scaled peaks (2.6 GB at 100000 rows) fit the free memory of the card but not the cap are declined; without the cap they run."""
    import mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency as C

    class _Pool:
        """A pool with a settable cap and some memory in use."""

        limit = 0

        def free_bytes(self):
            """No cached free bytes."""
            return 0

        def get_limit(self):
            """The cap, 0 for none."""
            return self.limit

        def used_bytes(self):
            """A little in use."""
            return 100 << 20

    class _Dev:
        """A 4 GB card with 3.5 GB free."""

        mem_info = (int(3.5 * 2**30), 4 * 2**30)

    import cupy as cp

    pool = _Pool()
    monkeypatch.setattr(cp.cuda, "Device", lambda *a, **k: _Dev())
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: pool)
    assert C._stages_fit_device(C.STAGE_VRAM_WEIGHTS_WITH_EARLY_A, 100_000) is True
    pool.limit = int(1.5 * 2**30)
    assert C._stages_fit_device(C.STAGE_VRAM_WEIGHTS_WITH_EARLY_A, 100_000) is False


def test_merge_handles_array_attributes():
    """A weighted fit keeps numpy arrays on the estimator (``==`` on them returns an array): merging must not raise, must keep an unchanged array,
    take a changed one, and append new items of a list holding arrays."""
    from types import SimpleNamespace

    import numpy as np

    from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency import merge_stage_state, snapshot

    w = np.arange(5.0)
    target = SimpleNamespace(w=w, codes={"a": np.ones(3)}, recs=[np.zeros(2)], tag="x")
    base = snapshot(target)
    stage = SimpleNamespace(w=w.copy(), codes={"a": np.ones(3), "b": np.full(2, 7.0)}, recs=[np.zeros(2), np.ones(2)], tag="x", new=np.ones(1))
    merge_stage_state(target, base, [stage])
    assert target.w is w  # equal values: the target keeps its own array
    assert set(target.codes) == {"a", "b"} and np.array_equal(target.codes["b"], np.full(2, 7.0))
    assert len(target.recs) == 2 and np.array_equal(target.recs[1], np.ones(2))
    assert np.array_equal(target.new, np.ones(1))
    changed = SimpleNamespace(w=np.arange(5.0) + 1, codes=target.codes, recs=target.recs, tag="x", new=target.new)
    merge_stage_state(target, snapshot(target), [changed])
    assert np.array_equal(target.w, np.arange(5.0) + 1)

"""Unit + regression tests for the FE/MRMR GPU VRAM absolute-cushion guard (``_fe_gpu_vram``).

Runs WITHOUT a real free GPU: ``cp.cuda.runtime.memGetInfo`` is monkeypatched so the free/total VRAM
is fully controlled, and the cupy-absent path is simulated by making ``import cupy`` raise. So these
tests are deterministic on any host (GPU present or not, contended or idle).
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

MOD = "mlframe.feature_selection.filters._fe_gpu_vram"

MB = 1024 * 1024
GB = 1024 * MB


@pytest.fixture()
def vram():
    """Fresh import of the cushion module with the once-per-process pool flag re-armed each test."""
    m = importlib.import_module(MOD)
    m._reset_fe_gpu_pool_limit_flag()
    yield m
    m._reset_fe_gpu_pool_limit_flag()


def _patch_meminfo(monkeypatch, free_b: int, total_b: int = 4 * GB):
    """Force cupy's memGetInfo to report a chosen (free, total) so the cushion runs without a real GPU."""
    import cupy as cp

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (int(free_b), int(total_b)))


def test_declines_when_free_below_cushion(monkeypatch, vram):
    """free < 1 GB cushion -> False (route CPU). The core near-full-card regression."""
    pytest.importorskip("cupy")
    _patch_meminfo(monkeypatch, free_b=322 * MB, total_b=4 * GB)  # the observed live wellbore condition
    assert vram.fe_gpu_has_vram_cushion(0) is False
    assert vram.fe_gpu_has_vram_cushion(10 * MB) is False


def test_allows_when_free_minus_needed_meets_cushion(monkeypatch, vram):
    """free - bytes_needed >= cushion -> True (GPU allowed)."""
    pytest.importorskip("cupy")
    _patch_meminfo(monkeypatch, free_b=3 * GB, total_b=4 * GB)
    assert vram.fe_gpu_has_vram_cushion(0) is True
    assert vram.fe_gpu_has_vram_cushion(1 * GB) is True  # 3 - 1 = 2 GB >= 1 GB cushion


def test_declines_when_needed_pushes_below_cushion(monkeypatch, vram):
    """Enough free, but bytes_needed eats the cushion -> False."""
    pytest.importorskip("cupy")
    _patch_meminfo(monkeypatch, free_b=1500 * MB, total_b=4 * GB)  # 1.5 GB free
    import cupy as cp

    class _EmptyPool:
        def free_bytes(self):
            return 0

        def free_all_blocks(self):
            raise AssertionError("must not free blocks when the pool holds none")

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _EmptyPool())
    assert vram.fe_gpu_has_vram_cushion(0) is True  # 1.5 GB >= 1 GB
    assert vram.fe_gpu_has_vram_cushion(1 * GB) is False  # 1.5 - 1.0 = 0.5 GB < 1 GB cushion


def test_pool_retained_blocks_released_before_declining(monkeypatch, vram):
    """Regression (2026-07-14 wellbore 100k): memGetInfo counts our own cupy pool's internally-FREE
    (retained) blocks as used -- a batch_pair_mi upload of 0.16GB was rejected at free=0.52GB/4GB while
    the pool held ~3GB of instantly-reusable blocks. The cushion must release the pool's free blocks and
    re-probe before declining, so a stale-pool reading can no longer knock the fit off the GPU path."""
    pytest.importorskip("cupy")
    import cupy as cp

    state = {"freed": False}

    class _FatPool:
        def free_bytes(self):
            return 3 * GB

        def free_all_blocks(self):
            state["freed"] = True

    def _meminfo():
        # 0.52 GB free before the pool is flushed; 3.5 GB free after (the observed live condition).
        return (int(3.5 * GB), 4 * GB) if state["freed"] else (int(0.52 * GB), 4 * GB)

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _FatPool())
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", _meminfo)
    assert vram.fe_gpu_has_vram_cushion(int(0.16 * GB)) is True
    assert state["freed"] is True


def test_pool_flush_still_declines_when_genuinely_full(monkeypatch, vram):
    """After the pool flush the card is STILL near-full (another process owns the memory) -> decline."""
    pytest.importorskip("cupy")
    import cupy as cp

    state = {"freed": False}

    class _SmallPool:
        def free_bytes(self):
            return 100 * MB

        def free_all_blocks(self):
            state["freed"] = True

    def _meminfo():
        return (620 * MB, 4 * GB) if state["freed"] else (520 * MB, 4 * GB)

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _SmallPool())
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", _meminfo)
    assert vram.fe_gpu_has_vram_cushion(int(0.16 * GB)) is False
    assert state["freed"] is True


def test_env_override_min_free_mb(monkeypatch, vram):
    """MLFRAME_FE_GPU_MIN_FREE_MB tightens/loosens the absolute floor."""
    pytest.importorskip("cupy")
    _patch_meminfo(monkeypatch, free_b=1500 * MB, total_b=8 * GB)  # big card so tiny-fraction doesn't clamp
    monkeypatch.setenv("MLFRAME_FE_GPU_MIN_FREE_MB", "2048")  # require 2 GB
    assert vram.fe_gpu_has_vram_cushion(0) is False  # 1.5 GB < 2 GB
    monkeypatch.setenv("MLFRAME_FE_GPU_MIN_FREE_MB", "512")  # require 0.5 GB
    assert vram.fe_gpu_has_vram_cushion(0) is True  # 1.5 GB >= 0.5 GB


def test_tiny_card_fraction_clamps_cushion(monkeypatch, vram):
    """On a hypothetical tiny card the cushion is min(abs floor, 50% of total), never > half the card."""
    pytest.importorskip("cupy")
    # total 1 GB -> tiny cushion = min(1 GB, 0.5 GB) = 0.5 GB; 0.6 GB free passes despite < 1 GB abs floor.
    _patch_meminfo(monkeypatch, free_b=600 * MB, total_b=1 * GB)
    assert vram.fe_gpu_has_vram_cushion(0) is True
    # 4 GB card keeps the FULL 1 GB floor (clamp does not bite): 0.9 GB free -> declined.
    _patch_meminfo(monkeypatch, free_b=900 * MB, total_b=4 * GB)
    assert vram.fe_gpu_has_vram_cushion(0) is False


def test_permissive_when_cupy_import_fails(monkeypatch, vram):
    """No cupy (non-GPU host) -> permissive True so those hosts are entirely unaffected."""
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("simulated: cupy unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert vram.fe_gpu_has_vram_cushion(0) is True
    assert vram.fe_gpu_has_vram_cushion(10 * GB) is True


def test_permissive_when_memgetinfo_raises(monkeypatch, vram):
    """A probe error must not block the GPU -> permissive True."""
    pytest.importorskip("cupy")
    import cupy as cp

    def _boom():
        raise RuntimeError("simulated memGetInfo failure")

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", _boom)
    assert vram.fe_gpu_has_vram_cushion(0) is True


def test_ensure_pool_limit_idempotent(monkeypatch, vram):
    """set_limit is called exactly ONCE per process (idempotent once-flag), with fraction= kwarg."""
    pytest.importorskip("cupy")
    import cupy as cp

    calls = []

    class _FakePool:
        def set_limit(self, size=None, fraction=None):
            calls.append((size, fraction))

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _FakePool())
    _patch_meminfo(monkeypatch, free_b=3 * GB, total_b=4 * GB)

    assert vram.ensure_fe_gpu_pool_limit() is True
    assert vram.ensure_fe_gpu_pool_limit() is False  # second call is a no-op
    assert vram.ensure_fe_gpu_pool_limit() is False
    assert len(calls) == 1
    assert calls[0][0] is None and calls[0][1] == pytest.approx(0.6)  # fraction=0.6 default, size unused


def test_ensure_pool_limit_env_fraction(monkeypatch, vram):
    """MLFRAME_FE_GPU_POOL_FRACTION overrides the cap fraction."""
    pytest.importorskip("cupy")
    import cupy as cp

    calls = []

    class _FakePool:
        def set_limit(self, size=None, fraction=None):
            calls.append(fraction)

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _FakePool())
    _patch_meminfo(monkeypatch, free_b=3 * GB, total_b=4 * GB)
    monkeypatch.setenv("MLFRAME_FE_GPU_POOL_FRACTION", "0.4")
    assert vram.ensure_fe_gpu_pool_limit() is True
    assert calls == [pytest.approx(0.4)]


def test_ensure_pool_limit_noop_without_cupy(monkeypatch, vram):
    """No cupy -> ensure returns False (nothing to cap), never raises."""
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("simulated: cupy unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert vram.ensure_fe_gpu_pool_limit() is False


# ================================================================================================
# Joint pool-cap / cushion bound (2026-07-09 fix): pool_cap + cushion must never exceed total VRAM,
# by construction, on ANY card size -- not just empirically fine on the 4 GB reference card.
# ================================================================================================


def test_pool_cap_plus_cushion_never_exceeds_total_on_small_card(monkeypatch, vram):
    """Regression sensor for the self-conflicting-arithmetic bug: on a small (1.5 GB) card the OLD
    independently-chosen fractions (0.6 pool + up-to-0.5-of-total cushion) summed past 100% of total
    VRAM. The fix must clamp the effective pool fraction so pool_cap_bytes + cushion_bytes <= total_b."""
    pytest.importorskip("cupy")
    import cupy as cp

    calls = []

    class _FakePool:
        def set_limit(self, size=None, fraction=None):
            calls.append(fraction)

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _FakePool())
    total_b = int(1.5 * GB)  # small card: old formula's cushion term alone would be 0.5*total = 0.75 GB
    _patch_meminfo(monkeypatch, free_b=int(1.2 * GB), total_b=total_b)
    # requested default fraction 0.6 would, under the OLD formula, coexist with a 0.5*total cushion
    # (0.6 + 0.5 = 1.1 > 1.0 of total) -- self-conflicting regardless of workload.
    assert vram.ensure_fe_gpu_pool_limit() is True
    assert len(calls) == 1
    effective_frac = calls[0]
    cushion_b = vram._cushion_bytes(total_b)
    pool_cap_bytes = effective_frac * total_b
    assert pool_cap_bytes + cushion_b <= total_b + 1, (  # +1 tolerates float rounding
        f"pool_cap({pool_cap_bytes}) + cushion({cushion_b}) exceeds total({total_b}) -- the two mechanisms are not jointly bounded"
    )


def test_pool_cap_unaffected_on_large_card(monkeypatch, vram):
    """On a normal-sized card (well above the cushion's clamp point) the requested fraction is honoured
    unchanged -- the joint bound must not needlessly tighten a card that was never at risk."""
    pytest.importorskip("cupy")
    import cupy as cp

    calls = []

    class _FakePool:
        def set_limit(self, size=None, fraction=None):
            calls.append(fraction)

    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _FakePool())
    _patch_meminfo(monkeypatch, free_b=8 * GB, total_b=16 * GB)
    assert vram.ensure_fe_gpu_pool_limit() is True
    assert calls == [pytest.approx(0.6)]  # requested default honoured exactly; cushion (<=1GB) is tiny vs 16GB


def test_cushion_bytes_shared_by_both_mechanisms(vram):
    """Both the per-dispatch cushion gate and the pool-cap must agree on the SAME cushion definition."""
    total_b = 4 * GB
    assert vram._cushion_bytes(total_b) == min(vram._min_free_mb() * MB, int(total_b * vram._TINY_CARD_CUSHION_FRACTION))


def test_regression_guard_declines_gpu_at_low_free(monkeypatch, vram):
    """Regression: a real guard (the _cmi_cuda batched-CMI gate) must decline the GPU on a near-full card.

    Pins the wired cushion end-to-end: with free VRAM below the cushion, ``_should_use_cuda`` returns False
    (routes CPU) EVEN for a shape that otherwise passes the size/shared-mem gates. Verifies the cushion is not
    bypassable by MLFRAME_FE_GPU_STRICT (checked before the STRICT override)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.info_theory import _cmi_cuda

    # Re-arm any prior GPU-failed poison so the gate is reachable.
    if hasattr(_cmi_cuda, "reset_cmi_gpu_circuit_breaker"):
        try:
            _cmi_cuda.reset_cmi_gpu_circuit_breaker()
        except Exception:
            pass
    monkeypatch.setattr(_cmi_cuda, "cupy_available", lambda: True)
    _patch_meminfo(monkeypatch, free_b=322 * MB, total_b=4 * GB)  # near-full shared card
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")  # STRICT must NOT bypass the cushion

    # A modest shape whose working set easily fits the RELATIVE cap but the card is near-full.
    assert _cmi_cuda._should_use_cuda(n=50_000, p=8, joint_size=64, nbins_x=4, nbins_y=4, nbins_z=4) is False

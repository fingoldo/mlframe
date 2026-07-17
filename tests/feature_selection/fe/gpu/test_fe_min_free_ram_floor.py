"""Pins the ABSOLUTE host-free-RAM floor on the FE candidate-buffer budget (2026-06-13).

The relative ``_FE_BUFFER_RAM_BUDGET_RATIO`` (0.3) cap left no guaranteed free headroom on
small-RAM hosts; ``_fe_effective_buffer_budget_bytes`` now carves an absolute reserve
(``MLFRAME_FE_MIN_FREE_RAM_GB`` / ``_FE_MIN_FREE_RAM_GB``, default 3 GiB) off ``available``
BEFORE the ratio/overhead/worker divide. These tests pin:
  (i)   forced-small ``available`` -> budget never consumes RAM below the reserve;
  (ii)  reserve == 0 (and psutil-missing) -> BYTE-IDENTICAL to the legacy formula;
  (iii) reserve unmet -> budget collapses but the caller's chunk-col floor keeps >= one pair
        (always makes progress, never 0/negative chunk).
"""

import importlib

import pytest

fe = importlib.import_module("mlframe.feature_selection.filters.feature_engineering")


def _legacy_budget(available_bytes, n_workers=1):
    """Byte-exact replica of the pre-floor formula (reserve == 0 must equal this)."""
    if available_bytes < 0:
        return -1
    nw = max(1, int(n_workers))
    raw = float(available_bytes) * fe._FE_BUFFER_RAM_BUDGET_RATIO
    return int(raw / (fe._FE_PEAK_OVERHEAD_FACTOR * nw))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_MIN_FREE_RAM_GB", raising=False)
    yield


def test_floor_leaves_reserve_free():
    """(i) With a small forced ``available`` the budget must fit inside ``available - reserve``,
    scaled by ratio/overhead/workers -- i.e. the buffer can never push free RAM below the reserve."""
    reserve = fe._fe_min_free_ram_bytes()
    assert reserve > 0  # default 1 GiB floor active
    available = reserve + 2 * 2**30  # 2 GiB usable above the floor
    budget = fe._fe_effective_buffer_budget_bytes(available, n_workers=1)
    usable = available - reserve
    # The whole budget*overhead*workers footprint must stay within usable -> leaves >= reserve free.
    assert budget <= usable, (budget, usable)
    assert budget * fe._FE_PEAK_OVERHEAD_FACTOR <= usable + 1  # +1 for int() floor slack
    # And it must equal the legacy formula applied to USABLE (not raw available).
    assert budget == _legacy_budget(usable, n_workers=1)


def test_reserve_is_host_global_not_per_worker():
    """(iii-adjacent) The reserve is subtracted ONCE before the per-worker divide, so more workers
    shrink the per-call budget but never multiply the reserve away."""
    reserve = fe._fe_min_free_ram_bytes()
    available = reserve + 8 * 2**30
    usable = available - reserve
    b1 = fe._fe_effective_buffer_budget_bytes(available, n_workers=1)
    b4 = fe._fe_effective_buffer_budget_bytes(available, n_workers=4)
    assert b1 == _legacy_budget(usable, n_workers=1)
    assert b4 == _legacy_budget(usable, n_workers=4)
    # 4 workers each get ~1/4 the per-call budget; their SUM still fits usable (reserve stays free).
    assert 4 * b4 * fe._FE_PEAK_OVERHEAD_FACTOR <= usable + 4


def test_reserve_zero_is_byte_identical_legacy(monkeypatch):
    """(ii) reserve == 0 (via env) -> byte-identical to the legacy formula for a range of inputs."""
    monkeypatch.setenv("MLFRAME_FE_MIN_FREE_RAM_GB", "0")
    assert fe._fe_min_free_ram_bytes() == 0
    for available in (0, 1, 1023, 2**20, 2**30, 7 * 2**30, 64 * 2**30):
        for nw in (1, 2, 8):
            assert fe._fe_effective_buffer_budget_bytes(available, n_workers=nw) == _legacy_budget(available, n_workers=nw), (available, nw)


def test_reserve_zero_via_constant_is_byte_identical(monkeypatch):
    """(ii) reserve == 0 via the module constant (no env) is also byte-identical legacy."""
    monkeypatch.setattr(fe, "_FE_MIN_FREE_RAM_GB", 0.0)
    assert fe._fe_min_free_ram_bytes() == 0
    for available in (0, 2**30, 16 * 2**30):
        assert fe._fe_effective_buffer_budget_bytes(available) == _legacy_budget(available)


def test_psutil_missing_returns_no_cap():
    """(ii) ``available < 0`` (no psutil) preserves legacy -1 ('no cap'), floor never engages."""
    assert fe._fe_effective_buffer_budget_bytes(-1, n_workers=1) == -1
    assert fe._fe_effective_buffer_budget_bytes(-1, n_workers=8) == -1


def test_reserve_unmet_clamps_to_zero_not_negative():
    """(iii) When available <= reserve the budget collapses to 0 (never negative); the caller's
    chunk-col floor (max(one_pair, ...)) then guarantees progress."""
    reserve = fe._fe_min_free_ram_bytes()
    assert fe._fe_effective_buffer_budget_bytes(reserve) == 0
    assert fe._fe_effective_buffer_budget_bytes(reserve // 2) == 0
    assert fe._fe_effective_buffer_budget_bytes(0) == 0
    # All non-negative -> caller's ``_eff_budget_bytes >= 0`` branch + ``max(one_pair, ...)`` floor holds.
    for available in (0, reserve // 2, reserve, reserve + 1):
        assert fe._fe_effective_buffer_budget_bytes(available) >= 0


def test_env_overrides_constant(monkeypatch):
    """The env var overrides the module constant for the reserve size."""
    monkeypatch.setattr(fe, "_FE_MIN_FREE_RAM_GB", 1.0)
    monkeypatch.setenv("MLFRAME_FE_MIN_FREE_RAM_GB", "4")
    assert fe._fe_min_free_ram_bytes() == int(4 * 2**30)
    # Garbage env value falls back to the constant (no crash).
    monkeypatch.setenv("MLFRAME_FE_MIN_FREE_RAM_GB", "not-a-number")
    assert fe._fe_min_free_ram_bytes() == int(1.0 * 2**30)

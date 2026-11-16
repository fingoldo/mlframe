"""The host conditional null drawn as contingency tables agrees in distribution with the row-shuffle loop and with the device null."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_cmi_redundancy_null as nul
from mlframe.feature_selection.filters._cmi_strata_tables_njit import StratifiedNull


def _codes(n: int, strata: int):
    """x depends on the stratum, y on x: ``(x, y, z)`` int64 codes."""
    rng = np.random.default_rng(0)
    z = rng.integers(0, strata, n).astype(np.int64)
    x = ((rng.integers(0, 10, n) + z) % 10).astype(np.int64)
    y = ((rng.integers(0, 5, n) + x // 4) % 5).astype(np.int64)
    return x, y, z


@pytest.mark.parametrize("n,strata", [(5000, 8), (20000, 32)])
def test_table_null_matches_the_shuffle_loop(monkeypatch, n, strata):
    """The mean and 95 % floor of the table null sit within sampling noise of the exact within-stratum shuffle loop (same Miller-Madow statistic): averaged over 4 seeds each, because one 200-draw mean varies by about 12 %."""
    x, y, z = _codes(n, strata)
    got: dict = {"1": [], "0": []}
    for mode in ("1", "0"):
        monkeypatch.setenv("MLFRAME_CMI_NULL_SHUFFLE", mode)
        for seed in range(1, 5):
            inp = nul._NullInputs(x, y, seed, 0)
            sup = nul._StratifiedSupport(inp, z)
            got[mode].append(nul._conditional_cpu_null(inp, sup, 200, 0.95))
    loop, tab = np.mean(got["1"], axis=0), np.mean(got["0"], axis=0)
    assert tab[1] == pytest.approx(loop[1], rel=0.25)
    assert tab[0] == pytest.approx(loop[0], rel=0.25)


def test_debiased_null_is_clamped_and_below_the_plug_in():
    """The corrected null is never negative and sits below the plug-in null (the plug-in carries the positive Miller-Madow bias)."""
    x, y, z = _codes(5000, 8)
    plug = StratifiedNull(x, y, z, 1).chunk(0, 100, 3)
    mm = StratifiedNull(x, y, z, 1, debias=True).chunk(0, 100, 3)
    assert (mm >= 0).all()
    assert mm.mean() < 0.2 * plug.mean()


def test_null_is_a_function_of_seed_and_permutation_index_only():
    """Any chunking of the permutations is the same data."""
    x, y, z = _codes(3000, 8)
    sn = StratifiedNull(x, y, z, 1, debias=True)
    whole = sn.chunk(0, 40, 9)
    np.testing.assert_array_equal(np.concatenate([sn.chunk(0, 15, 9), sn.chunk(15, 25, 9)]), whole)


def test_device_choice_is_forced_by_the_environment(monkeypatch):
    """``MLFRAME_CMI_PERM_NULL_DEVICE`` forces either side of the tuned choice."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as dev

    monkeypatch.setenv("MLFRAME_CMI_PERM_NULL_DEVICE", "1")
    assert dev.perm_null_on_device(10) is True
    monkeypatch.setenv("MLFRAME_CMI_PERM_NULL_DEVICE", "0")
    assert dev.perm_null_on_device(10**9) is False


def test_oversized_table_falls_back_to_the_loop(monkeypatch):
    """A table above the cell cap returns ``None`` from the table null, so the loop runs."""
    x, y, z = _codes(3000, 8)
    monkeypatch.setattr(nul, "_TABLE_NULL_MAX_CELLS", 10)
    inp = nul._NullInputs(x, y, 1, 0)
    sup = nul._StratifiedSupport(inp, z)
    assert nul._conditional_table_null(inp, sup, 20, 0.95) is None

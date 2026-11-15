"""The device synergy combo scorer equals the numba kernel; detect_synergy_combos returns the same combos with the device forced on or off."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_synergy_screen as screen
from mlframe.feature_selection.filters import _synergy_combo_gpu as g


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def _data(n=6000, p=7):
    """Columns of 2 to 6 levels and a binary target that is the parity of the first two (a pure 2-way synergy)."""
    rng = np.random.default_rng(7)
    levels = [2, 2, 3, 4, 5, 6, 2][:p]
    cols = [rng.integers(0, r, n).astype(np.int64) for r in levels]
    return cols, levels, ((cols[0] + cols[1]) % 2).astype(np.int64)


def _same_combos_and_values(got: list, want: list) -> None:
    """The same combos with the same joint MI to 1e-12, sorted by descending MI; combos whose MI is equal to the last bits (every triple holding the planted pair here) may swap places, because the order of exact ties is arbitrary."""
    assert {c for c, _ in got} == {c for c, _ in want}
    gv, wv = dict(got), dict(want)
    for c in wv:
        assert gv[c] == pytest.approx(wv[c], rel=1e-12, abs=1e-13)
    vals = [v for _, v in got]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_device_equals_numba_for_pairs_and_triples():
    """Every pair and triple matches the host kernel to 1e-12 relative."""
    _needs_gpu()
    cols, levels, y = _data()
    combos = [c for r in (2, 3) for c in itertools.combinations(range(len(cols)), r)]
    dev = g.combo_mm_mi_gpu(cols, combos, levels, y, 2)
    host = g._sweep_host(cols, combos, levels, y, 2)
    assert not np.isnan(dev).any()
    np.testing.assert_allclose(dev, host, rtol=1e-12, atol=1e-13)


def test_detect_synergy_combos_is_identical_with_the_device_forced(monkeypatch):
    """The detected combos and their joint MI (same order) are the same with the device off and on; the planted XOR pair is found."""
    _needs_gpu()
    cols, _, y = _data()
    monkeypatch.setattr(g, "synergy_device_chosen", lambda n, k: False)
    want = screen.detect_synergy_combos(cols, y, range(len(cols)))
    monkeypatch.setattr(g, "synergy_device_chosen", lambda n, k: True)
    calls = {"n": 0}
    real = g.combo_mm_mi_gpu

    def counting(*a, **k):
        """The device scorer, counting its calls so the test proves the device path ran."""
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(g, "combo_mm_mi_gpu", counting)
    got = screen.detect_synergy_combos(cols, y, range(len(cols)))
    assert calls["n"] == 1, "the device scorer must have run (a swallowed exception would send the sweep to the host)"
    _same_combos_and_values(got, want)
    assert (0, 1) in [c for c, _ in want]


def test_candidate_indices_that_are_not_dense_positions(monkeypatch):
    """Feature indices 5, 9, 12, 20 (not 0..3) reach the device through the dense remap and give the host's combos."""
    _needs_gpu()
    cols, _, y = _data(p=4)
    keyed = {5: cols[0], 9: cols[1], 12: cols[2], 20: cols[3]}
    monkeypatch.setattr(g, "synergy_device_chosen", lambda n, k: False)
    want = screen.detect_synergy_combos(keyed, y, list(keyed))
    monkeypatch.setattr(g, "synergy_device_chosen", lambda n, k: True)
    got = screen.detect_synergy_combos(keyed, y, list(keyed))
    _same_combos_and_values(got, want)
    assert (5, 9) in [c for c, _ in got]


def test_oversize_combos_fall_back_to_the_host(monkeypatch):
    """With a tiny table limit the device marks the combos NaN and the dispatcher returns the host values."""
    _needs_gpu()
    cols, levels, y = _data()
    combos = list(itertools.combinations(range(len(cols)), 2))
    monkeypatch.setattr(g, "_MAX_CELLS", 12)
    monkeypatch.setattr(g, "synergy_device_chosen", lambda n, k: True)
    assert np.isnan(g.combo_mm_mi_gpu(cols, combos, levels, y, 2)).any()
    got = g.combo_mm_mi_dispatch(g._sweep_host, cols, combos, levels, y, 2)
    np.testing.assert_allclose(got, g._sweep_host(cols, combos, levels, y, 2), rtol=1e-12, atol=1e-13)

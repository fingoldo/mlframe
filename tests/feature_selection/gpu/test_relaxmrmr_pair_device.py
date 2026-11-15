"""The device RelaxMRMR pair terms equal the numba pair loop; pair_interaction_sum returns the same sum with the device forced, and the device path demonstrably ran."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _relaxmrmr_pair_gpu as g
from mlframe.feature_selection.filters import _relaxmrmr_pair_loop as loop
from mlframe.feature_selection.filters._relaxmrmr_kernels import _cmi_mm_njit, _mi_mm_njit


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")


def _data(n=9000, s=6, levels=(4, 6, 3, 5, 7, 2)):
    """A candidate, a 3-class target tied to it and one selected feature, and the selected features' codes with mixed cardinalities."""
    rng = np.random.default_rng(11)
    sel = [rng.integers(0, levels[q], n).astype(np.int64) for q in range(s)]
    x = rng.integers(0, 5, n).astype(np.int64)
    y = ((x + sel[0]) % 3).astype(np.int64)
    return x, y, sel, np.array(levels[:s], dtype=np.int64)


def test_device_terms_equal_the_numba_pieces():
    """I(x; z | y) and I(x; z) of every pair agree with _cmi_mm_njit / _mi_mm_njit to 1e-12."""
    _needs_gpu()
    x, y, sel, ks = _data()
    pi, pj = np.triu_indices(len(sel), k=1)
    sel_mat = np.stack(sel)
    cmi, mi, fits = g.pair_terms_gpu(x, y, sel_mat, ks, 5, 3, pi.astype(np.int64), pj.astype(np.int64))
    assert fits.all()
    for q in range(len(pi)):
        z = sel[pi[q]] * ks[pj[q]] + sel[pj[q]]
        kz = int(ks[pi[q]] * ks[pj[q]])
        assert cmi[q] == pytest.approx(_cmi_mm_njit(x, z, y, 5, kz, 3), rel=1e-12, abs=1e-13)
        assert mi[q] == pytest.approx(_mi_mm_njit(x, z, 5, kz), rel=1e-12, abs=1e-13)


def test_pair_interaction_sum_matches_with_the_device_forced_and_it_ran(monkeypatch):
    """The sum over all pairs is the same with the device off and on (undersampled pairs excluded as on the host); the device scorer is called once."""
    _needs_gpu()
    x, y, sel, ks = _data()
    cmi_y = np.array([_cmi_mm_njit(c, x, y, int(k), 5, 3) for c, k in zip(sel, ks)])
    marg = np.array([_mi_mm_njit(c, x, int(k), 5) for c, k in zip(sel, ks)])
    monkeypatch.setattr(loop, "_MIN_PARALLEL_WORK", 1)
    monkeypatch.setattr(g, "pair_device_chosen", lambda n, k: False)
    want = loop.pair_interaction_sum(x, y, sel, ks, 5, 3, cmi_y, marg, 5.0)
    monkeypatch.setattr(g, "pair_device_chosen", lambda n, k: True)
    calls = {"n": 0}
    real = g.pair_terms_gpu

    def counting(*a, **k):
        """The device scorer, counting calls so the test proves the device path ran."""
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(g, "pair_terms_gpu", counting)
    got = loop.pair_interaction_sum(x, y, sel, ks, 5, 3, cmi_y, marg, 5.0)
    assert calls["n"] == 1
    assert got == pytest.approx(want, rel=1e-10, abs=1e-12)


def test_oversize_pairs_use_the_numba_loop(monkeypatch):
    """With a tiny table limit nothing fits the device and the numba loop gives the whole sum."""
    _needs_gpu()
    x, y, sel, ks = _data()
    cmi_y = np.array([_cmi_mm_njit(c, x, y, int(k), 5, 3) for c, k in zip(sel, ks)])
    marg = np.array([_mi_mm_njit(c, x, int(k), 5) for c, k in zip(sel, ks)])
    monkeypatch.setattr(loop, "_MIN_PARALLEL_WORK", 1)
    monkeypatch.setattr(g, "pair_device_chosen", lambda n, k: False)
    want = loop.pair_interaction_sum(x, y, sel, ks, 5, 3, cmi_y, marg, 5.0)
    monkeypatch.setattr(g, "pair_device_chosen", lambda n, k: True)
    monkeypatch.setattr(g, "_MAX_CELLS", 50)
    assert loop.pair_interaction_sum(x, y, sel, ks, 5, 3, cmi_y, marg, 5.0) == pytest.approx(want, rel=1e-12, abs=1e-13)

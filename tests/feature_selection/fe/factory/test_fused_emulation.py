"""Numpy emulation of the fused recompute-instead-of-store CUDA kernel versus the njit reference (the kernel itself was never run on a GPU)."""

import numpy as np

from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes._synthetic import NB, NCLS, make
from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes.offset_fused_cuda import (
    _OFFSET_FUSED_SRC,
    MODE_MIX,
    MODE_SHIFT,
    emulate_null,
    emulate_offset_fused,
)
from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes.offset_kernels import DEFAULT_FINE_BINS, score_offset_grid_b


def _grid():
    """3 forms x 4 offsets at n = 2000 with the flat parameter table the kernel takes."""
    U, V, T, yc = make(2000, K=3, G=4)
    P = np.zeros((12, 3))
    P[:, 0] = T.reshape(-1)
    return U, V, T, yc, P


def test_shift_mode_matches_njit_reference():
    """MODE_SHIFT emulation reproduces the CPU reference MI to rounding."""
    U, V, T, yc, P = _grid()
    mi, _ = emulate_offset_fused(U, V, P, 4, yc, NCLS, NB, MODE_SHIFT)
    ref, _ = score_offset_grid_b(U, V, T, yc, NCLS, NB, DEFAULT_FINE_BINS)
    assert np.abs(mi - ref.reshape(-1)).max() < 1e-12


def test_mix_mode_with_unit_alpha_equals_shift_up_to_rounding():
    """MODE_MIX with alpha = 1, beta = t, gamma = 0 is the shift family; only rounding-level differences are allowed."""
    U, V, T, yc, P = _grid()
    mi, _ = emulate_offset_fused(U, V, P, 4, yc, NCLS, NB, MODE_SHIFT)
    Pm = np.stack([np.ones(12), T.reshape(-1), np.zeros(12)], 1)
    mi_mix, _ = emulate_offset_fused(U, V, Pm, 4, yc, NCLS, NB, MODE_MIX)
    assert np.abs(mi_mix - mi).max() < 1e-6


def test_ties_match_reference():
    """Heavily tied operands give the same MI as the njit reference."""
    rng = np.random.default_rng(3)
    Ut, Vt = np.round(rng.random((1, 1500)) * 4), np.round(rng.random((1, 1500)) * 3)
    yt = rng.integers(0, 4, 1500)
    Pt = np.array([[0.0, 0, 0], [0.5, 0, 0]])
    a, _ = emulate_offset_fused(Ut, Vt, Pt, 2, yt, 4, NB)
    b, _ = score_offset_grid_b(Ut, Vt, Pt[:, :1].reshape(1, 2), yt.astype(np.int64), 4, NB, DEFAULT_FINE_BINS)
    assert np.abs(a - b.reshape(-1)).max() < 1e-12


def test_null_emulation_shape_and_range():
    """The permutation-null emulation returns one MI per (permutation, candidate), non-negative."""
    U, V, _, yc, P = _grid()
    _, edges = emulate_offset_fused(U, V, P, 4, yc, NCLS, NB, MODE_SHIFT)
    rng = np.random.default_rng(3)
    Yp = np.stack([rng.permutation(yc[:600]) for _ in range(3)]).astype(np.int8)
    nl = emulate_null(U[:, :600], V[:, :600], P, 4, Yp, NCLS, NB, edges)
    assert nl.shape == (3, 12)
    assert (nl >= 0).all()


def test_cuda_source_is_present():
    """The CUDA source string is shipped alongside the emulation (compiled only on a GPU box)."""
    assert "offset_fused_mi" in _OFFSET_FUSED_SRC

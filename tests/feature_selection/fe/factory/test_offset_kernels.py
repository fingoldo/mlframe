"""Parity of the fused offset-grid scorer prototypes with the materialise-then-score reference, edge cases and replay."""

import numba
import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes._synthetic import NB, NCLS, make, ref
from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes.offset_kernels import (
    DEFAULT_FINE_BINS,
    apply_offset_product,
    apply_offset_product_1d,
    candidate_codes_b,
    score_offset_grid_a,
    score_offset_grid_b,
)
from mlframe.feature_selection.filters._fe_edge_mi import _edge_bin_codes

N, K, G = 4000, 4, 4


@pytest.fixture(scope="module")
def case():
    """Small case-2 grid: (U, V, T, y codes, reference MI, reference candidate matrix)."""
    U, V, T, yc = make(N, K=K, G=G)
    r, M = ref(U, V, T, yc)
    return U, V, T, yc, r, M


def test_both_variants_match_reference_exactly(case):
    """Variant a (partition) and b (histogram refine) reproduce the reference MI to rounding, with no fallback gathers."""
    U, V, T, yc, r, _ = case
    a = score_offset_grid_a(U, V, T, yc, NCLS, NB)
    b, fallbacks = score_offset_grid_b(U, V, T, yc, NCLS, NB, DEFAULT_FINE_BINS)
    assert np.abs(a - r).max() < 1e-12
    assert np.abs(b - r).max() < 1e-12
    assert int(fallbacks.sum()) == 0


def test_bin_codes_match_repo_binning(case):
    """Single-candidate bin codes of variant b equal the repo's ``_edge_bin_codes`` code for code."""
    U, V, T, _, _, M = case
    for k, g in [(0, 0), (3, 2)]:
        codes = np.empty(N, np.int8)
        candidate_codes_b(U[k], V[k], T[k, g], NB, DEFAULT_FINE_BINS, codes)
        ref_codes = np.empty(N, np.int32)
        _edge_bin_codes(np.ascontiguousarray(M[:, k * G + g]), NB, ref_codes)
        assert np.array_equal(codes, ref_codes)


def _edge_inputs():
    """Degenerate inputs: heavy ties, constants, non-finite values, tiny n and a Cauchy-tailed operand."""
    rng = np.random.default_rng(1)
    return {
        "ties": (np.round(rng.random(500) * 5), np.round(rng.random(500) * 3)),
        "constant": (np.ones(500), np.ones(500)),
        "nan_inf": (np.where(rng.random(500) < 0.02, np.nan, rng.random(500)), np.where(rng.random(500) < 0.02, np.inf, rng.random(500))),
        "tiny_n": (rng.random(37), rng.random(37)),
        "heavy_tail": (rng.standard_cauchy(500), rng.random(500)),
    }


@pytest.mark.parametrize("name", list(_edge_inputs()))
def test_edge_cases_match_reference(name):
    """Ties, constants, NaN / inf, tiny n and heavy tails give the same MI as the reference (non-finite candidates scrubbed to 0)."""
    uu, vv = _edge_inputs()[name]
    y = np.random.default_rng(2).integers(0, 4, uu.size).astype(np.int64)
    UU, VV, TT = uu[None].copy(), vv[None].copy(), np.array([[0.0, 0.3, -1.0]])
    r, _ = ref(UU, VV, TT, y)
    assert np.abs(score_offset_grid_a(UU, VV, TT, y, 4, NB) - r).max() < 1e-12
    assert np.abs(score_offset_grid_b(UU, VV, TT, y, 4, NB, DEFAULT_FINE_BINS)[0] - r).max() < 1e-12


def test_replay_is_exact(case):
    """Recipe replay (batch and single column) equals the scrubbed numpy expression bit for bit."""
    U, V, T, _, _, _ = case
    expect = np.nan_to_num((U[0] + T[0, 3]) * V[0], nan=0, posinf=0, neginf=0)
    out = np.empty(N)
    assert np.array_equal(apply_offset_product_1d(U[0], V[0], T[0, 3], out), expect)
    batch = np.empty((K, N))
    apply_offset_product(U, V, T[:, 3].copy(), batch)
    assert np.array_equal(batch[0], expect)


def test_result_independent_of_thread_count(case):
    """The scorer returns identical numbers with 1 and 2 numba threads (one thread per candidate, fixed arithmetic)."""
    U, V, T, yc, _, _ = case
    saved = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = score_offset_grid_b(U, V, T, yc, NCLS, NB, DEFAULT_FINE_BINS)[0]
        numba.set_num_threads(min(2, saved))
        two = score_offset_grid_b(U, V, T, yc, NCLS, NB, DEFAULT_FINE_BINS)[0]
    finally:
        numba.set_num_threads(saved)
    assert np.array_equal(one, two)

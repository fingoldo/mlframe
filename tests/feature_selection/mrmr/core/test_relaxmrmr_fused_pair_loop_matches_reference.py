"""The fused RelaxMRMR pair loop computes exactly what the per-pair Python loop computed.

The 3-way interaction term sums a co-information over every pair of selected features. That loop ran in Python, three kernel round trips and
one fresh n-length composite array per pair, repeated for every candidate of every greedy round. It now runs as one parallel kernel over a
flat pair index, with a per-thread composite buffer.

Summation order changes when the pairs are summed from an array rather than accumulated in loop order, so the reference here is the same
arithmetic in the original order, compared at floating-point tolerance rather than bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._relaxmrmr_kernels import _cmi_mm_njit, _composite_codes_njit, _mi_mm_njit
from mlframe.feature_selection.filters._relaxmrmr_pair_loop import pair_interaction_sum


def _reference_pair_sum(x_int, y_int, sel_int, K_sel, K_x, K_y, cmi_y_mm, marg_mm, min_rows_per_cell) -> float:
    """The serial per-pair loop this replaced, kept verbatim as the thing the kernel has to agree with."""
    n_S = len(sel_int)
    n_rows = float(x_int.shape[0])
    inter = 0.0
    for i in range(n_S):
        for j in range(i + 1, n_S):
            K_i, K_j = K_sel[i], K_sel[j]
            if n_rows < float(min_rows_per_cell) * K_x * K_i * K_j * K_y:
                continue
            z_pair = _composite_codes_njit(sel_int[i], sel_int[j], K_j)
            cmi_ij = _cmi_mm_njit(x_int, z_pair, y_int, K_x, K_i * K_j, K_y)
            mi_x_zz = _mi_mm_njit(x_int, z_pair, K_x, K_i * K_j)
            inter += (cmi_y_mm[i] + cmi_y_mm[j] - cmi_ij) - (marg_mm[i] + marg_mm[j] - mi_x_zz)
    return inter


def _case(n_S: int, n: int = 4000, seed: int = 0, varying_k: bool = True):
    """A candidate, a target and ``n_S`` selected columns, with per-column cardinalities that differ when asked."""
    rng = np.random.default_rng(seed)
    K_x, K_y = 4, 3
    x = rng.integers(0, K_x, size=n).astype(np.int64)
    y = rng.integers(0, K_y, size=n).astype(np.int64)
    K_sel = [2 + (idx % 3 if varying_k else 0) for idx in range(n_S)]
    sel = [rng.integers(0, K_sel[idx], size=n).astype(np.int64) for idx in range(n_S)]
    marg = np.array([_mi_mm_njit(x, sel[idx], K_x, K_sel[idx]) for idx in range(n_S)])
    cmi_y = np.array([_cmi_mm_njit(x, sel[idx], y, K_x, K_sel[idx], K_y) for idx in range(n_S)])
    return x, y, sel, K_sel, K_x, K_y, cmi_y, marg


@pytest.mark.parametrize("n_S", [2, 3, 5, 8])
def test_the_fused_loop_matches_the_serial_one(n_S):
    """Across selected-set sizes, and with per-column cardinalities that differ, the totals agree."""
    args = _case(n_S)
    got = pair_interaction_sum(*args, 1.0)
    want = _reference_pair_sum(*args, 1.0)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12), f"n_S={n_S}: {got} vs {want}"


def test_a_uniform_cardinality_set_agrees_too():
    """Equal per-column cardinality is the common case and exercises a different composite stride."""
    args = _case(6, varying_k=False, seed=3)
    assert pair_interaction_sum(*args, 1.0) == pytest.approx(_reference_pair_sum(*args, 1.0), rel=1e-12, abs=1e-12)


def test_pairs_the_sample_cannot_support_are_skipped_the_same_way():
    """A high ``min_rows_per_cell`` makes most pairs inestimable; the kernel must drop exactly the ones the loop dropped."""
    args = _case(5, n=600, seed=1)
    for min_rows in (1.0, 5.0, 50.0, 500.0):
        got = pair_interaction_sum(*args, min_rows)
        want = _reference_pair_sum(*args, min_rows)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), f"min_rows={min_rows}: {got} vs {want}"
    assert pair_interaction_sum(*args, 1e9) == 0.0, "with no estimable pair the term must be exactly zero"


def test_fewer_than_two_selected_features_has_no_pair_term():
    """There is no pair to sum over, so the term is zero rather than an empty-reduction error."""
    for n_S in (0, 1):
        x, y, sel, K_sel, K_x, K_y, cmi_y, marg = _case(max(n_S, 1))
        assert pair_interaction_sum(x, y, sel[:n_S], K_sel[:n_S], K_x, K_y, cmi_y[:n_S], marg[:n_S], 1.0) == 0.0


def test_the_score_itself_is_unchanged_by_the_rewrite():
    """Through the public score, with the interaction term on: the value must match a recomputation from its parts."""
    from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score

    x, y, sel, K_sel, K_x, K_y, cmi_y, marg = _case(4, seed=7)
    alpha = 0.5
    score = relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=alpha, min_rows_per_cell=1.0)
    with_zero_alpha = relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=0.0, min_rows_per_cell=1.0)
    assert np.isfinite(score) and np.isfinite(with_zero_alpha)
    n_S = len(sel)
    expected_gap = alpha / (n_S * (n_S - 1) / 2.0) * _reference_pair_sum(x, y, sel, K_sel, K_x, K_y, cmi_y, marg, 1.0)
    assert score - with_zero_alpha == pytest.approx(expected_gap, rel=1e-10, abs=1e-12)


def test_both_sides_of_the_parallel_threshold_agree():
    """The small-table path and the parallel path are dispatched on work size, and must return the same number."""
    from mlframe.feature_selection.filters import _relaxmrmr_pair_loop as loop

    args = _case(6, n=3000, seed=11)
    forced_serial = loop._serial_pair_sum(*args, 1.0)
    assert forced_serial == pytest.approx(_reference_pair_sum(*args, 1.0), rel=1e-12, abs=1e-12)
    assert loop._pair_interaction_terms_njit is not None  # the parallel kernel is the other side of the dispatch
    assert loop.pair_interaction_sum(*args, 1.0) == pytest.approx(forced_serial, rel=1e-12, abs=1e-12)

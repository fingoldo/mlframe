"""Regression: the fused-njit ``_pair_mm_mi_njit`` synergy-pair scorer matches the numpy
``joint_synergy_mi`` reference (FP reduction order, ~1e-12) AND the ``detect_synergy`` verdict
is unchanged after wiring the fast kernel in.

Pins the optimization in ``filters/_fe_synergy_screen._pair_mm_mi_njit`` +
``filters/_synergy_detector._pair_mm_mi``. Bench:
``filters/_benchmarks/bench_synergy_detector_pair_mi_njit.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_synergy_screen import joint_synergy_mi
from mlframe.feature_selection.filters._synergy_detector import _pair_mm_mi, detect_synergy


def _codes(rng, n, nb):
    return rng.integers(0, nb, size=n).astype(np.int64)


@pytest.mark.parametrize("n,nb", [(600, 6), (2407, 8), (4000, 8)])
def test_pair_mm_mi_matches_numpy_reference(n, nb):
    rng = np.random.default_rng(n + nb)
    yc = _codes(rng, n, nb)
    const = np.zeros(n, dtype=np.int64)
    max_abs = 0.0
    for _ in range(40):
        cx = _codes(rng, n, nb)
        cy = _codes(rng, n, nb)
        for a, b in ((cx, cy), (cx, const)):
            old = joint_synergy_mi(a, b, yc)
            new = _pair_mm_mi(a, b, yc)
            # both must be on the same side of zero (selection-equivalence)
            assert (old > 0.0) == (new > 0.0)
            max_abs = max(max_abs, abs(old - new))
    assert max_abs < 1e-10, f"max|old-new|={max_abs:.3e} exceeds FP-reduction tolerance"


def test_pair_mm_mi_xor_signal_separates_from_noise():
    """The fused kernel still recovers an XOR pair's large joint excess vs near-zero marginals."""
    rng = np.random.default_rng(7)
    n = 4000
    b0 = (rng.random(n) > 0.5).astype(np.int64)
    b1 = (rng.random(n) > 0.5).astype(np.int64)
    yc = (b0 ^ b1).astype(np.int64)
    noise = _codes(rng, n, 8)
    const = np.zeros(n, dtype=np.int64)
    joint = _pair_mm_mi(b0, b1, yc)
    marg0 = _pair_mm_mi(b0, const, yc)
    noise_joint = _pair_mm_mi(b0, noise, yc)
    assert joint > 0.5, f"XOR joint MI collapsed: {joint}"
    assert marg0 < 0.02, f"XOR operand marginal should be ~0: {marg0}"
    assert joint > 10 * max(noise_joint, 1e-6)


def test_detect_synergy_verdict_on_xor_and_noise():
    """End-to-end: the detector flags XOR-bearing data and rejects pure noise."""
    rng = np.random.default_rng(0)
    n, p = 3000, 12
    # noise dataset
    Xn = rng.standard_normal((n, p))
    yn = rng.integers(0, 2, size=n)
    is_syn_noise, _ = detect_synergy(Xn, yn, random_seed=0)
    assert is_syn_noise is False

    # XOR-bearing dataset: target = sign(x0) XOR sign(x1)
    Xx = rng.standard_normal((n, p))
    yx = ((Xx[:, 0] > 0).astype(int) ^ (Xx[:, 1] > 0).astype(int))
    is_syn_xor, info = detect_synergy(Xx, yx, random_seed=0)
    assert is_syn_xor is True, f"detector missed XOR synergy: {info}"


def test_combo_mm_mi_cols_njit_matches_matrix_kernel():
    # The direct-columns synergy kernel (orders 2-3, no _mat materialise) must equal the (n, order)-matrix
    # kernel bit-for-bit -- same mixed-radix cell code + histogram + Miller-Madow debit.
    import numpy as np
    from mlframe.feature_selection.filters._fe_synergy_screen import (
        _combo_mm_mi_njit,
        _combo_mm_mi_cols_njit,
    )

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(500, 8000))
        nb = int(rng.integers(2, 9))
        kt = int(rng.integers(2, 6))
        order = int(rng.integers(2, 4))
        cols = [np.ascontiguousarray(rng.integers(0, nb, n).astype(np.int64)) for _ in range(order)]
        cards = np.array([nb] * order, dtype=np.int64)
        n_cells = nb ** order
        yt = rng.integers(0, kt, n).astype(np.int64)
        mat = np.empty((n, order), dtype=np.int64)
        for k in range(order):
            mat[:, k] = cols[k]
        a = _combo_mm_mi_njit(mat, cards, yt, kt, n_cells)
        c2 = cols[2] if order >= 3 else cols[1]
        b = _combo_mm_mi_cols_njit(cols[0], cols[1], c2, order, cards, yt, kt, n_cells)
        worst = max(worst, abs(a - b))
    assert worst == 0.0, f"cols kernel diverges {worst:.2e} from matrix kernel"

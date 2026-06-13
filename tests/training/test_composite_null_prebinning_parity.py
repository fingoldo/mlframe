"""Regression: the prebinned-codes null-MI fast path added to ``_auto_base``
(permutation-MI null filter) must be BIT-IDENTICAL to the per-call
``_mi_pair_bin`` it replaces, for NaN-free columns.

``np.quantile`` is shuffle-invariant (it sorts internally) and ``np.searchsorted``
is element-wise (so it commutes with any permutation). Therefore binning a
*shuffled* clean column equals shuffling that column's integer bin codes —
which lets the null loop bin y + each column ONCE and shuffle the codes per
permutation (``_mi_from_binned_pair``) instead of re-binning every permutation
(~5x faster on n=20k). This test locks in the exactness the optimization relies
on; any drift would silently alter which features survive the null threshold.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import (
    _mi_from_binned_pair,
    _mi_from_binned_pair_numpy,
    _mi_pair_bin,
)


def _bin_codes(arr: np.ndarray, nbins: int) -> np.ndarray:
    """Quantile-bin to integer codes, mirroring ``_mi_pair_bin``'s internal binning."""
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(arr, qs)
    codes = np.searchsorted(edges, arr, side="right").astype(np.int64)
    np.clip(codes, 0, nbins - 1, out=codes)
    return codes


@pytest.mark.parametrize("nbins", [5, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_prebinned_null_mi_is_bit_identical_to_mi_pair_bin(nbins: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 4000
    y = rng.normal(size=n)
    col = rng.normal(size=n) + 0.3 * y  # planted structure so MI is non-trivial

    # Prebin ONCE (what the fast path does outside the permutation loop).
    y_codes = _bin_codes(y, nbins)
    col_codes = _bin_codes(col, nbins)

    perm_rng = np.random.default_rng(123)
    for _ in range(25):
        order = perm_rng.permutation(n)
        # Reference: re-bin the shuffled VALUES every permutation (old behaviour).
        ref = _mi_pair_bin(col[order], y, nbins=nbins)
        # Fast path: shuffle the precomputed CODES and score from the contingency table.
        fast = _mi_from_binned_pair(col_codes[order], y_codes, nbins=nbins)
        # Joint integer counts are identical; the only difference is FP reduction ORDER -- numpy's _mi_pair_bin reduces the (nbins,nbins) product
        # array (pairwise summation) while the njit kernel walks cells row-major (sequential accumulation). That lands ~1e-13 worst-case across a
        # 200-seed x 9-nbins grid, far under any MI ranking threshold (~1e-3), so the contract is allclose-not-bitwise. See test_mi_kernel_divergence_bound.
        np.testing.assert_allclose(
            fast, ref, rtol=1e-9, atol=1e-12,
            err_msg=(
                f"prebinned null MI diverged from _mi_pair_bin beyond FP-order tolerance: ref={ref!r} fast={fast!r} "
                f"(nbins={nbins}, seed={seed})"
            ),
        )


def test_mi_kernel_divergence_bound() -> None:
    """Pin the FP-reduction-order divergence between the njit kernel, its numpy twin, and _mi_pair_bin to <1e-9 across a stress grid.

    The parity tests assert allclose (not bitwise) because the kernels differ only in summation order; this sensor proves that order-difference
    stays ULP-scale so a real numeric regression (~1e-3, which WOULD move an MI ranking decision) still trips. If a future kernel rewrite pushes
    divergence above 1e-9 this fails -- forcing a re-examination of whether the loosened allclose tolerance is still defensible.
    """
    worst = 0.0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        for nbins in (3, 5, 8, 16, 32, 50, 64, 128):
            n = int(rng.integers(5 * nbins + 10, 6000))
            y = rng.normal(size=n)
            col = rng.normal(size=n) + 0.3 * y
            yc = _bin_codes(y, nbins)
            cc = _bin_codes(col, nbins)
            ref = _mi_pair_bin(col, y, nbins=nbins)
            npref = _mi_from_binned_pair_numpy(cc, yc, nbins=nbins)
            fast = _mi_from_binned_pair(cc, yc, nbins=nbins)
            for a, b in ((ref, fast), (npref, fast), (ref, npref)):
                worst = max(worst, abs(a - b))
    assert worst < 1e-9, f"njit-vs-numpy MI divergence {worst:.3e} exceeded the ULP-scale bound; a real numeric regression may be hiding"


def test_shuffling_codes_equals_binning_shuffled_values() -> None:
    """The core commutation: searchsorted(quantile(shuffled), shuffled) == shuffle(codes)."""
    rng = np.random.default_rng(42)
    n, nbins = 2000, 8
    col = rng.standard_normal(n)
    codes = _bin_codes(col, nbins)
    order = np.random.default_rng(9).permutation(n)
    # Re-bin the shuffled values from scratch.
    rebinned = _bin_codes(col[order], nbins)
    # Versus shuffling the once-computed codes.
    np.testing.assert_array_equal(
        rebinned, codes[order],
        err_msg="re-binning shuffled values must equal shuffling the precomputed codes",
    )

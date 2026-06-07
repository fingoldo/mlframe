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

from mlframe.training.composite.discovery.screening import _mi_from_binned_pair, _mi_pair_bin


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
        assert ref == fast, (
            f"prebinned null MI diverged from _mi_pair_bin: ref={ref!r} fast={fast!r} "
            f"(nbins={nbins}, seed={seed}) — optimization is no longer bit-identical"
        )


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

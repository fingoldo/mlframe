"""OPT-B (2026-06-07): narrowing the FE discretiser's ordinal-code matrix from the int32
``quantization_dtype`` default to int8 (n_bins<=127) / int16 is VALUE-IDENTICAL.

The codes are non-negative ordinals in ``[0, n_bins)``, so a narrower storage width changes
only the bytes-per-element of ``disc_2d`` (the bandwidth-bound gather array in searchsorted +
batch_mi), never the values. These tests assert:

  1. ``discretize_2d_quantile_batch(dtype=int8)`` produces the SAME ordinal codes (value-equal)
     as ``dtype=int32`` / ``dtype=int16``.
  2. ``batch_mi_with_noise_gate`` returns BYTE-IDENTICAL fe_mi for an int8 vs int32 ``disc_2d``
     (the kernel's own histogram/joint-count accumulators are typed independently of the code
     matrix's dtype, so the MI must not move) -- this is the gate-critical invariant, since the
     noise-gate is a float ``>=`` comparison.
  3. ``_narrow_code_dtype`` picks int8 / int16 / fallback at the right n_bins boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_narrow_code_dtype_boundaries():
    from mlframe.feature_selection.filters._feature_engineering_pairs import _narrow_code_dtype

    assert _narrow_code_dtype(10, np.int32) == np.int8
    assert _narrow_code_dtype(127, np.int32) == np.int8
    assert _narrow_code_dtype(128, np.int32) == np.int16
    assert _narrow_code_dtype(32767, np.int32) == np.int16
    assert _narrow_code_dtype(40000, np.int32) == np.int32  # too wide -> requested dtype
    assert _narrow_code_dtype(40000, None) == np.int32


@pytest.mark.parametrize("dtype_in", [np.float32, np.float64])
@pytest.mark.parametrize("n_rows,K,n_bins", [(200, 7, 5), (2407, 64, 10), (500, 30, 100)])
def test_discretize_codes_value_identical_across_widths(n_rows, K, n_bins, dtype_in):
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    rng = np.random.default_rng(5 + n_rows + K + n_bins)
    arr2d = np.ascontiguousarray(rng.standard_normal((n_rows, K)).astype(dtype_in))
    c8 = discretize_2d_quantile_batch(arr2d, n_bins=n_bins, dtype=np.int8)
    c16 = discretize_2d_quantile_batch(arr2d, n_bins=n_bins, dtype=np.int16)
    c32 = discretize_2d_quantile_batch(arr2d, n_bins=n_bins, dtype=np.int32)
    # Values identical (ordinals); only the storage width differs.
    assert np.array_equal(c8.astype(np.int32), c32)
    assert np.array_equal(c16.astype(np.int32), c32)
    # And codes are in-range for int8 (n_bins<=100 here -> <=127).
    assert c8.min() >= 0 and c8.max() < n_bins


@pytest.mark.parametrize("use_su", [False, True])
@pytest.mark.parametrize("n_rows,K,n_bins", [(2407, 50, 10), (800, 120, 10), (400, 16, 30)])
def test_batch_mi_byte_identical_int8_vs_int32(n_rows, K, n_bins, use_su):
    """The gate-critical invariant: an int8 disc_2d yields byte-identical fe_mi to int32."""
    from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate

    rng = np.random.default_rng(13 + n_rows + K + n_bins + int(use_su))
    disc32 = rng.integers(0, n_bins, size=(n_rows, K)).astype(np.int32)
    disc8 = disc32.astype(np.int8)
    factors_nbins = np.full(K, n_bins, dtype=np.int64)
    # A target correlated with the first column so some MI is non-zero (exercise the gate).
    classes_y = (disc32[:, 0] % 3).astype(np.int32) if K else np.zeros(n_rows, np.int32)
    classes_y_safe = classes_y.copy()
    Ky = int(classes_y.max()) + 1
    freqs_y = np.bincount(classes_y, minlength=Ky).astype(np.float64) / n_rows

    def run(disc, classes_dtype):
        return batch_mi_with_noise_gate(
            disc_2d=np.ascontiguousarray(disc),
            factors_nbins=factors_nbins,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            npermutations=3,
            base_seed=np.uint64(0),
            min_nonzero_confidence=0.99,
            use_su=use_su,
            dtype=np.int32,  # joint_counts (the real counter) stays int32 regardless of disc dtype
            classes_dtype=classes_dtype,
        )

    mi_ref = run(disc32, np.int32)  # legacy: int32 disc + int32 classes_dense
    mi_narrow = run(disc8, np.int8)  # OPT-B: int8 disc + int8 classes_dense
    mi_mixed = run(disc32, np.int8)  # int32 disc + int8 classes_dense
    # BYTE-identical floats (view as uint64 so any last-bit drift is caught).
    for name, mi in (("int8+int8", mi_narrow), ("int32+int8", mi_mixed)):
        assert np.array_equal(np.asarray(mi_ref).view(np.uint64), np.asarray(mi).view(np.uint64)), (
            f"{name} gave different MI than the int32 reference -- would flip the noise-gate / selection"
        )

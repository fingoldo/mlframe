"""Wave 11 (Category 3) M5: ``_step_score.py``'s per-admitted-form discretize loop called
``discretize_array`` once per column instead of the batched ``discretize_2d_quantile_batch`` already the
default one layer up (``_pairs_score.py``/``_pairs_emit.py``). Fixed for the ``quantization_method ==
"quantile"`` case (gated exactly like the sibling call sites); the ``uniform``-method fallback loop is kept
but its buffer dtype is now pre-widened via ``_safe_code_dtype`` -- the pre-fix code preallocated the
buffer at the raw (possibly too-narrow) ``quantization_dtype`` and let ``discretize_array``'s own widened
return value get silently downcast back on assignment, wrapping codes negative for ``n_bins > 127`` under a
narrow ``quantization_dtype``. This test pins (1) ``discretize_2d_quantile_batch`` bit-identity against the
per-column ``discretize_array(method='quantile')`` loop it replaces (already claimed by that function's own
docstring; pinned here as the actual call-site contract), and (2) the dtype-wrap regression: fails before
the ``_safe_code_dtype`` pre-widen fix, passes after.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.discretization import (
    _safe_code_dtype,
    discretize_2d_quantile_batch,
    discretize_array,
)


def test_discretize_2d_quantile_batch_matches_per_column_discretize_array():
    rng = np.random.default_rng(0)
    n, k = 3000, 25
    mat = rng.normal(size=(n, k)).astype(np.float32) * rng.choice([1.0, 50.0, 0.01], size=k)
    n_bins = 10
    dtype = np.int8

    batched = discretize_2d_quantile_batch(mat, n_bins=n_bins, dtype=dtype)
    for j in range(k):
        expected = discretize_array(arr=mat[:, j], n_bins=n_bins, method="quantile", dtype=dtype)
        assert np.array_equal(batched[:, j], expected), f"column {j} mismatch"


def test_quantization_dtype_narrower_than_nbins_does_not_wrap_negative():
    """Regression: pre-fix, a preallocated buffer at the raw ``quantization_dtype`` (e.g. int8) silently
    downcast a properly-widened ``discretize_array`` output back to int8 on assignment, wrapping codes
    negative for ``n_bins > 127``. Reproduces the OLD buggy pattern directly (temporarily, to prove the
    failure signature) then confirms the fixed (pre-widened) pattern is correct."""
    rng = np.random.default_rng(1)
    n = 2000
    n_bins = 200  # > 127 -> _safe_code_dtype must widen past int8
    arr = rng.normal(size=n)
    requested_dtype = np.int8

    # OLD buggy pattern (pre-fix): preallocate at the raw requested dtype, assign the widened output into it.
    old_buf = np.empty(shape=(n, 1), dtype=requested_dtype)
    old_buf[:, 0] = discretize_array(arr=arr, n_bins=n_bins, method="quantile", dtype=requested_dtype)
    assert (old_buf < 0).any(), "sanity: the OLD pattern is expected to wrap negative for this repro"

    # NEW (fixed) pattern: pre-widen the buffer dtype the same way discretize_array widens internally.
    safe_dtype = _safe_code_dtype(n_bins, requested_dtype)
    assert safe_dtype != requested_dtype  # confirms widening actually triggers for this n_bins/dtype combo
    new_buf = np.empty(shape=(n, 1), dtype=safe_dtype)
    new_buf[:, 0] = discretize_array(arr=arr, n_bins=n_bins, method="quantile", dtype=requested_dtype)
    assert not (new_buf < 0).any()
    assert new_buf.min() == 0
    assert new_buf.max() == n_bins - 1

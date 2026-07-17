"""Op-parity regression for the FE materialise kernels (chunk njit + extval njit) vs the canonical
numpy binary ops, covering ALL registry op-codes 0-8.

Guards two confirmed 2026-06-22 divergences (found by the CPU/GPU-equivalency + hidden-flaw audits):

1. ``div`` (op 3) used the pre-2026-06-13 perturbed form ``x/(y + sign(y)*eps + eps)`` in the chunk
   kernels (CPU serial + parallel + the GPU CUDA twin) while the canonical ``_safe_div`` is the exact
   ``x/where(y==0, eps, y)``. The perturbation reaches ~2*eps/y as y -> 0, shifting re-binned MI ranks.

2. ``_materialise_extval_njit`` implemented only ops 0-5; ops 6/7/8 (abs_diff/signed/ratio_abs) fell into
   the ``else`` branch and were silently computed as ``min(a, b)`` -- selection-altering on any
   medium/maximal preset reaching the external-validation MI sweep.

Both assertions below FAIL on the pre-fix kernels and pass after.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise import (
    _materialise_chunk_njit,
    _materialise_extval_njit,
    _NJIT_BINARY_OP_CODES,
)
from mlframe.feature_selection.filters.feature_engineering import _safe_div

_OPS = ["mul", "add", "sub", "div", "max", "min", "abs_diff", "signed", "ratio_abs"]


def _ref(op, a, b):
    """Canonical numpy semantics matching the kernel docstrings (float64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if op == "mul":
        return a * b
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "div":
        return np.asarray(_safe_div(a, b), dtype=np.float64)
    if op == "max":
        return np.maximum(a, b)
    if op == "min":
        return np.minimum(a, b)
    if op == "abs_diff":
        return np.abs(a - b)
    if op == "signed":
        return np.sign(a) * np.abs(b)
    if op == "ratio_abs":
        return a / (np.abs(b) + 1.0)
    raise AssertionError(op)


@pytest.fixture
def ab():
    """Shared (a, b) pair of independent normal columns for the materialise-op parity tests."""
    rng = np.random.default_rng(20260622)
    n = 4000
    a = rng.normal(0, 3, n)
    b = rng.normal(0, 3, n)
    # Inject small + exact-zero denominators -- the regime where the perturbed div diverges.
    b[:50] = rng.uniform(1e-7, 1e-5, 50)  # tiny positive: perturbed != exact by ~2*eps/b
    b[50:60] = 0.0  # exact zero: eps floor must engage in both forms
    return a.astype(np.float64), b.astype(np.float64)


def test_extval_njit_all_ops_match_numpy(ab):
    """Extval njit all ops match numpy."""
    a, b = ab
    codes = np.array([_NJIT_BINARY_OP_CODES[o] for o in _OPS], dtype=np.int8)
    out = np.empty((a.shape[0], len(_OPS)), dtype=np.float64)
    _materialise_extval_njit(a, b[:, None], codes, out)
    for j, op in enumerate(_OPS):
        ref = _ref(op, a, b)
        # extval is exact float64; div/ratio_abs exact, the rest bit-exact ufuncs.
        np.testing.assert_allclose(
            out[:, j],
            ref,
            rtol=0,
            atol=1e-9,
            err_msg=f"extval op {op!r} diverged (ops 6/7/8 were silently computed as min before the fix)",
        )


def test_chunk_njit_all_ops_match_numpy_f32(ab):
    """Chunk njit all ops match numpy f32."""
    a, b = ab
    tv = np.ascontiguousarray(np.column_stack([a, b]).astype(np.float32))
    a_cols = np.zeros(len(_OPS), dtype=np.int64)  # operand 0 = a
    b_cols = np.ones(len(_OPS), dtype=np.int64)  # operand 1 = b
    codes = np.array([_NJIT_BINARY_OP_CODES[o] for o in _OPS], dtype=np.int8)
    out = np.empty((a.shape[0], len(_OPS)), dtype=np.float32)
    _materialise_chunk_njit(tv, a_cols, b_cols, codes, out)
    for j, op in enumerate(_OPS):
        # chunk path is float32 + nan_to_num(nan/inf -> 0); reference mirrors that.
        ref = _ref(op, tv[:, 0].astype(np.float64), tv[:, 1].astype(np.float64))
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # f32 tolerance; the tiny-denominator div would miss this by ~2*eps/b under the old perturbed form.
        np.testing.assert_allclose(
            out[:, j],
            ref,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"chunk op {op!r} diverged from canonical _safe_div/numpy",
        )


def test_div_exact_not_perturbed(ab):
    """Targeted: at a tiny positive denominator the exact _safe_div and the old perturbed form differ by
    a wide margin -- pin that the kernels use the exact form."""
    a, b = ab
    tiny = slice(0, 50)
    codes = np.array([_NJIT_BINARY_OP_CODES["div"]], dtype=np.int8)
    out = np.empty((a.shape[0], 1), dtype=np.float64)
    _materialise_extval_njit(a, b[:, None], codes, out)
    exact = _safe_div(a, b)
    perturbed = a / (b + np.sign(b) * 1e-9 + 1e-9)
    # The fixed kernel matches exact, NOT perturbed, on the tiny-denominator rows.
    assert np.max(np.abs(out[tiny, 0] - exact[tiny])) < 1e-6
    assert np.max(np.abs(exact[tiny] - perturbed[tiny])) > 1e-3  # the forms genuinely differ here

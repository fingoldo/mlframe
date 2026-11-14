"""The ``div`` op of the device materialise kernel equals the double-precision quotient rounded to float on every float pair, edge values included."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_materialise as M


def _reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``_safe_div`` as the host computes it: the quotient in double precision, the exact-zero denominator floored at 1e-9, rounded to float, then non-finite values scrubbed to 0."""
    den = np.where(b == 0.0, 1e-9, b.astype(np.float64))
    with np.errstate(all="ignore"):
        q = (a.astype(np.float64) / den).astype(np.float32)
    q = np.where(np.abs(q) < np.finfo(np.float32).tiny, np.copysign(np.float32(0.0), q), q)  # the kernel flushes denormal results to a signed zero
    return np.where(np.isfinite(q), q, np.float32(0.0)).astype(np.float32)


def test_div_equals_the_double_quotient_rounded_to_float():
    """Random magnitudes over 60 binades plus zeros, signed zeros, infinities, NaN and the smallest normal numbers on both sides: the same bits as the reference (denormal results flushed to a signed zero, as the kernel does)."""
    rng = np.random.default_rng(0)
    n = 60000
    specials = np.array(
        [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1.2e-38, -1.2e-38, 3.4e38, -3.4e38, 7.0, 3.0], dtype=np.float32
    )  # no denormal input: the kernel reads those as zero
    cols = rng.standard_normal((2, n)).astype(np.float32) * np.exp(rng.uniform(-30, 30, (2, n))).astype(np.float32)
    mask = rng.random((2, n)) < 0.08
    cols[mask] = rng.choice(specials, int(mask.sum()))
    tv = cp.asarray(np.ascontiguousarray(cols.T))  # (n, 2) row-major operand table
    out = M._fe_materialise_block_gpu(tv, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), np.array([3], dtype=np.int8))
    got = cp.asnumpy(out)[:, 0]
    want = _reference(cols[0], cols[1])
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))

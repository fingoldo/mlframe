"""The GPU basis-routing |corr| kernel must be exact on a column with a large offset (mrmr_audit_2026-09-14 NUM-2).

``_ABS_CORR_SRC`` accumulated raw power sums and formed the variance as ``n*sum(v^2) - (sum v)^2``. With a column mean far above its spread
the two terms agree in almost every digit, the difference cancels to rounding noise, and the kernel returned a corrupted correlation or the
``-1.0`` degenerate sentinel, which makes the host argmax skip the column as if it were constant. Basis evaluations that are not centred
(minmax / clip / even-degree terms) reach this regime. The CPU twins centre first; this GPU twin did not.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_basis import _gpu_batched_abs_corr


def _device_ok() -> bool:
    """True when a CUDA device is actually usable, not just importable."""
    try:
        cp.cuda.runtime.getDeviceCount()
        cp.asarray([1.0]).sum()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _device_ok(), reason="no usable CUDA device")


def _columns(n=20000, seed=0):
    """y plus three candidate columns that each correlate ~0.5 with it: centred, offset 1e7, offset 1e10."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    base = 0.5 * y + np.sqrt(1 - 0.25) * rng.normal(size=n)
    return y, np.column_stack([base, base + 1e7, base + 1e10])


def test_abs_corr_matches_numpy_on_offset_columns():
    """Every column's |corr| must equal numpy's centred Pearson value; an additive offset cannot change a correlation."""
    y, cand = _columns()
    expected = np.array([abs(np.corrcoef(cand[:, j], y)[0, 1]) for j in range(cand.shape[1])])
    got = cp.asnumpy(_gpu_batched_abs_corr(cp, cp.asarray(cand), cp.asarray(y)))
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert np.all(got > 0.4), f"an offset column was reported degenerate or corrupted: {got}"


def test_constant_column_is_still_marked_degenerate():
    """Control: a genuinely constant column (even at a large offset) must keep the -1.0 skip sentinel."""
    y, cand = _columns()
    cand = np.column_stack([cand[:, 0], np.full(cand.shape[0], 1e7)])
    got = cp.asnumpy(_gpu_batched_abs_corr(cp, cp.asarray(cand), cp.asarray(y)))
    assert got[1] == -1.0
    assert got[0] == pytest.approx(abs(np.corrcoef(cand[:, 0], y)[0, 1]), abs=1e-6)

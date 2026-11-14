"""The fused joint-count kernel of the resident permutation null builds the same contingency tables as a plain bincount, and the resident floor uses it."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _permutation_null_joint_kernel as K
from mlframe.feature_selection.filters import _permutation_null_resident as R


def _reference(x: np.ndarray, yp: np.ndarray, max_jc: int) -> np.ndarray:
    """Row ``c * pb + p`` = bincount of x[c] + yp[p], padded to ``max_jc`` cells."""
    cc = x.shape[0]
    pb = yp.shape[0]
    out = np.zeros((cc * pb, max_jc), dtype=np.int32)
    for c in range(cc):
        for p in range(pb):
            out[c * pb + p] = np.bincount((x[c].astype(np.int64) + yp[p].astype(np.int64)), minlength=max_jc)[:max_jc]
    return out


@pytest.mark.parametrize(
    "xdt,ydt,nbx,nby",
    [(np.int16, np.int8, 10, 2), (np.int64, np.int32, 20, 4), (np.uint8, np.uint8, 8, 3), (np.int32, np.int64, 64, 10)],
)
@pytest.mark.parametrize("n", [1, 257, 4099])
def test_tables_equal_a_plain_bincount_for_every_code_width(xdt, ydt, nbx, nby, n):
    """Mixed integer widths, a sample count that leaves a partial block, and a single sample: every table equals the bincount of x + y."""
    rng = np.random.default_rng(n + nbx)
    cc, pb = 3, 4
    max_jc = nbx * nby
    x = (rng.integers(0, nbx, (cc, n)) * nby).astype(xdt)
    yp = rng.integers(0, nby, (pb, n)).astype(ydt)
    got = K.perm_joint_counts(cp, cp.asarray(x), cp.asarray(yp), max_jc)
    assert got.shape == (cc * pb, max_jc) and got.dtype == cp.int32
    np.testing.assert_array_equal(cp.asnumpy(got), _reference(x, yp, max_jc))
    assert int(cp.asnumpy(got).sum()) == cc * pb * n


def test_a_table_over_the_shared_memory_limit_is_left_to_the_caller():
    """4 bytes per cell against the 48 KB limit: 12288 cells fit, 12289 do not (None, the caller keeps its scatter-add path)."""
    x = cp.zeros((1, 8), dtype=cp.int16)
    y = cp.zeros((1, 8), dtype=cp.int8)
    assert K.perm_joint_counts(cp, x, y, 12288) is not None
    assert K.perm_joint_counts(cp, x, y, 12289) is None


def test_the_resident_floor_runs_the_fused_kernel_and_equals_the_njit_floor(monkeypatch):
    """pooled_gain_floor_perms_cupy asks the fused kernel for its tables (it ran) and returns the njit floor to round-off."""
    from mlframe.feature_selection.filters import _permutation_null as P

    rng = np.random.default_rng(3)
    n, ncand, nperm = 3000, 4, 25
    nby = 3
    y = rng.integers(0, nby, n).astype(np.int64)
    nbx = np.array([6, 8, 5, 7])
    codes = [(rng.integers(0, int(b), n) * nby).astype(np.int64) for b in nbx]
    scaled = np.concatenate(codes)
    offsets = np.arange(ncand + 1, dtype=np.int64) * n
    joint_card = (nbx * nby).astype(np.int64)
    inv_n = 1.0 / n
    h_x = np.array([-(np.bincount(c // nby, minlength=int(b)) / n * np.log(np.maximum(np.bincount(c // nby, minlength=int(b)) / n, 1e-300))).sum() for c, b in zip(codes, nbx)])
    mm = np.zeros(ncand)
    py = np.bincount(y, minlength=nby) / n
    h_y = float(-(py * np.log(py)).sum())
    y_perms = np.stack([rng.permutation(y) for _ in range(nperm)]).astype(np.int64)
    calls = []
    real = R.perm_joint_counts
    monkeypatch.setattr(R, "perm_joint_counts", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    got = R.pooled_gain_floor_perms_cupy(scaled, offsets, joint_card, h_x, mm, h_y, y_perms, inv_n)
    assert calls, "the resident floor did not use the fused joint-count kernel"
    want = P._pooled_gain_floor_perms_njit(scaled, offsets, joint_card, h_x, mm, h_y, y_perms, inv_n)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

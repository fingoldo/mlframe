"""Regression: the VRAM-chunked resident perm-null helper must equal a single batched CMI call.

``conditional_perm_null_gpu`` scores CMI(perm_col; y | z) for every column of a resident (n, nperm) permutation
matrix. The conditional ``batched_cmi_gpu`` densifies a ``(nperm, Kx*Kyz)`` joint histogram; at the full-n
redundancy-gate calls the conditioning support is near-continuous (Kz ~ 1e5 -> Kyz multi-million), so the whole
batch is multi-GB (measured 7-11 GB at nperm=25) and OOMs a 4 GB card -- which used to force the gate onto the
800 MB host-key path + per-perm CPU loop. ``_batched_cmi_resident_chunked`` splits the perms into VRAM-sized
chunks; because each perm's CMI is INDEPENDENT of how many perms share a batched call, the chunked result must
equal the single-call result column-for-column. This pins that equivalence (incl. the chunk=1 path) so the
residency win never silently changes a null value.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as PNG
from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu


def _need_cuda() -> bool:
    """Need cuda."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _make_codes(seed=0, n=4000, nperm=12, Kx=6, Ky=3, Kz=60):
    """Make codes."""
    rng = np.random.default_rng(seed)
    Xp = cp.asarray(rng.integers(0, Kx, size=(n, nperm)).astype(np.int64))
    y = np.ascontiguousarray(rng.integers(0, Ky, size=n).astype(np.int64))
    z = np.ascontiguousarray(rng.integers(0, Kz, size=n).astype(np.int64))
    return Xp, y, z


def test_chunked_equals_single_batched():
    """Chunked equals single batched."""
    Xp, y, z = _make_codes()
    single = np.asarray(batched_cmi_gpu(Xp, y, z), dtype=np.float64)
    chunked = PNG._batched_cmi_resident_chunked(Xp, y, z)
    assert chunked.shape == (int(Xp.shape[1]),)
    # chunking only splits the batch -> column-for-column identical CMI (fp reduction order identical per column).
    assert np.allclose(single, chunked, rtol=0, atol=1e-9), single - chunked


def test_chunk_one_forced_matches(monkeypatch):
    # Report 1 KB free VRAM so the helper sizes chunk == 1 -> exercises the multi-iteration chunk loop and the
    # adaptive lower bound; the per-perm result must still equal the single batched call.
    """Chunk one forced matches."""
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (1024, 4 << 30), raising=True)
    Xp, y, z = _make_codes(seed=7)
    single = np.asarray(batched_cmi_gpu(Xp, y, z), dtype=np.float64)
    chunked = PNG._batched_cmi_resident_chunked(Xp, y, z)
    assert np.allclose(single, chunked, rtol=0, atol=1e-9)


def test_conditional_perm_null_gpu_reproducible_and_finite():
    # End-to-end: same (seed, salt) -> reproducible (floor, mean), both finite and non-negative (CMI >= 0).
    """Conditional perm null gpu reproducible and finite."""
    rng = np.random.default_rng(3)
    n, Kx, Kz = 5000, 5, 40
    x = rng.integers(0, Kx, size=n).astype(np.int64)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    z = rng.integers(0, Kz, size=n).astype(np.int64)
    order = np.argsort(z, kind="stable").astype(np.int64)
    sorted_z = z[order]
    z_rank = np.zeros(n, dtype=np.float64)
    z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])
    kw = dict(order=order, z_rank=z_rank, n_permutations=16, quantile=0.95, seed=42, salt=7)
    f1, m1 = PNG.conditional_perm_null_gpu(x, y, z, **kw)
    f2, m2 = PNG.conditional_perm_null_gpu(x, y, z, **kw)
    assert (f1, m1) == (f2, m2), "same (seed, salt) must be reproducible"
    for v in (f1, m1):
        assert np.isfinite(v) and v >= 0.0

"""The tiled fused joint-entropy kernel equals the unfused count+reduce reference, including column counts that leave a partial group.

``_batched_joint_entropy_and_k2`` runs the tiled kernel only when ``MLFRAME_FE_GPU_FUSE_CMI_ENTROPY`` forces it on (the default dispatch
picks it only for large joints), so the test sets that flag. The reference is the two-launch count+reduce path.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as C


@pytest.mark.parametrize("K", [1, 9, 13, 16])
@pytest.mark.parametrize("Kb", [2, 7])
def test_tiled_fused_entropy_equals_count_reduce(K, Kb, monkeypatch):
    """Entropies agree to round-off and the occupied-cell counts agree exactly with the unfused reference."""
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", "1")
    rng = np.random.default_rng(K * 10 + Kb)
    n, Kx = 5000, 6
    X = cp.asarray(rng.integers(0, Kx, size=(n, K), dtype=np.int64))
    b = cp.asarray(rng.integers(0, Kb, size=n, dtype=np.int64))
    h, k = C._batched_joint_entropy_and_k2(X, b, Kx, Kb, 1.0 / n)
    ref_h, ref_k = C._rows_entropy_and_k(C._batched_joint_counts2(X, b, Kx, Kb).astype(cp.float64), 1.0 / n)
    assert float(cp.abs(h - ref_h).max()) < 1e-12
    assert cp.array_equal(k, ref_k)

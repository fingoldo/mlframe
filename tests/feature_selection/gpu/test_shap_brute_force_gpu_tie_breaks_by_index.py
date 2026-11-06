"""The GPU subset scan breaks loss ties by combination index, as the CPU scan does.

With two identical feature columns, the subsets that contain either one have exactly the same loss. The top-1 boundary then falls
inside that tie, and the CPU keeps the earlier-enumerated subset. The GPU selection used an unstable argpartition/argsort and could
return the later one.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_gpu_top1_tie_resolves_to_lower_combination_index():
    """GPU and CPU pick the same subset when two identical columns tie for top-1."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu import brute_force_top_n_gpu
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    rng = np.random.default_rng(3)
    n, f = 400, 4
    phi = rng.normal(size=(n, f))
    phi[:, 2] = phi[:, 1]  # features 1 and 2 are identical: their single-feature losses tie exactly
    y = phi[:, 1].copy()
    base = np.zeros(n)

    gpu = brute_force_top_n_gpu(phi, base, y, classification=False, metric="mse", min_card=1, max_card=1, top_n=1)
    cpu = brute_force_top_n(phi, base, y, classification=False, metric="mse", min_card=1, max_card=1, top_n=1, parallel=False)

    # generate_combinations enumerates [3, 2, 1, 0], so the earliest tied subset is (2,); both backends must agree on it.
    assert gpu[0][1] == cpu[0][1]
    assert gpu[0][1] in ((1,), (2,))
    assert gpu[0][0] == pytest.approx(cpu[0][0], rel=1e-6)

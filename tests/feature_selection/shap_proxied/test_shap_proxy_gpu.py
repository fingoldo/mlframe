"""GPU subset-scan parity tests. Skipped unless cupy AND a CUDA device are present."""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")


def _has_device():
    """True iff cupy reports at least one CUDA device, gating the GPU subset-scan parity tests."""
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _has_device(), reason="no CUDA device available")]


@pytest.mark.parametrize("metric,classification", [("rmse", False), ("mae", False), ("brier", True), ("logloss", True)])
def test_gpu_matches_numba(metric, classification):
    """The GPU brute-force top-N subset scan matches the numba CPU version's winning subset and score, and agrees on the top-5 subset set, for every metric/task-type combo."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu import brute_force_top_n_gpu
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    rng = np.random.default_rng(0)
    phi = rng.normal(size=(400, 9))
    base = np.full(400, 0.1)
    margin = base + phi[:, [1, 4, 6]].sum(axis=1)
    if classification:
        y = (1.0 / (1.0 + np.exp(-margin)) > rng.random(400)).astype(float)
    else:
        y = margin + 0.02 * rng.normal(size=400)

    cpu = brute_force_top_n(phi, base, y, classification=classification, metric=metric, max_card=9, top_n=10)
    gpu = brute_force_top_n_gpu(phi, base, y, classification=classification, metric=metric, max_card=9, top_n=10)

    assert set(cpu[0][1]) == set(gpu[0][1])
    np.testing.assert_allclose(cpu[0][0], gpu[0][0], rtol=1e-6, atol=1e-9)
    # top-5 subset agreement
    assert {frozenset(c) for _, c in cpu[:5]} == {frozenset(c) for _, c in gpu[:5]}

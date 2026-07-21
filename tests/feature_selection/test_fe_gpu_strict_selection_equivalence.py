"""STRICT (GPU-resident FE) is selection-equivalent to the exact CPU path at production n.

The size-gated AUTO default engages STRICT above ~50k precisely because the FE selection converges to the CPU
result there (the small-n divergence is finite-sample MI-estimation variance that fades as n grows). This slow
integration test pins that contract: at n=50k the STRICT and CPU MRMR selections are identical. Skipped without a
GPU (STRICT is a no-op) and under MLFRAME_FAST (heavy fit).
"""

import os

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._fe_gpu_strict import _cuda_usable

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _cuda_usable(), reason="STRICT GPU-resident FE is a no-op without a CUDA device"),
]


def _make(n, seed):
    """Builds seeded synthetic test data; returns ``(X, y)``."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(22)})
    xi = (rng.standard_normal(n) * 3).astype(int)
    X["g0"] = xi.astype(float)
    s = X["f0"] * X["f1"] + 0.7 * X["f2"] - 0.5 * X["f3"] ** 2 + np.sin(3.3 * X["f4"]) + (xi % 3 == 0) * 0.8
    y = ((s + rng.standard_normal(n) * 0.4) > 0).astype(int).to_numpy()
    return X, y


def _select(X, y, strict, seed):
    """Test helper: os.environ['MLFRAME_FE_GPU_STRICT'] = strict; try: MRMR._FIT_CACHE.clear() except Exception: pass; m = MRMR(fe_ntop_features=8, interactions_max_order=2, n_...."""
    os.environ["MLFRAME_FE_GPU_STRICT"] = strict
    try:
        MRMR._FIT_CACHE.clear()
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        pass
    m = MRMR(fe_ntop_features=8, interactions_max_order=2, n_jobs=1, verbose=0, random_seed=seed, skip_retraining_on_same_content=False)
    m.fit(X, y)
    names = list(m.feature_names_in_)
    sup = np.asarray(m.support_).ravel()
    return tuple(sorted(names[i] for i in sup if i < len(names)))


def test_strict_matches_cpu_selection_at_production_n(monkeypatch):
    """Strict matches cpu selection at production n."""
    X, y = _make(50_000, 7000)
    cpu = _select(X, y, "0", 7000)
    strict = _select(X, y, "1", 7000)
    assert strict == cpu, f"STRICT {strict} != CPU {cpu} at n=50k (should be selection-equivalent)"

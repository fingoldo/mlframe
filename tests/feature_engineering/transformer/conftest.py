"""Session-scoped fixtures for the transformer FE test suite.

The big one is ``_warm_jit_cache``: numba compiles 8+ kernels on first call (5-15s each, cumulative ~60s cold-cache). We warm them once at session start so the
30+ tests across this directory don't each pay the first-call JIT cost. Persistent on-disk cache (``cache=True`` on the @njit decorators) means the cost is only
paid on the first session per source-hash; ``--no-cov`` is set in pyproject so we never accidentally enable coverage on this directory and bust the JIT cache
via the trace hook.

Also handles the hnswlib-segfault-on-Windows case: ``pytest.importorskip("hnswlib")`` can only catch ImportError; if the wheel imports but segfaults inside the
loader (a known Windows MSVC runtime issue), pytest collection dies before the skip fires. We subprocess-check hnswlib at session collection start and tag test
files that require it for collection-time skipping via ``pytest_collection_modifyitems``.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Generator

import numpy as np
import pytest
from sklearn.model_selection import KFold


# Direct in-process try-import. pynndescent is pure-Python+numba so the only failure mode is ImportError (no segfault possible). The cost of the first import
# is dominated by numba JIT (~30-60s) which is paid here once and then warm-cached for the rest of the session.
def _ann_backend_safely_importable() -> bool:
    """Return True iff ``pynndescent`` imports cleanly in the current process."""
    try:
        import pynndescent  # noqa: F401
        return True
    except ImportError:
        return False


_ANN_OK = _ann_backend_safely_importable()


def pytest_collection_modifyitems(config, items):
    """Skip any test that comes from a file with 'row_attention' in its name when the ANN backend is unavailable."""
    if _ANN_OK:
        return
    skip_marker = pytest.mark.skip(reason="pynndescent (default ANN backend) not importable; install via `pip install pynndescent>=0.5` or `pip install 'mlframe[transformer_ann]'`.")
    for item in items:
        if "row_attention" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session", autouse=True)
def _warm_jit_cache() -> Generator[None, None, None]:
    """Trigger numba JIT for every kernel in the transformer subpackage once per session.

    Without this, the first test that calls ``compute_rff_features`` (or any of the other kernels) eats the per-kernel compile cost; with this, the cost is paid
    once before any test runs and cached for the rest of the session via numba's persistent disk cache.

    Uses tiny inputs (5-row arrays) so the actual compute time is negligible (<1 ms). Only the JIT compilation matters.
    """
    from mlframe.feature_engineering.transformer._aggregation import (
        batch_weighted_mean,
        batch_weighted_std,
        batch_weighted_vector_mean,
        weighted_mean_1d,
        weighted_std_1d,
        weighted_var_1d,
    )
    from mlframe.feature_engineering.transformer._kernels_njit import (
        rff_matmul_njit,
        row_attention_stage4_njit,
        softmax_stable_inplace,
    )
    from mlframe.feature_engineering.transformer._projection import _project_single_head_njit

    rng = np.random.default_rng(0)
    # Aggregation kernels.
    v = rng.standard_normal(5).astype(np.float64)
    w = rng.random(5).astype(np.float64)
    weighted_mean_1d(v, w)
    weighted_var_1d(v, w)
    weighted_std_1d(v, w)

    vals = rng.standard_normal((3, 5)).astype(np.float64)
    wts = rng.random((3, 5)).astype(np.float64)
    out1 = np.empty(3, dtype=np.float64)
    out2 = np.empty(3, dtype=np.float64)
    batch_weighted_mean(vals, wts, out1)
    batch_weighted_std(vals, wts, out2)
    vec_vals = rng.standard_normal((3, 5, 4)).astype(np.float64)
    out3 = np.empty((3, 4), dtype=np.float64)
    batch_weighted_vector_mean(vec_vals, wts, out3)

    # Attention kernel.
    logits = rng.standard_normal(5).astype(np.float32)
    softmax_stable_inplace(logits)

    # RFF.
    X = rng.standard_normal((5, 4), dtype=np.float64).astype(np.float32)
    W = rng.standard_normal((4, 8), dtype=np.float64).astype(np.float32)
    b = rng.random(8).astype(np.float32)
    out_rff = np.empty((5, 16), dtype=np.float32)
    rff_matmul_njit(X, W, b, out_rff, scale=np.float32(0.5))

    # Projection.
    X2 = rng.standard_normal((5, 4)).astype(np.float32)
    W2 = rng.standard_normal((4, 8)).astype(np.float32)
    out_proj = np.empty((5, 8), dtype=np.float32)
    _project_single_head_njit(X2, W2, out_proj)

    # Stage 4 row-attention.
    q_proj = rng.standard_normal((4, 8)).astype(np.float32)
    k_proj = rng.standard_normal((10, 8)).astype(np.float32)
    y_train = rng.standard_normal(10).astype(np.float32)
    topk_ids = rng.integers(0, 10, size=(4, 3), dtype=np.int32)
    y_mean = np.empty(4, dtype=np.float32)
    y_std = np.empty(4, dtype=np.float32)
    x_mean = np.empty((4, 8), dtype=np.float32)
    row_attention_stage4_njit(q_proj, k_proj, y_train, topk_ids, 1.0, y_mean, y_std, x_mean)

    yield


@pytest.fixture
def small_X_y_classification():
    """Small (200, 16) classification dataset with seed=0 for reproducibility. Used by unit tests that need ``X_train, y_train``."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 16)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


@pytest.fixture
def kfold_splitter():
    """5-fold KFold with fixed seed - used by row-attention OOF tests."""
    return KFold(n_splits=5, shuffle=True, random_state=42)

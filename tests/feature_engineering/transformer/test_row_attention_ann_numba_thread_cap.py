"""The pynndescent ANN build must not ask numba for more threads than NUMBA_NUM_THREADS allows."""

from __future__ import annotations

import os
import subprocess  # nosec B404 - test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys

import pytest

pytest.importorskip("pynndescent")

_SNIPPET = """
import numpy as np
from mlframe.feature_engineering.transformer._row_attention_ann import build_hnsw_index, query_topk

rng = np.random.default_rng(0)
k_proj = rng.normal(size=(400, 16)).astype(np.float32)
index = build_hnsw_index(k_proj, num_threads=8, ann_backend="pynndescent", random_state=0)
labels, distances = query_topk(index, k_proj[:5], 3)
assert labels.shape == (5, 3) and distances.shape == (5, 3)
print("OK")
"""


def test_pynndescent_build_survives_a_numba_pool_smaller_than_the_requested_workers():
    """With NUMBA_NUM_THREADS=1 and num_threads=8 the build completes and queries return the requested neighbour count.

    numba.set_num_threads raises ValueError for a request above the pool size, which used to abort the whole row-attention feature build for a user who
    capped numba below the core count.
    """
    env = dict(os.environ, NUMBA_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", _SNIPPET], env=env, capture_output=True, text=True, timeout=900)  # nosec B603 - fixed argv, no shell
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "OK"

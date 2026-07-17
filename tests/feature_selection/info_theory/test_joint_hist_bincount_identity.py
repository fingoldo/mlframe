"""Identity regression for the bincount joint-histogram build (CPX9).

_mah / _pid_decomposition / _chao_shen replaced a pure-Python
``joint[xb[i],yb[i]]+=1`` loop with ``np.bincount(...).reshape(...)``. Integer
counts -> bit-identical joint table -> identical entropy/MI. This pins the
raveled-bincount construction against the explicit double/triple loop for both
2-D and 3-D cases, plus determinism of the public estimators.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._chao_shen import chao_shen_mi
from mlframe.feature_selection.filters._mah import mah_mi
from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition


@pytest.mark.parametrize("seed", [0, 3, 9])
def test_bincount_joint_2d_matches_loop(seed):
    rng = np.random.default_rng(seed)
    n, K_x, K_y = 4000, 16, 10
    xb = rng.integers(0, K_x, n).astype(np.int64)
    yb = rng.integers(0, K_y, n).astype(np.int64)
    loop = np.zeros((K_x, K_y), dtype=np.float64)
    for i in range(n):
        loop[xb[i], yb[i]] += 1.0
    vec = np.bincount(xb * K_y + yb, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)
    assert np.array_equal(loop, vec)


@pytest.mark.parametrize("seed", [0, 3, 9])
def test_bincount_joint_3d_matches_loop(seed):
    rng = np.random.default_rng(seed)
    n, K1, K2, K3 = 4000, 4, 5, 3
    x1 = rng.integers(0, K1, n).astype(np.int64)
    x2 = rng.integers(0, K2, n).astype(np.int64)
    y = rng.integers(0, K3, n).astype(np.int64)
    loop = np.zeros((K1, K2, K3), dtype=np.float64)
    for i in range(n):
        loop[x1[i], x2[i], y[i]] += 1.0
    vec = np.bincount((x1 * K2 + x2) * K3 + y, minlength=K1 * K2 * K3).reshape(K1, K2, K3).astype(np.float64)
    assert np.array_equal(loop, vec)


def test_estimators_deterministic():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.standard_normal(n)
    y = (x > 0).astype(np.int64)
    assert mah_mi(x, y) == mah_mi(x, y)
    xb = rng.integers(0, 8, n).astype(np.int64)
    assert chao_shen_mi(xb, y) == chao_shen_mi(xb, y)
    x1 = rng.integers(0, 3, n)
    x2 = rng.integers(0, 3, n)
    yy = rng.integers(0, 2, n)
    a = pid_decomposition(x1, x2, yy, 3, 3, 2)
    b = pid_decomposition(x1, x2, yy, 3, 3, 2)
    assert a == b

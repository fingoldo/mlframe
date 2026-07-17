"""Regression tests for the order-2 maxT pair-MI floor optimizations (2026-07-05):

1. Row-subsample cap (``MLFRAME_FE_PAIR_MAXT_MAX_ROWS``) on
   ``pooled_pair_permutation_null_joint_mi_floor`` -- speeds the floor compute
   while keeping SELECTION unchanged (survivors of ``pair_mi >= floor`` identical).
2. Resident-GPU pair-maxT circuit breaker -- once tripped, the GPU path is
   skipped for the rest of the process (mirrors the CMI breaker).
"""

import os

import numpy as np
import pytest

from mlframe.feature_selection.filters._permutation_null import (
    _pair_maxt_max_rows,
    pooled_pair_permutation_null_joint_mi_floor,
)
from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    batch_pair_mi_prange,
)


def _screen_data(n=30000, p=40, nbins_val=10, seed=7):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, nbins_val, size=(n, p)).astype(np.int64)
    # plant one strong synergy pair (0,1) and one weak-but-real pair (2,3)
    y = (data[:, 0] + data[:, 1]) % nbins_val
    mask = rng.random(n) < 0.5
    y2 = (data[:, 2] + data[:, 3]) % nbins_val
    y[mask] = y2[mask]
    nb = np.full(p, nbins_val, dtype=np.int64)
    fy = np.bincount(y, minlength=nbins_val).astype(np.float64) / n
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    pa = np.array([q[0] for q in pairs], np.int64)
    pb = np.array([q[1] for q in pairs], np.int64)
    return data, nb, y.astype(np.int64), fy, pa, pb


@pytest.fixture
def _restore_env():
    prev = os.environ.get("MLFRAME_FE_PAIR_MAXT_MAX_ROWS")
    yield
    if prev is None:
        os.environ.pop("MLFRAME_FE_PAIR_MAXT_MAX_ROWS", None)
    else:
        os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = prev


def test_max_rows_default_and_parsing(_restore_env):
    os.environ.pop("MLFRAME_FE_PAIR_MAXT_MAX_ROWS", None)
    assert _pair_maxt_max_rows() == 15000
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "0"
    assert _pair_maxt_max_rows() == 0  # disabled
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "8000"
    assert _pair_maxt_max_rows() == 8000
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "garbage"
    assert _pair_maxt_max_rows() == 15000  # invalid -> default


def test_cap_disabled_equals_full_n(_restore_env):
    data, nb, y, fy, pa, pb = _screen_data()
    kw = dict(factors_data=data, nbins=nb, pair_a=pa, pair_b=pb, classes_y=y, freqs_y=fy, n_permutations=15, quantile=0.95, random_seed=123)
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "0"
    full = pooled_pair_permutation_null_joint_mi_floor(**kw)
    # recompute -- deterministic
    assert pooled_pair_permutation_null_joint_mi_floor(**kw) == full


def test_cap_is_conservative_and_selection_equivalent(_restore_env):
    data, nb, y, fy, pa, pb = _screen_data()
    obs = batch_pair_mi_prange(data, pa, pb, nb, y, fy)  # observed pair-MI at FULL n (the gated value)
    kw = dict(factors_data=data, nbins=nb, pair_a=pa, pair_b=pb, classes_y=y, freqs_y=fy, n_permutations=15, quantile=0.95, random_seed=123)

    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "0"
    floor_full = pooled_pair_permutation_null_joint_mi_floor(**kw)
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "15000"
    floor_cap = pooled_pair_permutation_null_joint_mi_floor(**kw)

    # Capped floor is CONSERVATIVE (>= full-n floor: finite-sample MI bias ~1/n).
    assert floor_cap >= floor_full > 0.0
    # SELECTION-equivalence: the SAME pairs pass ``obs >= floor``.
    surv_full = set(np.nonzero(obs >= floor_full)[0].tolist())
    surv_cap = set(np.nonzero(obs >= floor_cap)[0].tolist())
    assert surv_cap == surv_full, f"selection changed: +{surv_cap - surv_full} -{surv_full - surv_cap}"
    # and the planted signal pairs survive under the cap
    idx = {(a, b): i for i, (a, b) in enumerate(zip(pa.tolist(), pb.tolist()))}
    assert obs[idx[(0, 1)]] >= floor_cap
    assert obs[idx[(2, 3)]] >= floor_cap


def test_cap_actually_shrinks_n_via_higher_floor(_restore_env):
    # Pin that the cap DOES engage (floor strictly higher than full-n) at an aggressive cap,
    # so a future "no-op cap" regression is caught.
    data, nb, y, fy, pa, pb = _screen_data()
    kw = dict(factors_data=data, nbins=nb, pair_a=pa, pair_b=pb, classes_y=y, freqs_y=fy, n_permutations=15, quantile=0.95, random_seed=123)
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "0"
    floor_full = pooled_pair_permutation_null_joint_mi_floor(**kw)
    os.environ["MLFRAME_FE_PAIR_MAXT_MAX_ROWS"] = "6000"
    floor_small = pooled_pair_permutation_null_joint_mi_floor(**kw)
    assert floor_small > floor_full


def test_pair_maxt_gpu_circuit_breaker_gates():
    from mlframe.feature_selection.filters._permutation_null_pair_resident import (
        pair_maxt_perm_null_gpu_enabled,
        reset_pair_maxt_gpu_circuit_breaker,
        trip_pair_maxt_gpu_circuit_breaker,
    )

    reset_pair_maxt_gpu_circuit_breaker()
    try:
        trip_pair_maxt_gpu_circuit_breaker()
        # once tripped, the GPU path is disabled regardless of STRICT / env
        assert pair_maxt_perm_null_gpu_enabled(100000, 1770) is False
    finally:
        reset_pair_maxt_gpu_circuit_breaker()

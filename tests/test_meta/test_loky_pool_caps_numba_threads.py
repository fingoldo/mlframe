"""The one process-pool backend mlframe builds (polynomial-pair FE) must cap every worker's numba prange pool, not only BLAS."""
from __future__ import annotations

from joblib._parallel_backends import LokyBackend


def test_inner_max_num_threads_caps_numba_threads_in_worker_env():
    """joblib's ``inner_max_num_threads`` exports NUMBA_NUM_THREADS (and the BLAS/OpenMP variables) to loky workers, so a pool of N workers does not start N full-width prange pools."""
    env = LokyBackend(inner_max_num_threads=1)._prepare_worker_env(n_jobs=4)
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"

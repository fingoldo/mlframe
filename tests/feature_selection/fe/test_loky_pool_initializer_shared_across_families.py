"""Regression test (2026-07-10, wellbore py-spy profiling finding): the CPU-only loky worker
initializer must be the SAME function object across every FE family that builds its own
``LokyBackend``, not a per-family duplicate with an identical body.

Why this matters: joblib/loky's ``get_reusable_executor`` keys pool reuse on the initializer
FUNCTION REFERENCE (among other config). Two behaviourally-identical-but-distinct callables (e.g.
``_step_pairmi.py``'s pair-MI sweep and ``polynom_pair_fe.py``'s polynomial-pair search each defining
their own local ``disable_cuda_in_worker``-alike function) defeat pool reuse: loky treats them as
DIFFERENT pool configurations and spawns a fresh 16-worker pool (process create + a full mlframe/numba
re-import per worker) for each, even when a warm, config-identical CPU-only pool from moments earlier
in the SAME fit is sitting idle. Profiling a production-representative wellbore fit (cProfile: 81.7% of
wall-clock in thread-wait; py-spy sampling with subprocess+idle threads: 51.5% of samples parked in
``multiprocessing.queues.get``) pointed squarely at this class of repeated-spawn overhead across the
fit's several loky-using FE stages.

This test pins the fix at the identity level: every module that builds a CPU-only ``LokyBackend`` must
import and pass the ONE shared ``mlframe.feature_selection.filters._joblib_safe.disable_cuda_in_worker``
-- not a local look-alike -- so loky's executor cache can actually recognize and reuse a warm pool.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._joblib_safe import disable_cuda_in_worker
from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairmi
from mlframe.feature_selection.filters import polynom_pair_fe


def test_step_pairmi_uses_the_shared_initializer():
    assert _step_pairmi.disable_cuda_in_worker is disable_cuda_in_worker


def test_polynom_pair_fe_uses_the_shared_initializer():
    assert polynom_pair_fe.disable_cuda_in_worker is disable_cuda_in_worker


def test_polynom_pair_fe_no_longer_defines_a_local_duplicate():
    """The pre-fix local duplicate (``_poly_worker_disable_cuda``) must be gone, not just unused --
    its continued presence would invite a future call site to accidentally reintroduce the split."""
    assert not hasattr(polynom_pair_fe, "_poly_worker_disable_cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])

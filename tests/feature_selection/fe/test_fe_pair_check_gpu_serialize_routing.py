"""Regression test for the GPU-contention serialize gate on ``check_prospective_fe_pairs``
(MRMR audit finding #27, confirmed pre-existing 2026-06-20 fix -- see the "GPU-FE SERIALIZE" comment in
``_mrmr_fe_step/_step_core.py``).

Finding #27 worried that ``check_prospective_fe_pairs``'s joblib ``backend="threading"`` fan-out branch
has no protection against N threads all contending for the single GPU when the per-pair candidate
materialise/binning runs on CUDA -- the same "joblib threading over GPU-bound tasks = contention, not
parallelism" pathology already fixed for the sibling ``compute_pair_mis_and_floor`` loky path. Reading
the current code shows this was ALREADY fixed (2026-06-20, predates the audit) via a different but
equally effective mechanism: route to the SERIAL-main-thread branch whenever the GPU-discretize path is
active, so only ONE thread ever touches the device, regardless of pair count or ``n_jobs``.

``_should_serialize_fe_pair_check`` (extracted 2026-07-09 for direct testability) is the exact predicate
gating that routing decision; these tests pin its contract so a future refactor cannot silently drop the
GPU-contention protection.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._mrmr_fe_step._step_core import _should_serialize_fe_pair_check


def test_gpu_active_forces_serial_even_with_many_pairs_and_workers():
    """The core regression: plenty of pairs to fill a large worker pool, but GPU-discretize is active ->
    must still route to serial (this is the exact scenario the audit worried had no protection)."""
    assert _should_serialize_fe_pair_check(n_prospective_pairs=5000, gpu_fe_active=True, serial_min_pairs_per_worker=16) is True


def test_gpu_inactive_with_plenty_of_pairs_uses_threading():
    """Gpu inactive with plenty of pairs uses threading."""
    assert _should_serialize_fe_pair_check(n_prospective_pairs=5000, gpu_fe_active=False, serial_min_pairs_per_worker=16) is False


def test_too_few_pairs_forces_serial_regardless_of_gpu():
    """Pre-existing narrow+tall protection (2026-06-08): unrelated to GPU, still must hold."""
    assert _should_serialize_fe_pair_check(n_prospective_pairs=1, gpu_fe_active=False, serial_min_pairs_per_worker=4) is True


def test_gpu_active_and_too_few_pairs_still_serial():
    """Gpu active and too few pairs still serial."""
    assert _should_serialize_fe_pair_check(n_prospective_pairs=1, gpu_fe_active=True, serial_min_pairs_per_worker=4) is True


def test_boundary_exactly_at_threshold_uses_threading():
    """n_prospective_pairs == threshold is NOT below it -> threading branch (GPU inactive)."""
    assert _should_serialize_fe_pair_check(n_prospective_pairs=4, gpu_fe_active=False, serial_min_pairs_per_worker=4) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])

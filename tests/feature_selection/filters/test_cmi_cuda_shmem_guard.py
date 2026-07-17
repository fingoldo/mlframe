"""Regression: the batched-CMI CUDA dispatch runs TWO kernels; the shared-mem gate must reject a launch when EITHER
exceeds the 48 KB/block limit. The original guard checked only kernel 1 (joint_size*4) and missed kernel 2
(cmi_from_joint: 256*8 + (nbx*nbz + nby*nbz + nbz)*4), which EXCEEDS kernel 1 when an axis is degenerate (nby=1 makes
nbx*nbz == joint_size, so kernel 2 = joint_size*4 + 2048 + 8*nbz). Such a case would pass the gate and launch-fail on
the device. This pins that _cmi_cuda_shmem_fits rejects it, while a comfortably-small config is still accepted and the
kernel-1-only overflow is still rejected."""

from mlframe.feature_selection.filters.info_theory._cmi_cuda import _cmi_cuda_shmem_fits

_CC = 48 * 1024


def test_degenerate_axis_kernel2_overflow_is_rejected():
    # nby=1, nbx=120, nbz=100 -> joint_size = 12000, kernel1 = 48000 <= 49152 (PASSES kernel-1 guard) ...
    nbx, nby, nbz = 120, 1, 100
    joint_size = nbx * nby * nbz
    assert joint_size * 4 <= _CC, "kernel-1 must fit so this exercises the kernel-2 guard specifically"
    # ... but kernel2 = 2048 + (12000 + 100 + 100)*4 = 50848 > 49152 -> MUST be rejected
    k2 = 256 * 8 + (nbx * nbz + nby * nbz + nbz) * 4
    assert k2 > _CC
    assert _cmi_cuda_shmem_fits(joint_size, nbx, nby, nbz) is False


def test_small_config_accepted():
    nbx, nby, nbz = 10, 10, 10
    assert _cmi_cuda_shmem_fits(nbx * nby * nbz, nbx, nby, nbz) is True


def test_kernel1_overflow_rejected():
    # joint_size*4 alone exceeds 48 KB
    nbx, nby, nbz = 40, 40, 40  # joint_size = 64000 -> *4 = 256000 > 49152
    assert _cmi_cuda_shmem_fits(nbx * nby * nbz, nbx, nby, nbz) is False


def test_legacy_no_nbins_checks_kernel1_only():
    # nbins unknown (0) -> only kernel 1 is guarded (back-compat for callers not passing nbins)
    assert _cmi_cuda_shmem_fits(10000) is True  # 40000 <= 49152
    assert _cmi_cuda_shmem_fits(20000) is False  # 80000 > 49152

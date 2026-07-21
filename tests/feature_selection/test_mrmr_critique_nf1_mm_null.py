"""Regression (MRMR critique N-F1): the permutation null MUST use the SAME estimator as the observed relevance.
The observed MI is Miller-Madow when mi_correction='miller_madow', but parallel_mi_prange_with_null computed the
null with plug-in MI -> the exceedance test compared plug-in shuffles against an MM-lowered observed
(over-rejection) and observed_mm - null_mean_plugin double-subtracted the bias. Now use_mm threads into the null.
No-op when MM is off (the default), so the plug-in path is unchanged.
"""

import numpy as np

from mlframe.feature_selection.filters.permutation import parallel_mi_prange_with_null


def _fixture(n=1500, kx=4, seed=0):
    """Builds seeded synthetic test data; returns ``(classes_x, freqs_x, y, freqs_y)``."""
    rng = np.random.default_rng(seed)
    classes_x = rng.integers(0, kx, n).astype(np.int32)
    y = ((classes_x + rng.integers(0, 2, n)) % 2).astype(np.int32)  # y correlated with x
    freqs_x = np.bincount(classes_x, minlength=kx).astype(np.float64) / n
    freqs_y = np.bincount(y, minlength=2).astype(np.float64) / n
    return classes_x, freqs_x, y, freqs_y


def test_null_kernel_mm_off_is_unchanged_default():
    """Null kernel mm off is unchanged default."""
    cx, fx, cy, fy = _fixture()
    # default (no use_mm) must equal explicit use_mm=False -> plug-in path bit-identical
    a = parallel_mi_prange_with_null(cx, fx, cy, fy, 32, 0.5, np.uint64(0), np.int32, False)
    b = parallel_mi_prange_with_null(cx, fx, cy, fy, 32, 0.5, np.uint64(0), np.int32, False, False)
    assert a == b


def test_null_kernel_applies_mm_when_requested():
    """Null kernel applies mm when requested."""
    cx, fx, cy, fy = _fixture()
    _, _, sum_plugin = parallel_mi_prange_with_null(cx, fx, cy, fy, 64, 0.5, np.uint64(7), np.int32, False, False)
    _, _, sum_mm = parallel_mi_prange_with_null(cx, fx, cy, fy, 64, 0.5, np.uint64(7), np.int32, False, True)
    # MM changes the per-shuffle null MI -> the accumulated null differs from the plug-in null (same seed/shuffles).
    assert abs(sum_mm - sum_plugin) > 1e-9, "use_mm did not change the null (MM correction not applied)"
    assert np.isfinite(sum_mm)

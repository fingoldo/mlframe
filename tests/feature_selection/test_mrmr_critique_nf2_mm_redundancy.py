"""Regression (MRMR critique N-F2 part 1): the Fleuret redundancy (conditional_mi) must carry the SAME Miller-Madow
bias correction as the MM relevance. Plug-in CMI over-estimates redundancy for high-cardinality candidates, biasing
the relevance-minus-redundancy objective against them. conditional_mi(use_mm=True) subtracts the analytic MM CMI
bias (k_xyz + k_z - k_xz - k_yz)/(2n); no-op (bit-identical) when use_mm=False (the default plug-in path).
"""

import numpy as np

from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi, _cmi_miller_madow_bias


def _fixture(kx, n=4000, seed=0):
    """Builds seeded synthetic test data; returns ``(fd, nb, np.array([0], np.int64), np.array([1], np.int64), np.array([2], np.int64))``."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, kx, n)
    y = rng.integers(0, 2, n)
    z = rng.integers(0, 6, n)
    fd = np.column_stack([x, y, z]).astype(np.int32)
    nb = np.array([kx, 2, 6], dtype=np.int64)
    return fd, nb, np.array([0], np.int64), np.array([1], np.int64), np.array([2], np.int64)


def test_conditional_mi_mm_subtracts_analytic_bias():
    """Conditional mi mm subtracts analytic bias."""
    fd, nb, xi, yi, zi = _fixture(kx=8)
    plug = conditional_mi(fd, xi, yi, zi, None, nb, dtype=np.int32, use_mm=False)
    mm = conditional_mi(fd, xi, yi, zi, None, nb, dtype=np.int32, use_mm=True)
    bias = _cmi_miller_madow_bias(fd, xi, yi, zi, nb, np.int32)
    assert bias >= -1e-12, "MM CMI bias must be >= 0 for nested supports"
    assert abs(mm - max(0.0, plug - bias)) < 1e-12, "use_mm must subtract exactly the analytic MM CMI bias"


def test_default_use_mm_false_is_plugin():
    """Default use mm false is plugin."""
    fd, nb, xi, yi, zi = _fixture(kx=8)
    a = conditional_mi(fd, xi, yi, zi, None, nb, dtype=np.int32)  # default (no use_mm)
    b = conditional_mi(fd, xi, yi, zi, None, nb, dtype=np.int32, use_mm=False)
    assert a == b, "default conditional_mi must equal explicit use_mm=False (plug-in, bit-identical)"


def test_mm_bias_grows_with_cardinality():
    # the N-F2 point: plug-in redundancy over-estimation scales with cardinality, so the MM correction is larger for
    # high-card candidates (which the plug-in objective was penalising).
    """Mm bias grows with cardinality."""
    _, nb8, xi, yi, zi = _fixture(kx=8)
    fd8, nb8, xi, yi, zi = _fixture(kx=8)
    fd40, nb40, *_ = _fixture(kx=40)
    b8 = _cmi_miller_madow_bias(fd8, xi, yi, zi, nb8, np.int32)
    b40 = _cmi_miller_madow_bias(fd40, xi, yi, zi, nb40, np.int32)
    assert b40 > b8 * 2, f"high-card bias should be much larger: Kx=40 {b40:.5f} vs Kx=8 {b8:.5f}"

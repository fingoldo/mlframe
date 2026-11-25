"""Cluster 09 regression: the noise-scale tag is rounded, not truncated (audit 2026-11-20, finding 09-F39).

``int(0.29 * 100)`` is 28 because of float error, so a scale of 0.29 was named ``n28``. The tag now rounds to the nearest hundredth.
"""

import numpy as np

from mlframe.feature_engineering.transformer import diffusion_noise as dn


def _column_names(noise_scale):
    """Run the diffusion features for one noise scale and return the output column names."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, 3))
    y = (X[:, 0] > 0.5).astype(int)
    out = dn.compute_diffusion_noise_features(X, y, X[:5], seed=0, noise_scales=(noise_scale,), n_virtuals_per_pos=2)
    return out.columns


def test_scale_of_0_29_is_tagged_n29_not_n28():
    """A noise scale of 0.29 is named ``n29`` in every column, so the tag is the rounded hundredth (09-F39)."""
    names = _column_names(0.29)
    assert names and all(name.startswith("diff_n29_") for name in names)


def test_scale_of_0_1_keeps_its_own_tag():
    """Control: an exactly representable scale keeps the tag it always had (09-F39)."""
    names = _column_names(0.1)
    assert names and all(name.startswith("diff_n10_") for name in names)

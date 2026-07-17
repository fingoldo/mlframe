"""Biz-value: ``auto_base_structural_boost_fraction`` scales the structural-affinity nudge.

``_auto_base`` adds a bounded MI nudge to structurally-obvious base columns (near-affine predictor of y, low-card integer
grouping, monotone/time column) so an OBVIOUS base is not buried when a noisier competitor's pairwise MI lands a hair
higher. The nudge magnitude is ``auto_base_structural_boost_fraction * mi_spread`` via ``boost_for_features`` -- the field
is the lever that decides how strong the tie-break is.

The win: a near-affine predictor of y receives a positive boost, and DOUBLING the fraction doubles that boost (so a
larger fraction can flip a near-MI-tie in favour of the structurally-correct base). A regression that ignores the
fraction (hardcodes the nudge) flattens the proportionality and FAILS the test.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._structural_hints import boost_for_features


def _affine_base_matrix(n=2000, seed=0):
    """Feature 'lin' is a near-affine predictor of y (structural linear_residual base); 'noise' is structureless."""
    rng = np.random.default_rng(seed)
    lin = rng.normal(size=n)
    y = 3.0 * lin + 1.0 + 0.01 * rng.normal(size=n)
    noise = rng.normal(size=n)
    x_matrix = np.column_stack([lin, noise])
    return x_matrix, y, ["lin", "noise"]


def test_biz_val_structural_boost_fraction_positive_on_affine_base():
    """The near-affine predictor of y gets a positive structural boost."""
    x_matrix, y, names = _affine_base_matrix()
    boost, _kinds = boost_for_features(x_matrix, y, names, mi_spread=1.0, max_boost_fraction=0.25)
    assert boost[0] > 0.0, "a near-affine predictor of y must receive a positive structural boost"


def test_biz_val_structural_boost_fraction_scales_linearly():
    """Doubling auto_base_structural_boost_fraction doubles the applied nudge."""
    x_matrix, y, names = _affine_base_matrix()
    b_quarter, _ = boost_for_features(x_matrix, y, names, mi_spread=1.0, max_boost_fraction=0.25)
    b_half, _ = boost_for_features(x_matrix, y, names, mi_spread=1.0, max_boost_fraction=0.50)
    assert b_half[0] > b_quarter[0], "a larger fraction must produce a larger boost"
    assert abs(b_half[0] - 2.0 * b_quarter[0]) < 1e-9, "boost must scale linearly with the fraction"

"""Cluster 39 regression: the SVD default width is one named constant used by both call sites (audit 2026-11-20, finding 39-F31).

The literal 256 used to be repeated at both call sites in apply.py. Both now read _SVD_DEFAULT_DIM, and the value is unchanged.
"""

from mlframe.training.feature_handling import apply as apply_mod


def test_constant_keeps_the_default_width():
    """The named default is 256, as before (39-F31)."""
    assert apply_mod._SVD_DEFAULT_DIM == 256

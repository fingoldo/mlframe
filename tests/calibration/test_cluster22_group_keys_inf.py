"""Cluster 22 regression: infinite group labels keep their own key instead of an undefined int64 cast (audit 2026-11-20, finding 22-F40)."""

import numpy as np

from mlframe.calibration.group_bias_correction import _canonical_group_key_series


def test_infinite_labels_get_their_own_string_keys():
    """inf and -inf map to 'inf' and '-inf', not to an integer key (22-F40)."""
    keys = _canonical_group_key_series(np.array([np.inf, -np.inf, 3.0])).tolist()
    assert keys == ["inf", "-inf", "3"]


def test_whole_finite_labels_still_stringify_as_integers():
    """Control: whole finite floats keep the int-style key so 3.0 and 3 match (22-F40)."""
    assert _canonical_group_key_series(np.array([3.0, 4.0])).tolist() == ["3", "4"]

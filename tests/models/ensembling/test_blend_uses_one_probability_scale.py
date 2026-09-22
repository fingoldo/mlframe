"""A blend must average member probabilities on one scale.

With calibrated probabilities enabled, each member contributed its calibrated mirror when it had one and its raw
probabilities otherwise, so isotonic-flattened and raw sigmoid vectors were averaged together.
"""

from __future__ import annotations

import types

import numpy as np

from mlframe.models.ensembling.process_method import _select_split_probs


def _member(raw, cal=None):
    ns = types.SimpleNamespace(val_probs=np.asarray(raw))
    if cal is not None:
        ns.calibrated_val_probs = np.asarray(cal)
    return ns


def test_a_mixed_member_set_falls_back_to_raw_for_everyone():
    members = [_member([0.2, 0.8], cal=[0.3, 0.7]), _member([0.1, 0.9])]
    got = _select_split_probs(members, "val", use_calibrated=True)
    np.testing.assert_array_equal(got[0], [0.2, 0.8])
    np.testing.assert_array_equal(got[1], [0.1, 0.9])


def test_a_fully_calibrated_set_uses_the_calibrated_probabilities():
    members = [_member([0.2, 0.8], cal=[0.3, 0.7]), _member([0.1, 0.9], cal=[0.15, 0.85])]
    got = _select_split_probs(members, "val", use_calibrated=True)
    np.testing.assert_array_equal(got[0], [0.3, 0.7])
    np.testing.assert_array_equal(got[1], [0.15, 0.85])


def test_calibration_off_means_raw():
    members = [_member([0.2, 0.8], cal=[0.3, 0.7])]
    np.testing.assert_array_equal(_select_split_probs(members, "val", use_calibrated=False)[0], [0.2, 0.8])

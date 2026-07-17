"""Sensor: splitting.py carve of index-level helpers into ``_split_helpers.py``.

Verifies re-export identity AND calls into the moved bodies so a missing
``from .parent import <name>`` (lazy NameError) would fail the suite, not just
pass an import-only check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_split_helpers_reexport_identity():
    """Split helpers reexport identity."""
    from mlframe.training import _split_helpers as sib
    from mlframe.training import splitting as parent

    assert parent._stratified_split is sib._stratified_split
    assert parent._carve_calib_from_train is sib._carve_calib_from_train


def test_stratified_split_body_callable():
    """Stratified split body callable."""
    from mlframe.training._split_helpers import _stratified_split

    idx = np.arange(20)
    y = np.array([0, 1] * 10)
    left, right = _stratified_split(idx, 0.25, y, random_state=0)
    assert len(right) == 5 and len(left) == 15
    # disjoint cover
    assert set(left.tolist()) | set(right.tolist()) == set(range(20))


def test_carve_calib_random_and_timeordered_bodies_callable():
    """Carve calib random and timeordered bodies callable."""
    from mlframe.training._split_helpers import _carve_calib_from_train

    train = np.arange(16)
    new_train, calib = _carve_calib_from_train(train, 0.25, n_total=20, timestamps=None, groups=None, rng=np.random.default_rng(0))
    assert len(calib) == 5 and len(new_train) == 11

    ts = pd.Series(pd.date_range("2020-01-01", periods=16))
    _new_train2, calib2 = _carve_calib_from_train(train, 0.25, n_total=20, timestamps=ts, groups=None, rng=np.random.default_rng(0))
    # oldest train rows form the calib slice under timestamps
    assert set(calib2.tolist()) == {0, 1, 2, 3, 4}

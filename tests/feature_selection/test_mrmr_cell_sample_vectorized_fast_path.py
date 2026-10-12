"""``_column_cell_sample``'s numeric fast path (one gather + one ``tobytes()`` call) must still
detect any content difference a frame's fingerprint needs to catch, exactly like the old
per-position ``np.asarray(scalar).tobytes()`` loop did -- only the implementation changed, not the
set of distinguishable contents (see the perf-win docstring on ``_column_cell_sample``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fingerprints import _column_cell_sample, _mrmr_compute_x_fingerprint


def test_numeric_fast_path_matches_the_legacy_per_position_loop():
    """The new fused numeric path returns the same content signal as the old per-position loop."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(5000)
    positions = [0, 1, 1234, 2500, 4999]

    def _legacy(arr, positions):
        """Reproduce the pre-fix per-position ``np.asarray(scalar).tobytes()`` loop verbatim."""
        return tuple(np.asarray(arr[p]).tobytes() for p in positions if p < len(arr))

    fast = _column_cell_sample(pd.DataFrame({"c": arr}), "c", positions)
    legacy = _legacy(arr, positions)
    assert fast == b"".join(legacy)


def test_numeric_fast_path_detects_a_single_changed_cell():
    """Two frames differing in exactly one sampled cell must not share a cell sample."""
    base = pd.DataFrame({"c": np.arange(2000.0)})
    changed = base.copy()
    positions = list(range(0, 2000, 4))
    changed.loc[positions[194], "c"] = -1.0
    assert _column_cell_sample(base, "c", positions) != _column_cell_sample(changed, "c", positions)


def test_object_dtype_column_still_uses_the_per_element_path():
    """A string/categorical (object-dtype) column keeps the per-element tuple-of-bytes shape, not a single blob."""
    df = pd.DataFrame({"c": ["mon", "tue", "wed", "thu"]})
    out = _column_cell_sample(df, "c", [0, 1, 2, 3])
    assert isinstance(out, tuple)
    assert out == tuple(np.asarray(v).tobytes() for v in ["mon", "tue", "wed", "thu"])


def test_fingerprint_still_distinguishes_frames_differing_only_in_a_numeric_column():
    """End-to-end: the frame fingerprint still separates two frames that differ in one numeric cell."""
    a = pd.DataFrame({"x": np.arange(2000.0), "y": np.arange(2000.0)})
    b = a.copy()
    b.loc[999, "x"] = -5.0
    assert _mrmr_compute_x_fingerprint(a) != _mrmr_compute_x_fingerprint(b)

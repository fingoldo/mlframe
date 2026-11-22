"""Audit 2026-11-20 finding 04-F22: probes are built into one frame in a single pass, not one column at a time.

The one-column-at-a-time loop fragmented the frame's blocks. The fix must keep the probe values identical to the
per-probe streams, so these tests pin both the values and the absence of fragmentation.
"""

import warnings

import numpy as np
import pandas as pd

from mlframe.data.datasets._rng import stream_for
from mlframe.data.datasets.from_real import PROBE_PREFIX, _resample_column, inject_matched_probes


def _frame(n=300, seed=0):
    """Two numeric columns plus one text column, so the probes must skip non-numeric donors."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.integers(0, 5, size=n).astype(float), "label": ["x"] * n},
    )


def test_probe_values_are_unchanged_by_the_single_pass_build():
    """Every probe is the resample of its donor under its own stream, exactly as before the restructure."""
    frame = _frame()
    target = np.zeros(len(frame))
    bed = inject_matched_probes(frame, target, n_probes=6, seed=3, source="s")
    assert bed.probe_columns == tuple(f"{PROBE_PREFIX}{i:04d}" for i in range(6))
    donors = {"a": frame["a"].to_numpy(), "b": frame["b"].to_numpy()}
    for index, col in enumerate(bed.probe_columns):
        donor = ("a", "b")[index % 2]
        expected = _resample_column(donors[donor], stream_for(3, "s", "matched_probe", col))
        np.testing.assert_array_equal(bed.frame[col].to_numpy(), expected)
    # The real columns come through untouched, in order, ahead of the probes.
    assert list(bed.frame.columns[: frame.shape[1]]) == list(frame.columns)
    pd.testing.assert_series_equal(bed.frame["a"], frame["a"])


def test_injection_does_not_fragment_the_frame_or_warn():
    """Many probes produce no PerformanceWarning from pandas block fragmentation."""
    frame = _frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bed = inject_matched_probes(frame, np.zeros(len(frame)), n_probes=200, seed=0, source="s")
    assert not [w for w in caught if issubclass(w.category, pd.errors.PerformanceWarning)]
    assert bed.frame.shape == (len(frame), frame.shape[1] + 200)


def test_donor_index_is_preserved_when_frame_has_a_custom_index():
    """A frame with a non-default index keeps its index on the probe columns, so rows still line up."""
    frame = _frame().set_index(pd.Index(np.arange(100, 100 + len(_frame()))))
    bed = inject_matched_probes(frame, np.zeros(len(frame)), n_probes=2, seed=1, source="s")
    assert bed.frame.index.equals(frame.index)

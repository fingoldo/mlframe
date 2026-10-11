"""The full frame is rebuilt from the split frames only when the split indices partition the full table."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._target_mix_target import assemble_full_frame


def test_partitioned_splits_rebuild_the_full_frame_in_row_order():
    """Splits with shuffled indices give back the original frame row for row."""
    full = pd.DataFrame({"a": np.arange(10.0)})
    tr, va, te = np.array([0, 2, 4, 6, 8, 9]), np.array([1, 5]), np.array([3, 7])
    out = assemble_full_frame(full.iloc[tr], full.iloc[va], full.iloc[te], tr, va, te, n_rows=10)
    assert out is not None
    assert np.array_equal(out["a"].to_numpy(), full["a"].to_numpy())


def test_a_missing_or_repeated_row_refuses_the_assembly():
    """A split that omits a row or repeats one returns None, not a silently misaligned frame."""
    full = pd.DataFrame({"a": np.arange(6.0)})
    tr, va, te = np.array([0, 1, 2]), np.array([3, 4]), np.array([4])
    assert assemble_full_frame(full.iloc[tr], full.iloc[va], full.iloc[te], tr, va, te, n_rows=6) is None
    tr2, va2, te2 = np.array([0, 1, 2]), np.array([3, 4]), np.array([5])
    assert assemble_full_frame(full.iloc[tr2], full.iloc[va2], full.iloc[te2], tr2, va2, te2, n_rows=6) is not None


def test_frame_length_mismatch_refuses_the_assembly():
    """A split frame whose length disagrees with its index returns None."""
    full = pd.DataFrame({"a": np.arange(4.0)})
    tr, va = np.array([0, 1]), np.array([2, 3])
    assert assemble_full_frame(full.iloc[:1], full.iloc[va], None, tr, va, np.array([], dtype=np.int64), n_rows=4) is None

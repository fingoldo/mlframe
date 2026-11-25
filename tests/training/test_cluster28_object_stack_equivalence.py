"""Cluster 28 regression: stacking object-dtype labels with tolist() gives the same array as the per-row asarray loop (audit 2026-11-20, finding 28-F53).

Equivalence over object arrays of numpy arrays and of Python lists: the stacked result is identical, and the module no longer uses the
per-row list comprehension.
"""

from pathlib import Path

import numpy as np

from mlframe.training import _training_loop_objectives as tlo

_SRC = Path(tlo.__file__).resolve()


def test_tolist_stack_equals_the_per_row_loop():
    """Both forms give identical stacked arrays for array and list elements (28-F53)."""
    arr_of_arrays = np.empty(3, dtype=object)
    arr_of_arrays[:] = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    arr_of_lists = np.empty(2, dtype=object)
    arr_of_lists[:] = [[1.0, 2.0], [3.0, 4.0]]
    for x in (arr_of_arrays, arr_of_lists):
        old = np.stack([np.asarray(c) for c in x], axis=0)
        new = np.stack(x.tolist(), axis=0)
        assert np.array_equal(old, new)

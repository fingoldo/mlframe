"""The group broadcast scrubs non-finite looked-up values on the distinct keys and equals the per-row reference, with missing keys, unseen keys, NaN and infinite values."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._grouped_coerce_shared import broadcast_lookup


def _reference(g_keys, lookup, glob):
    """The original form: map every row, then scrub the whole vector."""
    out = np.array([lookup.get(str(k), glob) for k in g_keys], dtype=_crit_np_dtype())
    return np.nan_to_num(out, nan=glob, posinf=glob, neginf=glob)


@pytest.mark.parametrize("with_missing", [False, True])
@pytest.mark.parametrize("glob", [0.5, -2.0])
def test_equals_the_per_row_reference(with_missing, glob):
    """Same values as mapping each row and scrubbing the vector."""
    rng = np.random.default_rng(0)
    keys = rng.choice(np.array(["a", "b", "c", "d", "unseen"], dtype=object), size=5000)
    if with_missing:
        keys[[3, 77, 1000]] = None
        keys[[5, 900]] = np.nan
    lookup = {"a": 1.5, "b": float("nan"), "c": float("inf"), "d": -float("inf"), "None": float("nan"), "nan": 7.0}
    got = broadcast_lookup(keys, lookup, glob)
    np.testing.assert_array_equal(got, _reference(keys, lookup, glob))
    assert got.dtype == _crit_np_dtype()

"""A pair-search chunk in which every transform fails still hands the consumer the four-field entry it unpacks."""
import logging

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_chunks import _compute_one_fe_chunk_part2


def _boom(a, b):
    """A binary op that always fails."""
    raise ValueError("bad op")


def test_all_failing_transforms_give_a_four_field_entry():
    """The early return stored (candidates, None, times); the consumer unpacks (candidates, mi, times, codes) and raised ValueError."""
    tv = np.random.default_rng(0).standard_normal((50, 2)).astype(np.float32)
    ta, tb = (0, "identity"), (1, "identity")
    done, out = _compute_one_fe_chunk_part2(
        None, {"boom": _boom}, [(0, 1)], {(0, 1): [(ta, tb)]}, {ta: 0, tb: 1}, "__prewarp__",
        False, tv, 8, np.int32, False, np.empty((50, 4), dtype=np.float32), logging.getLogger("t"), True, None,
    )
    assert done and len(out[(0, 1)]) == 4
    candidates, mi, _times, codes = out[(0, 1)]
    assert candidates == [] and mi is None and codes is None

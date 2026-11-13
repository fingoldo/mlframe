"""The blocked recompute applies each op to contiguous copies of the pair's two columns (a strided view made every ufunc 2.6x slower); the candidate block must hold exactly the values of the ops on the original columns."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters import feature_engineering as fe
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_recompute as rec
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_code_bins import scrub_nonfinite


def test_contiguous_column_is_a_cached_contiguous_copy_with_the_same_values():
    """One copy per column index, contiguous, equal to the strided column."""
    m = np.random.default_rng(0).standard_normal((50, 4)).astype(np.float32)
    cache: dict = {}
    c1 = rec._contiguous_column(cache, m, 2)
    assert c1.flags.c_contiguous and np.array_equal(c1, m[:, 2])
    assert rec._contiguous_column(cache, m, 2) is c1 and len(cache) == 1


@pytest.mark.parametrize("preset", ["minimal", "medium", "maximal"])
def test_the_candidate_block_equals_the_ops_on_the_original_columns(monkeypatch, preset):
    """Cleaned block rows are the registry ops applied to the strided columns, in registry order for every pair of the combination list."""
    rng = np.random.default_rng(1)
    n = 80
    tv = rng.standard_normal((n, 4)).astype(np.float32)
    tv[3, 0] = np.nan
    tv[7, 2] = 0.0
    names = [("a", "id"), ("b", "id"), ("c", "id"), ("d", "id")]
    vt = {nm: i for i, nm in enumerate(names)}
    combs = [(names[0], names[1]), (names[2], names[3]), (names[1], names[2])]
    binary = fe.create_binary_transformations(preset)
    st = SimpleNamespace(combs=combs, i=0, _local_times={}, _col_buf_1d=np.empty(n, dtype=np.float32))
    held = []

    def fake_bin(blk, meta, pending, *_a, **_k):
        """Keep the cleaned rows of the block that the recompute hands to the binner."""
        vals = np.array(blk[: len(meta)], copy=True)
        scrub_nonfinite(vals)
        held.append(vals)
        pending.clear()

    monkeypatch.setattr(rec, "_bin_held_block", fake_bin)
    monkeypatch.setattr(rec, "_score_pending_candidates", lambda *a, **k: [])
    monkeypatch.setattr(rec, "_replay_pending_scores", lambda *a, **k: None)
    rec._score_recompute_blocked(st, None, tv, vt, binary, "prewarp", 10, np.int8, "quantile", None, None, None, None, 0, 0.0, False, None, None, False, None, 0, None, None, None)
    got = np.concatenate(held, axis=0)
    want = []
    with np.errstate(all="ignore"):
        for pa, pb in combs:
            for fn in binary.values():
                row = np.asarray(fn(tv[:, vt[pa]], tv[:, vt[pb]]), dtype=np.float32)
                scrub_nonfinite(row)
                want.append(row)
    np.testing.assert_array_equal(got, np.stack(want))
    assert st.i == len(combs) * len(binary)

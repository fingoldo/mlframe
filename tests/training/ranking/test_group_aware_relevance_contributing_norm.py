"""Regression: group-aware relevance must normalise by the sum of CONTRIBUTING group
sizes (size>=4), not by all rows.

Only groups with >=4 rows enter the size-weighted MI accumulation; tiny queries are
skipped. Dividing the accumulated weight by ``sizes.sum()`` (all rows) shrinks the
size-weighted average toward 0 in proportion to the fraction of rows living in the skipped
tiny queries -- a systematic downward relevance bias. The fix divides by the sum of the
contributing groups' sizes only.
"""

import numpy as np

from mlframe.training.ranking._ranker_fs import group_aware_relevance


def test_padding_tiny_queries_does_not_shrink_relevance():
    """Padding tiny queries does not shrink relevance."""
    rng = np.random.default_rng(0)

    # Two large queries (size 50 each) where feature 0 is perfectly relevant within-query.
    big_sizes = [50, 50]
    blocks = []
    groups = []
    gid = 0
    for sz in big_sizes:
        y = rng.integers(0, 4, size=sz).astype(np.float64)
        x = y + rng.normal(scale=0.01, size=sz)  # high within-query MI
        blocks.append(np.column_stack([x, rng.normal(size=sz)]))
        groups.extend([gid] * sz)
        gid += 1

    arr_big = np.vstack(blocks)
    groups_big = np.asarray(groups)
    rel_big = group_aware_relevance(["f0", "f1"], arr_big, arr_big[:, 0].copy(), groups_big)
    # Use the relevance label = a copy of f0's within-query structure target.
    # Recompute with an explicit relevance label to keep it deterministic.
    y_big = np.concatenate([np.round(arr_big[groups_big == g, 0]) for g in (0, 1)])

    # Now PAD with many tiny (size<4) queries that contribute NOTHING to the accumulation.
    extra_rows = 300  # tiny queries -> these are skipped, but inflate sizes.sum() pre-fix
    tiny_x = rng.normal(size=(extra_rows, 2))
    tiny_groups = np.arange(100, 100 + extra_rows)  # all singletons (size 1 < 4)
    arr_pad = np.vstack([arr_big, tiny_x])
    groups_pad = np.concatenate([groups_big, tiny_groups])

    rel_big = group_aware_relevance(["f0", "f1"], arr_big, y_big, groups_big)
    y_pad = np.concatenate([y_big, rng.normal(size=extra_rows)])
    rel_pad = group_aware_relevance(["f0", "f1"], arr_pad, y_pad, groups_pad)

    # The contributing groups (the two size-50 queries) are IDENTICAL between the two
    # calls, so f0's relevance must be unchanged by the tiny-query padding.
    assert rel_pad["f0"] == rel_big["f0"], (rel_pad["f0"], rel_big["f0"])
    # Sanity: f0 is a meaningfully relevant feature.
    assert rel_big["f0"] > 0.1

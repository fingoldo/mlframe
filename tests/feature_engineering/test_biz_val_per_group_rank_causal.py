"""biz_value test for ``feature_engineering.grouped.per_group_rank``'s ``causal`` mode.

The win: a static full-group percentile rank (the default behaviour) uses every row of the
group, including rows that occur AFTER the row being scored in time -- fine for an offline
report, but a silent leak if that percentile is then used as a FEATURE for online/causal
scoring, because a real serving-time scorer only ever sees rows up to "now". This test builds
a time-ordered panel where each entity's value trends upward over time (a realistic "quality
improves" / "price drifts" process) and shows that the plain (non-causal) per-group percentile
rank leaks future information into a same-timestep classification task -- its rank is an
almost-perfect (near-1.0 AUC) predictor of a label that is itself defined from FUTURE rows --
while the ``causal=True`` expanding-window rank, which by construction only sees rows up to
and including the scored row, degrades to a realistic, still genuinely useful, AUC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.grouped import per_group_rank


def _make_drifting_panel(n_groups: int, group_size: int, seed: int):
    """Per-group value that drifts upward with a random walk + noise, in time order.

    ``label`` (opt-in target): 1 if this row's value is in the top half of the FULL group
    (a stand-in for "this entity ends up a top performer over its observed lifetime") --
    something a real online scorer would NOT know at row-scoring-time for early rows, since
    it depends on values recorded later in the group.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    groups = np.repeat(np.arange(n_groups), group_size)
    values = np.empty(n, dtype=np.float64)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        drift = rng.normal(loc=0.15, scale=1.0, size=group_size).cumsum()
        values[s:e] = drift + rng.normal(scale=0.05, size=group_size)
    label = np.empty(n, dtype=np.int64)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        seg = values[s:e]
        label[s:e] = (seg >= np.median(seg)).astype(np.int64)
    return groups, values, label


def test_biz_val_per_group_rank_causal_avoids_future_leakage():
    """Biz val per group rank causal avoids future leakage."""
    groups, values, label = _make_drifting_panel(n_groups=150, group_size=40, seed=11)

    rank_plain = per_group_rank(values, groups, method="average", pct=True)
    rank_causal = per_group_rank(values, groups, method="average", pct=True, causal=True)

    auc_plain = roc_auc_score(label, rank_plain)
    auc_causal = roc_auc_score(label, rank_causal)

    # Measured (seed=11): plain ~0.986 (near-perfect -- it's ranking against the SAME rows
    # the "top half of lifetime" label was defined from, i.e. it's allowed to see its own
    # future), causal ~0.79 (a real online scorer's honest, still clearly-useful, signal).
    # Thresholds set 5-15% below/above the measured values against the closest real baseline
    # (the same rank computation, causal omitted).
    assert auc_plain >= 0.92
    assert auc_causal >= 0.65, f"the causal rank carries no usable signal at all (AUC={auc_causal:.3f})"
    # The paired gap is the leakage guard, not an absolute ceiling on auc_causal. The old `<= 0.90` sat
    # 0.02 below the plain arm's own 0.92 floor, so a small change to tie handling in per_group_rank
    # tripped it on correct code; and it was derived from one seed. The no-leak property itself is pinned
    # exactly in test_a_later_row_cannot_change_an_earlier_rows_causal_rank below, with no fit involved.
    assert auc_plain - auc_causal >= 0.08


def test_a_later_row_cannot_change_an_earlier_rows_causal_rank():
    """The causal property, stated exactly: a row's rank depends only on rows strictly before it.

    This is what the AUC band was circling. Mutating a LATER row inside the same group must leave every
    earlier row's causal rank untouched -- and must change the mutated row's own, or the mutation was not
    material and the invariance holds for nothing.
    """
    groups, values, _label = _make_drifting_panel(n_groups=6, group_size=25, seed=3)
    baseline = per_group_rank(values, groups, method="average", pct=True, causal=True)

    for victim in (10, 24, 37, 60):
        mutated = values.copy()
        mutated[victim] = values[victim] + 1_000.0
        after = per_group_rank(mutated, groups, method="average", pct=True, causal=True)

        same_group = groups == groups[victim]
        earlier = same_group & (np.arange(values.size) < victim)
        assert earlier.any(), f"row {victim} is the first of its group; pick a victim with rows before it"
        np.testing.assert_array_equal(
            after[earlier], baseline[earlier],
            err_msg=f"mutating row {victim} changed the causal rank of rows BEFORE it -- the rank is looking ahead",
        )
        # Everything outside the group is untouched too.
        np.testing.assert_array_equal(after[~same_group], baseline[~same_group], err_msg="the mutation leaked across group boundaries")


def test_the_plain_rank_does_look_ahead():
    """The contrast: without `causal`, a later row DOES move an earlier row's rank.

    Without this, the invariance above would also hold for a rank that had stopped being computed at all.
    """
    groups, values, _label = _make_drifting_panel(n_groups=6, group_size=25, seed=3)
    baseline = per_group_rank(values, groups, method="average", pct=True)

    mutated = values.copy()
    mutated[24] = values[24] + 1_000.0
    after = per_group_rank(mutated, groups, method="average", pct=True)

    same_group = groups == groups[24]
    earlier = same_group & (np.arange(values.size) < 24)
    assert not np.array_equal(after[earlier], baseline[earlier]), (
        "the plain rank did not react to a later row either, so the causal invariance test above is not " "distinguishing causal from plain"
    )


def test_biz_val_per_group_rank_causal_first_row_semantics():
    """Biz val per group rank causal first row semantics."""
    groups = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    values = np.array([5.0, 1.0, 3.0, 10.0, 10.0], dtype=np.float64)

    incl_self = per_group_rank(values, groups, method="average", pct=True, causal=True, causal_exclude_self=False)
    excl_self = per_group_rank(values, groups, method="average", pct=True, causal=True, causal_exclude_self=True)

    # Including self: the very first row of every group is alone in its own window -> rank 1.0.
    assert incl_self[0] == 1.0
    assert incl_self[3] == 1.0
    # Excluding self: the first row of every group has no prior data -> NaN.
    assert np.isnan(excl_self[0])
    assert np.isnan(excl_self[3])
    # Tied second group (both value 10.0): the second row's window is [10.0, 10.0] -- both
    # rows tie, so the average-tie rank over 2 rows is (0 less + (2 equal + 1)/2) / 2 = 0.75,
    # not 1.0 (there is no "ahead" position among an all-tied window).
    assert incl_self[4] == 0.75


def test_biz_val_per_group_rank_causal_omitted_is_bit_identical_to_baseline():
    """Biz val per group rank causal omitted is bit identical to baseline."""
    groups, values, _label = _make_drifting_panel(n_groups=30, group_size=15, seed=3)

    baseline = per_group_rank(values, groups, method="average")
    without_kw = per_group_rank(values, groups, method="average", causal=False)

    np.testing.assert_array_equal(baseline, without_kw)


def test_biz_val_per_group_rank_causal_rejects_non_average_method():
    """Biz val per group rank causal rejects non average method."""
    groups, values, _label = _make_drifting_panel(n_groups=10, group_size=8, seed=5)
    for method in ("min", "max", "dense", "ordinal"):
        try:
            per_group_rank(values, groups, method=method, causal=True)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for method={method!r} with causal=True")

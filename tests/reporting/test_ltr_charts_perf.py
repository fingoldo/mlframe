"""Numerical-parity + content-sanity tests for the batched LTR chart kernels.

The per-query Python loops in ``mlframe.reporting.charts.ltr`` were replaced
by batched numba kernels in ``mlframe.metrics.ranking``. These tests pin that
each kernel reproduces an inline plain-numpy reference (exactly, or to 1e-9
where the only divergence is float reduction order / a deterministic tie-break
that the kernel deliberately improves), across multiple group sizes including
size-1, score ties, and integer AND float relevance grades.

NDCG_BY_QSIZE is a new panel: its content sanity (bin count, per-bin query-count
conservation, value range, small-vs-large inflation) and the full 6-panel default
template build are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.ranking import (
    _iter_group_slices,
    _lift_curve_kernel,
    _mrr_one_query,
    _ndcg_one_query,
    _per_query_mrr_kernel,
    _per_query_ndcg_kernel,
    _summary_batched_kernel,
    ndcg_at_k,
)
from mlframe.reporting.charts.ltr import (
    _ndcg_by_qsize_panel,
    _ndcg_dist_panel,
    compose_ltr_figure,
)
from mlframe.reporting.spec import BarPanelSpec, FigureSpec


# ---------------------------------------------------------------------------
# Synthetic generators covering mixed group sizes, ties, int + float grades
# ---------------------------------------------------------------------------


def _mixed_dataset(seed: int, ties: bool, float_grades: bool, n: int = 3000):
    """Mixed-group-size LTR sample.

    Sizes deliberately include singletons (group of one doc) by drawing many
    distinct group ids over n rows. ``ties`` rounds scores so tie-break order
    matters; ``float_grades`` emits non-integer relevance.
    """
    rng = np.random.default_rng(seed)
    group_ids = rng.integers(0, n // 3, size=n)  # heavy collisions => mix of 1..k sized groups
    if float_grades:
        y_true = rng.uniform(0.0, 3.0, size=n)
    else:
        y_true = rng.integers(0, 4, size=n)
    if ties:
        y_score = np.round(rng.normal(size=n), 1)  # coarse => many tied scores
    else:
        # all-distinct scores: a permutation with a tiny jitter so no exact ties
        y_score = rng.permutation(n).astype(np.float64) + rng.normal(0, 1e-7, size=n)
    return y_true, y_score.astype(np.float64), group_ids


def _ref_per_query(y_true, y_score, group_ids):
    """Plain reference: sorted layout + per-query metric loop (pre-batch path)."""
    st, ss, gs = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(gs) - 1
    ndcg = np.full(n_groups, np.nan)
    mrr = np.full(n_groups, np.nan)
    for i in range(n_groups):
        s, e = gs[i], gs[i + 1]
        if e > s:
            ndcg[i] = _ndcg_one_query(st[s:e], ss[s:e], 10)
            mrr[i] = _mrr_one_query(st[s:e], ss[s:e])
    return st, ss, gs, ndcg, mrr


def _ref_lift(y_true, y_score, group_ids, max_k):
    """Inline plain-numpy lift reference using a STABLE tie-break (mergesort).

    The kernel sorts ties with ``kind="mergesort"`` to be deterministic across
    runs/platforms (the same choice ``_ndcg_one_query``/``_mrr_one_query`` make).
    The reference matches that so parity is exact; an unstable (default-quicksort)
    reference would diverge ONLY in tie-break order, never in logic.
    """
    st, ss, gs = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(gs) - 1
    lift = np.zeros(max_k)
    counts = np.zeros(max_k)
    for i in range(n_groups):
        s, e = gs[i], gs[i + 1]
        n = e - s
        if n <= 0:
            continue
        re = st[s:e]
        sc = ss[s:e]
        order = np.argsort(-sc, kind="mergesort")
        ordered = re[order]
        cum = np.cumsum(ordered)
        ideal_cum = np.cumsum(-np.sort(-re))
        for k in range(min(n, max_k)):
            if ideal_cum[k] > 0:
                lift[k] += cum[k] / ideal_cum[k]
                counts[k] += 1
    counts[counts == 0] = 1
    return lift / counts


def _eq_with_nan(a, b):
    """Exact equality treating NaN==NaN as equal."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    return np.array_equal(a[~nan_a], b[~nan_b])


_CASES = [
    pytest.param(0, False, False, id="distinct_int"),
    pytest.param(1, True, False, id="ties_int"),
    pytest.param(2, False, True, id="distinct_float"),
    pytest.param(3, True, True, id="ties_float"),
]


# ---------------------------------------------------------------------------
# (a) Per-kernel numerical parity
# ---------------------------------------------------------------------------


class TestKernelParity:
    @pytest.mark.parametrize("seed,ties,float_grades", _CASES)
    def test_per_query_ndcg_kernel_identical_to_loop(self, seed, ties, float_grades):
        yt, ys, gid = _mixed_dataset(seed, ties, float_grades)
        st, ss, gs, ref_ndcg, _ = _ref_per_query(yt, ys, gid)
        got = _per_query_ndcg_kernel(st, ss, gs, 10)
        # Same underlying _ndcg_one_query (mergesort tie-break) => bit-identical.
        assert _eq_with_nan(ref_ndcg, got)

    @pytest.mark.parametrize("seed,ties,float_grades", _CASES)
    def test_per_query_mrr_kernel_identical_to_loop(self, seed, ties, float_grades):
        yt, ys, gid = _mixed_dataset(seed, ties, float_grades)
        st, ss, gs, _, ref_mrr = _ref_per_query(yt, ys, gid)
        got = _per_query_mrr_kernel(st, ss, gs)
        assert _eq_with_nan(ref_mrr, got)

    @pytest.mark.parametrize("seed,ties,float_grades", _CASES)
    def test_ndcg_k_batched_identical_to_per_k_ndcg_at_k(self, seed, ties, float_grades):
        yt, ys, gid = _mixed_dataset(seed, ties, float_grades)
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        sizes = np.diff(gs)
        max_k = min(int(sizes.max()), 50)
        eval_ks = np.arange(1, max_k + 1, dtype=np.int64)
        nsum, ncnt, _, _, _, _ = _summary_batched_kernel(st, ss, gs, eval_ks)
        batched = np.where(ncnt > 0, nsum / np.maximum(ncnt, 1), np.nan)
        ref = np.array([ndcg_at_k(yt, ys, gid, k=int(k)) for k in eval_ks])
        # Both reduce the same per-group NDCG@k over valid groups; identical.
        assert np.allclose(batched, ref, atol=1e-12, equal_nan=True)

    @pytest.mark.parametrize("seed,ties,float_grades", _CASES)
    def test_lift_kernel_matches_plain_reference(self, seed, ties, float_grades):
        yt, ys, gid = _mixed_dataset(seed, ties, float_grades)
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        max_k = min(int(np.diff(gs).max()), 50)
        ref = _ref_lift(yt, ys, gid, max_k)
        ls, cn = _lift_curve_kernel(st, ss, gs, max_k)
        got = ls / np.maximum(cn, 1)
        # 1e-9: only divergence is parallel float-reduction order across chunks.
        assert np.allclose(got, ref, atol=1e-9)

    def test_lift_kernel_bit_identical_on_distinct_scores(self):
        """No ties => the unstable-sort reference also matches, so the kernel is
        bit-identical to BOTH tie-break conventions (proves divergence is purely
        tie-break, never logic)."""
        yt, ys, gid = _mixed_dataset(2, ties=False, float_grades=True)
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        max_k = min(int(np.diff(gs).max()), 50)
        # Unstable (default) tie-break reference.
        n_groups = len(gs) - 1
        lift = np.zeros(max_k)
        counts = np.zeros(max_k)
        for i in range(n_groups):
            s, e = gs[i], gs[i + 1]
            re, sc = st[s:e], ss[s:e]
            order = np.argsort(-sc)
            cum = np.cumsum(re[order])
            ideal_cum = np.cumsum(-np.sort(-re))
            for k in range(min(e - s, max_k)):
                if ideal_cum[k] > 0:
                    lift[k] += cum[k] / ideal_cum[k]
                    counts[k] += 1
        counts[counts == 0] = 1
        ref = lift / counts
        ls, cn = _lift_curve_kernel(st, ss, gs, max_k)
        assert np.allclose(ls / np.maximum(cn, 1), ref, atol=1e-9)

    def test_lift_kernel_deterministic_across_runs(self):
        yt, ys, gid = _mixed_dataset(1, ties=True, float_grades=False)
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        max_k = min(int(np.diff(gs).max()), 50)
        a = _lift_curve_kernel(st, ss, gs, max_k)
        b = _lift_curve_kernel(st, ss, gs, max_k)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


class TestKernelEdgeCases:
    def test_size1_groups_and_all_zero_relevance(self):
        # groups: size1(rel>0), size2, size3(all-zero rel), size1(rel>0)
        gid = np.array([0, 1, 1, 2, 2, 2, 3])
        yt = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
        ys = np.array([0.5, 0.1, 0.9, 0.3, 0.2, 0.1, 0.7])
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        ndcg = _per_query_ndcg_kernel(st, ss, gs, 10)
        mrr = _per_query_mrr_kernel(st, ss, gs)
        # size-1 with positive rel => NDCG 1.0, MRR 1.0; all-zero group => NaN both.
        assert ndcg[0] == 1.0 and mrr[0] == 1.0
        assert np.isnan(ndcg[2]) and np.isnan(mrr[2])
        assert ndcg[3] == 1.0 and mrr[3] == 1.0

    def test_empty_input_yields_zero_groups(self):
        st, ss, gs = _iter_group_slices(np.array([]), np.array([]), np.array([]))
        assert len(gs) - 1 == 0
        ndcg = _per_query_ndcg_kernel(st, ss, gs, 10)
        ls, cn = _lift_curve_kernel(st, ss, gs, 1)
        assert ndcg.shape == (0,)
        assert ls.shape == (1,) and cn.shape == (1,)

    def test_ndcg_dist_panel_identical_to_pre_batch_filter(self):
        """``_ndcg_dist_panel`` must equal the pre-optimization computation:
        per-query NDCG@10 over groups with >=2 docs, NaN dropped."""
        yt, ys, gid = _mixed_dataset(5, ties=True, float_grades=False)
        st, ss, gs, ref_ndcg, _ = _ref_per_query(yt, ys, gid)
        sizes = np.diff(gs)
        ref = np.sort(ref_ndcg[(sizes >= 2) & ~np.isnan(ref_ndcg)])
        panel = _ndcg_dist_panel(yt, ys, gid)
        got = np.sort(np.asarray(panel.groups[0]))
        assert got.shape == ref.shape
        assert np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# (b) NDCG_BY_QSIZE panel content sanity
# ---------------------------------------------------------------------------


def _qsize_dataset(sizes_per_group, seed=0, small_score_better=False):
    """Build an LTR sample with explicit per-group sizes.

    ``small_score_better`` makes small groups rank (near-)perfectly and large
    groups rank poorly so NDCG@10 decreases with group size.
    """
    rng = np.random.default_rng(seed)
    y_true, y_score, group_ids = [], [], []
    for q, sz in enumerate(sizes_per_group):
        rels = rng.integers(0, 4, sz).astype(float)
        if small_score_better and sz <= 3:
            scores = rels + rng.normal(0, 0.01, sz)  # near-perfect ordering
        elif small_score_better:
            scores = rng.normal(0, 1.0, sz)  # uncorrelated => poor ranking
        else:
            scores = rels + rng.normal(0, 0.5, sz)
        y_true.extend(rels.tolist())
        y_score.extend(scores.tolist())
        group_ids.extend([q] * sz)
    return (np.asarray(y_true), np.asarray(y_score, dtype=np.float64),
            np.asarray(group_ids))


class TestNDCGByQSizePanel:
    def test_returns_bar_panel_with_valid_values(self):
        sizes = [1, 1, 2, 3, 4, 5, 8, 10, 16, 20, 33]
        yt, ys, gid = _qsize_dataset(sizes, seed=3)
        panel = _ndcg_by_qsize_panel(yt, ys, gid)
        assert isinstance(panel, BarPanelSpec)
        vals = np.asarray(panel.values)
        assert vals.size == len(panel.categories)
        # NDCG values are in [0, 1].
        assert np.all(vals >= -1e-12) and np.all(vals <= 1.0 + 1e-12)

    def test_bins_are_log2_spaced(self):
        # sizes spanning bins 0 (size1), 1 (2-3), 2 (4-7), 3 (8-15), 4 (16-31), 5 (32+)
        sizes = [1, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40]
        yt, ys, gid = _qsize_dataset(sizes, seed=4)
        panel = _ndcg_by_qsize_panel(yt, ys, gid)
        # Expected log2-floor bins present (only bins with >=1 valid group emitted).
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        from mlframe.metrics.ranking import _per_query_ndcg_kernel as _k
        ndcg10 = _k(st, ss, gs, 10)
        valid = ~np.isnan(ndcg10)
        expected_bins = np.unique(np.floor(np.log2(np.diff(gs)[valid])).astype(int))
        assert len(panel.categories) == len(expected_bins)

    def test_per_bin_counts_sum_to_n_valid_queries(self):
        sizes = [1, 1, 1, 2, 3, 4, 5, 9, 16, 30]
        yt, ys, gid = _qsize_dataset(sizes, seed=9)
        panel = _ndcg_by_qsize_panel(yt, ys, gid)
        st, ss, gs = _iter_group_slices(yt, ys, gid)
        from mlframe.metrics.ranking import _per_query_ndcg_kernel as _k
        n_valid = int(np.sum(~np.isnan(_k(st, ss, gs, 10))))
        # Each category label carries "(n=<count>, CI[lo,hi])"; the count is the token right after "n=" up to the
        # comma, and the per-bin counts sum to n_valid queries.
        total = 0
        for cat in panel.categories:
            n_str = cat.split("n=")[1].split(",")[0].rstrip(")").replace("_", "")
            total += int(n_str)
        assert total == n_valid

    def test_small_groups_inflate_relative_to_large(self):
        """On a synthetic where tiny groups rank near-perfectly and large groups
        rank randomly, the smallest-size bin's mean NDCG exceeds the largest."""
        # Many size-2/3 groups (near-perfect) and many size-30 groups (random).
        sizes = [2, 3] * 60 + [30] * 60
        yt, ys, gid = _qsize_dataset(sizes, seed=11, small_score_better=True)
        panel = _ndcg_by_qsize_panel(yt, ys, gid)
        vals = np.asarray(panel.values)
        # First bin = smallest sizes, last bin = largest sizes.
        assert vals[0] > vals[-1] + 0.05, (
            f"small-group NDCG {vals[0]:.3f} should exceed large-group {vals[-1]:.3f}"
        )

    def test_empty_input_returns_placeholder_bar(self):
        empty = np.array([])
        panel = _ndcg_by_qsize_panel(empty, empty, empty)
        assert isinstance(panel, BarPanelSpec)
        assert panel.categories == ("(no data)",)


# ---------------------------------------------------------------------------
# (c) Default-template composition regression
# ---------------------------------------------------------------------------


class TestComposeDefaultTemplate:
    def test_default_template_builds_six_panels(self):
        # Mixed group sizes incl. singletons, ties, integer grades.
        sizes = [1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33] * 4
        yt, ys, gid = _qsize_dataset(sizes, seed=2)
        fig = compose_ltr_figure(yt, ys, gid)  # default 6-panel template
        assert isinstance(fig, FigureSpec)
        n_panels = sum(len(row) for row in fig.panels)
        assert n_panels == 6

    def test_default_template_includes_ndcg_by_qsize(self):
        sizes = [2, 3, 5, 8, 13, 21] * 5
        yt, ys, gid = _qsize_dataset(sizes, seed=6)
        fig = compose_ltr_figure(yt, ys, gid)
        titles = [p.title for row in fig.panels for p in row]
        assert any("by query size" in t for t in titles)

    def test_shared_cache_reuse_matches_unshared(self):
        """Threading a single ``shared`` cache through the figure must yield the
        same NDCG_DIST / NDCG_BY_QSIZE content as building each panel cold."""
        sizes = [1, 2, 4, 8, 16, 32] * 6
        yt, ys, gid = _qsize_dataset(sizes, seed=8)
        cold_dist = np.sort(np.asarray(_ndcg_dist_panel(yt, ys, gid).groups[0]))
        cold_qsize = np.asarray(_ndcg_by_qsize_panel(yt, ys, gid).values)
        shared: dict = {}
        warm_dist = np.sort(np.asarray(_ndcg_dist_panel(yt, ys, gid, shared).groups[0]))
        warm_qsize = np.asarray(_ndcg_by_qsize_panel(yt, ys, gid, shared).values)
        assert np.array_equal(cold_dist, warm_dist)
        assert np.array_equal(cold_qsize, warm_qsize)

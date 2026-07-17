"""Bit-identity pins for two REJECTED report/metrics micro-leads.

Both leads were measured and rejected (see the sibling benches); these tests
pin the NUMERICAL equivalence the benches relied on, so a future re-attempt
knows the blocker is structural/perf, not correctness.

  * report-one-hot: a vectorized scatter one-hot equals the legacy
    ``column_stack`` of K ``==`` comparisons (rejected: slower at the report
    regime once the raw-label->column map is included). Bench:
    ``mlframe/training/_benchmarks/bench_report_one_hot.py``.
  * ks-shared-sort: KS computed from a reversed AUC-descending order equals
    KS from its own ascending sort (rejected: AUC sort is locked inside the
    batched/GPU ``compute_batch_aucs`` and threading an (N,K) order matrix
    back is a memory-costly restructure, not a micro-change). Bench:
    ``mlframe/metrics/_benchmarks/bench_ks_shared_sort.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._classification_extras import (
    _ks_statistic_kernel,
    ks_statistic,
)
from mlframe.training._benchmarks.bench_report_one_hot import (
    build_column_stack,
    build_searchsorted,
    build_searchsorted_allvalid,
    build_vectorized,
)


# ---------- Lead 1: report one-hot bit-identity ----------


@pytest.mark.parametrize("label_offset", [0, 10])
@pytest.mark.parametrize("k", [2, 3, 5])
def test_one_hot_vectorized_equals_column_stack(label_offset: int, k: int) -> None:
    rng = np.random.default_rng(7)
    n = 5_000
    classes = list(range(label_offset, label_offset + k))
    targets = rng.integers(label_offset, label_offset + k, size=n)
    ref = build_column_stack(targets, classes)
    assert np.array_equal(ref, build_vectorized(targets, classes))
    assert np.array_equal(ref, build_searchsorted(targets, classes))
    assert np.array_equal(ref, build_searchsorted_allvalid(targets, classes))
    # int8 dtype + (N, K) shape preserved.
    assert ref.dtype == np.int8 and ref.shape == (n, k)


def test_one_hot_unknown_label_yields_zero_row() -> None:
    # A target not in ``classes`` must map to an all-zero one-hot row in both
    # the legacy and the validity-masked vectorized paths.
    targets = np.array([0, 1, 2, 99], dtype=np.int64)
    classes = [0, 1, 2]
    ref = build_column_stack(targets, classes)
    assert np.array_equal(ref, build_searchsorted(targets, classes))
    assert ref[3].sum() == 0


# ---------- Lead 2: KS shared-sort bit-identity ----------


def _ks_via_desc_reverse(yt: np.ndarray, ys: np.ndarray) -> float:
    desc = np.argsort(ys)[::-1]  # the order the AUC path computes
    asc = np.ascontiguousarray(desc[::-1])
    return float(_ks_statistic_kernel(yt[asc], ys[asc]))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ks_from_reversed_desc_sort_is_bit_identical(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 20_000
    ys = rng.random(n)
    yt = (rng.random(n) < 0.3).astype(np.int64)
    base = ks_statistic(yt, ys)
    shared = _ks_via_desc_reverse(yt, ys)
    assert base == shared  # exact: same multiset, same tie-folding


def test_ks_shared_sort_identical_on_ties() -> None:
    # Tied scores are the dangerous case for any reorder. A reversed
    # descending order must still fold ties identically to the ascending sort.
    ys = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.9], dtype=np.float64)
    yt = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    assert ks_statistic(yt, ys) == _ks_via_desc_reverse(yt, ys)

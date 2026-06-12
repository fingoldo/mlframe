"""Microbench: per-class one-hot build in report_probabilistic_model_perf.

The multiclass one-vs-rest label matrix at the GPU batch-AUC fastpath is
built as ``np.column_stack([(targets == c).astype(int8) for c in classes])``
-- K passes over an N-vector + K temporaries + a column_stack copy. The
vectorized form maps labels to 0..K-1 once and scatters into a single
(N, K) int8 buffer with one fancy-index write.

Run: python -m mlframe.training._benchmarks.bench_report_one_hot

REJECT (measured, multi-size): at the report regime the labels are RAW
(``classes`` come from model.classes_ / np.unique(targets), not 0..K-1), so
the scatter form MUST first map label->column. With the map included the
scatter LOSES: n=100k/K=5 col_stack=873us vs best scatter (searchsorted)
2511us = 0.35x (3x slower); n=10k/K=3 = 0.29x. searchsorted only wins at
n=500k/K=20 (1.32x). The lead's "855->591us" win is the NO-MAP 0..K-1 case
(``out[arange,y]=1``): pure=985us vs col_stack=1369us = 1.4x -- but that
path does not apply here. The build runs ONCE per report (not in the
per-class loop), so even a 0.4ms saving is <0.5% of report wall. Not
shipped: a K int8 == + column_stack is already cache-friendly and cheap;
the 2D random-index scatter is the slow part.
"""
from __future__ import annotations

import time

import numpy as np


def build_column_stack(targets: np.ndarray, classes) -> np.ndarray:
    """Legacy path: K comparisons + column_stack."""
    return np.column_stack([(targets == c).astype(np.int8) for c in classes])


def build_vectorized(targets: np.ndarray, classes) -> np.ndarray:
    """Scatter path: label->index map once, single fancy-index write.

    Rows whose label is not in ``classes`` get an all-zero one-hot row,
    matching column_stack (no comparison would be True for them).
    """
    n = targets.shape[0]
    k = len(classes)
    out = np.zeros((n, k), dtype=np.int8)
    class_to_idx = {c: j for j, c in enumerate(classes)}
    idx = np.full(n, -1, dtype=np.int64)
    for j, c in enumerate(classes):
        idx[targets == c] = j
    valid = idx >= 0
    rows = np.nonzero(valid)[0]
    out[rows, idx[rows]] = 1
    return out


def build_searchsorted(targets: np.ndarray, classes) -> np.ndarray:
    """Map labels to column index via searchsorted on the sorted class array,
    then one masked fancy-index scatter. No Python per-class loop."""
    n = targets.shape[0]
    k = len(classes)
    cls = np.asarray(classes)
    order = np.argsort(cls, kind="stable")
    cls_sorted = cls[order]
    pos = np.searchsorted(cls_sorted, targets)
    pos = np.clip(pos, 0, k - 1)
    # map sorted position -> original class column
    col = order[pos]
    # validity: targets[i] actually equals classes[col[i]]
    valid = cls[col] == targets
    out = np.zeros((n, k), dtype=np.int8)
    rows = np.nonzero(valid)[0]
    out[rows, col[rows]] = 1
    return out


def build_searchsorted_allvalid(targets: np.ndarray, classes) -> np.ndarray:
    """Variant skipping the validity mask when caller guarantees every
    target is a known class (the report path: classes derive from the
    targets / model.classes_, so every target IS a class)."""
    k = len(classes)
    cls = np.asarray(classes)
    order = np.argsort(cls, kind="stable")
    cls_sorted = cls[order]
    pos = np.searchsorted(cls_sorted, targets)
    col = order[pos]
    out = np.zeros((targets.shape[0], k), dtype=np.int8)
    out[np.arange(targets.shape[0]), col] = 1
    return out


def _time(fn, *args, iters: int = 200) -> float:
    fn(*args)  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e6  # us/call


def main() -> None:
    rng = np.random.default_rng(0)
    for n, k, label_offset in [(100_000, 5, 0), (100_000, 5, 10), (500_000, 20, 0), (10_000, 3, 0)]:
        classes = list(range(label_offset, label_offset + k))
        targets = rng.integers(label_offset, label_offset + k, size=n)
        a = build_column_stack(targets, classes)
        b = build_vectorized(targets, classes)
        c = build_searchsorted(targets, classes)
        d = build_searchsorted_allvalid(targets, classes)
        ok = np.array_equal(a, b) and np.array_equal(a, c) and np.array_equal(a, d)
        t_cs = _time(build_column_stack, targets, classes)
        t_vec = _time(build_vectorized, targets, classes)
        t_ss = _time(build_searchsorted, targets, classes)
        t_ssa = _time(build_searchsorted_allvalid, targets, classes)
        print(
            f"n={n:>7} K={k:>2} off={label_offset:>2}  "
            f"col_stack={t_cs:8.1f}us  loop_scatter={t_vec:8.1f}us  "
            f"searchsorted={t_ss:8.1f}us  ss_allvalid={t_ssa:8.1f}us  "
            f"best_speedup={t_cs / min(t_ss, t_ssa):5.2f}x  identical={ok}"
        )


if __name__ == "__main__":
    main()

"""njit port of ``iterstrat.ml_stratifiers.IterativeStratification`` (Sechidis, Tsoumakas, Vlahavas 2011).

cProfile on a 2M-row multilabel-classification combo showed the pure-Python reference implementation
costing 35.2s tottime / 47.1s cumtime in ONE call -- 35% of the whole 100.8s profiled wall. The reference
algorithm is a strict SEQUENTIAL greedy state machine (each sample's fold assignment updates running
per-fold / per-label desired-count totals that gate every later assignment), so it cannot be parallelised
across samples -- but its per-sample body is pure numpy-per-element overhead (``np.where``, ``.max()``,
boolean masks, each re-dispatching and allocating on an array of size <= n), the classic GIL-bound
per-iteration-dispatch pattern from CLAUDE.md, just without the "GIL" part since there's no threading
here at all -- just raw Python/numpy call overhead repeated ``n`` times. Porting the exact control flow to
one njit function removes that overhead entirely.

TIE-BREAKING IS NOT BIT-IDENTICAL: the reference breaks ties (label selection order, fold selection
order) via ``sklearn.utils.check_random_state(...).choice(...)``; this port uses numba's own RNG
(``np.random.seed`` + ``np.random.randint`` inside the njit function, an independent stream from numpy's
global/legacy RandomState). Per CLAUDE.md's FE/MRMR selection-equivalence exception, the bar here is
STRATIFICATION QUALITY (per-fold per-label proportions matching the requested ``r``), not which specific
sample lands in which tied fold -- ties only arise between folds/labels that are, by the algorithm's own
invariant, equally desirable at that point, so a different tie-break cannot make the split systematically
worse. Validated via a many-trial sweep comparing per-fold label-count deviation from ``r`` against the
reference (see profiling/bench_iterative_stratification_njit.py and the regression test)."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _iterative_stratification_njit(labels: np.ndarray, r: np.ndarray, seed: int) -> np.ndarray:
    """njit twin of ``iterstrat.ml_stratifiers.IterativeStratification(labels, r, random_state)``.

    ``labels`` is ``(n, K)`` bool/int8 (0/1), ``r`` is ``(F,)`` float64 fold fractions summing to 1.
    Returns ``(n,)`` int64 fold ids in ``[0, F)``. See module docstring for the tie-break caveat.
    """
    np.random.seed(seed)
    n = labels.shape[0]
    k = labels.shape[1]
    f = r.shape[0]
    test_folds = np.full(n, -1, dtype=np.int64)

    c_folds = np.empty(f, dtype=np.float64)
    for j in range(f):
        c_folds[j] = r[j] * n

    label_totals = np.zeros(k, dtype=np.int64)
    for i in range(n):
        for j in range(k):
            if labels[i, j]:
                label_totals[j] += 1
    c_folds_labels = np.empty((f, k), dtype=np.float64)
    for j in range(f):
        for c in range(k):
            c_folds_labels[j, c] = r[j] * label_totals[c]

    unprocessed = n
    processed = np.zeros(n, dtype=np.bool_)
    num_labels = np.empty(k, dtype=np.int64)
    fold_max_buf = np.empty(f, dtype=np.int64)
    fold_sub_buf = np.empty(f, dtype=np.int64)

    while unprocessed > 0:
        for j in range(k):
            num_labels[j] = 0
        for i in range(n):
            if not processed[i]:
                for j in range(k):
                    if labels[i, j]:
                        num_labels[j] += 1
        total = 0
        for j in range(k):
            total += num_labels[j]

        if total == 0:
            # All remaining unprocessed samples carry no (unprocessed-relevant) label: distribute
            # them one at a time to whichever fold currently has the largest remaining desired count.
            for i in range(n):
                if processed[i]:
                    continue
                best_val = c_folds[0]
                n_max = 1
                fold_max_buf[0] = 0
                for j in range(1, f):
                    if c_folds[j] > best_val:
                        best_val = c_folds[j]
                        n_max = 1
                        fold_max_buf[0] = j
                    elif c_folds[j] == best_val:
                        fold_max_buf[n_max] = j
                        n_max += 1
                fold_idx = fold_max_buf[0] if n_max == 1 else fold_max_buf[np.random.randint(0, n_max)]
                test_folds[i] = fold_idx
                c_folds[fold_idx] -= 1.0
                processed[i] = True
                unprocessed -= 1
            break

        # Label with the fewest (but >0) remaining unprocessed examples, ties broken randomly.
        min_val = -1
        n_min = 0
        min_candidates = np.empty(k, dtype=np.int64)
        for j in range(k):
            if num_labels[j] > 0:
                if min_val == -1 or num_labels[j] < min_val:
                    min_val = num_labels[j]
                    n_min = 1
                    min_candidates[0] = j
                elif num_labels[j] == min_val:
                    min_candidates[n_min] = j
                    n_min += 1
        label_idx = min_candidates[0] if n_min == 1 else min_candidates[np.random.randint(0, n_min)]

        for i in range(n):
            if processed[i] or not labels[i, label_idx]:
                continue
            best_val = c_folds_labels[0, label_idx]
            n_max = 1
            fold_max_buf[0] = 0
            for j in range(1, f):
                v = c_folds_labels[j, label_idx]
                if v > best_val:
                    best_val = v
                    n_max = 1
                    fold_max_buf[0] = j
                elif v == best_val:
                    fold_max_buf[n_max] = j
                    n_max += 1
            if n_max == 1:
                fold_idx = fold_max_buf[0]
            else:
                sub_best = c_folds[fold_max_buf[0]]
                n_sub = 1
                fold_sub_buf[0] = fold_max_buf[0]
                for t in range(1, n_max):
                    cand = fold_max_buf[t]
                    if c_folds[cand] > sub_best:
                        sub_best = c_folds[cand]
                        n_sub = 1
                        fold_sub_buf[0] = cand
                    elif c_folds[cand] == sub_best:
                        fold_sub_buf[n_sub] = cand
                        n_sub += 1
                fold_idx = fold_sub_buf[0] if n_sub == 1 else fold_sub_buf[np.random.randint(0, n_sub)]

            test_folds[i] = fold_idx
            processed[i] = True
            unprocessed -= 1
            for j in range(k):
                if labels[i, j]:
                    c_folds_labels[fold_idx, j] -= 1.0
            c_folds[fold_idx] -= 1.0

    return test_folds

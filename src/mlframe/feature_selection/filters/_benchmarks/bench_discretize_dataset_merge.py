"""Bench: numeric+categorical block merge in categorize_dataset.

The mixed-frame path merges the numeric code block (n, n_num) with the categorical code block (n, n_cat)
once per fit. This bench measures np.append vs np.concatenate vs a preallocated-empty + two-slice write, to
confirm the naive preallocation is a wash (same one alloc + two block copies) - the documented conclusion
behind the bench-attempt-rejected note at the merge site. The only genuine win (an out=-slice discretiser
writing the numeric block straight into the combined buffer) is deferred; it touches the cached discretiser.

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_discretize_dataset_merge
"""
from __future__ import annotations

import time

import numpy as np


def _append(a, b):
    return np.append(a, b, axis=1)


def _concat(a, b):
    return np.concatenate([a, b], axis=1)


def _prealloc(a, b):
    out = np.empty((a.shape[0], a.shape[1] + b.shape[1]), dtype=a.dtype)
    out[:, : a.shape[1]] = a
    out[:, a.shape[1] :] = b
    return out


def _best_of(fn, a, b, reps=7):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(a, b)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n, n_num, n_cat in ((1_000_000, 100, 100), (2_000_000, 50, 50)):
        a = np.zeros((n, n_num), dtype=np.int8)
        b = np.ones((n, n_cat), dtype=np.int8)
        wa = _best_of(_append, a, b)
        wc = _best_of(_concat, a, b)
        wp = _best_of(_prealloc, a, b)
        print(f"n={n:>9d} num={n_num} cat={n_cat} | append {wa*1e3:7.2f}ms | concat {wc*1e3:7.2f}ms | prealloc {wp*1e3:7.2f}ms")


if __name__ == "__main__":
    main()

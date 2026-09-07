"""Bench for the REJECTED quicksort-with-tie-gate in ``_ScoreSort`` (reporting audit PERF-10).

``_ScoreSort.__init__`` sorts the score column with ``kind="stable"``, which costs about 2.4x a quicksort.
The proposal was to sort with quicksort, detect ties on the already-sorted array (an O(n) scan, no extra
sort), and re-sort stably only when ties exist.

It is rejected because the gate loses on exactly the inputs this package serves most: tree-model scores are
quantised and tie heavily, and there the gate pays for BOTH sorts. Run this to re-measure before reopening
it -- the numbers below are from one host and the crossover is what matters, not the absolute values.

    python _benchmarks/bench_score_sort_tie_gate.py
"""

from __future__ import annotations

import time

import numpy as np

N = 2_000_000
REPEATS = 3


def _stable(scores: np.ndarray) -> np.ndarray:
    """What ``_ScoreSort`` does today."""
    return np.argsort(scores, kind="stable")[::-1]


def _gated(scores: np.ndarray) -> np.ndarray:
    """Quicksort, then fall back to a stable sort when the sorted array turns out to hold ties."""
    order = np.argsort(scores, kind="quicksort")[::-1]
    if np.any(np.diff(scores[order]) == 0):
        order = np.argsort(scores, kind="stable")[::-1]
    return order


def _best(fn, scores: np.ndarray) -> float:
    """Best of ``REPEATS``, which is what a paired comparison on a shared machine can defend."""
    return min((lambda t0: (fn(scores), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(REPEATS))


def main() -> None:
    """Print the stable-vs-gated comparison on tie-free and tie-heavy score columns."""
    rng = np.random.default_rng(0)
    cases = {
        "continuous (tie-free)": rng.random(N),
        "tree-quantised (3dp, many ties)": np.round(rng.random(N), 3),
    }
    for name, scores in cases.items():
        stable, gated = _best(_stable, scores), _best(_gated, scores)
        print(f"{name:34s} stable {stable * 1000:7.1f} ms | gated {gated * 1000:7.1f} ms  ({stable / gated:.2f}x)")


if __name__ == "__main__":
    main()

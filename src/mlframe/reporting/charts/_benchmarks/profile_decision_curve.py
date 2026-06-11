"""cProfile harness for decision-curve analysis (charts/decision_curve.py).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_decision_curve``

``compute_net_benefit`` derives every TP(pt)/FP(pt) for all <=200 threshold
probabilities from ONE descending score sort plus a single cumulative-sum pass,
then a vectorised ``searchsorted`` maps each pt onto the sweep. The lone
``np.argsort`` over the n scores is the only super-linear step; everything else
is O(n) cumsum + O(P log n) searchsorted. At n=1e6 the sort is the irreducible
floor (a rank-threshold sweep needs a sorted score order), so there is no
actionable speedup beyond it -- the harness records that the sort dominates and
the spec build stays well under a second.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.decision_curve import build_decision_curve_spec, compute_net_benefit


def _make_data(n: int):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    raw = rng.standard_normal(n) + 1.5 * y
    score = 1.0 / (1.0 + np.exp(-raw))
    return y, score


def main(n: int = 1_000_000):
    y, score = _make_data(n)
    # warmup
    compute_net_benefit(y, score)

    t0 = time.perf_counter()
    for _ in range(5):
        build_decision_curve_spec(y, score)
    wall = (time.perf_counter() - t0) / 5.0
    print(f"decision-curve spec build @ n={n}: {wall*1000:.1f} ms/call (best-of-5 mean)")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        build_decision_curve_spec(y, score)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()

"""Wide-frame COST bench for the pairwise/n-way modular cheap scan -- gates the budget guard default.

Measures wall-time of ``cheap_modular_scan`` as a function of the number of integer-eligible columns p,
pairs-only (``max_triples=0``) vs pairs+triples, at n=2000. The cheap scan is O(p) self + O(C(p,2)) pairs
(x3 ops) + O(C(p,3)) triples, so the triple term is the wide-frame blow-up; this bench quantifies where
it stops being free relative to a typical MRMR fit so we can pick ``fe_pairwise_modular_max_int_cols``.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_pairwise_modular_cost``
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._pairwise_modular_fe import cheap_modular_scan

P_GRID = (10, 20, 30, 50, 75, 100)
N = 2000
REPEATS = 3
RESULTS_DIR = Path(__file__).parent / "_results"

# Match the prod orchestrator's combiner budgets so the bench reflects shipped cost, not the unbounded sweep.
PAIR_BUDGET = 24
TRIPLE_BUDGET = 12


def _frame(p: int, n: int = N, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = {f"c{i}": rng.integers(0, 100, n) for i in range(p)}
    y = rng.integers(0, 2, n)
    return pd.DataFrame(cols), y


def _time_scan(X, y, *, max_triples: int) -> float:
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cheap_modular_scan(X, y, max_pairs=PAIR_BUDGET, max_triples=max_triples, seed=0)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    # Warm the binned-MI / numba kernels so the first p doesn't carry JIT cost.
    Xw, yw = _frame(8)
    cheap_modular_scan(Xw, yw, max_pairs=PAIR_BUDGET, max_triples=TRIPLE_BUDGET, seed=0)

    rows = []
    for p in P_GRID:
        X, y = _frame(p)
        pairs_only = _time_scan(X, y, max_triples=0)
        pairs_triples = _time_scan(X, y, max_triples=TRIPLE_BUDGET)
        rows.append({
            "p": p, "n": N,
            "pairs_only_s": round(pairs_only, 4),
            "pairs_triples_s": round(pairs_triples, 4),
            "triple_overhead_s": round(pairs_triples - pairs_only, 4),
        })
        print(f"  p={p:3d}  pairs_only={pairs_only:.4f}s  pairs+triples={pairs_triples:.4f}s  "
              f"triple_overhead={pairs_triples - pairs_only:.4f}s")

    results = {
        "n": N, "repeats": REPEATS, "pair_budget": PAIR_BUDGET, "triple_budget": TRIPLE_BUDGET,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"pairwise_modular_cost_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()

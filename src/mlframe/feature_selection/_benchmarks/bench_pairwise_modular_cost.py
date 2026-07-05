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


BA_GRID = ((15, 2000), (30, 2000), (15, 20000), (30, 20000))


def _before_after() -> list[dict]:
    """Before/after of the batched-grid + early-reject-null scan vs the git-HEAD reference scan.

    The optimized scan (1) batches the residue grid into one multi-column MI call per effective-nbins group and (2) skips
    the 12-permutation null for combiners that cannot clear the baseline margin -- both bit-identical to the reference
    responded-set. This section records the measured speedup; numbers feed the regression test floor."""
    import subprocess
    import sys
    import types

    ref_src = subprocess.run(
        ["git", "show", "HEAD:src/mlframe/feature_selection/filters/_pairwise_modular_fe.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[3]),
    ).stdout
    if not ref_src.strip():
        print("  (before/after skipped: could not load HEAD reference scan)")
        return []
    ref = types.ModuleType("_ref_pwm_bench")
    ref.__package__ = "mlframe.feature_selection.filters"
    ref.__name__ = "mlframe.feature_selection.filters._pairwise_modular_fe"
    exec(compile(ref_src, "ref_pairwise_modular_fe.py", "exec"), ref.__dict__)

    def frame_sig(p, n, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 100, n); b = rng.integers(0, 100, n)
        y = ((a + b) % 7 >= 3).astype(int)
        cols = {"a": a, "b": b}
        for i in range(p - 2):
            cols[f"c{i}"] = rng.integers(0, 100, n)
        return pd.DataFrame(cols), y

    Xw, yw = frame_sig(5, 2000)
    ref.cheap_modular_scan(Xw, yw); cheap_modular_scan(Xw, yw)
    rows = []
    for p, n in BA_GRID:
        X, y = frame_sig(p, n)
        rt = min(_one(ref.cheap_modular_scan, X, y) for _ in range(REPEATS))
        nt = min(_one(cheap_modular_scan, X, y) for _ in range(REPEATS))
        rows.append({"p": p, "n": n, "ref_s": round(rt, 4), "new_s": round(nt, 4), "speedup": round(rt / nt, 3), "saved_s": round(rt - nt, 4)})
        print(f"  before/after p={p:3d} n={n:6d}  ref={rt:.4f}s  new={nt:.4f}s  speedup={rt / nt:.2f}x  saved={rt - nt:+.4f}s")
    return rows


def _one(fn, X, y) -> float:
    t0 = time.perf_counter()
    fn(X, y)
    return time.perf_counter() - t0


def main():
    # Warm the binned-MI / numba kernels so the first p doesn't carry JIT cost.
    Xw, yw = _frame(8)
    cheap_modular_scan(Xw, yw, max_pairs=PAIR_BUDGET, max_triples=TRIPLE_BUDGET, seed=0)

    print("== before/after: batched-grid + early-reject-null vs HEAD reference ==")
    before_after = _before_after()

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
        "rows": rows, "before_after": before_after,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"pairwise_modular_cost_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()

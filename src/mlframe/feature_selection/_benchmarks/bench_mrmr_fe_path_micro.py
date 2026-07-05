"""Isolated micro-benchmarks for the MRMR FE / validation-path perf levers (E1, E2, E6, E7, E8).

Each bench compares the OLD inline pattern against the NEW one on a realistic shape, warm + best-of-N,
and asserts the result is bit-identical (where the lever is a pure speedup). Run:

    python -m mlframe.feature_selection._benchmarks.bench_mrmr_fe_path_micro

Levers measured here (the surrounding MRMR fit is dominated by MI kernels, so these are isolated to
expose the per-pattern delta the full-fit wall would otherwise hide):

* E2  -- per-pair ``np.append(data, new_vals, axis=1)`` loop vs a single trailing ``np.concatenate``.
* E7  -- ``list(combinations(ids, 2))`` + two ``np.fromiter`` vs ``np.triu_indices`` over the id array.
* E8  -- ``apply(pd.to_numeric).fillna(0.0).to_numpy(float)`` vs ``np.asarray(float)`` + in-place NaN->0.
* E1/E6 -- copy-count check: removing a redundant full-frame ``.copy()`` is bit-identical by construction;
  we just time one extra ``.copy()`` on a representative frame to size the saving.

These are LEADS for the full-fit wall, not standalone verdicts -- the selection-identity regression tests in
``tests/feature_selection/mrmr/test_fe_path_perf_identity.py`` are the binding correctness gate.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd


def _best_of(fn, n_iter: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_e2_append_vs_concat(n: int = 50_000, k0: int = 220, n_pairs: int = 40, ncols_per: int = 1):
    """E2: appending ``n_pairs`` code blocks one-at-a-time (np.append) vs accumulate + one concatenate."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 8, size=(n, k0), dtype=np.int8)
    chunks = [rng.integers(0, 8, size=(n, ncols_per), dtype=np.int8) for _ in range(n_pairs)]

    def old():
        data = base
        for ch in chunks:
            data = np.append(data, ch, axis=1)
        return data

    def new():
        data = base
        acc = []
        for ch in chunks:
            acc.append(ch)
        if acc:
            data = np.concatenate([data, *acc], axis=1)
        return data

    assert np.array_equal(old(), new()), "E2: result mismatch"
    t_old = _best_of(old, 20)
    t_new = _best_of(new, 20)
    return {"lever": "E2_append_vs_concat", "n": n, "k0": k0, "n_pairs": n_pairs, "old_ms": t_old * 1e3, "new_ms": t_new * 1e3, "speedup": t_old / t_new}


def bench_e7_combinations_vs_triu(p: int = 1500):
    """E7: tuple-list + fromiter vs triu_indices over the id array (exhaustive-branch shape)."""
    ids_set = set(int(x) for x in np.random.default_rng(1).permutation(p)[:p])

    def old():
        pairs = list(combinations(ids_set, 2))
        a = np.fromiter((q[0] for q in pairs), dtype=np.int64, count=len(pairs))
        b = np.fromiter((q[1] for q in pairs), dtype=np.int64, count=len(pairs))
        return a, b, pairs

    def new():
        ids = list(ids_set)
        ids_arr = np.fromiter(ids, dtype=np.int64, count=len(ids))
        ia, ib = np.triu_indices(len(ids), k=1)
        return ids_arr[ia], ids_arr[ib], ids, ia, ib

    a_o, b_o, pairs_o = old()
    a_n, b_n, ids_n, ia_n, ib_n = new()
    assert np.array_equal(a_o, a_n) and np.array_equal(b_o, b_n), "E7: id-array mismatch"
    pairs_n = [(ids_n[ia_n[i]], ids_n[ib_n[i]]) for i in range(ia_n.shape[0])]
    assert pairs_o == pairs_n, "E7: pair-sequence mismatch"
    t_old = _best_of(lambda: old(), 5)
    t_new = _best_of(lambda: new(), 5)
    n_pairs = p * (p - 1) // 2
    return {"lever": "E7_combinations_vs_triu", "p": p, "n_pairs": n_pairs, "old_ms": t_old * 1e3, "new_ms": t_new * 1e3, "speedup": t_old / t_new}


def bench_e8_tonumeric_vs_asarray(n: int = 20_000, p: int = 120, nan_frac: float = 0.02):
    """E8: apply(to_numeric).fillna.to_numpy vs asarray(float) + in-place NaN->0 (all-numeric frame)."""
    rng = np.random.default_rng(2)
    M = rng.standard_normal((n, p))
    mask = rng.random((n, p)) < nan_frac
    M[mask] = np.nan
    df = pd.DataFrame(M, columns=[f"c{i}" for i in range(p)])

    def old():
        return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    def new():
        a = np.asarray(df, dtype=float)
        if np.isnan(a).any():
            a[np.isnan(a)] = 0.0
        return a

    assert np.array_equal(old(), new()), "E8: result mismatch"
    t_old = _best_of(old, 10)
    t_new = _best_of(new, 10)
    return {"lever": "E8_tonumeric_vs_asarray", "n": n, "p": p, "old_ms": t_old * 1e3, "new_ms": t_new * 1e3, "speedup": t_old / t_new}


def bench_e1e6_copy_cost(n: int = 50_000, p: int = 220):
    """E1/E6: cost of one redundant full-frame ``.copy()`` removed (sizes the per-block saving)."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"c{i}" for i in range(p)])
    t_copy = _best_of(lambda: df.copy(), 20)
    return {"lever": "E1E6_redundant_copy", "n": n, "p": p, "copy_ms": t_copy * 1e3}


def main() -> int:
    results = [
        bench_e2_append_vs_concat(),
        bench_e7_combinations_vs_triu(),
        bench_e8_tonumeric_vs_asarray(),
        bench_e1e6_copy_cost(),
    ]
    for r in results:
        if "speedup" in r:
            print(f"{r['lever']:32s} old={r['old_ms']:9.3f}ms new={r['new_ms']:9.3f}ms "
                  f"speedup={r['speedup']:.2f}x  ({ {k: v for k, v in r.items() if k not in ('lever','old_ms','new_ms','speedup')} })")
        else:
            print(f"{r['lever']:32s} { {k: v for k, v in r.items() if k != 'lever'} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def write_results(path: str | None = None) -> str:
    """Run all benches and persist a JSON results table under ``_results/`` (methodology: committed numbers)."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    results = [
        bench_e2_append_vs_concat(),
        bench_e7_combinations_vs_triu(),
        bench_e8_tonumeric_vs_asarray(),
        bench_e1e6_copy_cost(),
    ]
    out = {"capture_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results}
    p = Path(path) if path else Path(__file__).parent / "_results" / "mrmr_fe_path_micro.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return str(p)

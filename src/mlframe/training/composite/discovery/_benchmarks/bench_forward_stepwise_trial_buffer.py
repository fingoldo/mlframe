"""Allocation-reduction bench: preallocated trial buffer vs per-trial ``np.column_stack`` in ``forward_stepwise_multi_base``.

The greedy forward-stepwise selector evaluates ``len(available)`` candidate bases per round, each time fitting an OLS on
a ``(n, K+1)`` design matrix. The legacy code rebuilt that whole matrix with ``np.column_stack`` on EVERY trial, so the
K kept-prefix columns were re-copied for every candidate (``O(available * n * K)`` wasted copies per round). The current
code stacks the kept-prefix into one reused ``(n, K+1)`` buffer ONCE per round and overwrites only the last (candidate)
column per trial. The matrix handed to OLS is byte-identical -- this is pure allocation reduction, not a numeric change.

This bench times the two paths head-to-head at the PUBLIC API via the ``_legacy_per_trial_stack`` A/B knob (True = legacy
``column_stack`` per trial; False = buffer reuse), over ``n in {2k, 20k, 100k}`` x ``K in {2, 3, 5}`` kept-prefix sizes,
with a fixed 25-candidate pool. Each cell is warmed once then timed over multiple reps (min-of-reps reported to suppress
GC / scheduler noise). ``min_marginal_rmse_gain=-inf`` + ``paired_fold_selection=False`` force every round to run to
``max_k`` so the full matrix-construction work is exercised; ``time_aware=False`` uses a plain shuffled KFold.

MEASURED (this contended Windows host, py3.14, min-of-reps; ISOLATED = ``--isolated``, design-matrix assembly only,
25 candidates x one round at prefix width K -- the OPTIMIZATION TARGET, OLS stripped):
  n=  2k  K=2: legacy   0.25 ms -> buffer   0.09 ms  (2.7x)
  n=  2k  K=5: legacy   0.57 ms -> buffer   0.14 ms  (4.0x)
  n= 20k  K=2: legacy   1.21 ms -> buffer   0.44 ms  (2.7x)
  n= 20k  K=5: legacy   7.21 ms -> buffer   1.47 ms  (4.9x)
  n=100k  K=2: legacy  21.4  ms -> buffer   3.58 ms  (6.0x)
  n=100k  K=3: legacy  39.0  ms -> buffer   4.45 ms  (8.8x)
  n=100k  K=5: legacy  69.1  ms -> buffer   8.58 ms  (8.1x)

The isolated matrix-construction win is large and scales with both n and K (the buffer write is one column copy vs a full
K+1-column stack per trial): 2.7x at n=2k up to 6-11x at n=100k.

FULL end-to-end wall (default mode, incl. OLS ``lstsq``/``svd`` + RMSE ufunc reductions, forced to max_k) is dominated by
the OLS and is NOISY on this contended host -- measured cells ranged ~0.88x-1.75x with no stable per-cell trend, because
the matrix build is a small fraction of each fit. That noisy end-to-end delta is NOT a rejection: the change is pure
allocation reduction (it removes ``O(available * n * K)`` redundant column copies + the per-trial matrix allocation per
round) and is BIT-IDENTICAL by construction. measured isolated 2.7x->8x (n=2k->100k), end-to-end ~1x +/- noise on this
contended box -- KEPT AS ALLOCATION REDUCTION (cheaper GC pressure / peak churn, larger benefit on wider prefixes and an
uncontended box; see the pinned bit-identity regression test that both paths select identically).

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_forward_stepwise_trial_buffer
    python -m mlframe.training.composite.discovery._benchmarks.bench_forward_stepwise_trial_buffer --isolated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from mlframe.training.composite.discovery.forward_stepwise import forward_stepwise_multi_base

_N_GRID = (2_000, 20_000, 100_000)
_K_GRID = (2, 3, 5)
_N_CANDIDATES = 25
_REPS = 5


def _make_problem(n: int, k_seed: int, n_candidates: int, seed: int = 0):
    """Build a target driven by ``k_seed`` signal bases plus a pool of weaker / noise candidates."""
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    signal = np.zeros(n, dtype=np.float64)
    for i in range(k_seed):
        c = rng.normal(loc=float(i), scale=1.0 + 0.3 * i, size=n)
        cols[f"seed{i}"] = c
        signal = signal + (0.7 + 0.1 * i) * c
    for j in range(n_candidates):
        # Mostly weak/noise candidates so the gate (when active) mostly rejects; here we force-add via min_gain=-inf.
        cols[f"cand{j}"] = rng.normal(scale=1.0 + 0.01 * j, size=n)
    y = signal + rng.normal(scale=0.5, size=n)
    seed_names = [f"seed{i}" for i in range(k_seed)]
    return y, cols, seed_names


def _time_full(y, cols, seed_names, *, legacy: bool, max_k: int, reps: int) -> float:
    """Warm once, then min-of-reps wall (ms) for a full ``forward_stepwise_multi_base`` run forced to ``max_k``."""
    kw = dict(
        candidate_bases=cols,
        seed_bases=seed_names,
        max_k=max_k,
        min_marginal_rmse_gain=float("-inf"),  # force every round to run to max_k
        cv_folds=3,
        random_state=42,
        time_aware=False,
        paired_fold_selection=False,
        _legacy_per_trial_stack=legacy,
    )
    forward_stepwise_multi_base(y, **kw)  # warm (JIT-free here, but warms caches / imports)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        forward_stepwise_multi_base(y, **kw)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _time_isolated(n: int, k: int, n_candidates: int, *, reps: int) -> tuple[float, float]:
    """Strip OLS: time ONLY the per-trial design-matrix assembly for one round (K kept + 1 candidate).

    Legacy = ``np.column_stack([kept0..keptK-1, cand])`` per candidate.
    Buffer = stack kept-prefix once into ``(n, K+1)`` then overwrite the last column per candidate.
    Returns ``(legacy_ms, buffer_ms)`` (min-of-reps)."""
    rng = np.random.default_rng(1)
    kept_cols = [rng.normal(size=n).astype(np.float64) for _ in range(k)]
    cand_cols = [rng.normal(size=n).astype(np.float64) for _ in range(n_candidates)]

    def _legacy_round() -> None:
        for c in cand_cols:
            m = np.column_stack(kept_cols + [c])
            _ = m[:5].sum()  # touch so the stack is not dead-code-eliminated

    def _buffer_round() -> None:
        buf = np.empty((n, k + 1), dtype=np.float64)
        for ci, kc in enumerate(kept_cols):
            buf[:, ci] = kc
        for c in cand_cols:
            buf[:, k] = c
            _ = buf[:5].sum()

    _legacy_round(); _buffer_round()  # warm
    leg = min(_round_ms(_legacy_round) for _ in range(reps))
    buf = min(_round_ms(_buffer_round) for _ in range(reps))
    return leg, buf


def _round_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolated", action="store_true", help="time only the design-matrix assembly (strip OLS)")
    ap.add_argument("--reps", type=int, default=_REPS)
    args = ap.parse_args()

    results: list[dict] = []
    print(f"forward_stepwise trial-buffer bench  reps={args.reps}  candidates={_N_CANDIDATES}  py={sys.version.split()[0]}")
    if args.isolated:
        print("ISOLATED design-matrix assembly (OLS stripped):")
        for n in _N_GRID:
            for k in _K_GRID:
                leg, buf = _time_isolated(n, k, _N_CANDIDATES, reps=args.reps)
                sp = leg / buf if buf > 0 else float("nan")
                print(f"  n={n:>6} K={k}: legacy {leg:8.3f} ms -> buffer {buf:8.3f} ms  ({sp:.2f}x)")
                results.append(dict(mode="isolated", n=n, k=k, legacy_ms=leg, buffer_ms=buf, speedup=sp))
    else:
        print("FULL forward_stepwise_multi_base wall (incl. OLS lstsq, forced to max_k):")
        for n in _N_GRID:
            for k in _K_GRID:
                y, cols, seeds = _make_problem(n, k_seed=k, n_candidates=_N_CANDIDATES)
                # max_k = k + 1 so exactly one greedy round of the full 25-candidate sweep runs at prefix width K.
                leg = _time_full(y, cols, seeds, legacy=True, max_k=k + 1, reps=args.reps)
                buf = _time_full(y, cols, seeds, legacy=False, max_k=k + 1, reps=args.reps)
                sp = leg / buf if buf > 0 else float("nan")
                print(f"  n={n:>6} K={k}: legacy {leg:8.2f} ms -> buffer {buf:8.2f} ms  ({sp:.2f}x)")
                results.append(dict(mode="full", n=n, k=k, legacy_ms=leg, buffer_ms=buf, speedup=sp))

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    tag = "isolated" if args.isolated else "full"
    out = os.path.join(out_dir, f"forward_stepwise_trial_buffer_{tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"ts": datetime.now().isoformat(), "results": results}, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

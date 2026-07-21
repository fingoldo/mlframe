"""Bench: anchor local-slope O(n*K) full-refit-per-row vs O(A*K) cached-at-anchor.

``_anchor_features_for_segment``'s local-slope
OLS fit over the last K anchors was recomputed from scratch on EVERY row of the segment, even
though the rolling anchor window (and hence the slope) is only actually updated when a NEW
anchor arrives -- O(n*K) work where O(A*K) (A = anchor count) suffices. The fix caches the
slope and only recomputes it in the anchor branch.

OLD baseline is loaded from `git show HEAD:<anchor.py>` into a temp module so we A/B the REAL
prior code, never a from-memory rewrite.

Run:
    CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_engineering/_benchmarks/bench_anchor_slope_cache.py
"""
from __future__ import annotations

import subprocess  # nosec B404 - subprocess used below with fixed list args, no shell=True
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ANCHOR_REL = "src/mlframe/feature_engineering/anchor.py"


def _load_baseline_fn():
    """Materialise HEAD:anchor.py's _anchor_features_for_segment as a standalone callable (real prior code, not a from-memory rewrite)."""
    src = subprocess.check_output(["git", "show", f"HEAD:{ANCHOR_REL}"], cwd=REPO, text=True)  # nosec B603, B607 - fixed/trusted executable (git) with list args, no untrusted input, resolved via PATH intentionally
    start = src.index("def _anchor_features_for_segment")
    end = src.index("def add_anchor_extrapolation_features")
    fn_src = src[start:end]
    ns = {"np": np}
    exec(compile(fn_src, "anchor_baseline_fn", "exec"), ns)  # nosec B102 - trusted source (own git history), not user input
    return ns["_anchor_features_for_segment"]


def _make_data(n, anchor_frac, seed=0):
    rng = np.random.default_rng(seed)
    is_anchor = rng.random(n) < anchor_frac
    is_anchor[0] = True
    label = np.where(is_anchor, rng.standard_normal(n) * 10.0, np.nan)
    return label.astype(np.float64), is_anchor


def _best_of(fn, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    from mlframe.feature_engineering.anchor import _anchor_features_for_segment as new_fn

    old_fn = _load_baseline_fn()
    K = 5

    for n, anchor_frac in ((20_000, 0.02), (50_000, 0.02), (50_000, 0.15)):
        label, is_anchor = _make_data(n, anchor_frac)

        t_old, r_old = _best_of(lambda: old_fn(label, is_anchor, K))
        t_new, r_new = _best_of(lambda: new_fn(label, is_anchor, K))

        for key in r_old:
            assert np.allclose(r_old[key], r_new[key], equal_nan=True), f"mismatch on {key}"  # nosec B101 - internal invariant check in src/mlframe/feature_engineering/_benchmarks, not reachable with untrusted input

        print(f"n={n:>7}  anchor_frac={anchor_frac:<5} OLD={t_old*1e3:8.2f}ms  NEW={t_new*1e3:8.2f}ms  speedup={t_old/t_new:6.2f}x")


if __name__ == "__main__":
    main()

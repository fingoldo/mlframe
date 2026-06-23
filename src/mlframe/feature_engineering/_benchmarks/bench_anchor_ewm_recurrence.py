"""Bench CPX2: anchor EWMA O(A^2)-per-segment vs O(1)-per-step recurrence.

The prior `_anchor_ewm_core` recomputed, at EVERY row, decayed sums over ALL
anchors seen so far -> O(A) per row -> O(A^2) per segment. The new core keeps
running accumulators (S0, Sy, Su, Suy, Suu) in last-row-centred coordinates,
updated in O(1) per step.

OLD baseline is loaded from `git show HEAD:<anchor.py>` into a temp module so we
A/B the REAL prior code, never a from-memory rewrite.

Run:
    CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_engineering/_benchmarks/bench_anchor_ewm_recurrence.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ANCHOR_REL = "src/mlframe/feature_engineering/anchor.py"


def _load_baseline_module():
    """Materialise HEAD:anchor.py as an importable module (real prior code)."""
    src = subprocess.check_output(
        ["git", "show", f"HEAD:{ANCHOR_REL}"], cwd=REPO, text=True
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="anchor_baseline_"))
    # The module does `from .grouped import iter_group_segments`; rewrite to absolute.
    src = src.replace("from .grouped import", "from mlframe.feature_engineering.grouped import")
    p = tmpdir / "anchor_baseline.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("anchor_baseline", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_data(n, anchor_frac=0.15, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    is_anchor = rng.random(n) < anchor_frac
    label = np.where(is_anchor, np.cumsum(rng.standard_normal(n)) * 0.01, np.nan)
    group_ids = rng.integers(0, n_groups, size=n)
    group_ids.sort()  # contiguous-ish groups
    return label.astype(np.float64), is_anchor, group_ids


def _best_of(fn, reps=5):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def main():
    import mlframe.feature_engineering.anchor as new_mod

    base = _load_baseline_module()
    half_life = 30.0

    for n in (20_000, 100_000):
        label, is_anchor, gids = _make_data(n)
        # warm numba JIT on both
        base.anchor_ewm_features(label[:2000], is_anchor[:2000], gids[:2000], half_life_rows=half_life)
        new_mod.anchor_ewm_features(label[:2000], is_anchor[:2000], gids[:2000], half_life_rows=half_life)

        t_old, r_old = _best_of(lambda: base.anchor_ewm_features(label, is_anchor, gids, half_life_rows=half_life))
        t_new, r_new = _best_of(lambda: new_mod.anchor_ewm_features(label, is_anchor, gids, half_life_rows=half_life))

        # identity
        max_abs = 0.0
        for kk in r_old:
            a, b = r_old[kk], r_new[kk]
            m = np.isfinite(a) | np.isfinite(b)
            d = np.nanmax(np.abs(a[m] - b[m])) if m.any() else 0.0
            max_abs = max(max_abs, 0.0 if np.isnan(d) else d)
            # NaN positions must match exactly
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"NaN mask mismatch {kk}"

        print(f"n={n:>7}  OLD={t_old*1e3:8.2f}ms  NEW={t_new*1e3:8.2f}ms  "
              f"speedup={t_old/t_new:5.2f}x  max_abs_diff={max_abs:.2e}")


if __name__ == "__main__":
    main()

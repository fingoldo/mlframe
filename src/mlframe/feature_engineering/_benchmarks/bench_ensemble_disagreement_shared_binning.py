"""Bench: share the per-row bin-histogram between entropy + top2_mode_gap in predictor_disagreement_features.

OLD: predictor_consensus_entropy and predictor_top2_mode_gap each independently recomputed the IDENTICAL
equal-width binning + njit scatter-histogram (lo/hi/span/clip/_row_bin_histogram_njit) -- the heavy part of
both. NEW: the builder computes the counts ONCE via _bin_counts and feeds both _entropy_from_counts /
_top2_gap_from_counts. Bit-identical by construction (same binning math, evaluated once).

Run:  python bench_ensemble_disagreement_shared_binning.py
The OLD side is loaded from git (HEAD baseline) into a temp module so the A/B compares two real artifacts.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess  # nosec B404 - subprocess used below with fixed list args, no shell=True
import sys
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mlframe.feature_engineering import ensemble_features as ef_new


def _load_old_module() -> types.ModuleType:
    """Load the HEAD version of ensemble_features.py as a standalone module for the A/B baseline.

    Written to a real temp .py file because numba ``cache=True`` needs a file locator (it cannot
    cache a function compiled from a synthetic ``exec`` source string).
    """
    rel = "src/mlframe/feature_engineering/ensemble_features.py"
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    old_src = subprocess.check_output(["git", "-C", repo, "show", f"HEAD:{rel}"], text=True)  # nosec B603, B607 - fixed/trusted executable (git) with list args, no untrusted input, resolved via PATH intentionally
    fd, path = tempfile.mkstemp(suffix="_ensemble_features_old.py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(old_src)
    spec = importlib.util.spec_from_file_location("ensemble_features_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ef_old = _load_old_module()
    rng = np.random.default_rng(0)
    print(f"{'shape':>16} {'OLD ms':>10} {'NEW ms':>10} {'speedup':>8}  identity")
    for n, k in [(100_000, 8), (1_000_000, 8), (100_000, 20)]:
        preds = rng.standard_normal((n, k)).astype(np.float64)
        # warm both
        ef_old.predictor_disagreement_features(preds, emit_pairs=False)
        ef_new.predictor_disagreement_features(preds, emit_pairs=False)

        def best(fn) -> float:
            b = 1e18
            for _ in range(7):
                t = time.perf_counter()
                fn(preds, emit_pairs=False)
                b = min(b, time.perf_counter() - t)
            return b

        t_old = best(ef_old.predictor_disagreement_features)
        t_new = best(ef_new.predictor_disagreement_features)

        o = ef_old.predictor_disagreement_features(preds, emit_pairs=False)
        nw = ef_new.predictor_disagreement_features(preds, emit_pairs=False)
        ident = all(np.array_equal(o[key], nw[key]) for key in o)
        print(f"{f'n={n} k={k}':>16} {t_old*1000:>10.2f} {t_new*1000:>10.2f} {t_old/t_new:>8.2f}x  {'BIT-IDENTICAL' if ident else 'DIVERGENT!!'}")


if __name__ == "__main__":
    main()

"""Bench: MRMR.transform with the shared per-call recipe caches vs caches forced off, pandas and polars input.

Fits once per n on a hub-operand fixture, then times transform (warm, best-of-N) and traces its peak allocation both ways. Values are
checked equal before anything is reported.

Run: PYTHONPATH=src python -m mlframe.feature_selection.filters._benchmarks.bench_transform_recipe_caches
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import engineered_recipes
from mlframe.feature_selection.filters.mrmr import MRMR


def _hub_frame(n: int, seed: int = 0):
    """A target driven by several nonlinear functions of one hub column plus interactions with it."""
    rng = np.random.default_rng(seed)
    hub = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    y = ((hub**2 + np.sin(2 * hub) + hub * b + 0.5 * hub * c + 0.3 * rng.normal(size=n)) > 1.0).astype(np.int64)
    return pd.DataFrame({"hub": hub, "b": b, "c": c, "noise": rng.normal(size=n)}), y


def _best(fn, reps):
    """Best wall time of ``reps`` calls."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _peak(fn):
    """Peak traced bytes for one call."""
    tracemalloc.start()
    try:
        fn()
        _, pk = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pk


def main() -> None:
    """Print cached vs uncached transform time and peak for each input kind and n."""
    real = engineered_recipes.apply_recipe

    def uncached(recipe, frame, col_cache=None, basis_cache=None):
        """apply_recipe with the caches dropped."""
        return real(recipe, frame)

    print(f"{'input':>7} {'n':>9} {'recipes':>8} {'cached_ms':>10} {'uncached_ms':>12} {'speedup':>8} {'peak_cached_MB':>15} {'peak_uncached_MB':>17}")
    for n in (20_000, 200_000, 1_000_000):
        X, y = _hub_frame(n)
        MRMR._FIT_CACHE.clear()
        m = MRMR(random_seed=0, n_jobs=1, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2).fit(X.iloc[: min(n, 50_000)], y[: min(n, 50_000)])
        n_rec = len(getattr(m, "_engineered_recipes_", []) or [])
        inputs = [("pandas", X)]
        try:
            import polars as pl

            inputs.append(("polars", pl.from_pandas(X)))
        except ImportError:
            pass
        for kind, frame in inputs:
            m.transform(frame)  # warm
            out_c = m.transform(frame)
            engineered_recipes.apply_recipe = uncached
            try:
                out_u = m.transform(frame)
                t_u = _best(lambda: m.transform(frame), 5)
                p_u = _peak(lambda: m.transform(frame))
            finally:
                engineered_recipes.apply_recipe = real
            a_c = out_c.to_numpy() if hasattr(out_c, "to_numpy") else np.asarray(out_c)
            a_u = out_u.to_numpy() if hasattr(out_u, "to_numpy") else np.asarray(out_u)
            np.testing.assert_array_equal(a_c, a_u)
            t_c = _best(lambda: m.transform(frame), 5)
            p_c = _peak(lambda: m.transform(frame))
            print(f"{kind:>7} {n:>9} {n_rec:>8} {t_c * 1e3:>10.1f} {t_u * 1e3:>12.1f} {t_u / t_c:>8.2f} {p_c / 1e6:>15.1f} {p_u / 1e6:>17.1f}")


if __name__ == "__main__":
    main()

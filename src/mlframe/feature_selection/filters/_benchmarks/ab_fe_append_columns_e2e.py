"""End-to-end A/B for the pandas branch of ``_fe_frame_ops.fe_append_columns``: whole-frame ``concat`` vs shallow copy + assignment.

Runs ONE FE-heavy ``MRMR.fit`` in this process with the chosen strategy installed in every loaded module that imported the seam by name,
and prints wall time, peak process memory, and the selected feature names as JSON. Run each strategy in its own process (in-process state -
numba caches, warmed kernels, memo dicts - would otherwise contaminate the comparison), alternating, and compare.

Usage: ``python ab_fe_append_columns_e2e.py {concat|shallow} [--n 200000] [--p 30] [--seed 0]``
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd


def _shallow_append(X, cols):
    """Candidate seam body. Runs with ``_fe_frame_ops``'s globals once its code object is installed, so it may only use names defined there."""
    if not cols:
        return X
    if is_pandas(X):  # type: ignore[name-defined]  # noqa: F821 - resolved in _fe_frame_ops' globals after the code swap
        out = X.copy(deep=False)
        for name, vals in cols.items():
            out[name] = vals
        return out
    if is_polars(X):  # type: ignore[name-defined]  # noqa: F821
        return X.with_columns([pl.Series(name, np.asarray(vals)) for name, vals in cols.items()])  # type: ignore[name-defined]  # noqa: F821
    if isinstance(X, np.ndarray):
        extra = np.column_stack([np.asarray(v) for v in cols.values()])
        return np.hstack([X, extra])
    raise TypeError(f"fe_append_columns: unsupported frame type {type(X)!r}")


CALLS = {"n": 0}


def _install(strategy: str) -> None:
    """Swap the seam's CODE object, so every binding of the function - including ones imported lazily inside FE stages - runs the variant."""
    from mlframe.feature_selection.filters import _fe_frame_ops as ops

    if strategy == "shallow":
        ops.fe_append_columns.__code__ = _shallow_append.__code__
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    # Count real calls through the module attribute used by every lazy import, to prove the seam was exercised.
    original = ops.fe_append_columns

    def _counting(X, cols):
        """Count, then delegate to the (possibly swapped) seam."""
        CALLS["n"] += 1
        return original(X, cols)

    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "fe_append_columns", None) is original:
            mod.fe_append_columns = _counting  # type: ignore[attr-defined]


def _peak_rss_sampler(stop: threading.Event, box: dict) -> None:
    """Poll RSS every 50 ms and keep the maximum (portable; Windows peak_wset is read at the end as well)."""
    import psutil

    proc = psutil.Process()
    while not stop.is_set():
        box["peak"] = max(box.get("peak", 0), proc.memory_info().rss)
        stop.wait(0.05)


def main() -> None:
    """Run one fit with the requested strategy and print a JSON record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=["concat", "shallow"])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--p", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mlframe.feature_selection.filters.mrmr import MRMR

    # Import every FE stage module up front so each holds its by-name binding of the seam before the swap.
    import mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core  # noqa: F401

    _install(args.strategy)

    rng = np.random.default_rng(args.seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=args.n) for i in range(args.p)})
    X["cat"] = pd.Categorical(rng.integers(0, 8, size=args.n))
    y = X["x0"] * 1.2 + np.sin(2 * X["x1"]) + X["x2"] * X["x3"] * 0.8 + rng.normal(scale=0.3, size=args.n)

    import psutil

    box: dict = {}
    stop = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(stop, box), daemon=True)
    rss0 = psutil.Process().memory_info().rss
    sampler.start()
    t0 = time.perf_counter()
    m = MRMR(verbose=0, random_seed=args.seed).fit(X, y)
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join()
    mi = psutil.Process().memory_info()
    print(
        "AB_RESULT "
        + json.dumps(
            {
                "strategy": args.strategy,
                "seam_calls": CALLS["n"],
                "n": args.n,
                "p": args.p,
                "wall_s": round(wall, 2),
                "rss_start_mb": round(rss0 / 2**20, 1),
                "peak_rss_mb": round(max(box.get("peak", 0), getattr(mi, "peak_wset", 0)) / 2**20, 1),
                "selected": sorted(map(str, m.get_feature_names_out())),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

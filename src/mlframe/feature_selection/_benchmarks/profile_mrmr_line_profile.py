"""Line-level profiler for a named MRMR method, on synthetic data shaped like the wellbore-100k
production regime (99401 rows, ~500 mixed numeric/categorical columns).

Use this when cProfile shows a big ``tottime`` (self-time, excluding sub-calls) on a large
orchestration function with no single dominant callee -- line_profiler pinpoints the actual hot
line instead of guessing from a static read. It also disambiguates a genuine inline Python
hotspot from cProfile's blind spot for ``@njit``/GPU-async-kernel time getting folded into the
calling frame's tottime (see ``_fit_impl_core.py``'s own profiling notes, and CLAUDE.md's "GPU
profiling traps").

Run: ``python -m mlframe.feature_selection._benchmarks.profile_mrmr_line_profile [dotted.path.to.func]``
Defaults to ``mlframe.feature_selection.filters.mrmr.MRMR._fit_impl`` if no path is given.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
from line_profiler import LineProfiler


def _make_wellbore_like_frame(n_rows: int = 99_401, n_numeric: int = 400, n_categorical: int = 100, seed: int = 0):
    """Synthetic frame shaped like the wellbore-100k production regime: mostly-numeric columns plus
    a smaller block of low-cardinality categorical columns, a handful genuinely informative."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.1, 1.1, n_rows)
    b = rng.uniform(0.1, 1.1, n_rows)
    base = (a**2) / b
    cols = {}
    for k in range(n_numeric):
        cols[f"num_{k}"] = (base * (1.0 + 0.05 * rng.standard_normal(n_rows)) + 0.001 * k).astype(np.float32) if k < 5 else rng.standard_normal(n_rows).astype(np.float32)
    for k in range(n_categorical):
        cols[f"cat_{k}"] = rng.integers(0, rng.integers(2, 12), n_rows).astype(np.int32)
    X = pd.DataFrame(cols)
    y = pd.Series(base + 0.1 * rng.standard_normal(n_rows), name="y")
    return X, y


def _resolve_target(dotted: str):
    """Split ``pkg.mod.Class.method`` (or ``pkg.mod.func``) into (owner_object_or_class, attr_name)."""
    parts = dotted.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        mod_path = ".".join(parts[:split_at])
        try:
            obj: object = importlib.import_module(mod_path)
        except ImportError:
            continue
        for attr in parts[split_at:-1]:
            obj = getattr(obj, attr)
        return obj, parts[-1]
    raise ImportError(f"could not resolve module prefix of {dotted!r}")


def main() -> None:
    """CLI entrypoint: line-profile the requested target function against a synthetic wellbore-shaped fit."""
    target = sys.argv[1] if len(sys.argv) > 1 else "mlframe.feature_selection.filters.mrmr.MRMR._fit_impl"
    owner, attr_name = _resolve_target(target)
    orig_fn = getattr(owner, attr_name)

    lp = LineProfiler()
    lp.add_function(orig_fn)
    setattr(owner, attr_name, lp(orig_fn))

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _make_wellbore_like_frame()
    MRMR(verbose=0).fit(X, y)

    lp.print_stats()


if __name__ == "__main__":
    main()

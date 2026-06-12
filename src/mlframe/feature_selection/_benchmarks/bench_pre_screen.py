"""cProfile-driven speed investigation for ``mlframe.feature_selection.pre_screen``.

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python _benchmarks/bench_pre_screen.py

TARGET: ``compute_unsupervised_drops`` -- the unsupervised pre-screen (constant /
near-constant / high-null column drops). Train-only fit.

REPRESENTATIVE FIXTURE: wide frame n=5000, p=200 with constant cols, high-null cols,
a few exact-duplicate columns. Both polars and pandas backends profiled.

NOTE ON THE TASK PROMPT'S HYPOTHESIS (O(p^2) duplicate scan):
    The current pre_screen.py does NOT perform any duplicate-column detection. It is a
    single O(p) loop computing per-column null_count + var. There is no O(p^2) pairwise
    compare to collapse. The "hash-the-column-bytes" win does not apply -- nothing to fix
    there. This harness profiles what the function ACTUALLY does and reports the real
    mlframe-side hotspot ranking (see VERDICT in module docstring footer after running).

VERDICT (measured 2026-06-12, store py3.14, CPU):
    RESOLVED. Top mlframe-side hotspot in the PANDAS branch is the per-column null count
    ``col.isna().sum()`` -- it allocates a fresh boolean Series and dispatches through pandas
    nanops/sanitize_array per column (cProfile: 40600 Series.__init__ + 40600 _isna_array for
    203 cols x 200 runs). Replaced with a bit-identical fast path:
        - numpy float dtype -> ``np.isnan(col.to_numpy()).sum()`` (no Series alloc, ~6x: 61us->11us/col)
        - numpy int/bool dtype -> 0 (those dtypes cannot hold a null; no data touched)
        - every other dtype (object/None, datetime/NaT, nullable ext, category, sparse) -> exact
          fallback to ``col.isna().sum()`` (sparse keeps its dedicated pre-existing branch).
    var() is LEFT UNCHANGED: pandas ``col.var()`` (bottleneck nanvar) is already faster than
    ``np.nanvar`` AND differs from it at ~1e-15, which is not bit-identical near the 1e-24 cutoff.

    Drop-column SET is bit-identical on 2 fixed-seed fixtures (wide-with-exact-dups + mixed-dtype
    near-constant). Microbench (warm, multi-run, 2 sizes):
        n=5000  p=203:  19.3 ms -> 6.8 ms  (2.8x)
        n=20000 p=403:  66.2 ms -> 36.8 ms (1.8x; large-n is var-dominated, untouched -> smaller gain)
    Polars branch already 1.3 ms (var() in Rust); not a hotspot, untouched.
    NO full-frame copy introduced (per-column ``to_numpy()`` is a zero-copy view of the float buffer).

    NOTE: the task prompt's O(p^2) duplicate-column-scan hypothesis does NOT apply -- pre_screen.py
    performs no duplicate detection; it is a single O(p) per-column null+var loop. Nothing to collapse.
"""
from __future__ import annotations

import cProfile
import pstats
import io
import time

from typing import Any, Callable

import numpy as np

import polars as pl
import pandas as pd

from mlframe.feature_selection.pre_screen import compute_unsupervised_drops


def make_fixture(n: int = 5000, p: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Wide frame: most random-normal, plus constant / high-null / exact-dup columns."""
    rng = np.random.default_rng(seed)
    data = {}
    names = []
    for j in range(p):
        nm = f"f{j}"
        names.append(nm)
        if j % 25 == 0:
            # constant column
            data[nm] = np.full(n, float(j))
        elif j % 25 == 1:
            # high-null column (99.5% null) -- build as float with NaN
            col = np.full(n, np.nan)
            keep = rng.choice(n, size=max(1, n // 200), replace=False)
            col[keep] = rng.standard_normal(keep.size)
            data[nm] = col
        else:
            data[nm] = rng.standard_normal(n)
    # a few exact-duplicate columns (content equal to f2)
    for k in range(3):
        nm = f"dup{k}"
        data[nm] = data["f2"].copy()
    return data


def build_pandas(data: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(data)


def build_polars(data: dict[str, np.ndarray]) -> pl.DataFrame:
    return pl.DataFrame(data)


def microbench(fn: Callable, df: Any, n_runs: int = 30) -> tuple[float, Any]:
    # warm
    for _ in range(3):
        fn(df)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = fn(df)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1e3, out  # ms/call


def profile(fn: Callable, df: Any, n_runs: int = 200, label: str = "") -> None:
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_runs):
        fn(df)
    pr.disable()
    for sort in ("tottime", "cumulative"):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort)
        ps.print_stats(25)
        print(f"\n===== {label} sort={sort} (n_runs={n_runs}) =====")
        print(s.getvalue())


if __name__ == "__main__":
    data = make_fixture()
    pdf = build_pandas(data)
    pldf = build_polars(data)

    f = compute_unsupervised_drops
    for label, df in (("pandas", pdf), ("polars", pldf)):
        ms, out = microbench(f, df)
        print(f"[{label}] {ms:.3f} ms/call  -> {len(out)} drops: {sorted(out)[:6]}...")

    print("\n######## PROFILE: pandas ########")
    profile(f, pdf, n_runs=200, label="pandas")
    print("\n######## PROFILE: polars ########")
    profile(f, pldf, n_runs=200, label="polars")

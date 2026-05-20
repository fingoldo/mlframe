"""cProfile audit of ``optimise_hermite_pair`` -- find hotspots in the
multi-basis polynomial FE module.

Run::

    python -m mlframe.feature_selection._benchmarks.profile_hermite_fe

Outputs cProfile stats sorted by cumulative time. Calibrates against
cProfile attribution overhead (per ``feedback_profile_new_features``):
runs both with and without the profiler enabled and reports the
profiler-induced overhead so we know which numbers are real.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import warnings
from pathlib import Path

import numpy as np

# Wave 87 (2026-05-21): module-level filter removed; gated under __main__ below.


def _make_xor(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (np.sign(x1 * x2) > 0).astype(np.int64)
    return x1, x2, y


def _make_california_pair(n=2000, seed=42):
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y_cont = data.data, data.target
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    X = X[idx]
    y_cont = y_cont[idx]
    x1 = X[:, 0]  # MedInc
    x2 = X[:, 5]  # AveOccup
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    return x1, x2, y


def _run_basis(basis, x1, x2, y, n_trials=40, max_degree=4):
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair
    return optimise_hermite_pair(
        x1, x2, y,
        discrete_target=True,
        max_degree=max_degree,
        n_trials=n_trials,
        seed=42,
        basis=basis,
        baseline_uplift_threshold=0.0,  # always return result for profile
    )


def main():
    print("\n=== cProfile: optimise_hermite_pair, 4 bases x 2 scenarios ===\n")

    scenarios = {
        "xor (Gaussian, n=2000)": _make_xor(n=2000),
        "california (skewed, n=2000)": _make_california_pair(n=2000),
    }

    bases = ["hermite", "legendre", "chebyshev", "laguerre"]

    # Warmup -- numba JIT, sklearn validate caches, optuna init
    print("  Warmup pass...", flush=True)
    x1, x2, y = scenarios["xor (Gaussian, n=2000)"]
    _run_basis("hermite", x1, x2, y, n_trials=10, max_degree=3)

    # Calibration: time WITHOUT profiler enabled
    print("\n  Wall-time without profiler:")
    wall_no_profile = {}
    for sc_name, (x1, x2, y) in scenarios.items():
        t0 = time.perf_counter()
        for basis in bases:
            _run_basis(basis, x1, x2, y, n_trials=40, max_degree=4)
        wall_no_profile[sc_name] = time.perf_counter() - t0
        print(f"    {sc_name}: {wall_no_profile[sc_name]:.2f}s for 4 bases")

    # Now WITH profiler
    print("\n  Wall-time with profiler:")
    wall_with_profile = {}
    profilers = {}
    for sc_name, (x1, x2, y) in scenarios.items():
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        for basis in bases:
            _run_basis(basis, x1, x2, y, n_trials=40, max_degree=4)
        prof.disable()
        wall_with_profile[sc_name] = time.perf_counter() - t0
        profilers[sc_name] = prof
        overhead_pct = 100 * (wall_with_profile[sc_name] / wall_no_profile[sc_name] - 1)
        print(f"    {sc_name}: {wall_with_profile[sc_name]:.2f}s "
              f"(profiler overhead: {overhead_pct:+.1f}%)")

    # Print top hotspots per scenario
    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    for sc_name, prof in profilers.items():
        print(f"\n  --- Top hotspots: {sc_name} ---")
        s = io.StringIO()
        # Sort by cumulative time, top 30 functions, exclude C builtins.
        pstats.Stats(prof, stream=s).strip_dirs().sort_stats("cumulative").print_stats(30)
        text = s.getvalue()
        print(text)
        # Save full profile to file for offline analysis.
        out_file = out_dir / f"profile_hermite_fe_{sc_name.split()[0]}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# cProfile of optimise_hermite_pair, scenario: {sc_name}\n")
            f.write(f"# wall-time (no profiler): {wall_no_profile[sc_name]:.2f}s\n")
            f.write(f"# wall-time (with profiler): {wall_with_profile[sc_name]:.2f}s\n")
            f.write(f"# profiler overhead: {100 * (wall_with_profile[sc_name] / wall_no_profile[sc_name] - 1):+.1f}%\n\n")
            full_s = io.StringIO()
            pstats.Stats(prof, stream=full_s).strip_dirs().sort_stats("cumulative").print_stats(80)
            f.write(full_s.getvalue())
        print(f"  (full profile -> {out_file})")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

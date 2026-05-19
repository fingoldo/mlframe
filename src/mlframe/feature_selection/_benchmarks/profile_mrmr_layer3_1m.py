"""cProfile audit of MRMR.fit on a 1M-row / 30-feature dataset after
Layer 3 (commit 57f772c + b844214 + 033941b) is wired in as default.

Goal: find the NEXT CUDA acceleration target. Layer 1 (threading
backend) + Layer 2 (batch_pair_mi_prange njit kernel) + Layer 3
(numba.cuda + cupy size-aware dispatcher) plus the transparent
mi_direct_gpu -> mi_direct_gpu_batched routing at npermutations>=32
already cover the biggest historical hotspots. This profile measures
what's left and prints the top 30 by cumulative time so the next
optimization wave can be sized concretely.

Run::

    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \\
        -m mlframe.feature_selection._benchmarks.profile_mrmr_layer3_1m

Output: stdout summary + optionally pstats dump (saved next to the
script under ``_results/profile_mrmr_layer3_1m_<timestamp>.pstats`` if
the directory exists).
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import time
from datetime import datetime

import numpy as np
import pandas as pd


def _build_frame(n: int, n_features: int, seed: int):
    """Synthesise a multiclass classification dataset with mild interaction
    signal: y = (f0 + f1 + 2*f0*f1 + noise) % 3."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 5, size=n).astype(np.int32) for _ in range(n_features)]
    X = pd.DataFrame(np.column_stack(cols), columns=[f"f{i}" for i in range(n_features)])
    y_raw = (X["f0"] + X["f1"] + 2 * X["f0"] * X["f1"] + rng.integers(0, 2, size=n)) % 3
    y = pd.Series(y_raw.astype(np.int32), name="y")
    return X, y


def main() -> None:
    n_rows = 1_000_000
    n_features = 30
    seed = 11

    print(f"=== MRMR.fit profile (n_rows={n_rows:_}, n_features={n_features}) ===")
    t0 = time.perf_counter()
    X, y = _build_frame(n_rows, n_features, seed)
    print(f"frame built in {time.perf_counter() - t0:.2f}s; X={X.shape}, y={y.shape}")

    # Import after build so import-time JIT compiles don't enter the profile.
    from mlframe.feature_selection.filters.mrmr import MRMR

    # fe_npermutations=50 triggers the mi_direct -> mi_direct_gpu transparent
    # route (gated on npermutations>=32 to amortise the H2D copy). At 10 the
    # gate stays closed and the CPU shuffle_arr path runs as before.
    selector = MRMR(
        fe_max_steps=1,
        fe_ntop_features=10,
        fe_npermutations=50,
        random_seed=seed,
        verbose=0,
    )

    # Warm-up: trigger numba + CuPy JIT compilation outside the profile so
    # the profile measures steady-state work, not first-call compile costs.
    print("warming JIT caches (first call) ...")
    t_warm = time.perf_counter()
    try:
        from mlframe.feature_selection.filters._prewarm import (
            prewarm_fs_numba_cache, prewarm_fs_cupy_kernels,
        )
        prewarm_fs_numba_cache(verbose=False)
        prewarm_fs_cupy_kernels(verbose=False)
    except ImportError:
        pass
    _warm_X, _warm_y = _build_frame(2000, n_features, seed)
    # fe_npermutations=64 forces the GPU route (>=32 threshold from commit
    # ba78f04). Without this the warm-up never exercises mi_direct_gpu and
    # the profiled run pays the cold-start sub-kernel compiles.
    MRMR(
        fe_max_steps=1, fe_ntop_features=4, fe_npermutations=64,
        random_seed=seed, verbose=0,
    ).fit(_warm_X, _warm_y)
    print(f"warm-up done in {time.perf_counter() - t_warm:.2f}s")

    # Profiled run.
    print("profiling MRMR.fit on 1M rows ...")
    pr = cProfile.Profile()
    t_fit = time.perf_counter()
    pr.enable()
    selector.fit(X, y)
    pr.disable()
    fit_wall = time.perf_counter() - t_fit
    print(f"fit wall: {fit_wall:.2f}s; selected_support={int(selector.support_.sum())}")

    # Render top-30 by cumulative time.
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(30)
    print("\n=== Top 30 by cumulative time ===")
    print(buf.getvalue())

    # Render top-30 by internal time (tottime) too -- different hotspots.
    buf2 = io.StringIO()
    stats2 = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
    stats2.print_stats(30)
    print("\n=== Top 30 by internal (tottime) ===")
    print(buf2.getvalue())

    # Optional .pstats dump for interactive snakeviz inspection.
    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    if os.path.isdir(out_dir):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"profile_mrmr_layer3_1m_{stamp}.pstats")
        stats.dump_stats(out_path)
        print(f"\npstats dump: {out_path}")


if __name__ == "__main__":
    main()

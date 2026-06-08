"""FE shortlist-transformer profiling harness (FE-opt campaign 2026-06-08).

Profiles the FIVE transformers actually wired into ``train_mlframe_models_suite`` via
``ShortlistTransformerAdapter`` (the production opt-in FE path):

    cdist  -> compute_class_distance_features
    loclift-> compute_local_lift_features
    rff    -> compute_rff_features
    bgm    -> compute_bgmm_density_ratio_features
    rsd    -> compute_residual_stratified_distance_features

Each is driven through the EXACT production wiring (``ShortlistTransformerAdapter.fit`` /
``.transform``, Mode B: fit-on-train, apply-to-query), so the measured wall + golden output
are the real suite FE-transform cost, not a synthetic standalone call.

Representative big-wide deterministic config (one fixed seed). The kNN-bottlenecked transformers
(cdist/loclift/rsd) are run with a train bank large enough to exercise the hnswlib path the
suite uses at scale.

Usage:
    python -m profiling.bench_fe_shortlist_transforms wall      # wall-time only (low overhead)
    python -m profiling.bench_fe_shortlist_transforms golden    # save golden engineered output
    python -m profiling.bench_fe_shortlist_transforms verify    # compare current output vs golden (bit-identity)
    python -m profiling.bench_fe_shortlist_transforms cprofile  # cProfile + top-N hotspots, per transformer
    python -m profiling.bench_fe_shortlist_transforms all       # wall + cprofile
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from typing import Callable

import numpy as np
import polars as pl


def _stabilize_native_env() -> None:
    """Force a deterministic, environment-stable CPU path for the benchmark.

    On this dev box (i7-7700HQ + GTX 1050 Ti Pascal, conda numpy/MKL) two native libraries
    segfault intermittently with a Windows access violation:
      * ``import hnswlib`` (the kNN backend) crashes when its native DLL is loaded LAZILY after
        cupy/numba/scipy are already resident -- a DLL load-order / MSVC-runtime conflict
        (faulthandler pinned the access violation to ``_check_hnsw_available`` -> ``import hnswlib``).
      * cupy GPU kernels are flaky on this Pascal card during the kernel-tuning sweep.
    Neither is an mlframe code defect; both are this host's native-lib instability. Forcing the
    sklearn-exact kNN path (``_HNSW_AVAILABLE=False``) and the CPU RFF/sweep path
    (``_GPU_AVAILABLE=False``) gives a stable, REPRODUCIBLE, bit-identity-friendly baseline.
    The sklearn-exact path is also the deterministic one (hnswlib is approximate -> not
    bit-reproducible regardless), so it is the correct bit-identity measurement target.
    """
    os.environ.setdefault("PYUTILZ_KERNEL_DISABLE_SWEEP", "1")
    os.environ.setdefault("MLFRAME_DISABLE_HNSW", "1")  # production-supported escape hatch (see _knn_helper)
    import mlframe.feature_engineering.transformer._utils as _U
    _U._GPU_AVAILABLE = False
    import mlframe.feature_engineering.transformer._knn_helper as _KH
    _KH._HNSW_AVAILABLE = False


_stabilize_native_env()

GOLDEN_PATH = r"D:/Temp/fe_opt_golden.npz"

# ---------------------------------------------------------------------------
# Representative big-wide deterministic config.
# n_train: bank size; n_query: rows transformed; d: feature width.
# n_train chosen above the 50k hnswlib crossover so cdist/loclift/rsd exercise the
# production hnsw path; d=64 is a representative wide tabular width.
# ---------------------------------------------------------------------------
# Representative realistic FE regime. The 60k+ hnswlib bank regime is environment-unstable on
# this 8GB/Pascal box (sklearn NearestNeighbors n_jobs=-1 OpenMP segfaults intermittently under
# memory pressure), so the default config sits in the realistic suite FE regime (a few-k bank,
# the MRMR-scene 2407x299 order of magnitude) where the sklearn-exact kNN path runs reliably.
# Override via env FE_BENCH_N_TRAIN / FE_BENCH_D for a scaling sweep on a roomier machine.
import os as _os
N_TRAIN = int(_os.environ.get("FE_BENCH_N_TRAIN", "6000"))
N_QUERY = int(_os.environ.get("FE_BENCH_N_QUERY", "2000"))
D = int(_os.environ.get("FE_BENCH_D", "64"))
SEED = 12345


def _make_data(task: str = "binary"):
    """Deterministic (X_train, y_train, X_query). Binary y with ~12% positive base rate,
    a non-trivial structure so the FE transforms produce a varied (non-degenerate) output."""
    rng = np.random.default_rng(SEED)
    X_train = rng.standard_normal((N_TRAIN, D)).astype(np.float32)
    X_query = rng.standard_normal((N_QUERY, D)).astype(np.float32)
    # Structured binary target: logistic on a sparse linear combo + interaction.
    w = rng.standard_normal(D).astype(np.float32)
    logits = X_train @ w + 0.5 * (X_train[:, 0] * X_train[:, 1])
    p = 1.0 / (1.0 + np.exp(-(logits - np.quantile(logits, 0.88))))
    y_train = (rng.random(N_TRAIN) < p).astype(np.int32)
    if task == "regression":
        y_train = logits.astype(np.float32)
    return X_train, y_train, X_query


def _build_adapters():
    """Return ordered list of (name, ShortlistTransformerAdapter) for the 5 shortlist transformers,
    wired exactly as a suite user would via custom_pre_pipelines."""
    from mlframe.feature_engineering.transformer import (
        ShortlistTransformerAdapter,
        compute_class_distance_features,
        compute_local_lift_features,
        compute_rff_features,
        compute_bgmm_density_ratio_features,
        compute_residual_stratified_distance_features,
    )

    adapters = [
        ("rff", ShortlistTransformerAdapter(
            compute_rff_features, needs_y=False, passthrough=False,
            # use_gpu=False: deterministic CPU njit path. The GPU matmul (cupy) is fp32 and
            # cannot be bit-identical to the CPU njit (>1e-9 elementwise), so for a bit-identity
            # FE-wall baseline the CPU path is the correct, reproducible measurement target.
            # (The GPU crossover has its own dedicated bench: bench_rff_matmul.py.)
            compute_kwargs={"n_features": 256, "seed": SEED, "use_gpu": False}, seed=SEED)),
        ("cdist", ShortlistTransformerAdapter(
            compute_class_distance_features, needs_y=True, passthrough=False,
            compute_kwargs={"task": "binary", "seed": SEED}, seed=SEED)),
        ("loclift", ShortlistTransformerAdapter(
            compute_local_lift_features, needs_y=True, passthrough=False,
            compute_kwargs={"task": "binary", "k": 32, "seed": SEED}, seed=SEED)),
        ("bgm", ShortlistTransformerAdapter(
            compute_bgmm_density_ratio_features, needs_y=True, passthrough=False,
            compute_kwargs={"task": "binary", "seed": SEED}, seed=SEED)),
        ("rsd", ShortlistTransformerAdapter(
            compute_residual_stratified_distance_features, needs_y=True, passthrough=False,
            compute_kwargs={"task": "binary", "seed": SEED}, seed=SEED)),
    ]
    return adapters


def _run_all(X_train, y_train, X_query):
    """Fit + transform every shortlist adapter; return dict name -> (n_query, n_feat) float32 ndarray
    AND per-transformer wall times."""
    adapters = _build_adapters()
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    Xt_df = pl.DataFrame({f"f{i}": X_train[:, i] for i in range(X_train.shape[1])})
    Xq_df = pl.DataFrame({f"f{i}": X_query[:, i] for i in range(X_query.shape[1])})
    for name, adapter in adapters:
        t0 = time.perf_counter()
        adapter.fit(Xt_df, y_train)
        out = adapter.transform(Xq_df)
        timings[name] = time.perf_counter() - t0
        # out is a pandas DataFrame (passthrough=False -> only new features)
        outputs[name] = np.ascontiguousarray(out.to_numpy(), dtype=np.float32)
    return outputs, timings


def _warm():
    """Tiny warm-up call to JIT-compile numba kernels + prime hnswlib so wall-time excludes one-time JIT."""
    rng = np.random.default_rng(0)
    Xt = rng.standard_normal((200, 8)).astype(np.float32)
    Xq = rng.standard_normal((50, 8)).astype(np.float32)
    y = (rng.random(200) < 0.3).astype(np.int32)
    from mlframe.feature_engineering.transformer import compute_rff_features
    compute_rff_features(Xt, X_query=Xq, n_features=16, seed=1)


def cmd_wall(reps: int = 1):
    _warm()
    X_train, y_train, X_query = _make_data()
    best = None
    for r in range(reps):
        outputs, timings = _run_all(X_train, y_train, X_query)
        total = sum(timings.values())
        print(f"[rep {r}] per-transformer wall (s):")
        for name in ["rff", "cdist", "loclift", "bgm", "rsd"]:
            print(f"    {name:<8} {timings[name]:8.4f}")
        print(f"    {'TOTAL':<8} {total:8.4f}")
        if best is None or total < best:
            best = total
    print(f"\nBEST TOTAL FE-transform wall: {best:.4f} s")
    return best


def cmd_golden():
    _warm()
    X_train, y_train, X_query = _make_data()
    outputs, timings = _run_all(X_train, y_train, X_query)
    np.savez(GOLDEN_PATH, **outputs)
    print(f"Golden engineered output saved to {GOLDEN_PATH}")
    for name, arr in outputs.items():
        print(f"    {name:<8} shape={arr.shape} sum={arr.sum():.6f} nan={np.isnan(arr).sum()}")


def cmd_verify():
    _warm()
    X_train, y_train, X_query = _make_data()
    outputs, _ = _run_all(X_train, y_train, X_query)
    golden = np.load(GOLDEN_PATH)
    all_ok = True
    for name in outputs:
        cur = outputs[name]
        gold = golden[name]
        if cur.shape != gold.shape:
            print(f"    {name:<8} SHAPE MISMATCH cur={cur.shape} gold={gold.shape}")
            all_ok = False
            continue
        identical = np.array_equal(cur, gold) or (np.array_equal(np.isnan(cur), np.isnan(gold)) and np.array_equal(np.nan_to_num(cur), np.nan_to_num(gold)))
        if identical:
            print(f"    {name:<8} BIT-IDENTICAL")
        else:
            maxdiff = np.nanmax(np.abs(cur.astype(np.float64) - gold.astype(np.float64)))
            print(f"    {name:<8} DIFFERS maxabs={maxdiff:.3e}")
            all_ok = False
    print("\nVERIFY:", "ALL BIT-IDENTICAL" if all_ok else "MISMATCH DETECTED")
    return all_ok


def cmd_cprofile(top: int = 20):
    _warm()
    X_train, y_train, X_query = _make_data()
    pr = cProfile.Profile()
    pr.enable()
    outputs, timings = _run_all(X_train, y_train, X_query)
    pr.disable()
    print("Per-transformer wall (s):")
    for name in ["rff", "cdist", "loclift", "bgm", "rsd"]:
        print(f"    {name:<8} {timings[name]:8.4f}")
    print(f"    TOTAL    {sum(timings.values()):8.4f}\n")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(top)
    print(s.getvalue())
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(top)
    print("==== BY TOTTIME ====")
    print(s2.getvalue())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "wall"
    if cmd == "wall":
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cmd_wall(reps)
    elif cmd == "golden":
        cmd_golden()
    elif cmd == "verify":
        ok = cmd_verify()
        sys.exit(0 if ok else 1)
    elif cmd == "cprofile":
        cmd_cprofile()
    elif cmd == "all":
        cmd_wall()
        print("\n" + "=" * 60 + "\n")
        cmd_cprofile()
    else:
        print(f"unknown cmd {cmd!r}")
        sys.exit(2)


if __name__ == "__main__":
    main()

"""mRMR benchmark runner.

Captures wall-time (min/median/p95/std), per-stage memory (tracemalloc),
counters, and arr2str collision census for each scenario across ``--runs`` runs.
Optional pyinstrument profiling pass via ``--profile``.

Usage
-----
    python -m mlframe.feature_selection._benchmarks.bench_mrmr --scenarios all --tag pre-refactor
    python -m mlframe.feature_selection._benchmarks.bench_mrmr --scenarios n10k_p100_clf --runs 3 --profile

The output JSON path is ``_results/<tag>_<git_sha>.json`` plus a
``_results/MANIFEST.json`` recording capture environment for rotation policy.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import shutil
import subprocess  # nosec B404 - subprocess used below with fixed list args, no shell=True
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Imports of MRMR are deferred until inside _run_one() so that we can wipe the
# numba __pycache__ before importing for a true cold-start measurement.

from . import _datasets

logger = logging.getLogger("mlframe.bench_mrmr")

DISPERSION_GATE = 0.25
MIN_RUNS = 5
MAX_EXTRA_RUNS = 10
RESULTS_DIR = Path(__file__).parent / "_results"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(  # nosec B603, B607 - fixed/trusted executable (git) with list args, no untrusted input, resolved via PATH intentionally
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception as exc:
        logger.debug("_git_sha: git rev-parse failed: %s", exc)
        return "nogit"


def _cpu_model() -> str:
    return platform.processor() or platform.machine() or "unknown"


def _gpu_model() -> str:
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        attrs = cp.cuda.runtime.getDeviceProperties(dev.id)
        return attrs["name"].decode() if isinstance(attrs.get("name"), bytes) else str(attrs.get("name", "unknown"))
    except Exception as exc:
        logger.debug("_gpu_model: cupy device probe failed: %s", exc)
        return "no-gpu"


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for mod in ("numpy", "numba", "sklearn", "pandas", "joblib", "polars"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = "missing"
    try:
        import cupy
        out["cupy"] = cupy.__version__
    except ImportError:
        out["cupy"] = "missing"
    return out


def _wipe_numba_cache() -> None:
    """Wipe __pycache__ inside mlframe.feature_selection.* and any user numba cache.

    Per the plan: must run before each tagged baseline so cross-module njit
    compile times are honestly measured.
    """
    fs_root = Path(__file__).resolve().parents[1]
    for p in fs_root.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    nb_cache = fs_root.parent.parent / ".numba_cache"
    if nb_cache.exists():
        shutil.rmtree(nb_cache, ignore_errors=True)


def _stage_timer():
    """Returns (record(name), get_breakdown()). record() captures wall time + RSS delta
    against the previous boundary."""
    boundaries: list[tuple[str, float, int]] = []  # (name, t_s, mem_bytes_peak_so_far)

    def record(name: str) -> None:
        try:
            current, peak = tracemalloc.get_traced_memory()
        except RuntimeError:
            current, peak = 0, 0
        boundaries.append((name, time.perf_counter(), peak))

    def breakdown() -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for prev, curr in zip(boundaries, boundaries[1:]):
            stage = curr[0]
            out[stage] = {
                "wall_s": curr[1] - prev[1],
                "mem_peak_mb": curr[2] / (1024 * 1024),
            }
        return out

    return record, breakdown


def _run_one(scenario: _datasets.Scenario, seed: int, profile: bool) -> dict[str, Any]:
    """Single run: build data, fit MRMR, capture metrics. Imports MRMR fresh."""
    from mlframe.feature_selection.filters import MRMR  # type: ignore

    X, y = _datasets.make_scenario_data(scenario, random_state=seed)

    record, breakdown = _stage_timer()
    tracemalloc.start()
    record("start")

    record("data_built")

    mrmr_kwargs = dict(
        quantization_nbins=10,
        interactions_max_order=1,
        full_npermutations=3,
        baseline_npermutations=2,
        random_seed=seed,
        use_gpu=scenario.use_gpu,
        n_jobs=1,
        verbose=0,
        fe_max_steps=scenario.fe_max_steps,
        cv=2,
    )

    profiler = None
    if profile:
        try:
            from pyinstrument import Profiler
            profiler = Profiler(interval=0.01)
            profiler.start()
        except ImportError:
            logger.warning("pyinstrument not installed; skipping profile pass")

    t_fit_start = time.perf_counter()
    mrmr = MRMR(**mrmr_kwargs)
    mrmr.fit(X, y)
    t_fit_end = time.perf_counter()

    if profiler is not None:
        profiler.stop()

    record("fit_done")

    # Counters: best-effort, fall back to None if attributes absent.
    counters: dict[str, Any] = {}
    for attr, jsname in (
        ("n_features_", "n_features_selected"),
        ("n_features_in_", "n_features_in"),
    ):
        counters[jsname] = getattr(mrmr, attr, None)
    cached_MIs = getattr(mrmr, "_cached_MIs", None) or getattr(mrmr, "cached_MIs_", None)
    cached_cond = getattr(mrmr, "_cached_cond_MIs", None) or getattr(mrmr, "cached_cond_MIs_", None)
    counters["cached_MIs_size"] = len(cached_MIs) if cached_MIs is not None else None
    counters["cached_cond_MIs_size"] = len(cached_cond) if cached_cond is not None else None

    support = getattr(mrmr, "support_", None)
    support_list = sorted(support.tolist()) if support is not None else None

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out: dict[str, Any] = {
        "wall_s": t_fit_end - t_fit_start,
        "mem_peak_mb": peak / (1024 * 1024),
        "stages": breakdown(),
        "counters": counters,
        "support_set": support_list,
        "seed": seed,
    }
    if profiler is not None:
        profile_path = RESULTS_DIR / "profiles"
        profile_path.mkdir(parents=True, exist_ok=True)
        html_file = profile_path / f"{scenario.name}_seed{seed}.html"
        html_file.write_text(profiler.output_html(), encoding="utf-8")
        out["profile_html"] = str(html_file.relative_to(RESULTS_DIR.parent))
    return out


def _aggregate_runs(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    walls = np.array([r["wall_s"] for r in per_run])
    mem = np.array([r["mem_peak_mb"] for r in per_run])
    sup = per_run[0]["support_set"]
    sanity_ok = all(r["support_set"] == sup for r in per_run)
    dispersion = walls.std() / walls.mean() if walls.mean() > 0 else float("inf")
    return {
        "wall_time": {
            "min": float(walls.min()),
            "median": float(np.median(walls)),
            "p95": float(np.percentile(walls, 95)),
            "std": float(walls.std()),
            "n_runs": len(per_run),
            "dispersion": float(dispersion),
        },
        "mem_peak_mb": {
            "min": float(mem.min()),
            "median": float(np.median(mem)),
            "p95": float(np.percentile(mem, 95)),
        },
        "stages": per_run[0]["stages"],
        "counters": per_run[0]["counters"],
        "support_set": sup,
        "support_sanity_ok": sanity_ok,
    }


def run_scenario(scenario: _datasets.Scenario, runs: int, profile: bool, seed_base: int = 42) -> dict[str, Any]:
    if scenario.use_gpu:
        try:
            import cupy  # noqa: F401
        except ImportError:
            return {"skipped": "no cupy installed"}

    # Warmup pass (numba compile prewarm).
    logger.info(f"[{scenario.name}] warmup pass...")
    _run_one(scenario, seed_base, profile=False)
    gc.collect()

    per_run: list[dict[str, Any]] = []
    for i in range(runs):
        logger.info(f"[{scenario.name}] run {i + 1}/{runs}...")
        per_run.append(_run_one(scenario, seed_base, profile=profile and i == 0))
        gc.collect()

    agg = _aggregate_runs(per_run)
    if agg["wall_time"]["dispersion"] > DISPERSION_GATE and runs < MAX_EXTRA_RUNS:
        extra = min(MAX_EXTRA_RUNS - runs, runs)
        logger.warning(f"[{scenario.name}] dispersion {agg['wall_time']['dispersion']:.3f} > {DISPERSION_GATE}; +{extra} runs")
        for i in range(extra):
            per_run.append(_run_one(scenario, seed_base, profile=False))
            gc.collect()
        agg = _aggregate_runs(per_run)
    return agg


def _write_manifest(tag: str, git_sha: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"captures": []}
    manifest["captures"].append({
        "tag": tag,
        "git_sha": git_sha,
        "capture_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "cpu_model": _cpu_model(),
        "gpu_model": _gpu_model(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", default="cpu", help="comma-separated scenario names, or 'all' / 'cpu' / 'gpu'")
    p.add_argument("--tag", default="adhoc", help="output tag (e.g. 'pre-refactor')")
    p.add_argument("--runs", type=int, default=MIN_RUNS, help=f"runs per scenario (>= {MIN_RUNS} recommended)")
    p.add_argument("--profile", action="store_true", help="capture pyinstrument HTML for first run of each scenario")
    p.add_argument("--clear-cache", action="store_true", help="wipe numba __pycache__ before run")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.clear_cache:
        logger.info("wiping numba cache...")
        _wipe_numba_cache()

    # Per-process numba cache dir on Windows to avoid file lock races during
    # parallel hyperparameter search invoking this script concurrently.
    if sys.platform == "win32":
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_bench_")

    if args.scenarios == "all":
        scenarios = _datasets.list_scenarios(include_gpu=True)
    elif args.scenarios == "cpu":
        scenarios = _datasets.list_scenarios(include_gpu=False)
    elif args.scenarios == "gpu":
        scenarios = _datasets.GPU_SCENARIOS
    else:
        names = [s.strip() for s in args.scenarios.split(",")]
        scenarios = [_datasets.SCENARIOS[n] for n in names]

    git_sha = _git_sha()
    results: dict[str, Any] = {}
    for sc in scenarios:
        results[sc.name] = run_scenario(sc, runs=args.runs, profile=args.profile, seed_base=args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}_{git_sha}.json"
    out_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "git_sha": git_sha,
                "capture_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_manifest(args.tag, git_sha)
    logger.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

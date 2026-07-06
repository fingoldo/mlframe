"""Param-oracle CPU-backend selection + HW-tuned crossover lookup for polyeval.

Houses the ``polyeval`` njit-vs-njit_par CPU-backend ParamOracle (the
"learning to optimize" migration, gated OFF by default) and the
kernel-tuning-cache threshold lookup ``_lookup_polyeval_thresholds``. The
polyeval kernels + the ``polyeval_dispatch`` router stay in the parent
``hermite_fe``; ``benchmark_polyeval_cpu_backends`` lazy-imports the kernel
dicts from the parent so the kernel registry is never split from the kernels.
"""
from __future__ import annotations

import os as _os

import numpy as np

# Thresholds in array length n. Tunable via env var.
_PAR_THRESHOLD = int(_os.environ.get("MLFRAME_POLYEVAL_PAR_THRESHOLD", "50000"))
_CUDA_THRESHOLD = int(_os.environ.get("MLFRAME_POLYEVAL_CUDA_THRESHOLD", "500000"))


def _lookup_polyeval_thresholds(basis: str, n: int) -> tuple[int, int]:
    """Wave 23 P2 (2026-05-20): consult kernel_tuning_cache for HW-tuned
    (par_threshold, cuda_threshold) crossovers; fall back to the
    source-code defaults (which are env-var-overridable for tests)."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        _cache = KernelTuningCache.load_or_create()
        _entry = _cache.lookup("polyeval", basis=basis, n_samples=n)
        _par = int(_entry["par_threshold"]) if _entry and "par_threshold" in _entry else _PAR_THRESHOLD
        _cuda = int(_entry["cuda_threshold"]) if _entry and "cuda_threshold" in _entry else _CUDA_THRESHOLD
        return _par, _cuda
    except Exception:
        return _PAR_THRESHOLD, _CUDA_THRESHOLD


# --- Param-Oracle CPU-backend migration (proof-of-concept) -----------------
# The njit-vs-njit_par CPU crossover is the FIRST kernel_tuning_cache decision
# migrated to the ParamOracle ("learning to optimize") path. It is gated OFF by
# default (MLFRAME_POLYEVAL_ORACLE unset/"0") so the legacy threshold path stays
# byte-identical. When enabled, a ParamOracle keyed on the array-size fingerprint
# picks njit vs njit_par from RECORDED wall-times instead of a hardcoded crossover.
#
# The cuda branch stays EXACTLY on kernel_tuning_cache (_lookup_polyeval_thresholds), HW-tuned per host by
# _run_sweep_polyeval. On transfer-bound laptop GPUs (e.g. RTX 500 Ada) the cheap degree-N Horner kernel
# never beats njit_par because the H2D+D2H round trip dwarfs the compute, so the sweep persists a sentinel
# cuda_threshold above the swept range -- the dispatcher then never routes to the slower cuda path on that host.
# The oracle here governs ONLY the {njit, njit_par} CPU choice.

_POLYEVAL_ORACLE_FN_NAME = "polyeval_cpu_backend"
_POLYEVAL_ORACLE_PARAM_SPACE = {"backend": ["njit", "njit_par"]}
_polyeval_oracle_singleton = None


def _polyeval_oracle_enabled() -> bool:
    return _os.environ.get("MLFRAME_POLYEVAL_ORACLE", "0").strip() not in ("", "0")


def _polyeval_size_fingerprint(n: int) -> dict:
    """Stat-only fingerprint for the CPU-backend choice: array length only.
    Buckets at half-decade resolution (handled downstream by the oracle), so
    n=200 and n=210 collapse to one region but n=200 and n=500k do not."""
    return {"n": int(n), "p": 1, "dtype_kind": "f"}


def get_polyeval_oracle():
    """Lazily build (once per process) the ParamOracle that governs the CPU
    njit/njit_par backend choice. Seeds cold-start observations from the
    existing ``polyeval`` kernel_tuning_cache regions (read-only bridge) so the
    migration inherits any HW-tuned history rather than starting blind."""
    global _polyeval_oracle_singleton
    if _polyeval_oracle_singleton is not None:
        return _polyeval_oracle_singleton
    from mlframe.utils import ParamOracle
    oracle = ParamOracle(
        "polyeval_cpu_backend.parquet",
        param_space=_POLYEVAL_ORACLE_PARAM_SPACE,
        minimize="elapsed_s",
        mode="inference",
        min_observations=1,
    )
    # Read-only import of any njit/njit_par history the kernel cache holds. The
    # legacy 'polyeval' KTC entry stores par_threshold/cuda_threshold, not a
    # per-size backend label, so this is usually a no-op today; the bridge is
    # exercised by Layer-103 tests with synthetic KTC data and is ready for the
    # day a per-size CPU sweep is recorded.
    try:
        oracle.read_ktc_regions(
            "polyeval_cpu_backend", param_field="backend",
            fixed_fp={"p": 1, "dtype_kind": "f"},
            fn_name=_POLYEVAL_ORACLE_FN_NAME,
        )
    except Exception:  # nosec B110 - best-effort path
        pass
    _polyeval_oracle_singleton = oracle
    return _polyeval_oracle_singleton


def benchmark_polyeval_cpu_backends(basis: str, sizes=(200, 500_000), repeats: int = 3, oracle=None) -> dict:
    """Sweep njit vs njit_par at the given array sizes, timing each, and record
    the wall-times into the CPU-backend ParamOracle. Populates the oracle so
    later ``inference`` calls recommend the empirically faster backend per size.

    Returns ``{(n, backend): median_elapsed_s}`` for inspection. CPU-only: never
    touches the cuda path (unbenchable here)."""
    import time as _time
    if oracle is None:
        oracle = get_polyeval_oracle()
    from . import _NJIT_FUNCS, _NJIT_PAR_FUNCS
    c = np.array([0.3, -0.7, 0.2, 0.5, -0.1], dtype=np.float64)
    funcs = {"njit": _NJIT_FUNCS[basis], "njit_par": _NJIT_PAR_FUNCS[basis]}
    results: dict = {}
    for n in sizes:
        x = np.linspace(-1.0, 1.0, int(n)).astype(np.float64)
        for backend, fn in funcs.items():
            fn(x, c)  # warm the numba compile so it doesn't pollute the timing
            times = []
            for _ in range(max(1, repeats)):
                t0 = _time.perf_counter()
                fn(x, c)
                times.append(_time.perf_counter() - t0)
            med = float(sorted(times)[len(times) // 2])
            fp = _polyeval_size_fingerprint(n)
            oracle.record(fp, {"backend": backend}, {"elapsed_s": med}, fn_name=_POLYEVAL_ORACLE_FN_NAME)
            results[(int(n), backend)] = med
    return results


def _polyeval_oracle_pick_cpu_backend(n: int) -> str:
    """Ask the oracle which CPU backend (njit | njit_par) is faster for size
    ``n``. Falls back to ``njit`` if the oracle has no usable recommendation."""
    oracle = get_polyeval_oracle()
    fp = _polyeval_size_fingerprint(n)
    combo = oracle.recommend(fp, fn_name=_POLYEVAL_ORACLE_FN_NAME)
    backend = combo.get("backend") if isinstance(combo, dict) else None
    return backend if backend in ("njit", "njit_par") else "njit"

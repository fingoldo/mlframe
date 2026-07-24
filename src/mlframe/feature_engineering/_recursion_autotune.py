"""Auto-tune sweep for the per-group recursion FE kernels.

Mirrors ``mlframe.feature_selection._benchmarks.kernel_tuning_cache``: on
first use for a host, measure serial vs parallel njit at a grid of
``(n_samples, n_groups)`` points and persist the winning backend per
region into the shared ``pyutilz.performance.kernel_tuning.cache`` JSON. The
dispatcher then reads those regions instead of the pre-sweep fallback.

Triggered lazily (``run_auto_tune=True`` on the dispatcher) or via the
CLI ``python -m mlframe.feature_engineering._recursion_autotune``.
"""
from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# (n_groups, rows_per_group) grid spanning the realistic regimes: few big
# series, many medium, lots of small. Region caps are derived from these.
_SWEEP_GRID = [
    (1, 5_000),
    (4, 2_000),
    (16, 2_000),
    (64, 2_000),
    (256, 1_000),
    (1024, 500),
]
_N_ITERS = 3
# Bump when the bayesian recursion kernels or the sweep grid/semantics change,
# so cached tunings re-validate (the code_version hash already tracks the kernel
# source; salt covers grid/threshold changes the hash can't see).
_RECURSION_SALT = 1


def recursion_code_version(kernel_name: str) -> str | None:
    """code_version for a recursion kernel: hashes the bayesian kernel body +
    salt. None if pyutilz/code-versioning is unavailable."""
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version
        from . import bayesian

        fn = bayesian.bocpd_features if kernel_name == "fe_bocpd" else bayesian.online_bayesian_linear_regression
        return str(compute_code_version(fn, salt=_RECURSION_SALT))
    except Exception as exc:
        logger.debug("recursion_code_version: code-versioning unavailable for %r: %s", kernel_name, exc)
        return None


def _make_data(n_groups: int, rows_per_group: int, seed: int = 0):
    """Synthesize a group-structured observation series (a random walk with ~2% missing) plus a design matrix, for autotune sweep timing."""
    rng = np.random.default_rng(seed)
    N = n_groups * rows_per_group
    well = np.repeat(np.arange(n_groups), rows_per_group)
    md = np.tile(np.arange(rows_per_group, dtype=float), n_groups)
    obs = np.cumsum(rng.normal(0, 0.1, N)) + md * 0.01
    obs[rng.random(N) < 0.02] = np.nan
    X = np.column_stack([np.ones(N), md / (md.max() + 1e-9)])
    return obs, X, well


def _time_backend(kernel_name: str, backend: str, obs, X, well) -> float:
    """Median wall-time (over ``_N_ITERS`` warm runs, after one JIT-warmup call) of the given recursion kernel forced onto ``backend`` ("serial"/"parallel"), used to pick the faster backend per grid point."""
    from . import bayesian
    # Select the backend via the explicit _force_backend arg rather than toggling the process-global
    # MLFRAME_FE_RECURSION_BACKEND env: the old set/del-in-finally globally flipped the backend that any
    # concurrently-running FE caller in another thread would read mid-computation.
    if kernel_name == "fe_bocpd":
        def fn():
            """Run one bocpd_features call on the fixed synthetic data, forced onto the closed-over backend."""
            return bayesian.bocpd_features(obs, group_ids=well, hazard=1 / 250, _force_backend=backend)
    else:
        def fn():
            """Run one online_bayesian_linear_regression call on the fixed synthetic data, forced onto the closed-over backend."""
            return bayesian.online_bayesian_linear_regression(obs, X, group_ids=well, _force_backend=backend)
    fn()  # warmup (jit compile)
    ts = []
    for _ in range(_N_ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _run_sweep(kernel_name: str) -> list[dict]:
    """Measure serial vs parallel across the grid; return cache regions."""
    regions: list[dict] = []
    for n_groups, rpg in _SWEEP_GRID:
        obs, X, well = _make_data(n_groups, rpg)
        N = obs.size
        t_serial = _time_backend(kernel_name, "serial", obs, X, well)
        t_parallel = _time_backend(kernel_name, "parallel", obs, X, well)
        choice = "parallel" if t_parallel < t_serial else "serial"
        regions.append({
            "n_samples_max": int(N),
            "n_groups_max": int(n_groups),
            "backend_choice": choice,
            "serial_ms": round(t_serial * 1e3, 3),
            "parallel_ms": round(t_parallel * 1e3, 3),
        })
        logger.info("autotune %s: N=%d groups=%d -> %s (serial=%.2fms par=%.2fms)",
                    kernel_name, N, n_groups, choice, t_serial * 1e3, t_parallel * 1e3)
    # Catch-all: extrapolate the largest-grid choice to everything bigger.
    regions.append({
        "n_samples_max": None, "n_groups_max": None,
        "backend_choice": regions[-1]["backend_choice"],
    })
    return regions


def ensure_recursion_tuning(kernel_name: str, *, force: bool = False) -> None:
    """Populate the kernel_tuning_cache for ``kernel_name`` if absent.

    No-op when a tuning already exists (unless ``force``). Best-effort: any
    failure (pyutilz missing, numba missing) is logged and swallowed so the
    dispatcher silently falls back to its heuristic.
    """
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        cache = KernelTuningCache()
        if not force and cache.has(kernel_name):
            return
        regions = _run_sweep(kernel_name)
        cache.update(kernel_name, axes=["n_samples", "n_groups"], regions=regions)
        logger.info("autotune %s: cached %d regions", kernel_name, len(regions))
    except Exception as e:
        logger.warning("ensure_recursion_tuning(%s) failed: %s", kernel_name, e)


def _cli() -> None:
    """Command-line entrypoint: force-runs ``ensure_recursion_tuning`` for each requested kernel and logs the result."""
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Auto-tune FE recursion kernels.")
    p.add_argument("--force", action="store_true", help="re-run even if cached")
    p.add_argument("--kernels", nargs="*", default=["fe_bocpd", "fe_oblr"])
    args = p.parse_args()
    for kn in args.kernels:
        ensure_recursion_tuning(kn, force=args.force)


if __name__ == "__main__":
    _cli()


# Register both recursion kernels with the @kernel_tuner registry so
# retune_all / ``mlframe-tune-kernels`` discover + batch-tune them. CPU-only
# (serial vs parallel njit; no GPU variant). Wrapped so a missing pyutilz /
# circular import never breaks the dispatcher.
from pyutilz.performance.kernel_tuning.registry import kernel_tuner
from . import bayesian

for _kn, _ref in (("fe_bocpd", bayesian.bocpd_features), ("fe_oblr", bayesian.online_bayesian_linear_regression)):
    kernel_tuner(
        kernel_name=_kn,
        variant_fns=(_ref,),
        tuner=(lambda kn=_kn: _run_sweep(kn)),
        axes={"n_samples": [n * r for n, r in _SWEEP_GRID], "n_groups": [g for g, _ in _SWEEP_GRID]},
        fallback={"backend_choice": "serial"},
        env_key="MLFRAME_FE_RECURSION_BACKEND",
        gpu_capable=False,
        salt=_RECURSION_SALT,
        cli_label=_kn,
    )


__all__ = ["ensure_recursion_tuning", "recursion_code_version"]

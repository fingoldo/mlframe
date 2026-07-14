"""Measurement-backed backend lookup for the composite-target corr + collinear dispatchers.

Both ``_corr_numba.safe_abs_corr_all_dispatch`` and
``_collinear_numba.near_collinear_keep_mask_fast`` choose between a pure-numpy
reference and a ``numba`` kernel by a size gate.  The raw gates (corr: ``n >= 20k``
AND ``F >= 64``; collinear: ``B >= 10`` AND ``n >= 256``) were hardcoded dev-box
guesses.  Per the project's "integrate with kernel_tuning_cache, do NOT hardcode"
rule this module routes the choice through the same
``pyutilz.performance.kernel_tuning.cache.KernelTuningCache`` infrastructure that
already powers ``joint_hist_batched`` / ``plugin_mi_classif_dispatch`` -- consulting
the per-host cache for the measured (n, F/B) crossover and falling back to the
hardcoded threshold whenever pyutilz / the cache is unavailable or the kernel has
not been tuned on this host yet.

Both backends are bit-identical by construction (the dispatchers re-decide every
borderline column/pair with the exact numpy primitives), so the lookup only ever
trades runtime, never the returned result -- which is why this consult-or-fallback
design is safe regardless of which backend fires.

Env-var force-overrides (escape hatch + A/B + tests):
  * ``MLFRAME_COMPOSITE_CORR_BACKEND=numba|numpy``
  * ``MLFRAME_COMPOSITE_COLLINEAR_BACKEND=numba|numpy``

KTC sweep status: ``_run_sweep_composite_corr`` / ``_run_sweep_composite_collinear``
below measure both backends across an (n_rows, n_cols) grid on THIS host and persist
the crossover via ``ensure_composite_corr_tuning`` / ``ensure_composite_collinear_tuning``.
Unlike the njit-vs-cuda sweeps in ``kernel_tuning_cache/_auto_tune_sweeps_a.py``, these
are pure-CPU numpy-vs-numba crossovers (no GPU context) -- cheap and safe to run inline,
so ``run_auto_tune=True`` triggers a real (not placeholder) sweep on a cache miss.

Measured on this dev box (multi-session shared machine, real contention present):
``composite_collinear_dispatch`` shows numba winning at EVERY measured cell down to
n=500/cols=10 (74.9ms numpy vs 1.1ms numba) -- expected, since the numpy reference
walks columns in a Python loop (not vectorised), unlike the corr reference. This
matches the existing hardcoded gate's much lower ``_MIN_ROWS=256``/``_MIN_COLS=10``.
``composite_corr_dispatch`` (numpy reference IS vectorised: matmul/einsum) shows a
NOISY, non-monotonic crossover under this host's real contention -- e.g. numpy still
edges out numba at n=50,000/cols=64 (12.3ms vs 16.9ms) and n=100,000/cols=128 (59.5ms
vs 60.5ms), both well inside the hardcoded gate's "numba" region -- rather than the
clean single-threshold crossover the hardcoded gate assumes. Per the project's A/B
policy (median-of-N, quiet host), this sweep should be RE-RUN on an idle host before
its regions are trusted as the production default; no dispatcher call site has been
flipped to ``run_auto_tune=True`` here, so the hardcoded gate stays authoritative
until a clean re-sweep lands. The plumbing (sweep + persist + consult) is complete
and tested; only the "trust this host's numbers" decision is deliberately deferred.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

_CORR_ENV = "MLFRAME_COMPOSITE_CORR_BACKEND"
_COLLINEAR_ENV = "MLFRAME_COMPOSITE_COLLINEAR_BACKEND"
_VALID_BACKENDS = ("numba", "numpy")


def _forced_backend(env_key: str) -> str | None:
    """Return a validated forced backend from ``env_key`` or ``None``.

    A bare / unset / unrecognised value is treated as "no override" (the lookup
    then proceeds normally) so a typo can never silently pin the slow path.
    """
    val = os.environ.get(env_key, "").strip().lower()
    return val if val in _VALID_BACKENDS else None


def _get_cache():
    """Return the shared per-process KernelTuningCache singleton or ``None``.

    Delegates to the FS-side singleton so this module and the hot-path filters share
    ONE instance (one ``nvidia-smi`` probe per process).  Any import miss / init
    failure returns ``None`` -> the caller uses its hardcoded fallback.
    """
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
    except Exception:  # pyutilz / FS package unavailable -> hardcoded fallback.
        return None
    try:
        return get_kernel_tuning_cache()
    except Exception:  # pragma: no cover - defensive; singleton already guards.
        return None


def _median_call_ms(callable_no_args, n_iters: int) -> float:
    """Median wall time (ms) of ``n_iters`` calls to the zero-arg ``callable_no_args``."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        callable_no_args()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _run_sweep_composite_corr(n_iters: int = 3) -> list[dict]:
    """Sweep ``(n_rows, n_cols)`` for ``safe_abs_corr_all_dispatch``'s numpy-vs-numba crossover.

    Pure CPU numpy-vs-numba (no GPU context), so a real sweep is cheap and safe to run inline on a
    cache miss -- unlike the njit-vs-cuda sweeps in ``kernel_tuning_cache/_auto_tune_sweeps_a.py``,
    no contention modelling is needed here.
    """
    from ._corr_numba import _abs_corr_all_kernel, _HAS_NUMBA
    from .screening import _safe_abs_corr_all

    if not _HAS_NUMBA:
        return []
    rng = np.random.default_rng(7)
    n_axis = (2_000, 10_000, 20_000, 50_000, 100_000)
    col_axis = (8, 32, 64, 128)
    # Warm the JIT once so the first measured cell isn't paying compile cost.
    _wx = rng.normal(size=(500, 4))
    _wy = _wx[:, 0] - _wx[:, 0].mean()
    try:
        _abs_corr_all_kernel(_wx, _wy, float(np.dot(_wy, _wy)), 1e-9)
    except Exception as exc:
        logger.debug("composite corr sweep warmup failed: %s", exc)
        return []

    def _corr_cell(n: int, k: int) -> dict | None:
        """Time numpy vs numba at one ``(n, k)`` cell, returning a region dict or ``None`` on failure."""
        try:
            y = rng.normal(size=n)
            X = np.ascontiguousarray(rng.normal(size=(n, k)), dtype=np.float64)
            y_dev = y - y.mean()
            var_y = float(np.dot(y_dev, y_dev))

            t_numpy = _median_call_ms(lambda y=y, X=X: _safe_abs_corr_all(y, X), n_iters)
            t_numba = _median_call_ms(lambda X=X, y_dev=y_dev, var_y=var_y: _abs_corr_all_kernel(X, y_dev, var_y, 1e-9), n_iters)
            backend = "numba" if t_numba < t_numpy else "numpy"
            logger.info("auto_tune composite_corr n=%d k=%d -> %s (numpy=%.3fms numba=%.3fms)", n, k, backend, t_numpy, t_numba)
            return {
                "n_samples_max": int(n), "n_cols_max": int(k),
                "backend_choice": backend, "numpy_ms": round(t_numpy, 4), "numba_ms": round(t_numba, 4),
            }
        except Exception as exc:
            logger.debug("composite corr sweep skipped n=%d k=%d: %s", n, k, exc)
            return None

    regions: list[dict] = [r for n in n_axis for k in col_axis if (r := _corr_cell(n, k)) is not None]
    if not regions:
        return []
    largest = max(regions, key=lambda r: (r["n_samples_max"], r["n_cols_max"]))
    regions.append({"n_samples_max": None, "n_cols_max": None, "backend_choice": largest["backend_choice"]})
    return regions


def _run_sweep_composite_collinear(n_iters: int = 3) -> list[dict]:
    """Sweep ``(n_rows, n_cols)`` for ``near_collinear_keep_mask_fast``'s numpy-vs-numba crossover."""
    from ._collinear_numba import _HAS_NUMBA, _column_stats_allfinite, _keep_mask_kernel_allfinite
    from ._eval_stats import _near_collinear_keep_mask_numpy

    if not _HAS_NUMBA:
        return []
    rng = np.random.default_rng(7)
    n_axis = (500, 2_000, 10_000, 50_000)
    col_axis = (10, 30, 60, 120)
    _wfm = rng.normal(size=(200, 5))
    try:
        _wmean, _wvar = _column_stats_allfinite(_wfm)
        _keep_mask_kernel_allfinite(_wfm, _wmean, _wvar, 0.98, 1e-9)
    except Exception as exc:
        logger.debug("composite collinear sweep warmup failed: %s", exc)
        return []

    def _collinear_cell(n: int, k: int) -> dict | None:
        """Time numpy vs numba at one ``(n, k)`` cell, returning a region dict or ``None`` on failure."""
        try:
            fm = np.ascontiguousarray(rng.normal(size=(n, k)), dtype=np.float64)
            mean, var = _column_stats_allfinite(fm)

            t_numpy = _median_call_ms(lambda fm=fm: _near_collinear_keep_mask_numpy(fm, corr_threshold=0.98), n_iters)
            t_numba = _median_call_ms(
                lambda fm=fm, mean=mean, var=var: _keep_mask_kernel_allfinite(fm, mean, var, 0.98, 1e-9), n_iters,
            )
            backend = "numba" if t_numba < t_numpy else "numpy"
            logger.info("auto_tune composite_collinear n=%d k=%d -> %s (numpy=%.3fms numba=%.3fms)", n, k, backend, t_numpy, t_numba)
            return {
                "n_samples_max": int(n), "n_cols_max": int(k),
                "backend_choice": backend, "numpy_ms": round(t_numpy, 4), "numba_ms": round(t_numba, 4),
            }
        except Exception as exc:
            logger.debug("composite collinear sweep skipped n=%d k=%d: %s", n, k, exc)
            return None

    regions: list[dict] = [r for n in n_axis for k in col_axis if (r := _collinear_cell(n, k)) is not None]
    if not regions:
        return []
    largest = max(regions, key=lambda r: (r["n_samples_max"], r["n_cols_max"]))
    regions.append({"n_samples_max": None, "n_cols_max": None, "backend_choice": largest["backend_choice"]})
    return regions


def _CORR_SWEEP() -> list[dict]:
    """Zero-arg tuner adapter for ``_lookup_backend``'s ``run_auto_tune`` path (corr kernel)."""
    return _run_sweep_composite_corr()


def _COLLINEAR_SWEEP() -> list[dict]:
    """Zero-arg tuner adapter for ``_lookup_backend``'s ``run_auto_tune`` path (collinear kernel)."""
    return _run_sweep_composite_collinear()


def _lookup_backend(
    kernel_name: str,
    *,
    dims: dict,
    axes: list[str],
    fallback: str,
    run_auto_tune: bool = False,
) -> str:
    """Consult the KTC for ``kernel_name`` at ``dims``; return ``numba``/``numpy``.

    Returns ``fallback`` when pyutilz / the cache is unavailable, the kernel is not
    tuned on this host, or anything goes wrong -- so the dispatch degrades cleanly to
    the hardcoded size gate. ``run_auto_tune=True`` triggers a REAL sweep
    (``_run_sweep_composite_corr`` / ``_collinear``, pure CPU, cheap) on a cache miss
    instead of just falling through, and persists the result for subsequent calls.
    """
    cache = _get_cache()
    if cache is None or cache is False:
        return fallback
    try:
        if run_auto_tune:
            tuner = _CORR_SWEEP if kernel_name == "composite_corr_dispatch" else _COLLINEAR_SWEEP
            result = cache.get_or_tune(
                kernel_name,
                dims=dims,
                tuner=tuner,
                axes=axes,
                fallback={"backend_choice": fallback},
            )
        else:
            # Sweep is DEFERRED -- use a pure cache LOOKUP, never ``get_or_tune``. get_or_tune logs a
            # "sweep starting" banner and runs the (no-op) tuner on every miss, which is misleading
            # (no tuning happens) and pointless (the result is always the fallback until a sweep is
            # persisted). ``lookup`` consults a persisted region if one exists and otherwise returns
            # None -> fallback, silently.
            result = cache.lookup(kernel_name, **dims)
            if result is None:
                return fallback
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc in _VALID_BACKENDS:
            return bc
    except Exception as exc:  # any cache hiccup -> hardcoded fallback, never raise.
        logger.debug("composite KTC lookup for %s failed: %s", kernel_name, exc)
    return fallback


def choose_corr_backend(
    n_rows: int,
    n_cols: int,
    *,
    min_rows: int,
    min_cols: int,
    run_auto_tune: bool = False,
) -> str:
    """Pick ``"numba"`` or ``"numpy"`` for the abs-corr-all dispatch.

    Order: env-var force-override -> KTC measured crossover -> hardcoded size gate
    (``numba`` iff ``n_rows >= min_rows`` AND ``n_cols >= min_cols``).
    """
    forced = _forced_backend(_CORR_ENV)
    if forced is not None:
        return forced
    fallback = "numba" if (n_rows >= min_rows and n_cols >= min_cols) else "numpy"
    return _lookup_backend(
        "composite_corr_dispatch",
        dims={"n_samples": n_rows, "n_cols": n_cols},
        axes=["n_samples", "n_cols"],
        fallback=fallback,
        run_auto_tune=run_auto_tune,
    )


def choose_collinear_backend(
    n_rows: int,
    n_cols: int,
    *,
    min_rows: int,
    min_cols: int,
    run_auto_tune: bool = False,
) -> str:
    """Pick ``"numba"`` or ``"numpy"`` for the near-collinear keep-mask dispatch.

    Order: env-var force-override -> KTC measured crossover -> hardcoded size gate
    (``numba`` iff ``n_cols >= min_cols`` AND ``n_rows >= min_rows``).
    """
    forced = _forced_backend(_COLLINEAR_ENV)
    if forced is not None:
        return forced
    fallback = "numba" if (n_cols >= min_cols and n_rows >= min_rows) else "numpy"
    return _lookup_backend(
        "composite_collinear_dispatch",
        dims={"n_samples": n_rows, "n_cols": n_cols},
        axes=["n_samples", "n_cols"],
        fallback=fallback,
        run_auto_tune=run_auto_tune,
    )


def ensure_composite_corr_tuning(force: bool = False) -> list[dict] | None:
    """Run (or reuse the cached) ``composite_corr_dispatch`` sweep and persist it.

    ``force=True`` re-sweeps even if a cached entry exists. Returns the persisted
    regions, or ``None`` if pyutilz/the cache is unavailable.
    """
    cache = _get_cache()
    if cache is None or cache is False:
        return None
    if not force and cache.has("composite_corr_dispatch"):
        return None  # already tuned; nothing to do
    regions = _run_sweep_composite_corr()
    if regions:
        cache.update("composite_corr_dispatch", axes=["n_samples", "n_cols"], regions=regions)
    return regions


def ensure_composite_collinear_tuning(force: bool = False) -> list[dict] | None:
    """Run (or reuse the cached) ``composite_collinear_dispatch`` sweep and persist it.

    ``force=True`` re-sweeps even if a cached entry exists. Returns the persisted
    regions, or ``None`` if pyutilz/the cache is unavailable.
    """
    cache = _get_cache()
    if cache is None or cache is False:
        return None
    if not force and cache.has("composite_collinear_dispatch"):
        return None  # already tuned; nothing to do
    regions = _run_sweep_composite_collinear()
    if regions:
        cache.update("composite_collinear_dispatch", axes=["n_samples", "n_cols"], regions=regions)
    return regions


__all__ = [
    "choose_corr_backend", "choose_collinear_backend",
    "ensure_composite_corr_tuning", "ensure_composite_collinear_tuning",
]

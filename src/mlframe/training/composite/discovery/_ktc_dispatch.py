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

KTC sweep status: the lookup + fallback + env-override are wired here; the full
auto-tune sweep (``run_auto_tune=True`` -> ``_run_sweep_*``) is DEFERRED -- these
are CPU njit-vs-numpy crossovers, cheap to sweep but not yet measured on a quiet
host, so until a sweep is persisted every call falls through to the hardcoded
fallback below (identical behaviour to the pre-KTC code).  When a sweep lands, add
``_run_sweep_composite_corr`` / ``_collinear`` next to ``_run_sweep_mi_classif_dispatch``
and pass ``run_auto_tune=True`` from the dispatchers.
"""
from __future__ import annotations

import logging
import os

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
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache
    except Exception:  # pyutilz / FS package unavailable -> hardcoded fallback.
        return None
    try:
        return get_kernel_tuning_cache()
    except Exception:  # pragma: no cover - defensive; singleton already guards.
        return None


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
    the hardcoded size gate.  ``run_auto_tune`` is plumbed for a future sweep but
    defaults off (the sweep is deferred), so a cache miss falls straight to the
    fallback without paying a sweep cost.
    """
    cache = _get_cache()
    if cache is None or cache is False:
        return fallback
    try:
        result = cache.get_or_tune(
            kernel_name,
            dims=dims,
            tuner=(lambda: []),  # sweep deferred: a miss falls to the fallback.
            axes=axes,
            fallback={"backend_choice": fallback},
        )
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


__all__ = ["choose_corr_backend", "choose_collinear_backend"]

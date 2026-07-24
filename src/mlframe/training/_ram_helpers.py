"""RAM and GPU memory management utilities."""

from __future__ import annotations

import gc, logging, sys, threading

import psutil

logger = logging.getLogger(__name__)

# Lock for the adaptive baseline so concurrent callers (e.g. parallel
# joblib worker pools and the main thread both invoking
# `maybe_clean_ram_adaptive`) read-modify-write the baseline atomically.
_MAYBE_CLEAN_LOCK = threading.Lock()

def _caller_logger() -> logging.Logger:
    """Return the logger bound to the module that called the public helper
    which in turn called us. Used so progress lines like "Done. RAM usage:"
    or the "PHASE N" banner are attributed to the caller's module (e.g.
    ``mlframe.training.core``) instead of this utils module -- matches what
    a reader expects when scanning log origins.
    """
    try:
        # Stack: [_caller_logger] -> [public helper] -> [real caller]
        frame = sys._getframe(2)
        return logging.getLogger(frame.f_globals.get("__name__", __name__))
    except Exception as exc:
        logger.debug("_caller_logger: stack walk failed, attributing to this module instead: %s", exc)
        return logger

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import polars as pl
from pyutilz.system import clean_ram, get_own_memory_usage


def log_ram_usage() -> None:
    """Log current RAM usage, attributed to the caller's module."""
    _caller_logger().info(f"Done. RAM usage: {get_own_memory_usage():.1f}GB.")


# Adaptive clean_ram: skip gc.collect + trim when RSS hasn't grown meaningfully
# since the last clean. On small-DF runs (<50MB), gc.collect alone costs ~0.4s/call;
# 10 unconditional calls is ~44% of a 10s training. Baseline is refreshed after
# every real clean, so growth is measured from the most recent state.
_MAYBE_CLEAN_BASELINE_MB: float = 0.0
_MAYBE_CLEAN_MIN_GROWTH_MB: float = 500.0


def maybe_clean_ram_adaptive() -> None:
    """Call pyutilz.clean_ram only when process RSS has grown by
    ``_MAYBE_CLEAN_MIN_GROWTH_MB`` since the previous clean. Cheap
    short-circuit replacement for bare ``clean_ram()`` on hot training
    paths where small-DF runs don't justify a 0.4s gc.collect per call.

    Thread-safe: a module-level lock serializes baseline read-modify-write
    so concurrent callers from joblib workers don't race the float assignment
    (CPython float store is atomic but the if/clean/re-read sequence is not).
    """
    global _MAYBE_CLEAN_BASELINE_MB
    try:
        rss_mb = psutil.Process().memory_info().rss / 1024**2
    except Exception:
        clean_ram()
        return
    with _MAYBE_CLEAN_LOCK:
        if _MAYBE_CLEAN_BASELINE_MB == 0.0:
            _MAYBE_CLEAN_BASELINE_MB = rss_mb
            return
        if rss_mb - _MAYBE_CLEAN_BASELINE_MB > _MAYBE_CLEAN_MIN_GROWTH_MB:
            clean_ram()
            try:
                _MAYBE_CLEAN_BASELINE_MB = psutil.Process().memory_info().rss / 1024**2
            except Exception:
                _MAYBE_CLEAN_BASELINE_MB = rss_mb


def clean_ram_and_gpu(verbose: bool = False) -> None:
    """
    Clean both CPU RAM and GPU memory.

    Combines pyutilz.clean_ram() with GPU memory cleanup.
    Call this after model training to free memory before training next model.

    Args:
        verbose: If True, log memory stats after cleanup
    """

    # Clean CPU RAM first
    clean_ram()

    # Clean GPU memory if PyTorch CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            # Synchronize all CUDA streams before cleanup. cuda.synchronize() can raise
            # on a kernel error (out-of-memory mid-launch, async illegal-access from a
            # prior op). Pre-fix the raise would skip empty_cache() + gc.collect() below
            # so the GPU memory pool stayed full and the next allocation OOM'd with no
            # signal that the prior error caused it. Wrap in try/finally so the cache
            # release runs even when synchronize fails.
            try:
                torch.cuda.synchronize()
            except RuntimeError as _sync_err:
                logger.warning(
                    "torch.cuda.synchronize raised %s: %s; proceeding with empty_cache " "anyway to reclaim what we can. Likely a prior kernel error.",
                    type(_sync_err).__name__,
                    _sync_err,
                )
            # Empty the CUDA memory cache. Also robust to its own RuntimeError (very
            # rare: only on a broken CUDA context).
            try:
                torch.cuda.empty_cache()
            except RuntimeError as _empty_err:
                logger.warning(
                    "torch.cuda.empty_cache raised %s: %s; CUDA context may be broken.",
                    type(_empty_err).__name__, _empty_err,
                )
            # Force garbage collection again after GPU cleanup
            gc.collect()

            if verbose:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info("GPU memory after cleanup: %.2fGB allocated, %.2fGB reserved", allocated, reserved)
    except ImportError:
        pass  # PyTorch not installed


def estimate_df_size_mb(df) -> float:
    """Estimated in-memory size of a Polars/pandas DataFrame in MB.

    Returns `inf` for unsupported types so downstream OOM-protection thresholds
    trip correctly (Arrow/Modin/Dask inputs otherwise silently lose `clean_ram`
    heuristic's size-proportional growth check).
    """
    if isinstance(df, pl.DataFrame):
        return float(df.estimated_size("mb"))
    if isinstance(df, pd.DataFrame):
        return float(df.memory_usage(deep=True).sum() / 1024**2)
    return float("inf")


def get_process_rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        return float(psutil.Process().memory_info().rss / 1024**2)
    except Exception as exc:
        logger.debug("get_process_rss_mb: RSS probe failed, reporting 0.0: %s", exc)
        return 0.0


def should_clean_ram(baseline_rss_mb: float, df_size_mb: float, min_growth_mb: float = 500.0) -> bool:
    """True iff a clean_ram call (~0.6s) is likely justified.

    Triggers when either:
      - RSS grew beyond baseline by max(min_growth_mb, 30% of DF size) -- accumulated
        temp state worth collecting; OR
      - free system RAM < 2x DF size -- OOM risk, gc may release Arrow buffers.

    The ``free_mb < 2 * df_size_mb`` branch only matters when df_size is
    a meaningful fraction of system RAM (gigabyte-scale frames). For
    small frames the 2*df_size threshold is far below typical free RAM
    (we'd need free<10MB on a 5MB frame to trip), so the
    ``psutil.virtual_memory()`` call is pure overhead - 3ms per call
    on Windows. Short-circuit it when df_size is small enough that the
    OOM branch is moot. cProfile of fuzz combo c0134 traced 13ms across
    4 calls of ``should_clean_ram`` to ``virtual_memory()`` alone.
    """
    try:
        rss_mb = psutil.Process().memory_info().rss / 1024**2
    except Exception as e:
        logger.debug("should_clean_ram: RSS measurement failed, falling back to clean", exc_info=e)
        return True  # can't measure -> fall back to cleaning
    growth = rss_mb - baseline_rss_mb
    if growth > max(min_growth_mb, 0.3 * df_size_mb):
        return True
    # Only check free RAM when df is large enough for the 2*df_size
    # threshold to be a plausible OOM trigger. 256 MiB is a safe lower
    # bound: at this size the threshold is 512 MiB, comparable to
    # typical pressure on small VMs; smaller frames cannot OOM the host.
    if df_size_mb < 256:
        return False
    try:
        free_mb = psutil.virtual_memory().available / 1024**2
    except Exception as e:
        logger.debug("should_clean_ram: free-RAM measurement failed, falling back to clean", exc_info=e)
        return True
    return bool(free_mb < 2 * df_size_mb)


def maybe_clean_ram_and_gpu(
    baseline_rss_mb: float,
    df_size_mb: float,
    verbose: bool = False,
    reason: str = "",
) -> float:
    """Call clean_ram_and_gpu only when RAM metrics indicate it's worthwhile.

    On small DFs this avoids 0.6s of pure overhead per call; on large production
    DFs (or when the process is growing) it still fires at every site.

    Returns the (possibly refreshed) baseline RSS in MB. After a fire, baseline
    is re-captured so subsequent `growth = rss - baseline` checks are not
    monotonically inflated by already-cleaned state. Callers should assign
    the return back to their local baseline variable.
    """
    if should_clean_ram(baseline_rss_mb, df_size_mb):
        clean_ram_and_gpu(verbose=verbose)
        if verbose:
            if reason:
                logger.info("  clean_ram fired (%s)", reason)
            else:
                logger.info("  clean_ram fired")
        return get_process_rss_mb()
    return baseline_rss_mb

"""CUDA attempt of ``discretize_2d_array`` (VRAM guard, row-chunked fallback, full-upload fast path), carved out of the package ``__init__`` to keep it under the module-size budget.

The CUDA entry points are passed in by the facade (not imported here) so tests that monkeypatch ``discretization.discretize_2d_array_cuda`` /
``discretize_2d_array_cuda_row_chunked`` keep intercepting the calls.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.discretization")


def try_discretize_2d_cuda(
    arr: np.ndarray, n_bins: int, method: str, dtype: type, cuda_fn: Callable[..., np.ndarray], cuda_row_chunked_fn: Callable[..., np.ndarray]
) -> Optional[np.ndarray]:
    """Run the CUDA discretisation (full upload, or row-chunked when the VRAM cushion is missing); None when CUDA is unavailable or every GPU attempt failed (caller runs CPU prange)."""
    try:
        from .._gpu_policy import cuda_available_for_run

        if cuda_available_for_run():
            # VRAM guard: ``discretize_2d_array_cuda`` H2D-uploads the WHOLE ``arr``
            # unconditionally (``d_arr = cp.asarray(arr)``), then ``cp.percentile`` needs a
            # comparably-sized internal sort/partition scratch buffer on top - at production scale
            # (millions of rows) this can consume a small card's entire VRAM. On Windows/WDDM an
            # oversized upload can transparently over-subscribe device memory via host-paging instead
            # of raising a catchable CUDA OOM, so the kernel then grinds through PCIe-paged memory for
            # minutes before the OS silently kills the process - the try/except below never fires
            # because there is no exception to catch. Reject BEFORE attempting the upload, mirroring
            # every other GPU-FE dispatch site's ``_fe_gpu_vram.fe_gpu_has_vram_cushion`` guard.
            _bytes_needed = arr.nbytes * 2 + (arr.shape[0] * arr.shape[1] * np.dtype(dtype).itemsize)
            _vram_ok = True
            _free_gb = _total_gb = None
            _free_b: Optional[int] = None
            try:
                import cupy as _cp_probe

                _free_b, _total_b = _cp_probe.cuda.runtime.memGetInfo()
                _free_gb, _total_gb = _free_b / 1024**3, _total_b / 1024**3
            except Exception as exc:
                logger.debug("discretize_2d_array: memGetInfo probe failed (%s)", exc)
            try:
                from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion

                _vram_ok = fe_gpu_has_vram_cushion(_bytes_needed)
            except Exception as exc:
                logger.debug("discretize_2d_array: VRAM cushion probe failed (%s); permissive", exc)
            if not _vram_ok:
                # Never silent (explicit user feedback): log the full sizing context - requested GB,
                # shape/dtype, and free/total VRAM - so a production run is diagnosable from the log
                # alone, not just "GPU skipped" with no numbers.
                logger.warning(
                    "discretize_2d_array: GPU upload REJECTED -- requested %.2fGB upload+scratch "
                    "(n_rows=%d, n_cols=%d, in_dtype=%s, out_dtype=%s) exceeds the safe VRAM budget "
                    "(free=%s, total=%s) -- trying a row-chunked GPU path before falling back to CPU prange",
                    _bytes_needed / 1024**3,
                    arr.shape[0],
                    arr.shape[1],
                    arr.dtype,
                    np.dtype(dtype),
                    f"{_free_gb:.2f}GB" if _free_gb is not None else "unknown",
                    f"{_total_gb:.2f}GB" if _total_gb is not None else "unknown",
                )
                try:
                    # Thread the ALREADY-PROBED free-VRAM value through (memGetInfo is a read-only counter
                    # query - no GPU state changes between the probe above and here) so the row-chunked
                    # fallback's OWN internal memGetInfo call is skipped: this reject-path used to call it
                    # 3x total (this probe, the cushion check's internal probe, and the row-chunked one)
                    # for one decision. ``None`` (probe failed above) keeps the row-chunked function's own
                    # self-probe unchanged.
                    result = cuda_row_chunked_fn(
                        arr=arr,
                        n_bins=n_bins,
                        method=method,
                        dtype=dtype,
                        free_bytes=_free_b,
                    )
                    if method == "quantile":
                        # The row-chunked path derives quantile edges from a random SUBSAMPLE, while both
                        # other backends compute exact full-column order statistics. That is a documented,
                        # selection-equivalence-validated trade -- but WHICH path runs is decided by a
                        # transient free-VRAM probe, i.e. by what another process happens to be holding.
                        # So the same frame can be binned exactly on one run and approximately on the next,
                        # and the old message ("GPU speed preserved, VRAM-safe") never said the edges
                        # changed at all. Say it, at a level a normal run actually shows.
                        logger.warning(
                            "discretize_2d_array: completed via row-chunked CUDA -- quantile bin edges are "
                            "APPROXIMATE (from a %s-row subsample) because free VRAM was insufficient for the "
                            "exact path. Bin codes, and therefore selection, may differ from a run where the "
                            "cushion probe passed.",
                            "default-sized",
                        )
                    else:
                        # min/max is reducible across chunks, so this path is bit-identical here.
                        logger.info(
                            "discretize_2d_array: completed via row-chunked CUDA (GPU speed preserved, VRAM-safe; edges bit-identical for method=%s)",
                            method,
                        )
                    return result
                except Exception as exc:
                    logger.warning(
                        "discretize_2d_array: row-chunked CUDA also failed (%s: %s) -- falling back to CPU prange",
                        type(exc).__name__,
                        exc,
                    )
            if _vram_ok:
                try:
                    return cuda_fn(
                        arr=arr,
                        n_bins=n_bins,
                        method=method,
                        dtype=dtype,
                    )
                except Exception as exc:
                    logger.debug(
                        "discretize_2d_array: CUDA fastpath failed (%s: %s); " "falling back to CPU prange",
                        type(exc).__name__,
                        exc,
                    )
    except ImportError:
        pass
    return None

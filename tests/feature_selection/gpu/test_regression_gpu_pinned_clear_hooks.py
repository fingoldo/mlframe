"""Regression tests for the pinned-host-memory release hooks (LEAK-P2).

The clear hooks let page-locked host buffers be freed at fit completion instead of
living for the whole process. These verify the CPU-side bookkeeping; the actual
cupy DMA path is GPU-only (DOC-verified by reading on a no-GPU host).
"""

from __future__ import annotations

from collections import OrderedDict

from mlframe.feature_engineering.transformer import _kernels_cupy
from mlframe.feature_selection.filters import _gpu_resident_select


def test_clear_pinned_buffers_empties_pool_and_reports_count():
    """Clear pinned buffers empties pool and reports count."""
    _kernels_cupy._PINNED_BUFFERS.clear()
    _kernels_cupy._PINNED_BUFFERS[("t", "a", (1,), "f4")] = object()
    _kernels_cupy._PINNED_BUFFERS[("t", "b", (2,), "f4")] = object()
    released = _kernels_cupy.clear_pinned_buffers()
    assert released == 2
    assert len(_kernels_cupy._PINNED_BUFFERS) == 0
    # Idempotent: second clear releases nothing.
    assert _kernels_cupy.clear_pinned_buffers() == 0


def test_clear_pinned_d2h_drops_thread_local_buffer():
    """Clear pinned d2h drops thread local buffer."""
    # ``bufs`` (plural, a dict keyed by DMA slot) is the real attribute ``clear_pinned_d2h`` reads/clears
    # (see ``_gpu_resident_materialise.py``'s ``_pinned_view`` / ``clear_pinned_d2h``) -- a stale singular
    # ``buf`` here silently no-ops the whole test (``getattr(..., "bufs", None)`` reads None regardless).
    _gpu_resident_select._PINNED_D2H_TLS.bufs = {0: object()}
    assert _gpu_resident_select.clear_pinned_d2h() is True
    assert getattr(_gpu_resident_select._PINNED_D2H_TLS, "bufs", None) is None
    # Idempotent: nothing left to drop.
    assert _gpu_resident_select.clear_pinned_d2h() is False

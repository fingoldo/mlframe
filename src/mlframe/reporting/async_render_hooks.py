"""Glue between ``ReportRenderQueue`` and the suite's save chokepoints (``render_and_save`` and raw-Figure saves).

The queue itself knows nothing about figures. This module owns the per-thread "active queue" that the chokepoints consult,
the worker-side task functions, and the decision of whether a suite should render asynchronously at all.

What always stays synchronous (and why):

* an interactive session (notebook/REPL) -- inline ``show`` output must appear in order, in the cell that asked for it;
* ``keep_handles=True`` -- the caller wants the live figure object back;
* a render with no ``base_path`` -- nothing is saved, so there is nothing to defer;
* figures that hold a pyplot manager -- pyplot's global registry is not thread-safe, only explicit ``Figure`` objects move.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np

from mlframe.reporting._async_render import ReportRenderQueue, physical_core_count

logger = logging.getLogger(__name__)

_TLS = threading.local()

# Fewer physical cores than this and the render worker would fight training for CPU; 'auto' stays synchronous below it.
AUTO_MIN_PHYSICAL_CORES = 3


def active_render_queue() -> Optional[ReportRenderQueue]:
    """The queue the calling thread should defer saves to, or ``None`` to save synchronously."""
    return getattr(_TLS, "queue", None)


def set_active_render_queue(queue: Optional[ReportRenderQueue]) -> Optional[ReportRenderQueue]:
    """Install ``queue`` for the calling thread (``None`` clears it); returns the previous one so callers can restore it."""
    prev = active_render_queue()
    _TLS.queue = queue
    return prev


@contextmanager
def render_queue_scope(queue: Optional[ReportRenderQueue]) -> Iterator[Optional[ReportRenderQueue]]:
    """Make ``queue`` the active queue of this thread for the duration of the block."""
    prev = set_active_render_queue(queue)
    try:
        yield queue
    finally:
        set_active_render_queue(prev)


def resolve_async_render(setting: Any, *, save_charts: bool, data_dir: Optional[str]) -> bool:
    """Whether a suite should render in the background.

    ``True`` / ``False`` are taken literally (``True`` still needs charts to be saved, since with nothing written there is nothing
    to defer). ``"auto"`` turns it on when charts are saved under a data_dir and the machine has at least
    ``AUTO_MIN_PHYSICAL_CORES`` physical cores.
    """
    if setting is False or setting is None:
        return False
    if not (save_charts and data_dir):
        return False
    if setting is True:
        return True
    if str(setting).strip().lower() == "auto":
        return physical_core_count() >= AUTO_MIN_PHYSICAL_CORES
    return False


def _render_spec_task(spec: Any, output: Any, base_path: str, subfolders: bool) -> str:
    """Worker body for one FigureSpec: render + save synchronously with the submit-time layout; raise if a backend dropped the chart."""
    from mlframe.reporting.renderers.save import render_and_save_now

    failed: list = []
    render_and_save_now(spec, output, base_path, interactive=False, format_subfolders=subfolders, failed_backends=failed)
    if failed:
        raise RuntimeError(f"render failed on backend(s) {failed}")
    return base_path


def _save_figure_task(fig: Any, path: str) -> str:
    """Worker body for one raw matplotlib ``Figure`` already drawn-to in the submitting thread: save it and drop it."""
    try:
        fig.savefig(_ensure_parent(path), bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # nosec B110 - a figure with no pyplot manager has nothing to close
            logger.debug("figure close after async save failed", exc_info=True)
    return path


def _label_of(path: str) -> str:
    """File name used to label a queued artifact in failure messages (``figure`` for an empty path)."""
    base = os.path.basename(path)
    return base if base else "figure"


def _ensure_parent(path: str) -> str:
    """Create ``path``'s directory (idempotent) and return ``path``."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


# A spec is plot payload (binned curves, a 5k-point scatter), a few hundred KB; copying it at submit makes the queued render
# independent of anything the training loop does to the arrays afterwards. A spec bigger than this is handed over by reference
# instead -- the copy would cost more than the isolation is worth, and the arrays are treated as read-only.
SPEC_SNAPSHOT_MAX_BYTES = 32 * 1024 * 1024


def _tree_nbytes(obj: Any, limit: int, _depth: int = 0) -> int:
    """ndarray bytes reachable from ``obj`` through dataclasses, tuples, lists and dicts; stops counting once past ``limit``."""
    if _depth > 8:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    total = 0
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        children: Any = (getattr(obj, f.name, None) for f in dataclasses.fields(obj))
    elif isinstance(obj, (tuple, list)):
        children = obj
    elif isinstance(obj, dict):
        children = obj.values()
    else:
        return 0
    for c in children:
        total += _tree_nbytes(c, limit, _depth + 1)
        if total > limit:
            return total
    return total


def snapshot_spec(spec: Any, max_bytes: int = SPEC_SNAPSHOT_MAX_BYTES) -> Any:
    """A private copy of ``spec`` when its arrays are small enough to be worth isolating, else ``spec`` itself."""
    try:
        if _tree_nbytes(spec, max_bytes) <= max_bytes:
            return copy.deepcopy(spec)
    except Exception:
        logger.debug("spec snapshot failed; queueing it by reference", exc_info=True)
    return spec


def submit_render(queue: ReportRenderQueue, spec: Any, output: Any, base_path: str, subfolders: bool) -> None:
    """Queue one FigureSpec render; the file name in ``base_path`` labels any failure."""
    if queue.backend == "thread":
        spec = snapshot_spec(spec)
    queue.submit(_render_spec_task, spec, output, base_path, subfolders, name=_label_of(base_path))


def submit_figure_save(queue: ReportRenderQueue, fig: Any, path: str) -> bool:
    """Queue the save of an explicit (non-pyplot) ``Figure``. Returns False when the figure must be saved synchronously instead."""
    if queue.backend != "thread":
        return False  # a live Figure does not cross a process boundary cheaply
    if getattr(getattr(fig, "canvas", None), "manager", None) is not None:
        return False  # pyplot-registered: its registry is global and not thread-safe
    queue.submit(_save_figure_task, fig, path, name=_label_of(path))
    return True


def start_suite_render_queue(reporting_config: Any, *, save_charts: bool, data_dir: Optional[str], verbose: bool = True) -> Optional[ReportRenderQueue]:
    """Build the queue a suite renders through, or ``None`` when ``ReportingConfig.async_render`` resolves to off.

    Never raises: a queue that cannot be built leaves the suite on inline rendering, which is always correct.
    """
    try:
        setting = getattr(reporting_config, "async_render", False)
        if not resolve_async_render(setting, save_charts=save_charts, data_dir=data_dir):
            return None
        if str(setting).strip().lower() == "auto":
            from mlframe.reporting.renderers.save import _detect_interactive_session

            if _detect_interactive_session():
                return None  # inline display must stay in order; nothing would be deferred anyway
        backend = str(getattr(reporting_config, "async_render_backend", "thread"))
        cap_mb = float(getattr(reporting_config, "async_render_max_pending_mb", 512.0))
        queue = ReportRenderQueue(backend=backend, workers=getattr(reporting_config, "async_render_workers", None), max_pending_mb=cap_mb)
        if backend == "process":
            queue.warm()
        if verbose:
            logger.info(
                "[async-render] report rendering runs in the background: backend=%s, workers=%d, queued-bytes cap=%.0f MB "
                "(ReportingConfig.async_render=False to render inline)",
                backend, queue.workers, cap_mb,
            )
        return queue
    except Exception:
        logger.warning("[async-render] could not start the background render queue; rendering inline", exc_info=True)
        return None


def join_suite_render_queue(queue: Optional[ReportRenderQueue], metadata: Optional[Dict[str, Any]] = None, *, final: bool = False) -> None:
    """Wait for every queued artifact, log the one-line summary, and stamp ``metadata["async_render"]``.

    Runs on the suite thread. ``final=True`` (the suite's ``finally``) also closes the queue and unlinks its shared memory.
    """
    if queue is None:
        return
    try:
        summary = queue.close() if final else queue.join()
    except KeyboardInterrupt:
        queue.close(wait=False)
        raise
    if summary.submitted > queue.logged_submitted:
        # Each failure was already logged as a WARNING by the queue when it happened; the summary restates the names once.
        (logger.warning if summary.failed else logger.info)("[async-render] %s", summary.line())
        queue.logged_submitted = summary.submitted
    if isinstance(metadata, dict):
        metadata["async_render"] = {
            "backend": queue.backend,
            "workers": queue.workers,
            "artifacts": summary.completed,
            "failed": [f.name for f in summary.failures],
            "worker_seconds": round(summary.busy_seconds, 3),
            "overlapped_seconds": round(summary.overlapped_seconds, 3),
        }


def render_queued_mark() -> int:
    """Artifacts submitted to the active queue so far (0 without one): taken before a model's report, read back by ``log_render_queued``."""
    queue = active_render_queue()
    return queue.submitted if queue is not None else 0


def log_render_queued(mark: int) -> None:
    """Log how many artifacts the model just reported were queued since ``mark`` and how many are still pending (no-op without a queue)."""
    queue = active_render_queue()
    if queue is not None:
        logger.info("  render queued: %d artifact(s) for this model, %d pending in the background", queue.submitted - mark, queue.pending())

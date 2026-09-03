"""Save dispatch: render once per backend in the PlotOutputSpec, save in
all requested formats. In an interactive IPython / Jupyter session, the
figures of INTERACTIVE backends are also shown inline (session detected via
``__IPYTHON__`` builtin or ``sys.ps1``); see ``_SAVE_ONLY_BACKENDS`` for why
matplotlib is written to disk but never rendered into the cell.

File-naming policy:
- Single backend × single format: ``<base_path>.<fmt>`` (e.g. ``plot.png``).
  Mirrors the historical single-output convention, kept working for existing callers.
- Otherwise: ``<base_path>.<backend>.<fmt>`` so the operator sees which
  backend produced which file (e.g. ``plot.plotly.html`` +
  ``plot.matplotlib.pdf``).
- With ``format_subfolders`` on, each file then lands in a per-format
  SUBFOLDER of that directory (``png/plot.matplotlib.png``,
  ``html/plot.plotly.html``), so a suite emitting an interactive and a static
  copy of every figure does not leave the two kinds interleaved in one
  listing. The SUITE turns this on (``ReportingConfig.plot_format_subfolders``,
  default True); a direct library call stays flat unless it asks.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from mlframe.reporting.output import PlotOutputSpec
from mlframe.reporting.renderers.base import get_renderer
from mlframe.reporting.renderers._render_timings import chart_type_of, record_chart_render
from mlframe.reporting.spec import FigureSpec
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# Per-thread inline-display override. Backing the programmatic override with thread-local storage (rather than
# a process-global os.environ write) isolates concurrent suites: one thread forcing inline display no longer
# flips the mode of an unrelated suite running in another thread. The decision is taken on the main thread
# (see render_and_save "Main-thread post-processing"), so the set and the read are always same-thread. The
# MLFRAME_PLOT_INLINE_DISPLAY env var remains an external fallback when no thread-local override is set.
_INLINE_OVERRIDE = threading.local()
_UNSET = object()


def _thread_inline_override():
    """Tri-state per-thread override: True / False if set on this thread, else None (fall back to env)."""
    val = getattr(_INLINE_OVERRIDE, "value", _UNSET)
    return None if val is _UNSET else val

# Per-format output subfolders. A suite emitting both backends writes an interactive HTML and a static PNG of every
# figure into ONE directory, so a 4-model x VAL/TEST x N-ensemble run leaves hundreds of files of two kinds
# interleaved, and the operator who wants "the PNGs" has to filter by extension by eye. Writing each format into its
# own subfolder (``<dir>/html/name.html``, ``<dir>/png/name.png``) keeps the two kinds apart at no cost to either.
# Thread-local for the same reason the inline-display override is (concurrent suites must not flip each other's mode),
# with an env fallback for runs that cannot reach the config.
_SUBFOLDER_OVERRIDE = threading.local()
# LIBRARY default is flat, and the SUITE turns it on via ``ReportingConfig.plot_format_subfolders`` (default
# True). The mixed-directory problem this solves is a property of a suite run writing hundreds of figures in
# two formats; a direct ``render_and_save`` caller has its own on-disk contract and should not have its layout
# changed underneath it by a library upgrade.
_FORMAT_SUBFOLDERS_DEFAULT = False


def _thread_subfolder_override() -> Optional[bool]:
    """Tri-state per-thread override for the per-format subfolder layout: True / False if set, else None."""
    val = getattr(_SUBFOLDER_OVERRIDE, "value", _UNSET)
    return None if val is _UNSET else bool(val)


def set_format_subfolders(enabled: Optional[bool]) -> None:
    """Set (or clear, with ``None``) this thread's per-format-subfolder override."""
    if enabled is None:
        if hasattr(_SUBFOLDER_OVERRIDE, "value"):
            del _SUBFOLDER_OVERRIDE.value
    else:
        _SUBFOLDER_OVERRIDE.value = bool(enabled)


def get_format_subfolders() -> Optional[bool]:
    """This thread's per-format-subfolder override, or None when unset (so the caller can restore it)."""
    return _thread_subfolder_override()


def _use_format_subfolders() -> bool:
    """Whether saved files go into a per-format subfolder: thread override, else env var, else the default."""
    override = _thread_subfolder_override()
    if override is not None:
        return bool(override)
    env = os.environ.get("MLFRAME_PLOT_FORMAT_SUBFOLDERS")
    if env is not None:
        env_lo = env.strip().lower()
        if env_lo in ("1", "true", "yes", "on"):
            return True
        if env_lo in ("0", "false", "no", "off"):
            return False
    return _FORMAT_SUBFOLDERS_DEFAULT


def resolve_output_path(base_path: str, backend: str, fmt: str, *, multi_output: bool, subfolders: Optional[bool] = None) -> str:
    """Full path for one saved figure, honouring the per-format subfolder layout.

    The filename is unchanged by the layout, so a caller that knows the flat name can find the file by prepending
    the format directory rather than by re-deriving the name.
    """
    stem = f"{base_path}.{backend}.{fmt}" if multi_output else f"{base_path}.{fmt}"
    if not (subfolders if subfolders is not None else _use_format_subfolders()):
        return stem
    directory, name = os.path.split(stem)
    return os.path.join(directory, fmt, name)


# Static (non-interactive) export formats: no hover, so plotly legends must be enabled for these to be readable.
_STATIC_FORMATS = frozenset({"png", "svg", "pdf", "jpg", "jpeg"})

# Backends that are NEVER shown inline in a notebook cell -- they are save-only artifact producers.
#
# The default ``plot_outputs`` requests both backends ("plotly[html] + matplotlib[png]"), and inline display
# used to fire for each, so every chart appeared TWICE in the cell: the interactive plotly figure and a
# static matplotlib duplicate of the same data. The static copy adds nothing a reader can act on in a
# notebook (plotly already renders there, with hover), so matplotlib stays a file-producing backend: its PNG
# is still written to disk exactly as before.
_SAVE_ONLY_BACKENDS = frozenset({"matplotlib"})


def _backend_is_needed(backend: str, *, will_save: bool, interactive: bool, keep_handles: bool) -> bool:
    """Whether ``backend`` has any consumer for its figure, i.e. whether rendering it is worth the time.

    A backend earns its render when the figure will be written to disk, returned to the caller via
    ``keep_handles``, or shown inline. A save-only backend (see ``_SAVE_ONLY_BACKENDS``) is never shown, so
    with saving switched off it has NO consumer at all -- rendering it then is pure cost, which on a
    multi-model suite is seconds of matplotlib work whose output is discarded unseen.
    """
    if keep_handles or will_save:
        return True
    return interactive and backend not in _SAVE_ONLY_BACKENDS

# Process-wide counter of charts dropped by the multi-backend render thread (timeout OR exception). On timeout the
# worker thread is abandoned and the chart is silently lost; this counter (mirror of plotly's kaleido oneshot stats)
# lets the suite-end summary quote how many diagnostics went missing instead of failing silently.
_RENDER_FAILURE_COUNT = 0
_RENDER_TIMEOUT_COUNT = 0
_RENDER_EXCEPTION_COUNT = 0


def get_render_failure_stats() -> "Dict[str, int]":
    """Returns ``{"total", "timeouts", "exceptions"}`` charts dropped by ``render_and_save`` backend threads.

    Mirrors ``plotly.get_kaleido_oneshot_stats``; the suite-end summary surfaces this so silently-dropped
    diagnostics are visible. Cleared via ``reset_render_failure_stats``.
    """
    return {"total": _RENDER_FAILURE_COUNT, "timeouts": _RENDER_TIMEOUT_COUNT, "exceptions": _RENDER_EXCEPTION_COUNT}


def reset_render_failure_stats() -> None:
    """Zero the module-level render-failure counters read by ``get_render_failure_stats``. Call at the start of a suite so failure counts don't accumulate across independent runs in the same process."""
    global _RENDER_FAILURE_COUNT, _RENDER_TIMEOUT_COUNT, _RENDER_EXCEPTION_COUNT
    _RENDER_FAILURE_COUNT = 0
    _RENDER_TIMEOUT_COUNT = 0
    _RENDER_EXCEPTION_COUNT = 0


def _record_render_failure(timed_out: bool) -> None:
    """Increment the shared failure counters for one dropped chart, bucketing into the timeout or exception sub-count depending on how the backend thread failed."""
    global _RENDER_FAILURE_COUNT, _RENDER_TIMEOUT_COUNT, _RENDER_EXCEPTION_COUNT
    _RENDER_FAILURE_COUNT += 1
    if timed_out:
        _RENDER_TIMEOUT_COUNT += 1
    else:
        _RENDER_EXCEPTION_COUNT += 1


def _detect_interactive_session() -> bool:
    """True iff we're inside an IPython kernel or interactive Python REPL.

    Resolution order:
    1. ``MLFRAME_PLOT_INLINE_DISPLAY`` env var, if set — explicit override.
       Truthy (``1`` / ``true`` / ``yes`` / ``on``) → True; falsy
       (``0`` / ``false`` / ``no`` / ``off``) → False. Case-insensitive.
       Useful for batch jupyter runs (papermill, nbconvert, scheduled
       notebooks) that want save-only despite ``__IPYTHON__`` being set,
       or for non-standard runtimes where auto-detection misfires.
    2. ``__IPYTHON__`` builtin (set by IPython / Jupyter kernels).
    3. ``sys.ps1`` (set by the bare REPL).

    The naive ``"IPython" in sys.modules`` heuristic is unreliable —
    matplotlib + many ML libraries drag IPython in as a transitive dep
    even from plain Python scripts (giving false positives).

    A per-thread override set via ``set_inline_display_mode`` takes precedence over all of the below.
    """
    _override = _thread_inline_override()
    if _override is not None:
        return bool(_override)
    env = os.environ.get("MLFRAME_PLOT_INLINE_DISPLAY")
    if env is not None:
        env_lo = env.strip().lower()
        if env_lo in ("1", "true", "yes", "on"):
            return True
        if env_lo in ("0", "false", "no", "off"):
            return False
        # Unrecognized value falls through to auto-detect rather than
        # silently treating as True — operator typo shouldn't accidentally
        # force inline display.
    try:
        return bool(__IPYTHON__)  # type: ignore[name-defined]
    except NameError:
        import sys
        return hasattr(sys, "ps1")


def set_inline_display_mode(mode):
    """Set the inline-display override for the CURRENT thread.

    Stored in thread-local state (not the process-wide env), so concurrent suites in different threads do
    not clobber each other's mode. Takes precedence over the ``MLFRAME_PLOT_INLINE_DISPLAY`` env var, which
    still applies (per-thread) when no override is set. ``mode``:
      - ``True``  → force inline display (overrides auto-detect).
      - ``False`` → force save-only (overrides auto-detect).
      - ``None``  → clear the override; ``_detect_interactive_session``
        falls back to the env var then ``__IPYTHON__`` / ``sys.ps1`` auto-detect.

    Used by ``train_mlframe_models_suite`` to honor
    ``ReportingConfig.plot_inline_display``.
    """
    if mode not in (None, True, False):
        raise ValueError(f"set_inline_display_mode(mode={mode!r}): expected True, " "False, or None")
    if mode is None:
        _INLINE_OVERRIDE.value = _UNSET
    else:
        _INLINE_OVERRIDE.value = mode


def get_inline_display_mode():
    """Return the current inline-display override for this thread.

    Returns the same tri-state set_inline_display_mode accepts:
      - the per-thread override (``True`` / ``False``) if one is set on this thread;
      - else ``True`` / ``False`` parsed from MLFRAME_PLOT_INLINE_DISPLAY if set;
      - else ``None`` (auto-detect path).

    Used by the suite's snapshot+restore wrap so a per-suite override is reverted
    at suite finish rather than leaking into the next suite call.
    """
    _override = _thread_inline_override()
    if _override is not None:
        return _override
    _val = os.environ.get("MLFRAME_PLOT_INLINE_DISPLAY")
    if _val is None:
        return None
    _norm = _val.strip().lower()
    if _norm in ("1", "true", "yes", "y", "on"):
        return True
    if _norm in ("0", "false", "no", "n", "off"):
        return False
    # Unrecognised env value -> treat as unset.
    return None


def render_and_save(
    spec: FigureSpec,
    output: PlotOutputSpec,
    base_path: str,
    *,
    keep_handles: bool = False,
    interactive: Optional[bool] = None,
    format_subfolders: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Render the spec on each backend in ``output`` and save in all formats.

    Parameters
    ----------
    spec : FigureSpec
        Pure-data chart spec (rendered once per backend).
    output : PlotOutputSpec
        Parsed DSL describing backends + formats.
    base_path : str
        Filesystem path stem (no extension). Each saved file appends
        ``.<fmt>`` (single backend / single format) or
        ``.<backend>.<fmt>`` (multiple).
    keep_handles : bool
        When True, return a dict mapping ``backend -> native fig handle``
        so callers can show / further-tweak the figures. Default False
        releases handles for matplotlib (frees memory; matplotlib leaks
        ~1MB per figure in long-running suites).
    format_subfolders : bool, optional
        When True, each file is written to a per-format subdirectory of
        ``base_path``'s directory (``png/plot.png``, ``html/plot.html``)
        instead of alongside every other format. ``None`` (default) reads
        the thread override the suite sets from
        ``ReportingConfig.plot_format_subfolders``, then
        ``MLFRAME_PLOT_FORMAT_SUBFOLDERS``, then the module default (False --
        see that constant for why the library stays flat).
    interactive : bool, optional
        When True, also call ``renderer.show(fig)`` per backend so the
        figure renders inline in the notebook cell (in addition to the
        on-disk save). When ``None`` (default), auto-detected via
        ``__IPYTHON__`` builtin / ``sys.ps1``. When ``False``, save-only.

    Returns
    -------
    dict or None
        ``{backend: native_fig}`` when ``keep_handles=True``, else None.
    """
    if interactive is None:
        interactive = _detect_interactive_session()
    _subfolders = _use_format_subfolders() if format_subfolders is None else bool(format_subfolders)

    # An empty ``base_path`` is how callers say "do not persist this figure"; with nothing to write, a
    # save-only backend has no consumer left and is skipped entirely rather than rendered and discarded.
    will_save = bool(base_path)
    _backends = [(b, f) for b, f in output.backends if _backend_is_needed(b, will_save=will_save, interactive=bool(interactive), keep_handles=keep_handles)]
    handles: Dict[str, Any] = {}
    if not _backends:
        return handles if keep_handles else None

    # Filtering cannot change the on-disk names: a backend is only dropped when nothing is being saved.
    multi_output = (len(_backends) > 1) or any(len(fmts) > 1 for _, fmts in _backends)

    # Parallelize render+save across backends: each builds its OWN renderer
    # + fig from the frozen FigureSpec (no shared mutable state). Both Agg
    # (matplotlib) and write_html (plotly) release the GIL during the heavy
    # C work, so ThreadPoolExecutor yields real wall-clock speedup. show()
    # + plt.close() stay on the main thread (pyplot global state + Jupyter
    # display hooks are not thread-friendly).

    def _do_backend(backend: str, fmts) -> "Tuple[str, Any]":
        """Render ``spec`` once on ``backend`` and save it to every format in ``fmts``; runs on a worker thread so multiple backends render+save concurrently (see the note above on GIL release during Agg/write_html)."""
        # Timed per backend rather than per call: the two backends render concurrently, so one wall time for both
        # would attribute the slower one's cost to the faster one as well.
        _t0 = time.perf_counter()
        renderer = get_renderer(backend)
        # plotly legends default off (hover identifies series interactively); enable them when a static format
        # is in this backend's save set since a png/svg/pdf export has no hover.
        if backend == "plotly" and (set(fmts) & _STATIC_FORMATS):
            fig = renderer.render(spec, static_legend=True)
        else:
            fig = renderer.render(spec)
        # ``will_save`` gates the WRITE too, not just the backend selection above. A backend kept alive by
        # ``keep_handles`` or an interactive session still reached this loop with an empty ``base_path``, and
        # ``resolve_output_path`` then composed a name out of the extension alone -- writing ``.matplotlib.png``
        # and ``.html`` into the process's working directory. Dot-prefixed, so ``ls`` never showed them.
        for fmt in fmts if will_save else ():
            path = resolve_output_path(base_path, backend, fmt, multi_output=multi_output, subfolders=_subfolders)
            _dir = os.path.dirname(path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            renderer.save(fig, path, fmt)
        record_chart_render(chart_type_of(base_path), time.perf_counter() - _t0, backend=backend)
        return backend, fig

    if len(_backends) > 1:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
        # max_workers = backend count; each task = one render+save pipeline.
        with ThreadPoolExecutor(max_workers=len(_backends)) as _ex:
            _futures = [_ex.submit(_do_backend, backend, fmts) for backend, fmts in _backends]
            _results = []
            for f in _futures:
                try:
                    _results.append(f.result(timeout=60))
                except _FutureTimeout:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                    _record_render_failure(timed_out=True)
                    log_throttle(
                        logger, "render_save_backend_future_timeout", logging.WARNING,
                        "render_and_save: backend future exceeded 60s; the worker thread is abandoned and one "
                        "chart is dropped. See get_render_failure_stats(). ", exc_info=True,
                    )
                except Exception:
                    _record_render_failure(timed_out=False)
                    log_throttle(
                        logger, "render_save_backend_future_failed", logging.WARNING,
                        "render_and_save: backend future failed; one render output dropped. " "See get_render_failure_stats().",
                        exc_info=True,
                    )
    else:
        # Single-backend path: skip the thread pool overhead, but still go through the SAME try/except +
        # _record_render_failure bookkeeping the multi-backend path uses a few lines above -- this is the
        # COMMON case (most call sites request one backend), and a bare list comprehension here previously
        # neither counted a rendering exception in get_render_failure_stats() (undercounting the suite-end
        # "N charts silently dropped" summary) nor caught it, so it propagated to whatever called
        # render_and_save -- some call sites outside this cluster invoke it with no wrapping try/except of
        # their own.
        _results = []
        for backend, fmts in _backends:
            try:
                _results.append(_do_backend(backend, fmts))
            except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate (see the multi-backend branch above)
                _record_render_failure(timed_out=False)
                log_throttle(
                    logger, "render_save_single_backend_failed", logging.WARNING,
                    "render_and_save: single-backend render failed; one render output dropped. " "See get_render_failure_stats().",
                    exc_info=True,
                )

    # Main-thread post-processing: interactive show + cleanup. Both touch
    # pyplot / Jupyter display hooks that are NOT thread-safe.
    for backend, fig in _results:
        # Save-only backends are never shown inline: with the default two-backend ``plot_outputs`` the cell
        # otherwise received the interactive plotly figure AND a static matplotlib duplicate of the same
        # data. Their files are written exactly as before -- only the redundant cell output is dropped.
        if interactive and backend not in _SAVE_ONLY_BACKENDS:
            try:
                renderer = get_renderer(backend)
                renderer.show(fig)
            except Exception as e:
                logger.debug(
                    "render_and_save: %s renderer.show() failed (%s: %s); "
                    "on-disk save unaffected", backend, type(e).__name__, e,
                )
        if keep_handles:
            handles[backend] = fig
        elif backend == "matplotlib":
            # Unconditional now: the figure was never handed to a display hook, so nothing else holds a
            # reference. Previously the interactive branch left it open, leaking ~1MB per chart in a
            # notebook session precisely where suites emit the most figures.
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:  # nosec B110 - optional dependency import guard
                logger.debug("matplotlib figure close failed: %s", e)

    return handles if keep_handles else None


__all__ = [
    "render_and_save",
    "set_inline_display_mode",
    "get_inline_display_mode",
    "get_render_failure_stats",
    "reset_render_failure_stats",
]

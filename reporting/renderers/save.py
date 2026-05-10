"""Save dispatch: render once per backend in the PlotOutputSpec, save in
all requested formats. In an interactive IPython / Jupyter session, the
figures are ALSO shown inline before save so the operator sees the
plot in the notebook cell (verified detected via ``__IPYTHON__`` builtin
or ``sys.ps1``).

File-naming policy:
- Single backend × single format: ``<base_path>.<fmt>`` (e.g. ``plot.png``).
  Mirrors the pre-2026-05-08 single-output convention.
- Otherwise: ``<base_path>.<backend>.<fmt>`` so the operator sees which
  backend produced which file (e.g. ``plot.plotly.html`` +
  ``plot.matplotlib.pdf``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from mlframe.reporting.output import PlotOutputSpec
from mlframe.reporting.renderers.base import get_renderer
from mlframe.reporting.spec import FigureSpec

logger = logging.getLogger(__name__)


def _detect_interactive_session() -> bool:
    """True iff we're inside an IPython kernel or interactive Python REPL.

    ``__IPYTHON__`` is a builtin set by IPython / Jupyter kernels;
    ``sys.ps1`` is set by the bare REPL. The naive
    ``"IPython" in sys.modules`` heuristic is unreliable — matplotlib
    + many ML libraries drag IPython in as a transitive dep even from
    plain Python scripts (giving false positives).
    """
    try:
        return bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        import sys
        return hasattr(sys, "ps1")


def render_and_save(
    spec: FigureSpec,
    output: PlotOutputSpec,
    base_path: str,
    *,
    keep_handles: bool = False,
    interactive: Optional[bool] = None,
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

    multi_output = (len(output.backends) > 1) or any(
        len(fmts) > 1 for _, fmts in output.backends
    )
    handles: Dict[str, Any] = {}

    for backend, fmts in output.backends:
        renderer = get_renderer(backend)
        fig = renderer.render(spec)
        for fmt in fmts:
            if multi_output:
                path = f"{base_path}.{backend}.{fmt}"
            else:
                path = f"{base_path}.{fmt}"
            renderer.save(fig, path, fmt)
        # Inline display for interactive sessions. Done BEFORE the
        # matplotlib close-on-release path below so plt.close() doesn't
        # destroy the figure mid-display. plotly's fig.show() in
        # IPython routes through plotly.io.renderers (default
        # 'notebook' inside Jupyter, 'browser' from REPL); matplotlib's
        # show() is also IPython-aware.
        if interactive:
            try:
                renderer.show(fig)
            except Exception as e:
                # Non-fatal: save already completed. Log once per backend
                # so the user knows inline display failed but the on-disk
                # artifact is still there.
                logger.debug(
                    "render_and_save: %s renderer.show() failed (%s: %s); "
                    "on-disk save unaffected", backend, type(e).__name__, e,
                )
        if keep_handles:
            handles[backend] = fig
        elif backend == "matplotlib" and not interactive:
            # Close matplotlib figs explicitly to release memory.
            # Skip when interactive — Jupyter inline backend keeps a
            # reference for the cell render; closing here would erase it.
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    return handles if keep_handles else None


__all__ = ["render_and_save"]

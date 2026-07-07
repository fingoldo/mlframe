"""Renderer Protocol + factory.

Backends are lazy-imported (matplotlib + plotly are heavy modules) so
importing the renderer package doesn't pull both in unconditionally.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mlframe.reporting.spec import FigureSpec


@runtime_checkable
class Renderer(Protocol):
    """Backend renderer Protocol.

    Implementations build a native figure (matplotlib ``Figure`` or
    plotly ``go.Figure``) from the spec, save to a file in a given
    format, and optionally show interactively.
    """

    backend: str  # "matplotlib" | "plotly"

    def render(self, spec: FigureSpec, *, static_legend: bool = False) -> Any:
        """Build native figure handle from the spec.

        ``static_legend`` is honoured by the plotly backend (see
        ``PlotlyRenderer.render``); other backends accept and ignore it.
        """
        ...

    def save(self, fig: Any, path: str, fmt: str) -> None:
        """Write figure to ``path`` in format ``fmt``.

        Raises ``ValueError`` if the renderer doesn't support the
        format (caller should consult ``BACKEND_FORMATS`` first).
        """
        ...

    def show(self, fig: Any) -> None:
        """Open the figure in an interactive viewer."""
        ...


def get_renderer(backend: str) -> Renderer:
    """Lazy-import the requested backend renderer."""
    backend = backend.lower()
    if backend == "matplotlib":
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        return MatplotlibRenderer()
    if backend == "plotly":
        from mlframe.reporting.renderers.plotly import PlotlyRenderer
        return PlotlyRenderer()
    raise ValueError(f"unknown renderer backend {backend!r}; " "supported: matplotlib, plotly")


__all__ = ["Renderer", "get_renderer"]

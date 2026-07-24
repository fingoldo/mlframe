"""Low-level plotly colour / axis helpers carved out of ``plotly.py`` to keep the renderer module under the house
LOC limit. Pure helpers (colormap-name mapping, rgba coercion, subplot axis-ref); no PlotlyRenderer dependency."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Map matplotlib colormap BASE names to plotly's named colorscales so the plotly backend renders the same scale as matplotlib.
# Keys are lowercased; the resolver lowercases its input and strips a trailing ``_r`` (reversed), re-appending it to the plotly name.
_MPL_TO_PLOTLY = {
    "rdylbu": "RdYlBu",
    "rdylgn": "RdYlGn",
    "rdbu": "RdBu",
    "reds": "Reds",
    "blues": "Blues",
    "greens": "Greens",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "magma": "Magma",
    "inferno": "Inferno",
    "coolwarm": "RdBu_r",  # mpl coolwarm has no plotly twin; RdBu_r is the closest diverging blue-low/red-high match
}


def _axis_ref(fig, row: int, col: int) -> str:
    """x-axis reference string for the subplot at (row, col), e.g. ``"x"`` / ``"x4"`` — for scaleanchor."""
    try:
        n_cols = len(fig._grid_ref[0])
        idx = (row - 1) * n_cols + col
    except Exception:
        idx = 1
    return "x" if idx == 1 else f"x{idx}"


def _rgba(color: str, alpha: float) -> str:
    """Best-effort named/hex color -> rgba() string with the given alpha; leaves rgb()/rgba() as-is."""
    c = str(color)
    if c.startswith("rgba(") or c.startswith("rgb("):
        return c
    try:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(c)
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"
    except Exception as exc:
        logger.debug("_rgba: color conversion failed for %r, passing through unchanged: %s", color, exc)
        return c


def _mpl_to_plotly_cmap(name: str) -> str:
    """Map a matplotlib colormap name to a plotly colorscale name.

    Case-insensitive (matplotlib resolves names case-insensitively); a trailing ``_r`` is stripped, the base looked up, and ``_r``
    re-appended so the plotly scale is reversed the same way matplotlib reverses. Falls back to 'Viridis' for genuinely-unknown
    names with a WARN — the goal is zero warnings for cmaps the charts actually request.
    """
    key = str(name).lower()
    reversed_suffix = ""
    if key.endswith("_r"):
        key = key[:-2]
        reversed_suffix = "_r"
    base = _MPL_TO_PLOTLY.get(key)
    if base is not None:
        # XOR the request's reversal with any reversal baked into the mapped name (e.g. coolwarm -> RdBu_r).
        want_reversed = bool(reversed_suffix) ^ base.endswith("_r")
        plain = base[:-2] if base.endswith("_r") else base
        return plain + "_r" if want_reversed else plain
    logger.warning(
        "Unknown colormap %r; falling back to plotly 'Viridis'. "
        "Add the mapping to mlframe.reporting.renderers._plotly_color._MPL_TO_PLOTLY "
        "to silence this warning.",
        name,
    )
    return "Viridis"

"""Plot-output DSL: ``"plotly[html,png] + matplotlib[pdf]"``.

The DSL describes one or more (backend, formats) pairs separated by ``+``.
Each backend clause is ``<backend>[<fmt1>,<fmt2>,...]``. Whitespace is
tolerated everywhere.

Examples:
    "plotly[html]"                       — single backend, single format
    "plotly[html,png]"                   — single backend, two formats
    "plotly[html] + matplotlib[pdf]"     — two backends; render twice
    "matplotlib[png]"                    — back-compat with pre-2026-05-08 default

Validation:
- Backend ∈ {"matplotlib", "plotly"} (room for "bokeh" later)
- Per-backend format allowlist (matplotlib can't write html, plotly can't
  write jpeg, etc.)
- No duplicate backends in one DSL
- No duplicate formats within one backend
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

# Per-backend allowed output formats.
# matplotlib: ``Figure.savefig`` -> static raster + vector
# plotly: write_html (interactive) + write_image (kaleido) + to_json
BACKEND_FORMATS: Dict[str, FrozenSet[str]] = {
    "matplotlib": frozenset({"png", "pdf", "svg", "jpg", "jpeg"}),
    "plotly": frozenset({"html", "png", "svg", "pdf", "json"}),
}


@dataclass(frozen=True)
class PlotOutputSpec:
    """Parsed output specification.

    Attributes
    ----------
    raw : str
        Original DSL string (kept for diagnostics + round-trip).
    backends : tuple of (backend_name, formats)
        Parsed and validated. ``formats`` is a frozenset for stable equality
        + hashability.
    """

    raw: str
    backends: Tuple[Tuple[str, FrozenSet[str]], ...]


# Match a single backend clause: "plotly[html,png]" with optional whitespace.
# Group 1 = backend name (alpha only), Group 2 = format list inside [].
_CLAUSE_RE = re.compile(r"^\s*([A-Za-z]+)\s*\[\s*([A-Za-z0-9_,\s]+?)\s*\]\s*$")
# Split the plot_outputs DSL on '+' with optional surrounding whitespace.
_CLAUSE_SPLIT_RE = re.compile(r"\s*\+\s*")


def parse_plot_output_dsl(s: str) -> PlotOutputSpec:
    """Parse and validate a plot-output DSL string.

    Raises ``ValueError`` on any of:
    - empty string
    - unparseable clause (missing ``[`` / ``]`` etc.)
    - unknown backend
    - unknown format for the chosen backend
    - duplicate backends in one DSL
    - duplicate formats within one backend
    """
    if not s or not s.strip():
        raise ValueError("plot_outputs DSL is empty")

    raw = s
    # Split on '+' (allowing whitespace around it). Keep order so callers
    # can rely on (matplotlib first → primary) when both are requested.
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(s) if c.strip()]
    if not clauses:
        raise ValueError(f"plot_outputs DSL has no clauses: {s!r}")

    parsed: List[Tuple[str, FrozenSet[str]]] = []
    seen_backends: set = set()
    for clause in clauses:
        m = _CLAUSE_RE.match(clause)
        if m is None:
            raise ValueError(f"plot_outputs clause {clause!r} is malformed; " "expected '<backend>[<fmt1>,<fmt2>,...]'.")
        backend = m.group(1).lower()
        if backend not in BACKEND_FORMATS:
            raise ValueError(f"plot_outputs backend {backend!r} not supported. " f"Allowed: {sorted(BACKEND_FORMATS)}")
        if backend in seen_backends:
            raise ValueError(f"plot_outputs lists backend {backend!r} more than once.")
        seen_backends.add(backend)

        # Parse format list.
        raw_fmts = [f.strip().lower() for f in m.group(2).split(",") if f.strip()]
        if not raw_fmts:
            raise ValueError(f"plot_outputs clause {clause!r} declares no formats.")
        if len(raw_fmts) != len(set(raw_fmts)):
            dupes = sorted({f for f in raw_fmts if raw_fmts.count(f) > 1})
            raise ValueError(f"plot_outputs clause {clause!r} has duplicate format(s): {dupes}")

        # Per-backend format compat.
        allowed = BACKEND_FORMATS[backend]
        unknown = [f for f in raw_fmts if f not in allowed]
        if unknown:
            raise ValueError(f"plot_outputs backend {backend!r} does not support format(s) " f"{unknown}. Allowed for {backend}: {sorted(allowed)}.")

        parsed.append((backend, frozenset(raw_fmts)))

    return PlotOutputSpec(raw=raw, backends=tuple(parsed))


__all__ = [
    "PlotOutputSpec",
    "parse_plot_output_dsl",
    "BACKEND_FORMATS",
]

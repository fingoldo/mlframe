"""Decile gains/lift/KS table for a binary scorer, and its figure.

Carved out of ``binary.py``, which had grown past the house carve band. This pair is the one part of that module
that does NOT go through the spec layer: the figure is drawn with direct matplotlib calls, because it is a TABLE
rather than a plot and no PanelSpec expresses a table. Keeping it beside the spec-based panel builders made the
module read as if the spec layer were optional; keeping it in its own module makes the exception explicit and
contained. Both names are re-exported from ``binary`` so no call site changes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._binary_shared import _finite_binary

logger = logging.getLogger(__name__)


def binary_decile_table(y_true: np.ndarray, y_score: np.ndarray, *, n_deciles: int = 10) -> Dict[str, np.ndarray]:
    """Per-decile gains/lift/KS table for a binary scorer (score-sorted, decile 1 = highest scores).

    Returns a dict of length-``n_deciles`` arrays:
        decile        : 1..n_deciles (1 = top-scored group)
        count         : rows in the decile
        positives     : positive rows in the decile
        response_rate : positives / count
        gain          : cumulative fraction of all positives captured by deciles 1..d
        lift          : (cumulative positive rate through decile d) / overall prevalence
        cum_ks        : cumulative |%positives - %negatives| captured through decile d (the decile-resolution KS)

    The gains/lift toolkit that completes the existing lift curve; the integrator surfaces this
    as a metrics table rather than a chart panel.
    """
    yt, ys = _finite_binary(y_true, y_score)
    n = len(yt)
    out = {
        "decile": np.arange(1, n_deciles + 1, dtype=np.int64),
        "count": np.zeros(n_deciles, dtype=np.int64),
        "positives": np.zeros(n_deciles, dtype=np.int64),
        "response_rate": np.full(n_deciles, np.nan),
        "gain": np.full(n_deciles, np.nan),
        "lift": np.full(n_deciles, np.nan),
        "cum_ks": np.full(n_deciles, np.nan),
    }
    if n == 0:
        return out
    order = np.argsort(ys, kind="stable")[::-1]
    y_desc = yt[order].astype(np.int64)
    # Split the score-sorted rows into ~equal deciles via integer boundaries (handles n not divisible by n_deciles).
    bounds = (np.arange(n_deciles + 1) * n / n_deciles).round().astype(np.int64)
    n_pos_total = int(y_desc.sum())
    n_neg_total = n - n_pos_total
    prevalence = n_pos_total / n if n else 0.0
    cum_pos = 0
    cum_neg = 0
    cum_count = 0
    for d in range(n_deciles):
        lo, hi = bounds[d], bounds[d + 1]
        seg = y_desc[lo:hi]
        cnt = len(seg)
        pos = int(seg.sum())
        out["count"][d] = cnt
        out["positives"][d] = pos
        if cnt > 0:
            out["response_rate"][d] = pos / cnt
        cum_pos += pos
        cum_neg += cnt - pos
        cum_count += cnt
        if n_pos_total > 0:
            out["gain"][d] = cum_pos / n_pos_total
        if prevalence > 0 and cum_count > 0:
            out["lift"][d] = (cum_pos / cum_count) / prevalence
        frac_pos = cum_pos / n_pos_total if n_pos_total > 0 else 0.0
        frac_neg = cum_neg / n_neg_total if n_neg_total > 0 else 0.0
        out["cum_ks"][d] = abs(frac_pos - frac_neg)
    return out


# Columns drawn in the decile table, in order: (table-header, source-key-or-None, value-formatter).
_DECILE_TABLE_COLUMNS: Tuple[Tuple[str, str, Callable], ...] = (
    ("decile", "decile", lambda v: f"{int(v)}"),
    ("n", "count", lambda v: f"{int(v):,}"),
    ("positives", "positives", lambda v: f"{int(v):,}"),
    ("response", "response_rate", lambda v: "-" if not np.isfinite(v) else f"{v:.1%}"),
    ("cum gain", "gain", lambda v: "-" if not np.isfinite(v) else f"{v:.1%}"),
    ("lift", "lift", lambda v: "-" if not np.isfinite(v) else f"{v:.2f}"),
    ("cum KS", "cum_ks", lambda v: "-" if not np.isfinite(v) else f"{v:.3f}"),
)


def binary_decile_table_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_deciles: int = 10,
    highlight_top: int = 3,
    title: str = "Decile gain / lift table",
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """Render the score-sorted decile gain/lift/KS table (decile 1 = top scores) as a styled matplotlib table figure.

    The tabular complement to the GAIN curve: stakeholders read the exact per-decile capture / lift / cumulative-KS
    numbers a curve only shows graphically. All numbers come from ONE call to ``binary_decile_table`` (a single
    O(n log n) score sort) -- no per-decile rescans. The top ``highlight_top`` deciles are tinted, the cumulative-gain
    column carries a light value-proportional shade, and a TOTAL row sums n / positives with the overall response rate.

    Edge cases mirror the iter-1 guard style: a single-class target (gain/lift undefined) or fewer than ``n_deciles``
    finite rows renders a centered annotation instead of a misleading table. Returns a matplotlib ``Figure`` (the
    SHAP-style direct-matplotlib path; the heavy aggregation stays spec-pure in ``binary_decile_table``).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    yt, ys = _finite_binary(y_true, y_score)
    n = len(yt)
    n_pos = int(yt.sum()) if n else 0

    def _annotated(msg: str):
        """Render a bare, title-only figure carrying a centered explanatory message in place of a table (degenerate-input fallback)."""
        fig = Figure(figsize=(8.0, 2.4) if figsize is None else figsize)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return fig

    if n == 0:
        return _annotated("Decile table undefined\n(no finite (label, score) pairs)")
    if n_pos == 0 or n_pos == n:
        return _annotated("Decile gain / lift undefined\n(only one class present)")
    # With fewer rows than deciles every decile would hold <=1 row -- the per-decile rates are noise; bin to n rows.
    eff_deciles = n_deciles if n >= n_deciles else max(1, n)
    note = "" if n >= n_deciles else f" (n={n} < {n_deciles}: {eff_deciles} bins)"

    tbl = binary_decile_table(yt, ys, n_deciles=eff_deciles)
    n_rows = len(tbl["decile"])

    col_headers = [c[0] for c in _DECILE_TABLE_COLUMNS]
    cells: List[List[str]] = [[fmt(tbl[key][d]) for _, key, fmt in _DECILE_TABLE_COLUMNS] for d in range(n_rows)]
    total_pos = int(tbl["positives"].sum())
    total_n = int(tbl["count"].sum())
    total_resp = total_pos / total_n if total_n else float("nan")
    # TOTAL row: cumulative gain/KS are 100% / 0 by construction at the full population; lift is 1.0 (the baseline).
    total_row = ["TOTAL", f"{total_n:,}", f"{total_pos:,}", "-" if not np.isfinite(total_resp) else f"{total_resp:.1%}", "100.0%", "1.00", "0.000"]
    cells.append(total_row)

    fig = Figure(figsize=(8.0, 0.42 * (n_rows + 3)) if figsize is None else figsize)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title + note, fontsize=11)
    table = ax.table(cellText=cells, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    gain_col = col_headers.index("cum gain")
    gain_vals = tbl["gain"]
    header_color = "#34495e"
    highlight = "#fff3cd"
    total_color = "#d6eaf8"
    gain_shade = (0.66, 0.78, 0.91)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == n_rows + 1:
            cell.set_facecolor(total_color)
            cell.set_text_props(fontweight="bold")
        else:
            d = row - 1
            if col == gain_col and np.isfinite(gain_vals[d]):
                a = 0.18 + 0.55 * float(gain_vals[d])
                cell.set_facecolor((gain_shade[0], gain_shade[1], gain_shade[2], a))
            elif d < highlight_top:
                cell.set_facecolor(highlight)
            else:
                cell.set_facecolor("white")
    return fig


__all__ = ["binary_decile_table", "binary_decile_table_figure"]

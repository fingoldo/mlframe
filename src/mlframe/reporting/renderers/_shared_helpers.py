"""Renderer-agnostic helpers shared by the matplotlib and plotly renderers.

Both renderers thin heatmap tick labels the same way and need the same
finite value-range over a matrix for cell-text color resolution; the single
implementation lives here so the two backends can't drift.
"""
from __future__ import annotations

import numpy as np

# A density heatmap bins into ~80x80 cells, so one tick per cell-label overlaps into unreadable soup. Above this
# many labels, show at most this many evenly-spaced ticks (the rest of the grid is still drawn).
_HEATMAP_MAX_TICKS = 8


def _thin_tick_positions(n: int, max_ticks: int = _HEATMAP_MAX_TICKS):
    """Evenly-spaced tick indices for an axis of ``n`` labels, always including the first and last."""
    if n <= max_ticks:
        return list(range(n))
    return sorted({round(i * (n - 1) / (max_ticks - 1)) for i in range(max_ticks)})


def _finite_range(mat):
    """``(vmin, vmax)`` over finite entries, or ``None`` when the matrix is empty / all non-finite.

    Heatmap cell-text color resolution needs a real value range; ``np.nanmin`` raises on an empty array and
    returns NaN on an all-NaN matrix, so callers gate the per-cell text loop on a non-None result.
    """
    a = np.asarray(mat, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def _per_series_flags(flag, n: int):
    """Normalize a per-series bool flag (single bool / tuple / None) into a length-n bool list."""
    if flag is None:
        return [False] * n
    if isinstance(flag, (tuple, list, np.ndarray)):
        seq = list(flag)
        return [bool(seq[i]) if i < len(seq) else False for i in range(n)]
    return [bool(flag)] * n


# Panel-title wrapping. Both backends fold a long diagnostic title onto several lines; the chars-per-line
# budget below was calibrated for one panel of ``_TITLE_REF_WIDTH_IN`` inches but used to be applied at ANY
# panel width, so a wide figure folded its title into a narrow ragged column using a fraction of the space.
_PANEL_TITLE_WRAP_CHARS = 46
_TITLE_REF_WIDTH_IN = 6.0


def panel_title_wrap_chars(figsize, cols: int = 1) -> int:
    """Chars-per-line budget for a panel title, scaled to that panel's actual width.

    ``figsize`` is the whole figure's (width, height) in inches; ``cols`` the grid's column count, so the
    per-panel width is ``figsize[0] / cols``. Returns the calibration constant unchanged for a panel of the
    reference width, and grows/shrinks proportionally either side of it. A non-subscriptable / malformed
    ``figsize`` falls back to the reference width rather than raising -- this only controls text layout.
    """
    try:
        panel_w = float(figsize[0]) / max(int(cols), 1)
    except (TypeError, IndexError, ValueError, ZeroDivisionError):
        panel_w = _TITLE_REF_WIDTH_IN
    # Floor keeps a very narrow panel's title from degenerating into one word per line.
    return max(20, round(_PANEL_TITLE_WRAP_CHARS * panel_w / _TITLE_REF_WIDTH_IN))


def wrap_title_lines(text, width: int) -> list:
    """Wrap ``text`` to ``width`` chars/line, wrapping each ``\n``-delimited segment INDEPENDENTLY.

    Preserving explicit breaks matters: ``textwrap.wrap`` treats a newline as ordinary whitespace, so
    feeding it a title that already carries deliberate breaks silently collapses and re-flows them. Callers
    build these titles with intentional structure (e.g. one line per metric family), which must survive.
    """
    import textwrap

    out: list = []
    for line in str(text).split("\n"):
        out.extend(textwrap.wrap(line, width=width, break_long_words=False) or [""])
    return out


def epoch_ns_ticks(x_values, n_ticks: int = 6):
    """``(tickvals, ticktext)`` rendering an epoch-NANOSECOND x axis as human-readable dates.

    Spec builders that plot a metric against time hand the renderers ``int64`` nanoseconds (a numeric x is
    what lets vspans / regime shading share the same coordinate space) and set ``x_is_time`` to say "these
    are timestamps". Before this helper existed, ``x_is_time`` only ROTATED the tick labels -- nothing ever
    converted the numbers back -- so a time axis rendered as ``1.62e18 ... 1.78e18``, which carries no
    usable information for a reader.

    Returns ``(None, None)`` when there is nothing to format, so callers leave the axis untouched. That
    includes the case where x is ALREADY datetime-like: ``x_is_time`` marks both representations (builders
    may pass real ``datetime`` objects instead of epoch integers), and both renderers format genuine
    datetime axes natively -- only the numeric-epoch form needs help.
    """
    try:
        arr = np.asarray(x_values, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None, None  # datetime objects / strings: the backend's own date axis handles these
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    span_days = (hi - lo) / 8.64e13  # ns per day
    # Pick the coarsest format that still separates adjacent ticks, so labels stay short and unambiguous.
    if span_days > 730:
        fmt = "%Y-%m"
    elif span_days > 2:
        fmt = "%Y-%m-%d"
    else:
        fmt = "%m-%d %H:%M"
    tickvals = np.linspace(lo, hi, max(2, int(n_ticks)))
    import datetime as _dt

    # Explicit UTC rather than the deprecated naive ``utcfromtimestamp``; these axes are wall-clock labels,
    # so a fixed reference zone keeps them stable regardless of the machine rendering the figure.
    ticktext = [_dt.datetime.fromtimestamp(v / 1e9, tz=_dt.timezone.utc).strftime(fmt) for v in tickvals]
    return tickvals, ticktext

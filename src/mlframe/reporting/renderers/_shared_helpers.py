"""Renderer-agnostic helpers shared by the matplotlib and plotly renderers.

Both renderers thin heatmap tick labels the same way and need the same
finite value-range over a matrix for cell-text color resolution; the single
implementation lives here so the two backends can't drift.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# A density heatmap bins into ~80x80 cells, so one tick per cell-label overlaps into unreadable soup. Above this
# many labels, show at most this many evenly-spaced ticks (the rest of the grid is still drawn).
_HEATMAP_MAX_TICKS = 8
# Renderer-level safety nets for a spec carrying raw large-n data. Builders are expected to pre-sample /
# pre-bin, but a renderer is public API and must not embed n values into an output file.
#
# These three were declared INDEPENDENTLY in both renderers with identical values. Two copies of a number
# whose whole purpose is that both backends behave the same is a drift waiting to happen: changing one and
# not the other yields two different charts from one spec, and nothing anywhere would flag it. Single
# definition here; both backends import it.
_SCATTER_MAX_POINTS = 50_000
_HIST_PREBIN_THRESHOLD = 50_000
_HEATMAP_CELL_TEXT_MAX = 400
# Cap for ONE bar-category label. Both backends rotate these labels already, so the cap is a safety valve
# against a pathological generated name (a 200-char column) blowing out the axis, not routine shortening.
_BAR_LABEL_MAXLEN = 60


def truncate_bar_label(label: Any, maxlen: int = _BAR_LABEL_MAXLEN) -> str:
    """Shorten one bar-category label to ``maxlen`` chars, ellipsis-suffixed.

    Single definition on purpose: both renderers need byte-identical label text or the same spec yields two
    differently-labelled charts, and two copies of a truncation rule is exactly the drift this module exists
    to prevent (see the shared threshold constants above).
    """
    s = str(label)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "..."


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


def heatmap_value_to_index(lo: float, hi: float, n_bins: int):
    """Return ``value -> bin-index`` for a heatmap axis binned over ``[lo, hi]`` into ``n_bins`` cells.

    Both renderers draw the y=x reference and the robust trend line in BIN-INDEX space while
    ``robust_fit_endpoints`` returns VALUE space, so both need this map. It lives here because they diverged:
    the plotly side rounded AND clamped the result to a category label, which MOVES an extrapolated endpoint to
    the axis edge and therefore changes the drawn segment's SLOPE -- defeating the panel's stated purpose, which
    is to make a systematic slope bias visible. matplotlib applied the affine map alone and let the axis limits
    clip the segment, which keeps the slope of the visible portion correct.

    The map is deliberately NOT clamped: a caller that needs clipping sets axis limits, which clips the drawn
    line without moving its endpoints.
    """
    span = float(hi) - float(lo)
    scale = (float(n_bins) - 1.0) / span if span > 0 else 0.0

    def _to_index(v: float) -> float:
        """Position of ``v`` on the bin-index axis; may fall outside ``[0, n_bins - 1]`` for an extrapolation."""
        return (float(v) - float(lo)) * scale

    return _to_index


def histogram_bar_extent(bin_centers: Any, width: Any) -> Tuple[Optional[float], Optional[float]]:
    """``(left_edge_of_first_bar, right_edge_of_last_bar)`` for a pre-binned bar panel.

    ``width`` is a scalar for evenly-spaced bins and a per-bar array for uneven ones, so halving it wholesale
    raises on the array case -- which is how a per-bar width first reached production as a ``TypeError`` from a
    plotly worker rather than as a wrong-looking chart. Both backends anchor their Normal-overlay grid on this
    pair and both got it wrong the same way, so the arithmetic lives here once.
    """
    centres = np.asarray(bin_centers)
    if centres.size == 0:
        return None, None
    arr = isinstance(width, np.ndarray)
    first = float(width[0]) if arr else float(width)
    last = float(width[-1]) if arr else float(width)
    return float(centres[0] - first / 2.0), float(centres[-1] + last / 2.0)


def select_per_point(value: Any, mask: np.ndarray, n: int) -> Any:
    """Narrow ``value`` to ``mask`` when it is a per-point array of length ``n``; return it untouched otherwise.

    Splitting a scatter into two traces (filled observations, hollow weak ones) has to carry every per-point
    field along -- sizes, colours -- while leaving the scalar style fields alone. Both backends need exactly
    this test, and getting it wrong is silent: a size array that is not narrowed either raises or, worse,
    pairs each point with another point's size.
    """
    if isinstance(value, np.ndarray) and len(value) == n:
        return value[mask]
    return value


def low_evidence_mask(indices: Any, n: int) -> np.ndarray:
    """Boolean mask over ``n`` points marking those a builder flagged as resting on too little data.

    Shared so the two backends cannot disagree about WHICH points are unreadable: a bin drawn hollow in the PNG
    and solid in the interactive HTML is one chart contradicting itself. Out-of-range indices are dropped rather
    than raising, because this only controls emphasis.
    """
    mask = np.zeros(max(int(n), 0), dtype=bool)
    if indices is None or mask.size == 0:
        return mask
    idx = np.asarray(indices, dtype=np.int64)
    mask[idx[(idx >= 0) & (idx < mask.size)]] = True
    return mask


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


# Per-fontsize character-advance tables. One table serves every string at that size, so the font is touched
# once per size per process rather than once per line. Guarded because ``render_and_save`` renders its backends
# CONCURRENTLY in a thread pool, and both call in here: without the lock two threads racing on the same new size
# each build their own table (matplotlib's font machinery is not reentrant, and the wasted build is the benign
# half of the problem).
_CHAR_ADVANCE_CACHE: dict = {}
_CHAR_ADVANCE_LOCK = threading.Lock()
# Left+right breathing room reserved when fitting text to the figure width, in inches.
_TEXT_SIDE_MARGIN_IN = 0.35


def _char_advances(fontsize: float) -> dict:
    """Per-character horizontal advance (in points) for the active font at ``fontsize``, built once and cached.

    This replaced measuring each candidate LINE with ``matplotlib.textpath.TextPath``, which rasterises glyph
    outlines: at ~8 ms a call, greedily wrapping one 40-word headline cost **4.2 seconds**, and a suite rendering
    hundreds of charts hit the renderer's 60 s per-figure timeout. Advances are what a font actually lays text
    out with, the table is 95 entries built in ~14 ms per size, and summing it is ~10 us per line -- validated
    against matplotlib's own ``get_text_width_height_descent`` to within 0.11% on realistic titles, which is far
    inside the margin already reserved at each side of the figure.
    """
    cached: "dict | None" = _CHAR_ADVANCE_CACHE.get(fontsize)
    if cached is not None:
        return cached
    from matplotlib import font_manager, ft2font

    with _CHAR_ADVANCE_LOCK:
        # Re-checked inside the lock: another thread may have built this size while this one waited.
        under_lock: "dict | None" = _CHAR_ADVANCE_CACHE.get(fontsize)
        if under_lock is not None:
            return under_lock
        # The STORE happens here, under the lock that guards it, rather than inside the builder. A builder that
        # writes a cache while its CALLER holds the lock is the "locked elsewhere, unlocked here" shape: the
        # module greps as lock-aware, so a later caller that forgets the lock still reads as correct.
        table = _build_char_advances(fontsize, font_manager, ft2font)
        _CHAR_ADVANCE_CACHE[fontsize] = table
        return table


def _build_char_advances(fontsize: float, font_manager: Any, ft2font: Any) -> dict:
    """Load the font once and tabulate every printable-ASCII advance. Called under ``_CHAR_ADVANCE_LOCK``."""
    font = ft2font.FT2Font(font_manager.findfont(font_manager.FontProperties(size=fontsize)))
    font.set_size(fontsize, 72)  # 72 dpi so the advance comes out in points directly
    # ``LoadFlags`` replaced the module-level constants in matplotlib 3.10; the old spelling still resolves on
    # older releases, so both are tried by name -- only one of the two exists in any given install.
    try:
        flags = ft2font.LoadFlags.NO_HINTING
    except AttributeError:
        flags = ft2font.LOAD_NO_HINTING  # pre-3.10 spelling, deprecated but still present
    table: dict = {}
    for code in range(32, 127):
        table[chr(code)] = float(font.load_char(code, flags=flags).linearHoriAdvance) / 65536.0
    # Anything outside printable ASCII (a degree sign, a plus-minus, CJK) falls back to the widest letter rather
    # than to an average: over-estimating breaks a line early, under-estimating runs it off the canvas.
    table["\x00default"] = max(table.values())
    return table


def _measured_text_width_pt(text: str, fontsize: float) -> float:
    """Width of ``text`` in points, measured on the ACTIVE font rather than counted in characters.

    A character budget is a guess about the font: proportional faces put ``i`` and ``W`` an order of
    magnitude apart, so one constant is simultaneously too generous for a title full of capitals and far too
    stingy for a run of digits and punctuation -- which is exactly what a metrics headline is. Summing real
    advances removes the guess without paying for glyph outlines.
    """
    table = _char_advances(fontsize)
    default = table["\x00default"]
    return float(sum(table.get(ch, default) for ch in text))


def _split_overlong_word(word: str, fontsize: float, budget_pt: float) -> list[str]:
    """Break one token that is wider than the whole line into measured pieces.

    A generated column name, a file path or a serialised param dict carries no spaces, so a space-only wrapper
    leaves it whole and it runs straight out of the panel. The cut point is found by bisection on the MEASURED
    prefix width -- about log2(len) measurements rather than one per character.
    """
    pieces: list[str] = []
    rest = word
    while rest and _measured_text_width_pt(rest, fontsize) > budget_pt:
        lo, hi = 1, len(rest)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _measured_text_width_pt(rest[:mid], fontsize) <= budget_pt:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(rest[:lo])
        rest = rest[lo:]
    if rest:
        pieces.append(rest)
    return pieces


def wrap_text_to_width(text: Any, *, fontsize: float, width_in: float, fallback_chars: int = 90, break_long_words: bool = False) -> list[str]:
    """Wrap ``text`` to the real width of a ``width_in``-inch figure, measuring the font instead of counting characters.

    The fixed character budgets this replaces (90 for a suptitle, 110 for a caption) were calibrated once and
    then applied at every figure width and font size, so a wide figure folded its headline into a narrow
    ragged column with a third of the width unused -- the text was being broken by an assumption, not by the
    edge of the canvas.

    An explicit line break in the text is honoured: each segment is wrapped independently, so a deliberate
    break survives. Any measurement failure falls back to the character budget: this only controls text
    layout and must never raise.
    """
    lines_in = str(text).splitlines() or [""]
    try:
        budget_pt = max(float(width_in) - 2.0 * _TEXT_SIDE_MARGIN_IN, 1.0) * 72.0
        out: list[str] = []
        for segment in lines_in:
            words = segment.split()
            if not words:
                out.append("")
                continue
            if break_long_words:
                expanded: list[str] = []
                for word in words:
                    expanded.extend(_split_overlong_word(word, fontsize, budget_pt))
                words = expanded
            current = words[0]
            for word in words[1:]:
                candidate = current + " " + word
                if _measured_text_width_pt(candidate, fontsize) <= budget_pt:
                    current = candidate
                else:
                    out.append(current)
                    current = word
            out.append(current)
        return out
    except Exception:
        logger.debug("measured text wrapping failed; falling back to the character budget", exc_info=True)
        return wrap_title_lines(text, fallback_chars)


def panel_title_wrap_chars(figsize: Any, cols: int = 1) -> int:
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


def wrap_annotation_text(text: Any, panel_width_in: float, fontsize: float) -> str:
    """Wrap free-text panel content to the panel's own character budget, breaking over-long tokens.

    Neither backend wraps this correctly on its own. matplotlib's ``wrap=True`` measures against the FIGURE box, not
    the axes -- for centred text it allows ``2 * min(dist_to_fig_left, dist_to_fig_right)``, measured at 643.8 px
    inside a 532.9 px panel, 21% too wide -- and it only ever breaks at spaces, so a single long token
    (``DummyClassifier(strategy=prior)``, a file path, a metric dict) is never broken at all and runs straight into
    the neighbouring panel. plotly does not wrap free text whatsoever, and paints annotations ABOVE traces, so the
    overflow lands visually on top of whatever sits beside it.

    The line is now bounded by MEASURED glyph widths rather than by an assumed ~0.6 em average advance. That
    assumption is wrong in both directions on the text this actually wraps: a metric dict is mostly digits and
    punctuation (narrower, so the panel went under-filled) while a class name in CamelCase is mostly capitals
    (wider, so it still overflowed the panel the budget was supposed to protect).
    """
    usable_in = max(float(panel_width_in) * 0.92, 0.5)  # leave a small margin inside the panel
    lines = wrap_text_to_width(
        text,
        fontsize=fontsize,
        width_in=usable_in + 2.0 * _TEXT_SIDE_MARGIN_IN,  # the helper reserves its own side margins; this keeps the 0.92 factor as the only one
        fallback_chars=max(12, int(usable_in / (max(float(fontsize), 1.0) * 0.6 / 72.0))),
        break_long_words=True,
    )
    return "\n".join(lines)


def wrap_title_lines(text: Any, width: int) -> list[str]:
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


def epoch_ns_ticks(x_values: Any, n_ticks: int = 6) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
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

"""Class-structure (group x time) heatmap: a RedHat-style leakage / structure diagnostic.

Each cell is the positive-class rate (binary target) or mean target (regression) over the rows that fall in a given
group and a given equal-population time bin. Genuine signal shows as a smooth field; leakage / structure shows as a
sharp group-band (a whole row far from the rest) or a time-band (a whole column), e.g. a group whose label was
assigned after the outcome was known, or a period where the labelling policy changed.

The only length-n work is the 2-D accumulate of ``sum(y)`` and ``count`` over the (group, time) cell grid, done in a
single njit pass (``class_structure_matrix``); everything else is over the tiny ``n_groups x n_time`` matrix. Columns
are pulled as narrow ndarray views, never as a whole-frame copy, so a 100+ GB carrier is safe.

cProfile verdict (profile_class_structure_heatmap, best-of-3 walltime): the accumulate kernel is a single memory-bound
pass; njit is bit-identical to the two-``np.bincount`` numpy path and wins at every profiled n. See the bench docstring
for the njit-vs-numpy numbers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec

# Default cap on the number of distinct group rows; the rest fold into a single "other" row so a high-cardinality group
# column (thousands of ids) still yields a readable heatmap rather than an unrenderable wall of rows.
DEFAULT_MAX_GROUPS: int = 30
# Equal-population time bins: enough columns to reveal a time-band without shrinking per-cell support into noise.
DEFAULT_N_TIME_BINS: int = 20
# A cell below this many rows is blanked: at 1-2 rows the positive rate is 0.00 or 1.00 by construction, which paints
# a fully-saturated cell indistinguishable from a real leakage band backed by tens of thousands of rows.
_CELL_SUPPORT_FLOOR: int = 20
# Past this many cells the per-cell numbers overlap into an unreadable smear (same ceiling as the other heatmaps).
_CELL_TEXT_MAX: int = 400

try:
    import numba

    @numba.njit(cache=True, fastmath=False)
    def _accumulate_group_time(group_codes: np.ndarray, time_codes: np.ndarray, y: np.ndarray, n_groups: int, n_time: int):
        """njit variant: accumulate per-(group, time)-cell ``sum(y)`` and row ``count`` in one length-n pass; returns ``(sums, counts)``."""
        # Single length-n pass building the per-cell (sum of y, count) over the group x time grid. Row-order float64
        # accumulation, so the sums are bit-identical to a two-``np.bincount`` walk on the flattened cell index.
        sums = np.zeros((n_groups, n_time), dtype=np.float64)
        counts = np.zeros((n_groups, n_time), dtype=np.float64)
        for i in range(group_codes.shape[0]):
            g = group_codes[i]
            t = time_codes[i]
            sums[g, t] += y[i]
            counts[g, t] += 1.0
        return sums, counts

    _HAS_NUMBA = True
except ImportError:  # numba unavailable: fall back to a flattened two-bincount numpy accumulate.
    _HAS_NUMBA = False

    def _accumulate_group_time(group_codes, time_codes, y, n_groups, n_time):
        """numpy fallback (no numba): same ``(sums, counts)`` accumulate via two ``np.bincount`` calls over the flattened cell index."""
        flat = group_codes * n_time + time_codes
        ncells = n_groups * n_time
        counts = np.bincount(flat, minlength=ncells).astype(np.float64)
        sums = np.bincount(flat, weights=y, minlength=ncells)
        return sums.reshape(n_groups, n_time), counts.reshape(n_groups, n_time)


def class_structure_matrix(group_codes: np.ndarray, time_codes: np.ndarray, y: np.ndarray, n_groups: int, n_time: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(rate_matrix, count_matrix)`` of shape ``(n_groups, n_time)``.

    ``rate_matrix[g, t]`` is ``mean(y)`` over the rows whose group code is ``g`` and time code is ``t`` (the
    positive-class rate for a binary ``y``); empty cells are ``nan``. ``count_matrix[g, t]`` is that cell's support.
    This is the only length-n kernel in the module.
    """
    gc = np.ascontiguousarray(group_codes, dtype=np.int64)
    tc = np.ascontiguousarray(time_codes, dtype=np.int64)
    yv = np.ascontiguousarray(y, dtype=np.float64)
    # _accumulate_group_time's njit path has no bounds checking (numba default): a length mismatch between gc/tc/yv,
    # or a code outside [0, n_groups)/[0, n_time), silently reads/writes past the allocated (n_groups, n_time) buffer
    # -- a memory-corrupting "Windows fatal exception: access violation" that kills the whole process, bypassing even
    # a caller's try/except (caught live via an ensembling COARSE-gate path handing this a shorter y than group/time
    # codes). Validate the contract here so a violation raises a normal, catchable ValueError instead.
    if not (gc.shape[0] == tc.shape[0] == yv.shape[0]):
        raise ValueError(f"class_structure_matrix: length mismatch group_codes={gc.shape[0]} time_codes={tc.shape[0]} y={yv.shape[0]}")
    if gc.shape[0] > 0 and (gc.min() < 0 or gc.max() >= n_groups):
        raise ValueError(f"class_structure_matrix: group_codes out of range [0, {n_groups})")
    if tc.shape[0] > 0 and (tc.min() < 0 or tc.max() >= n_time):
        raise ValueError(f"class_structure_matrix: time_codes out of range [0, {n_time})")
    sums, counts = _accumulate_group_time(gc, tc, yv, int(n_groups), int(n_time))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(counts > 0.0, sums / counts, np.nan)
    return rate, counts


def _pull_column(df: Any, col: Any) -> np.ndarray:
    """Pull one column of ``df`` (pandas / polars / ndarray-of-columns) as a narrow ndarray view, no frame copy."""
    if hasattr(df, "columns") and not isinstance(df, np.ndarray):
        c = df[col]
        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)
    arr = np.asarray(df)
    return arr[:, col] if arr.ndim == 2 else arr


def _equal_population_codes(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Rank-based equal-population bin codes in ``0..n_bins-1``; ties are split positionally, robust to any dtype scale."""
    n = values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    codes = (ranks * n_bins) // n
    np.clip(codes, 0, n_bins - 1, out=codes)
    return np.asarray(codes.astype(np.int64))


def _time_codes(df: Any, y_len: int, timestamps: Optional[np.ndarray], time_col: Optional[Any], n_time_bins: int) -> np.ndarray:
    """Equal-population time codes from explicit ``timestamps``, an integer ``time_col``, or row order as a fallback."""
    if timestamps is not None:
        vals = np.asarray(timestamps)
    elif time_col is not None:
        vals = _pull_column(df, time_col)
    else:
        vals = np.arange(y_len, dtype=np.int64)
    if vals.dtype.kind == "M":  # datetime64 -> integer nanoseconds for ranking
        vals = vals.astype("datetime64[ns]").astype(np.int64)
    if vals.dtype.kind in "OUS":
        raise TypeError(
            f"class_structure_heatmap: the time axis has dtype {vals.dtype!r}. Parse it to datetime64 or a numeric "
            "epoch first -- a string column cannot be ranked into equal-population time bins."
        )
    numeric = np.ascontiguousarray(vals, dtype=np.float64)
    # argsort places NaN LAST, so rows with a missing timestamp silently landed in the most-recent bin and read as a
    # genuine late-period signal. Send them to their own trailing bin instead of contaminating a real period.
    if not np.isfinite(numeric).all():
        numeric = np.where(np.isfinite(numeric), numeric, np.nanmax(numeric[np.isfinite(numeric)], initial=0.0) + 1.0)
    return _equal_population_codes(numeric, n_time_bins)


def _time_bin_labels(
    df: Any,
    y_len: int,
    timestamps: Optional[np.ndarray],
    time_col: Optional[Any],
    time_codes: np.ndarray,
    n_time_bins: int,
) -> Tuple[str, ...]:
    """Column labels naming each equal-population bin's TIME SPAN, falling back to the bin index for a row-order axis.

    A bare "0".."19" is a bin index, and a bin index is never a unit: a suspicious column could not be mapped back to
    a date range, which is the first thing a reader wants once a column looks anomalous.
    """
    if timestamps is None and time_col is None:
        return tuple(f"rows\nbin {i}" for i in range(n_time_bins))
    vals = np.asarray(timestamps) if timestamps is not None else _pull_column(df, time_col)
    is_datetime = vals.dtype.kind == "M"
    numeric = vals.astype("datetime64[ns]").astype(np.int64) if is_datetime else np.asarray(vals, dtype=np.float64)

    def fmt(v: float) -> str:
        """One bin edge as a date (datetime axis) or a compact number (numeric axis)."""
        if is_datetime:
            return str(np.datetime64(int(v), "ns").astype("datetime64[D]"))
        return f"{v:.3g}"

    labels: List[str] = []
    for b in range(n_time_bins):
        block = numeric[time_codes == b]
        block = block[np.isfinite(block)] if not is_datetime else block
        labels.append(f"{fmt(float(block.min()))}\n..{fmt(float(block.max()))}" if block.size else f"bin {b}\n(empty)")
    return tuple(labels)


def _group_codes_capped(group_arr: np.ndarray, max_groups: int) -> Tuple[np.ndarray, List[str], int]:
    """Map raw group values to row codes, keeping the ``max_groups`` largest groups and folding the rest into "other"."""
    encodable = group_arr if group_arr.dtype.kind in "iuf" else group_arr.astype(str)
    labels, inv = np.unique(encodable, return_inverse=True)
    inv = inv.astype(np.int64)
    n_unique = labels.shape[0]
    if n_unique <= max_groups:
        row_labels = [str(v) for v in labels]
        return inv, row_labels, n_unique
    counts = np.bincount(inv, minlength=n_unique)
    order = np.argsort(counts)[::-1]
    keep = order[:max_groups]
    new_code = np.full(n_unique, max_groups, dtype=np.int64)  # default row = the "other" bucket
    new_code[keep] = np.arange(max_groups, dtype=np.int64)
    row_labels = [str(labels[gi]) for gi in keep] + ["other"]
    return new_code[inv], row_labels, max_groups + 1


def class_structure_panel(df: Any, y: np.ndarray, *, group: Any, timestamps: Optional[np.ndarray] = None,
                          time_col: Optional[Any] = None, max_groups: int = DEFAULT_MAX_GROUPS,
                          n_time_bins: int = DEFAULT_N_TIME_BINS) -> HeatmapPanelSpec:
    """HeatmapPanelSpec of the per-(group, time-bin) positive-class rate / mean target over ``df`` and ``y``.

    ``group`` is a column key into ``df``; ``timestamps`` (or an integer ``time_col``, else row order) is binned into
    ``n_time_bins`` equal-population columns; groups past the ``max_groups`` largest fold into a single "other" row.

    Cells below :data:`_CELL_SUPPORT_FLOOR` rows are blanked. A single-row cell renders a saturated 1.00 that is
    visually indistinguishable from a genuine 50k-row leakage band, and this chart's whole purpose is spotting those
    bands, so an unsupported cell must not compete with them for the reader's attention. Every cell's row count is
    carried in its tooltip either way.
    """
    yv = np.ascontiguousarray(np.asarray(y), dtype=np.float64)
    n = yv.shape[0]
    group_arr = _pull_column(df, group)
    group_codes, row_labels, n_rows = _group_codes_capped(group_arr, int(max_groups))
    time_codes = _time_codes(df, n, timestamps, time_col, int(n_time_bins))
    rate, counts = class_structure_matrix(group_codes, time_codes, yv, n_rows, int(n_time_bins))
    low_support = counts < _CELL_SUPPORT_FLOOR
    rate = np.where(low_support, np.nan, rate)
    # Equal-population bins are unequal in TIME, so a bare index cannot be mapped back to a period. Label with the
    # bin's own time span where a real time vector exists; fall back to the index when the axis is row order.
    col_labels = _time_bin_labels(df, n, timestamps, time_col, time_codes, int(n_time_bins))
    n_blank = int(low_support.sum())
    return HeatmapPanelSpec(
        matrix=rate,
        row_labels=tuple(row_labels),
        col_labels=col_labels,
        title=(
            "Class structure by group x time"
            + (f"\n({n_blank} of {counts.size} cells hold under {_CELL_SUPPORT_FLOOR} rows and are blanked)" if n_blank else "")
        ),
        xlabel="time bin (equal population, so unequal in time)",
        ylabel="group",
        colormap="magma",
        cell_text=rate if rate.size <= _CELL_TEXT_MAX else None,
        text_format=".2f",
        colorbar_label="P(y=1) / mean y",
        cell_hovertext=np.asarray([[f"{int(c):,} rows ({c / max(n, 1):.2%} of all)" for c in row] for row in counts], dtype=object),
    )


def compose_class_structure_figure(df: Any, y: np.ndarray, *, group: Any, timestamps: Optional[np.ndarray] = None,
                                   time_col: Optional[Any] = None, max_groups: int = DEFAULT_MAX_GROUPS,
                                   n_time_bins: int = DEFAULT_N_TIME_BINS,
                                   suptitle: str = "Class structure by group x time") -> FigureSpec:
    """One-panel FigureSpec wrapping :func:`class_structure_panel`."""
    panel = class_structure_panel(df, y, group=group, timestamps=timestamps, time_col=time_col, max_groups=max_groups, n_time_bins=n_time_bins)
    return FigureSpec(
        suptitle=suptitle,
        panels=((panel,),),
        figsize=(7.0, 5.0),
        caption=(
            "Each cell is the positive-class rate (or mean target) for one group within one EQUAL-POPULATION time "
            "bin, so every column holds the same number of rows and the columns are NOT equal in time -- the column "
            "labels give each bin's actual span. A whole row far from the rest is a group-level anomaly (a label "
            "assigned after the outcome was known?); a whole column is a period-level one (a labelling-policy "
            "change?). Hover shows each cell's row count, and cells below the support floor are left blank so a "
            "one-row cell cannot masquerade as a leakage band."
        ),
    )


__all__ = [
    "DEFAULT_MAX_GROUPS",
    "DEFAULT_N_TIME_BINS",
    "class_structure_matrix",
    "class_structure_panel",
    "compose_class_structure_figure",
]

"""Row/column density-sort leak detection for anonymized tabular data.

COMPETITION/EXPLORATORY USE ONLY -- NOT FOR PRODUCTION.

Several top Kaggle solutions (e.g. Santander Value Prediction Challenge) discovered
that anonymized, shuffled feature matrices sometimes hide a row-shifted time-series
structure: sorting columns by non-zero/non-null density, then sorting rows by density,
reveals consecutive rows that are near-duplicates of each other shifted by one time
step (row[i+1] shares most of its non-null values with row[i], offset by one column).
Matching such pairs let competitors read future target values directly off adjacent
rows -- a leak.

This module implements a generic version of that recipe: sort rows and columns by
density, then flag consecutive row pairs whose non-null/non-zero value sets overlap
unusually strongly as *candidate* leak pairs.

Real production data is not deliberately anonymized/shuffled this way, so this
diagnostic has no production analog. It must never be imported by production mlframe
code or wired into default pipelines -- see ``mlframe.competition`` package docstring
and ``MLFRAME_IDEAS_competitions.md`` ("Row/column sort-by-density leak-detection
technique").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["sort_by_density_leak_scan", "LeakScanResult", "find_shifted_column_groups"]


@dataclass
class LeakScanResult:
    """Result bundle of :func:`sort_by_density_leak_scan`.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring.
    """

    row_order: np.ndarray
    col_order: np.ndarray
    row_density: np.ndarray
    col_density: np.ndarray
    overlap_scores: np.ndarray
    candidate_pairs: list[tuple[int, int, float]] = field(default_factory=list)


def _nonnull_nonzero_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask, True where a cell is non-null and (for numeric dtypes) non-zero."""
    values = df.to_numpy(dtype=object)
    mask = np.empty(values.shape, dtype=bool)
    for j in range(values.shape[1]):
        col = df.iloc[:, j]
        notna = col.notna().to_numpy()
        if pd.api.types.is_numeric_dtype(col):
            nonzero = col.fillna(0).to_numpy() != 0
            mask[:, j] = notna & nonzero
        else:
            mask[:, j] = notna
    return mask


def sort_by_density_leak_scan(
    df: pd.DataFrame,
    target: np.ndarray | None = None,
    *,
    overlap_threshold: float = 0.8,
    min_shared_values: int = 3,
    max_col_shift: int = 1,
    neighbor_window: int = 5,
) -> dict:
    """Sort rows/columns by density and flag candidate row-shift leak pairs.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring. Not a production
    diagnostic: it targets deliberately anonymized/shuffled competition data where
    row-shifted duplication is a known leak pattern, not real-world data pipelines.

    Sorts columns by their non-null/non-zero density (fraction of populated cells),
    then sorts rows the same way. Within the density-sorted order, computes a Jaccard-style
    overlap of the *positions* of shared non-null values between each consecutive row pair
    (row[i], row[i+1]) and flags pairs above ``overlap_threshold`` as candidate leak pairs
    (row[i+1] plausibly being row[i] shifted by one time step in an anonymized panel).

    Args:
        df: input frame (numeric and/or categorical columns).
        target: optional 1-D array aligned with ``df``'s rows; if given, it is
            concatenated as the first column before sorting (mirrors the Santander
            trick of prepending the label so label/feature matches surface together).
        overlap_threshold: minimum shared-value overlap fraction (0..1) for a
            consecutive density-sorted row pair to be flagged as a leak candidate.
        min_shared_values: minimum number of jointly-populated cells required before a
            pair is considered (avoids spurious high overlap between two near-empty rows).
        max_col_shift: largest column-shift (in original column order) tried when
            comparing a row to nearby density-sorted rows; the overlap reported is the
            best over shifts in ``-max_col_shift..max_col_shift``, which is what actually
            catches the "row[i+1] = row[i] shifted by one time step" leak pattern
            (a shift of 0 alone only catches exact-duplicate rows, not shifted ones).
        neighbor_window: rows with identical/near-identical density sort into ties broken
            arbitrarily; each row is compared against the next ``neighbor_window`` rows
            in density-sorted order (not just its immediate successor) so a shifted-leak
            partner landing a few ties away is still found.

    Returns:
        dict with keys ``row_order`` (original row indices in density-sorted order),
        ``col_order`` (original column indices in density-sorted order), ``row_density``,
        ``col_density`` (density of each row/column, indexed by *original* row/column
        position -- to read them in sorted order index with ``row_order``/``col_order``),
        ``overlap_scores`` (overlap of each density-sorted row with its successor, last
        entry NaN), and ``candidate_pairs`` (list of ``(row_idx_a, row_idx_b, overlap)``
        tuples in *original* row-index space, sorted by descending overlap).
    """
    work = df.copy()
    if target is not None:
        work.insert(0, "__target__", np.asarray(target))

    mask = _nonnull_nonzero_mask(work)
    n_rows, n_cols = mask.shape

    col_density = mask.mean(axis=0)
    col_order = np.argsort(col_density, kind="stable")
    mask_sorted_cols = mask[:, col_order]

    row_density = mask_sorted_cols.mean(axis=1)
    # secondary key: sum of numeric values ignoring NaN. A row and its column-shifted
    # partner share almost the same multiset of values, so this breaks density ties in a
    # way that pulls shifted-duplicate pairs adjacent instead of leaving tie order to
    # input-shuffle chance.
    numeric_values = work.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    row_value_sum = np.nansum(numeric_values, axis=1)
    row_order = np.lexsort((row_value_sum, row_density))
    # overlap/shift comparison uses ORIGINAL column order (density-sorted column order
    # scrambles column adjacency, which is what a "shifted by one time step" match needs).
    mask_by_row_order = mask[row_order, :]

    mask_int = mask_by_row_order.astype(np.int32)
    row_popcount = mask_int.sum(axis=1)

    overlap_scores = np.full(n_rows, np.nan, dtype=float)
    pair_best: dict[tuple[int, int], float] = {}
    for i in range(n_rows):
        j_hi = min(i + 1 + neighbor_window, n_rows)
        if j_hi <= i + 1:
            continue
        block = mask_int[i + 1 : j_hi]  # (w, n_cols)
        block_popcount = row_popcount[i + 1 : j_hi]
        best_overlap_block = np.zeros(block.shape[0], dtype=float)
        for shift in range(-max_col_shift, max_col_shift + 1):
            if shift >= 0:
                a_seg = mask_int[i, : n_cols - shift]
                b_block_seg = block[:, shift:]
            else:
                a_seg = mask_int[i, -shift:]
                b_block_seg = block[:, : n_cols + shift]
            shared = b_block_seg @ a_seg  # popcount of AND, per window row
            union = block_popcount + row_popcount[i] - shared
            valid = (union > 0) & (shared >= min_shared_values)
            overlap = np.where(valid, shared / np.where(union == 0, 1, union), 0.0)
            best_overlap_block = np.maximum(best_overlap_block, overlap)
        overlap_scores[i] = best_overlap_block.max() if best_overlap_block.size else np.nan

        hits = np.nonzero(best_overlap_block >= overlap_threshold)[0]
        for offset in hits:
            j = i + 1 + int(offset)
            score = float(best_overlap_block[offset])
            orig_a = int(row_order[i])
            orig_b = int(row_order[j])
            key = (min(orig_a, orig_b), max(orig_a, orig_b))
            if score > pair_best.get(key, 0.0):
                pair_best[key] = score

    candidate_pairs = [(a, b, score) for (a, b), score in pair_best.items()]
    candidate_pairs.sort(key=lambda t: t[2], reverse=True)

    return {
        "row_order": row_order,
        "col_order": col_order,
        "row_density": row_density,
        "col_density": col_density,
        "overlap_scores": overlap_scores,
        "candidate_pairs": candidate_pairs,
    }


def find_shifted_column_groups(
    df: pd.DataFrame,
    *,
    max_lag: int = 3,
    corr_threshold: float = 0.95,
    min_overlap_rows: int = 10,
) -> dict:
    """Best-effort structural leak-hunting: group numeric columns that look like the
    same underlying series shifted by a small row-lag.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring. This is a deliberately
    MINIMAL, best-effort stand-in for the fuller ``TemporalShiftGroupDetector`` idea
    described in ``MLFRAME_IDEAS_competitions.md`` ("Leak/duplicate-row detection via
    cross-column group reconstruction"), which called for cross-correlation search over
    column *and* row permutations to reconstruct de-anonymized panel groups (e.g. the
    113 groups of 40 shifted features found in Santander Value Prediction). This
    implementation only searches small fixed row-lags between column pairs (no row
    reordering search), which is enough to catch the simplest "column B is column A
    shifted down by k rows" pattern but will miss leaks requiring row permutation.

    For every pair of numeric columns, computes the Pearson correlation between column
    A and column B shifted by ``lag`` rows (for ``lag`` in ``-max_lag..max_lag``), takes
    the best-scoring lag, and unions any pair whose best correlation exceeds
    ``corr_threshold`` into the same candidate group (simple union-find over columns).

    Args:
        df: input frame; only numeric columns are considered.
        max_lag: largest row-shift (in either direction) tried between each column pair.
        corr_threshold: minimum |Pearson r| at the best lag to union two columns.
        min_overlap_rows: minimum number of jointly-non-null rows required at a given
            lag before its correlation is trusted (avoids spurious correlations from
            tiny overlaps).

    Returns:
        dict with keys ``groups`` (list of lists of original column names, one list per
        connected group with 2+ members) and ``pairwise_best`` (list of
        ``(col_a, col_b, best_lag, best_corr)`` tuples for every pair considered,
        regardless of threshold, for inspection).
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n = len(df)
    arrays = {c: df[c].to_numpy(dtype=float) for c in numeric_cols}

    parent = {c: c for c in numeric_cols}

    def find(c: str) -> str:
        """Return the union-find root of column ``c``, path-compressing along the way."""
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a: str, b: str) -> None:
        """Merge the union-find groups containing columns ``a`` and ``b``."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pairwise_best: list[tuple[str, str, int, float]] = []
    for i, col_a in enumerate(numeric_cols):
        a = arrays[col_a]
        for col_b in numeric_cols[i + 1 :]:
            b = arrays[col_b]
            best_lag = 0
            best_corr = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    a_seg, b_seg = a[: n - lag], b[lag:]
                else:
                    a_seg, b_seg = a[-lag:], b[: n + lag]
                if a_seg.size < min_overlap_rows:
                    continue
                valid = np.isfinite(a_seg) & np.isfinite(b_seg)
                if int(valid.sum()) < min_overlap_rows:
                    continue
                a_valid, b_valid = a_seg[valid], b_seg[valid]
                if a_valid.std() == 0 or b_valid.std() == 0:
                    continue
                corr = float(np.corrcoef(a_valid, b_valid)[0, 1])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            pairwise_best.append((col_a, col_b, best_lag, best_corr))
            if abs(best_corr) >= corr_threshold:
                union(col_a, col_b)

    grouped: dict[str, list[str]] = {}
    for c in numeric_cols:
        grouped.setdefault(find(c), []).append(c)
    groups = [members for members in grouped.values() if len(members) >= 2]

    return {"groups": groups, "pairwise_best": pairwise_best}

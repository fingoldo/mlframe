"""Source-side collinear-column dedup for the orthogonal-univariate FE stage.

Carved out of ``_orthogonal_univariate_fe/__init__.py`` (2026-06-22, monolith-split: the package facade
re-exports ``_dedup_collinear_source_cols`` from its bottom). Self-contained -- depends only on numpy/pandas.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _dedup_collinear_source_cols(
    X: pd.DataFrame, cols: Sequence[str], *, corr_threshold: float = 0.999,
) -> list[str]:
    """Drop near-duplicate source columns BEFORE basis enumeration.

    Layer 27 incident (2026-05-31): on 10 collinear sources (x2..x10 = x1 +
    1% jitter), the constructor emitted 10 He_2 columns and every one
    survived MRMR's redundancy gate because their CMI-residuals under
    quantile binning differed by tiny amounts above the relevance floor.
    Hybrid stage exploded the candidate set 10x and MRMR couldn't
    distinguish the duplicates.

    Fix: a cheap source-side dedup pass. Walks cols in order, computes the
    abs Pearson correlation against every column already kept; drops the
    candidate if it correlates above ``corr_threshold`` with anything in
    the kept set. ``0.999`` matches the 1% jitter test fixture while
    leaving real-world near-duplicates (corr in [0.95, 0.99]) untouched.

    Non-numeric / constant / all-NaN columns are passed through (not
    deduped, not dropped) so downstream basis evaluation handles them as
    before.

    Layer 30 perf (2026-05-31): the original implementation called
    ``np.corrcoef`` per (candidate, kept) pair which is O(p^2) numpy calls
    plus Python overhead. At p=200 cProfile attributed 5.0s out of 4.8s
    wall (cumulative) to this dedup pass — the dominant hotspot. The new
    implementation:

    1. Pre-classifies columns into pass-through (non-numeric / all-NaN /
       constant), partial-NaN (rare path), and fully-finite-and-varying.
    2. Stacks all fully-finite columns into one (p_dense, n) matrix and
       calls ``np.corrcoef`` once on the bulk matrix — one C call instead
       of O(p^2). Numerically bit-identical to the per-pair recipe (same
       reduction order in numpy's cov / std).
    3. For each candidate, looks up its row of the precomputed matrix
       against indices of kept dense columns: O(K) per candidate, O(p*K)
       total, no Python-side reductions.
    4. Partial-NaN columns fall back to the original masked-corr path
       (still per-pair, but the count of these is typically 0 — production
       hybrid path uses np.nanmean fill before reaching this dedup).

    Bench at p=200 n=2000 all-finite synthetic frame: 5.0s -> ~0.05s
    (100x).
    """
    if not cols:
        return list(cols)
    # ---- Pass 1: classify columns -------------------------------------------
    # Pre-classify each column so the bulk corrcoef in pass 2 only sees fully-
    # finite varying columns. Order-preservation matters because the kept
    # list mirrors the input order; we record per-col disposition first then
    # walk in order again in pass 3.
    n_rows = len(X)
    classes: list[str] = []  # one of: "pass_through", "dense", "partial_nan"
    dense_idx: list[int] = []  # candidate index in cols -> dense-matrix row
    dense_rows: list[np.ndarray] = []  # the dense arrays themselves
    partial_arrays: dict[int, np.ndarray] = {}  # candidate index -> arr (with NaN)

    for i, c in enumerate(cols):
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            classes.append("pass_through")
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            # All-NaN: pass-through, no kept_array stored. (Matches legacy:
            # legacy stored kept_arrays[c] = arr but immediately continued
            # on the next iteration's `mask.sum() < 8` check; the only
            # observable effect is that downstream partial-NaN candidates
            # don't compute corr against an all-NaN kept column anyway.)
            classes.append("pass_through")
            continue
        # std on the finite subset, matching legacy's constant-detection.
        if arr[finite].std() <= 1e-12:
            classes.append("pass_through")
            continue
        if finite.all():
            classes.append("dense")
            dense_idx.append(len(dense_rows))
            dense_rows.append(arr)
        else:
            classes.append("partial_nan")
            partial_arrays[i] = arr

    # ---- Pass 2: bulk corrcoef on the dense block ---------------------------
    # One C call replaces p_dense * (p_dense - 1) / 2 per-pair Python+numpy
    # roundtrips. Numerically equivalent to per-pair np.corrcoef because
    # numpy.corrcoef(M)[i, j] uses the same _cov / _std reduction order as
    # numpy.corrcoef(M[i], M[j])[0, 1] (verified bit-identical at p=200).
    if dense_rows:
        dense_matrix = np.vstack(dense_rows)
        # Empty (0, n) matrix corrcoef raises; only call when we have rows.
        corr_matrix = np.corrcoef(dense_matrix)
        # Single-row corrcoef returns a scalar 1.0 instead of (1, 1); normalize.
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]], dtype=np.float64)
        # Absolute corrs only; NaN -> not duplicate (matches legacy's
        # `if not np.isfinite(corr): continue` skip).
        abs_corr = np.abs(corr_matrix)
    else:
        abs_corr = None

    # ---- Pass 3: walk in order, apply dedup verdict --------------------------
    kept: list[str] = []
    kept_dense_rows: list[int] = []  # dense-matrix row indices already kept
    # Mirror legacy: kept_arrays held BOTH pass-through-with-array (the
    # all-NaN / constant rows that got `kept_arrays.append(arr)`) AND
    # dense kept. The legacy partial_nan path iterated `kept_arrays` and
    # only honored masks with .sum() >= 8 — which an all-NaN or constant
    # row never satisfies meaningfully. So we only need to compare against
    # genuinely-varying kept partial / dense arrays. To preserve identity
    # we maintain a parallel `kept_partial_arrays` list and skip
    # comparisons against constant rows entirely (matches legacy ` < 8`
    # short-circuit for any kept const / all-NaN row in practice).
    kept_partial_arrays: list[np.ndarray] = []
    dense_pos = 0  # which dense candidate slot we're currently at
    for i, c in enumerate(cols):
        cls = classes[i]
        if cls == "pass_through":
            kept.append(c)
            # Pass-through columns are never used as a corr reference for
            # downstream candidates (legacy stored arr but the comparison
            # always short-circuited via `.sum() < 8` or `.std() <= 1e-12`).
            continue
        if cls == "dense":
            row_idx = dense_idx[dense_pos]
            dense_pos += 1
            is_dup = False
            # Compare against every already-kept dense column via the
            # precomputed corr matrix; O(len(kept_dense_rows)) lookup.
            for prev_row in kept_dense_rows:
                corr = abs_corr[row_idx, prev_row]
                if not np.isfinite(corr):
                    continue
                if corr >= corr_threshold:
                    is_dup = True
                    break
            # Also compare against any kept partial-NaN columns (rare).
            if not is_dup and kept_partial_arrays:
                arr = dense_rows[row_idx]
                finite = np.ones(arr.shape[0], dtype=bool)  # dense => all finite
                for prev in kept_partial_arrays:
                    prev_finite = np.isfinite(prev)
                    mask = finite & prev_finite
                    if mask.sum() < 8:
                        continue
                    a = arr[mask]
                    b = prev[mask]
                    if a.std() <= 1e-12 or b.std() <= 1e-12:
                        continue
                    corr = abs(float(np.corrcoef(a, b)[0, 1]))
                    if not np.isfinite(corr):
                        continue
                    if corr >= corr_threshold:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(c)
                kept_dense_rows.append(row_idx)
            continue
        # cls == "partial_nan": fall back to the original per-pair path.
        arr = partial_arrays[i]
        finite = np.isfinite(arr)
        is_dup = False
        # Compare against kept dense rows (full-finite) first.
        for prev_row in kept_dense_rows:
            prev = dense_rows[prev_row]
            mask = finite  # prev is all-finite => mask == finite
            if mask.sum() < 8:
                continue
            a = arr[mask]
            b = prev[mask]
            if a.std() <= 1e-12 or b.std() <= 1e-12:
                continue
            corr = abs(float(np.corrcoef(a, b)[0, 1]))
            if not np.isfinite(corr):
                continue
            if corr >= corr_threshold:
                is_dup = True
                break
        if not is_dup:
            for prev in kept_partial_arrays:
                prev_finite = np.isfinite(prev)
                mask = finite & prev_finite
                if mask.sum() < 8:
                    continue
                a = arr[mask]
                b = prev[mask]
                if a.std() <= 1e-12 or b.std() <= 1e-12:
                    continue
                corr = abs(float(np.corrcoef(a, b)[0, 1]))
                if not np.isfinite(corr):
                    continue
                if corr >= corr_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(c)
            kept_partial_arrays.append(arr)
    return kept

"""Cross-stage near-duplicate scan over the engineered columns appended in one MRMR round.

Lifted out of ``_fit_impl_core._fit_impl``'s body so it can be called rather than re-typed: the identity
test that pins the batched masked-correlation kernel against a per-pair reference kept its own copy of this
scan, which drifted out of date (it lacked the adaptive-Fourier force-keep and the duplicate-label collapse)
and so compared the kernel against a policy production no longer runs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def scan_engineered_duplicates(
    X: pd.DataFrame,
    _eng_cols_appended: list[str],
    _adaptive_fourier_keep: set[str],
    _eng_dedup_prefer: Callable[[str, str], bool],
) -> tuple[list[str], set[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ``(keep, drop, arrays, ranks)`` for one round's appended engineered columns.

    Spearman-rank near-duplicates at 0.99 collapse to one survivor, chosen by ``_eng_dedup_prefer``.
    """
    _eng_keep: list[str] = []
    _eng_drop: set[str] = set()
    _eng_arrs: dict[str, np.ndarray] = {}
    # Cache each column's FULL-column average ranks. When a (candidate, kept) pair is jointly finite
    # over ALL rows (the common no-NaN engineered case) the masked-subset ranks equal these full ranks,
    # so we reuse them instead of re-sorting both columns per pair - removing the O(K^2) rank-sorts the
    # dedup did (only the O(K^2) corrcoef remains). Bit-identical: same arrays -> same average ranks.
    _eng_ranks: dict[str, np.ndarray] = {}
    # Per-column full-finiteness (zero NaN), cached alongside the ranks: the fast-path condition below
    # (``_mask.all()``) can only ever be True when BOTH sides are fully finite, so a fully-finite
    # candidate's O(K) kept-column comparisons can be BATCHED in one parallel call (see
    # ``_eng_dedup_batch_corr.one_vs_many_abs_corr_masked``) instead of K separate ``np.corrcoef``
    # calls - only for the subset of kept columns that are themselves fully finite (and hence carry a
    # buffer row); a NaN-containing kept/candidate pair keeps the original per-pair masked path.
    # APPEND-ONLY rank buffer (2026-07-13, bench-attempt-rejected note in _eng_dedup_batch_corr.py):
    # a naive per-candidate ``np.vstack`` of the CURRENT kept ranks re-copies O(K) rows on EVERY
    # candidate (O(K^2 * n) total memcpy, the SAME order as the corrcoef calls it replaces - measured
    # a NET LOSS). This buffer is written ONCE per fully-finite column (when first admitted) and never
    # copied again; the kernel takes a zero-copy VIEW of it plus a boolean "still live" mask.
    _eng_rank_buf = np.empty((len(_eng_cols_appended), len(X)), dtype=np.float64)
    _eng_row_of: dict[str, int] = {}
    _eng_next_free_row = 0
    _eng_fully_finite: dict[str, bool] = {}
    for _c in _eng_cols_appended:
        if _c in _eng_drop:
            continue
        if _c in _adaptive_fourier_keep:
            # Force-keep adaptive Fourier columns; record their array so
            # later candidates can still be deduped AGAINST them.
            _col_view_a = X[_c]
            if isinstance(_col_view_a, pd.DataFrame):
                _col_view_a = _col_view_a.iloc[:, 0]
            _eng_keep.append(_c)
            _eng_arrs[_c] = np.asarray(_col_view_a.to_numpy(), dtype=np.float64)
            continue
        # Defense in depth: if X carries duplicate column labels (a
        # caller-side data-quality issue we don't want to silently
        # mask but can't crash on either), ``X[_c]`` returns a
        # DataFrame; collapse to the first column so rank/corrcoef
        # downstream see a 1-D array and the cross-stage dedup
        # still runs.
        _col_view = X[_c]
        if isinstance(_col_view, pd.DataFrame):
            _col_view = _col_view.iloc[:, 0]
        _arr_c = np.asarray(_col_view.to_numpy(), dtype=np.float64)
        _fin_c = np.isfinite(_arr_c)
        _eng_fully_finite[_c] = bool(_fin_c.all())
        if not _fin_c.any() or _arr_c[_fin_c].std() <= 1e-12:
            _eng_keep.append(_c)
            _eng_arrs[_c] = _arr_c
            continue
        # Rank-correlate (Spearman) rather than Pearson: MRMR's plug-in
        # MI scorer quantile-bins each column before computing MI, so
        # two engineered columns related by ANY monotone reshape (square
        # vs |x| vs log|x|) project to identical bin sequences and carry
        # identical information about y. Pearson at 0.999 catches only
        # the perfect linear case (e.g. x^2 vs x^2-1); Spearman at 0.99
        # catches the full monotone-equivalent family that MRMR's
        # downstream gate cannot distinguish.
        # Full-column ranks of the candidate, cached (reused below when a pair is fully finite).
        _ranks_c = pd.Series(_arr_c).rank(method="average").to_numpy()
        _eng_ranks[_c] = _ranks_c
        _colliding_kept: list[str] = []
        # BATCHED FAST PATH: the ``_mask.all()`` fast-path condition below can only ever
        # be True when BOTH the candidate and the kept column are fully finite - so when the candidate
        # itself is fully finite, every currently-kept column that is ALSO fully finite (and hence
        # already has a row in the append-only rank buffer) can be compared in ONE batched+parallel
        # call instead of one ``np.corrcoef`` call per kept column. Kept columns with any NaN (rare
        # per this loop's own comment) fall through to the unchanged per-pair path below, unaffected.
        _fast_kept_set: set = set()
        if _eng_fully_finite[_c] and _arr_c.shape[0] >= 8 and _eng_next_free_row > 0:
            from ._eng_dedup_batch_corr import one_vs_many_abs_corr_masked
            _active_mask = np.zeros(_eng_next_free_row, dtype=np.bool_)
            _row_to_kc: dict[int, str] = {}
            for _kc in _eng_keep:
                _r = _eng_row_of.get(_kc)
                if _r is not None:
                    _active_mask[_r] = True
                    _row_to_kc[_r] = _kc
            if _active_mask.any():
                _fast_corrs = one_vs_many_abs_corr_masked(_ranks_c, _eng_rank_buf[:_eng_next_free_row], _active_mask)
                for _r, _kc in _row_to_kc.items():
                    _fast_kept_set.add(_kc)
                    if _fast_corrs[_r] >= 0.99:
                        _colliding_kept.append(_kc)
        for _kept_col in _eng_keep:
            if _kept_col in _fast_kept_set:
                continue
            _arr_k = _eng_arrs[_kept_col]
            _mask = _fin_c & np.isfinite(_arr_k)
            if _mask.sum() < 8:
                continue
            _a, _b = _arr_c[_mask], _arr_k[_mask]
            if _a.std() <= 1e-12 or _b.std() <= 1e-12:
                continue
            if bool(_mask.all()):
                # No-NaN fast path: masked subset == full column, so reuse the cached full-column ranks
                # (identical values) instead of re-sorting both columns for this pair.
                _ranks_a = _ranks_c
                _ranks_b = _eng_ranks.get(_kept_col)
                if _ranks_b is None:
                    _ranks_b = pd.Series(_arr_k).rank(method="average").to_numpy()
                    _eng_ranks[_kept_col] = _ranks_b
            else:
                _ranks_a = pd.Series(_a).rank(method="average").to_numpy()
                _ranks_b = pd.Series(_b).rank(method="average").to_numpy()
            if _ranks_a.std() <= 1e-12 or _ranks_b.std() <= 1e-12:
                continue
            _rank_corr = abs(float(np.corrcoef(_ranks_a, _ranks_b)[0, 1]))
            if np.isfinite(_rank_corr) and _rank_corr >= 0.99:
                _colliding_kept.append(_kept_col)
        if _colliding_kept:
            # Keep-higher-MI: the candidate displaces every colliding kept column it out-scores, and is itself dropped only if some colliding kept column wins.
            # ``_eng_dedup_prefer`` returns False when MI is unavailable, so an unscored cluster degrades exactly to the original first-appended policy (candidate dropped).
            _cand_loses = any(not _eng_dedup_prefer(_c, _kept_col) for _kept_col in _colliding_kept)
            if _cand_loses:
                _eng_drop.add(_c)
            else:
                for _kept_col in _colliding_kept:
                    _eng_drop.add(_kept_col)
                    _eng_keep.remove(_kept_col)
                    _eng_arrs.pop(_kept_col, None)
                _eng_keep.append(_c)
                _eng_arrs[_c] = _arr_c
                if _eng_fully_finite[_c]:
                    _eng_rank_buf[_eng_next_free_row] = _ranks_c
                    _eng_row_of[_c] = _eng_next_free_row
                    _eng_next_free_row += 1
        else:
            _eng_keep.append(_c)
            _eng_arrs[_c] = _arr_c
            if _eng_fully_finite[_c]:
                _eng_rank_buf[_eng_next_free_row] = _ranks_c
                _eng_row_of[_c] = _eng_next_free_row
                _eng_next_free_row += 1
    return _eng_keep, _eng_drop, _eng_arrs, _eng_ranks

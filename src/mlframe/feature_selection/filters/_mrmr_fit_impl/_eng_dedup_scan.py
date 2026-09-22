"""Cross-stage near-duplicate scan over the engineered columns appended in one MRMR round.

Lifted out of ``_fit_impl_core._fit_impl``'s body so it can be called rather than re-typed: the identity
test that pins the batched masked-correlation kernel against a per-pair reference kept its own copy of this
scan, which drifted out of date (it lacked the adaptive-Fourier force-keep and the duplicate-label collapse)
and so compared the kernel against a policy production no longer runs.
"""
from __future__ import annotations

from functools import cmp_to_key
from typing import Callable

import numpy as np
import pandas as pd

from .._fe_frame_ops import FE_EAGER_MATERIALIZE_MAX_BYTES

# Byte budget for the dedup rank buffer. Module-level so a test can shrink it; read at call time.
_RANK_BUF_MAX_BYTES = FE_EAGER_MATERIALIZE_MAX_BYTES
# Raw-value near-constant test, relative to the column's own magnitude (an absolute std floor treats tiny-scale columns as constant).
# Rank vectors keep their absolute floor: ranks are >= 1, so it cannot misfire there, and it matches the batched kernel.
_REL_TOL = 32.0 * np.finfo(np.float64).eps


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
    # The buffer is allocated LAZILY and grown geometrically, bounded by ``_RANK_BUF_MAX_BYTES``. It used to be a
    # K x n float64 ``np.empty`` at function entry, sized by the TOTAL appended-column count before any column was
    # inspected: 3.2 GB at the ~200-column / n=2M case this module cites, 160 GB at n=100M -- and on Windows
    # ``np.empty`` commits pages against the paging file (the documented WinError 1455 failure). Rows are only ever
    # written for fully-finite ADMITTED columns, so most of that allocation was never touched. A column that cannot
    # fit under the budget just gets no buffer row, and the per-pair path below already compares every kept column
    # without one -- so keep/drop is unchanged, only the batching coverage shrinks.
    _eng_rank_buf: np.ndarray | None = None
    # Each row's mean and centred sum-of-squares, cached when the row is appended: the buffer is append-only, so these are constants of the
    # row, and recomputing them per candidate cost two extra passes over every row of every comparison.
    _eng_row_mean: list[float] = []
    _eng_row_ss: list[float] = []
    _eng_rank_cap = 0
    _eng_row_of: dict[str, int] = {}
    _eng_next_free_row = 0
    _n_rows = len(X)
    _eng_rank_hard_cap = len(_eng_cols_appended) if _n_rows == 0 else min(len(_eng_cols_appended), int(_RANK_BUF_MAX_BYTES) // (8 * _n_rows))

    def _store_rank_row(ranks: np.ndarray) -> int | None:
        """Append ``ranks`` as the next buffer row, growing within budget; ``None`` when the budget is exhausted."""
        nonlocal _eng_rank_buf, _eng_rank_cap, _eng_next_free_row
        if _eng_next_free_row >= _eng_rank_hard_cap:
            return None
        _buf = _eng_rank_buf
        if _buf is None or _eng_next_free_row == _eng_rank_cap:
            _new_cap = min(_eng_rank_hard_cap, max(4, _eng_rank_cap * 2))
            _grown = np.empty((_new_cap, _n_rows), dtype=np.float64)
            if _buf is not None and _eng_next_free_row:
                _grown[:_eng_next_free_row] = _buf[:_eng_next_free_row]
            _buf = _grown
            _eng_rank_buf, _eng_rank_cap = _grown, _new_cap
        _row = _eng_next_free_row
        _buf[_row] = ranks
        from ._eng_dedup_batch_corr import row_mean_and_centred_ss

        _m, _ss = row_mean_and_centred_ss(_buf[_row])
        _eng_row_mean.append(float(_m))
        _eng_row_ss.append(float(_ss))
        _eng_next_free_row += 1
        return _row

    _eng_fully_finite: dict[str, bool] = {}
    # Near-duplicate at a 0.99 rank correlation is NOT transitive: A~B and B~C with A!~C is routine at that threshold. A single pass that
    # compares each candidate only against what is currently kept therefore returns a survivor SET that depends on the order candidates were
    # emitted in (A,B,C keeps {A, C}; B,A,C keeps {B} alone), and that order is whatever the upstream families happened to append, which
    # shifts whenever one of their top_k does. Processing strongest-first makes the survivor set a function of the values instead: the
    # preferred column of any colliding cluster is always already kept when the others arrive, so it is never evicted by a weaker one.
    # ``_eng_dedup_prefer`` reports False both ways when MI is unavailable, which sorts as equal and leaves those columns in emission order,
    # exactly the previous behaviour for an unscored cluster.
    def _prefer_cmp(_a: str, _b: str) -> int:
        """Order two candidates strongest-first, treating an unscored pair as equal so the sort stays stable."""
        if _a == _b:
            return 0
        if _eng_dedup_prefer(_a, _b):
            return -1
        if _eng_dedup_prefer(_b, _a):
            return 1
        return 0

    _scan_order = sorted(_eng_cols_appended, key=cmp_to_key(_prefer_cmp))
    for _c in _scan_order:
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
        if not _fin_c.any() or _arr_c[_fin_c].std() <= _REL_TOL * float(np.abs(_arr_c[_fin_c]).max()):
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
        if _eng_fully_finite[_c] and _arr_c.shape[0] >= 8 and _eng_next_free_row > 0 and _eng_rank_buf is not None:
            from ._eng_dedup_batch_corr import one_vs_many_abs_corr_masked
            _active_mask = np.zeros(_eng_next_free_row, dtype=np.bool_)
            _row_to_kc: dict[int, str] = {}
            for _kc in _eng_keep:
                _r = _eng_row_of.get(_kc)
                if _r is not None:
                    _active_mask[_r] = True
                    _row_to_kc[_r] = _kc
            if _active_mask.any():
                _fast_corrs = one_vs_many_abs_corr_masked(
                    _ranks_c,
                    _eng_rank_buf[:_eng_next_free_row],
                    _active_mask,
                    np.asarray(_eng_row_mean[:_eng_next_free_row], dtype=np.float64),
                    np.asarray(_eng_row_ss[:_eng_next_free_row], dtype=np.float64),
                )
                for _r, _kc in _row_to_kc.items():
                    _fast_kept_set.add(_kc)
                    if _fast_corrs[_r] >= 0.99:
                        _colliding_kept.append(_kc)
        for _kept_col in _eng_keep:
            if _kept_col in _fast_kept_set:
                continue
            _arr_k = _eng_arrs[_kept_col]
            # A kept column already known to be fully finite contributes an all-True mask; skip the O(n) isfinite pass for it.
            _mask = _fin_c if _eng_fully_finite.get(_kept_col, False) else (_fin_c & np.isfinite(_arr_k))
            if _mask.sum() < 8:
                continue
            _a, _b = _arr_c[_mask], _arr_k[_mask]
            if _a.std() <= _REL_TOL * float(np.abs(_a).max()) or _b.std() <= _REL_TOL * float(np.abs(_b).max()):
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
                    _row_c = _store_rank_row(_ranks_c)
                    if _row_c is not None:
                        _eng_row_of[_c] = _row_c
        else:
            _eng_keep.append(_c)
            _eng_arrs[_c] = _arr_c
            if _eng_fully_finite[_c]:
                _row_c = _store_rank_row(_ranks_c)
                if _row_c is not None:
                    _eng_row_of[_c] = _row_c
    return _eng_keep, _eng_drop, _eng_arrs, _eng_ranks

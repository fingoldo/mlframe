"""Cross-stage near-duplicate scan over the engineered columns appended in one MRMR round.

Lifted out of ``_fit_impl_core._fit_impl``'s body so it can be called rather than re-typed: the identity
test that pins the batched masked-correlation kernel against a per-pair reference kept its own copy of this
scan, which drifted out of date (it lacked the adaptive-Fourier force-keep and the duplicate-label collapse)
and so compared the kernel against a policy production no longer runs.
"""
from __future__ import annotations

from functools import cmp_to_key, partial
from typing import Callable

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
import pandas as pd

from .._fe_frame_ops import FE_EAGER_MATERIALIZE_MAX_BYTES

# Byte budget for the dedup rank buffer. Module-level so a test can shrink it; read at call time.
_RANK_BUF_MAX_BYTES = FE_EAGER_MATERIALIZE_MAX_BYTES
# Raw-value near-constant test, relative to the column's own magnitude (an absolute std floor treats tiny-scale columns as constant).
# Rank vectors keep their absolute floor: ranks are >= 1, so it cannot misfire there, and it matches the batched kernel.
# Intentional float64: relative tolerance defined against float64 machine epsilon
_REL_TOL = 32.0 * np.finfo(np.float64).eps


# Near-duplicate at a 0.99 rank correlation is NOT transitive: A~B and B~C with A!~C is routine at that threshold. A single pass that
# compares each candidate only against what is currently kept therefore returns a survivor SET that depends on the order candidates were
# emitted in (A,B,C keeps {A, C}; B,A,C keeps {B} alone), and that order is whatever the upstream families happened to append, which
# shifts whenever one of their top_k does. Processing strongest-first makes the survivor set a function of the values instead: the
# preferred column of any colliding cluster is always already kept when the others arrive, so it is never evicted by a weaker one.
# ``_eng_dedup_prefer`` reports False both ways when MI is unavailable, which sorts as equal and leaves those columns in emission order,
# exactly the previous behaviour for an unscored cluster.
def _strongest_first_cmp(_a: str, _b: str, _eng_dedup_prefer: Callable[[str, str], bool]) -> int:
    """Order two candidates strongest-first, treating an unscored pair as equal so the sort stays stable.

    Args:
        _a: the first candidate name.
        _b: the second candidate name.
        _eng_dedup_prefer: the two-argument predicate reporting whether the first candidate is preferred.

    Returns:
        -1, 0 or 1 in the ``cmp`` convention.
    """
    if _a == _b:
        return 0
    if _eng_dedup_prefer(_a, _b):
        return -1
    if _eng_dedup_prefer(_b, _a):
        return 1
    return 0


class _RankRowBuffer:
    """An append-only matrix of per-column rank rows, with each row's mean and centred sum-of-squares cached as it lands.

    Carved out of ``scan_engineered_duplicates`` to keep it under its length ceiling. The buffer is append-only, so a row's moments are
    constants of that row: caching them on append replaces two extra passes over every row on every later comparison. Growth is doubling,
    bounded by a hard cap derived from the byte budget, and ``store`` reports exhaustion rather than raising so the caller can fall back
    to the unbatched comparison for the rest of the scan.

    Parameters
    ----------
    n_rows
        Length of one rank row, i.e. the number of samples.
    hard_cap
        The most rows that may ever be stored, from the byte budget.
    """

    def __init__(self, *, n_rows: int, hard_cap: int) -> None:
        self._n_rows = int(n_rows)
        self._hard_cap = int(hard_cap)
        self.matrix: np.ndarray | None = None
        self.means: list[float] = []
        self.ss: list[float] = []
        self.used = 0
        self._cap = 0

    def store(self, ranks: np.ndarray) -> int | None:
        """Append ``ranks`` as the next row, growing within budget.

        Args:
            ranks: the rank row to store.

        Returns:
            The row index it landed on, or ``None`` when the row budget is exhausted.
        """
        if self.used >= self._hard_cap:
            return None
        buf = self.matrix
        if buf is None or self.used == self._cap:
            new_cap = min(self._hard_cap, max(4, self._cap * 2))
            grown = np.empty((new_cap, self._n_rows), dtype=_crit_np_dtype())
            if buf is not None and self.used:
                grown[: self.used] = buf[: self.used]
            buf = grown
            self.matrix, self._cap = grown, new_cap
        row = self.used
        buf[row] = ranks
        from ._eng_dedup_batch_corr import row_mean_and_centred_ss

        mean, ss = row_mean_and_centred_ss(buf[row])
        self.means.append(float(mean))
        self.ss.append(float(ss))
        self.used += 1
        return row


_PRE_RANK_MAX_BYTES = 512 * 1024 * 1024  # the ranks of all fully finite candidates are held until the scan reaches them (it keeps them anyway); more than this are ranked one by one


def _precompute_full_ranks(X: pd.DataFrame, scan_order: list, fourier_keep: set) -> "dict[str, np.ndarray]":
    """Average ranks of every fully finite, non-constant candidate column, computed together by one parallel kernel (the scan ranked them one at a time with pandas, 2.2 s of a 100000-row fit)."""
    n = len(X)
    if n < 8 or len(scan_order) < 4:
        return {}
    # the batch obeys the rank-buffer byte budget (a column over it is ranked one by one when the scan reaches it), so no allocation here exceeds what the scan itself may hold
    max_cols = min(int(_PRE_RANK_MAX_BYTES), int(_RANK_BUF_MAX_BYTES)) // (8 * n)
    if max_cols < 2:
        return {}
    names: list = []
    arrays = []
    for c in scan_order:
        if c in fourier_keep or c not in X.columns:
            continue
        v = X[c]
        if isinstance(v, pd.DataFrame):
            v = v.iloc[:, 0]
        a = np.asarray(v.to_numpy(), dtype=_crit_np_dtype())
        if not np.isfinite(a).all() or a.std() <= _REL_TOL * float(np.abs(a).max()):
            continue
        names.append(c)
        arrays.append(np.ascontiguousarray(a))
        if len(names) >= max_cols:
            break
    if len(names) < 2:
        return {}
    from numba.typed import List as _TypedList

    from ._eng_dedup_batch_corr import average_ranks_many

    cols = _TypedList()
    for a in arrays:
        cols.append(a)
    ranks = np.empty((len(names), n), dtype=np.float64)  # Intentional float64: rank values kept at the width pandas ranks have
    average_ranks_many(cols, ranks)
    return {c: ranks[i] for i, c in enumerate(names)}


def _column_1d(X: pd.DataFrame, name: str) -> pd.Series:
    """``X[name]`` as a Series; with duplicate column labels the first of them."""
    col = X[name]
    return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col


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
    _eng_row_of: dict[str, int] = {}
    _n_rows = len(X)
    _rank_rows = _RankRowBuffer(
        n_rows=_n_rows,
        hard_cap=len(_eng_cols_appended) if _n_rows == 0 else min(len(_eng_cols_appended), int(_RANK_BUF_MAX_BYTES) // (8 * _n_rows)),
    )

    _eng_fully_finite: dict[str, bool] = {}
    _scan_order = sorted(_eng_cols_appended, key=cmp_to_key(partial(_strongest_first_cmp, _eng_dedup_prefer=_eng_dedup_prefer)))
    _pre_ranks = _precompute_full_ranks(X, _scan_order, _adaptive_fourier_keep)
    _orders = _CandidateOrderPrefetch(X, _scan_order)
    for _c in _scan_order:
        if _c in _eng_drop:
            continue
        if _c in _adaptive_fourier_keep:
            # Force-keep adaptive Fourier columns; record their array so
            # later candidates can still be deduped AGAINST them.
            _eng_keep.append(_c)
            _eng_arrs[_c] = np.asarray(_column_1d(X, _c).to_numpy(), dtype=_crit_np_dtype())
            continue
        # Defense in depth: if X carries duplicate column labels (a
        # caller-side data-quality issue we don't want to silently
        # mask but can't crash on either), ``X[_c]`` returns a
        # DataFrame; collapse to the first column so rank/corrcoef
        # downstream see a 1-D array and the cross-stage dedup
        # still runs.
        _arr_c = np.asarray(_column_1d(X, _c).to_numpy(), dtype=_crit_np_dtype())
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
        _ranks_c = _pre_ranks.pop(_c, None)
        if _ranks_c is None:
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
        if _eng_fully_finite[_c] and _arr_c.shape[0] >= 8 and _rank_rows.used > 0 and _rank_rows.matrix is not None:
            _active_mask = np.zeros(_rank_rows.used, dtype=np.bool_)
            _row_to_kc: dict[int, str] = {}
            _collect_kept_engineered_rows(_eng_keep, _eng_row_of, _active_mask, _row_to_kc)
            _masked_corr_prefilter(_active_mask, _ranks_c, _rank_rows, _row_to_kc, _fast_kept_set, _colliding_kept)
        _pair_collisions(_colliding_kept, _eng_keep, _fast_kept_set, _arr_c, _fin_c, _ranks_c, _eng_arrs, _eng_ranks, _eng_fully_finite, cand_order=lambda _c=_c: _orders.get(_c))
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
                    _row_c = _rank_rows.store(_ranks_c)
                    if _row_c is not None:
                        _eng_row_of[_c] = _row_c
        else:
            _eng_keep.append(_c)
            _eng_arrs[_c] = _arr_c
            if _eng_fully_finite[_c]:
                _row_c = _rank_rows.store(_ranks_c)
                if _row_c is not None:
                    _eng_row_of[_c] = _row_c
    return _eng_keep, _eng_drop, _eng_arrs, _eng_ranks


class _CandidateOrderPrefetch:
    """Finite-row sort orders of the scan's candidates, computed ``_PREFETCH`` candidates ahead on the edge-jobs pool (``np.argsort`` releases the GIL).

    The masked Spearman of ``_pair_collisions`` needs the candidate's own stable sort, about 8 ms per 100000-row column, which ran serially once per candidate
    (1.3 s of a 100000 x 100 fit). The order depends on the candidate column alone, so the next block of candidates in scan order is sorted together when the
    first of them asks; candidates that never ask cost at most one block."""

    _PREFETCH = 32

    def __init__(self, X, scan_order):
        self._X = X
        self._names = list(scan_order)
        self._pos = {c: i for i, c in enumerate(self._names)}
        self._orders: dict = {}

    def get(self, name):
        """The order of candidate ``name`` (``None`` for a name outside the scan)."""
        hit = self._orders.get(name)
        if hit is not None:
            return hit
        i = self._pos.get(name)
        if i is None:
            return None
        from .._edge_pool import map_on_edge_pool
        from ._eng_dedup_batch_corr import finite_sort_order

        todo = [c for c in self._names[i : i + self._PREFETCH] if c not in self._orders and c in self._X.columns]

        def _one(c):
            """The finite-row order of candidate ``c`` at the criterion dtype, as the scan builds the column."""
            return finite_sort_order(np.ascontiguousarray(np.asarray(_column_1d(self._X, c).to_numpy(), dtype=_crit_np_dtype())), cache=False)

        for c, order in zip(todo, map_on_edge_pool(_one, todo, min_items=2)):
            self._orders[c] = order
        if i >= self._PREFETCH:
            self._orders.pop(self._names[i - self._PREFETCH], None)  # the scan never asks for a candidate behind it again
        return self._orders.get(name)


def _pair_collisions(_colliding_kept, _eng_keep, _fast_kept_set, _arr_c, _fin_c, _ranks_c, _eng_arrs, _eng_ranks, _eng_fully_finite, cand_order=None):
    """Append each kept column the candidate Spearman-collides with (|rho| >= 0.99) to ``_colliding_kept``, skipping batched ones.

    Pairs that are not fully finite are ranked over their jointly finite rows; they are collected and scored together by one parallel kernel (two pandas ranks per pair, one pair at a time,
    were about 5 s of a 100000-row fit), and the colliding columns are appended in the order of ``_eng_keep`` as before."""
    _decided: list = []  # (kept column, True/False collides) or (kept column, None) for a pair left to the masked kernel
    _masked: list = []
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
            _decided.append((_kept_col, None))
            _masked.append(_kept_col)
            continue
        if _ranks_a.std() <= 1e-12 or _ranks_b.std() <= 1e-12:
            continue
        _rank_corr = abs(float(np.corrcoef(_ranks_a, _ranks_b)[0, 1]))
        _decided.append((_kept_col, bool(np.isfinite(_rank_corr) and _rank_corr >= 0.99)))
    _masked_hit: set = set()
    if _masked:
        from numba.typed import List as _TypedList

        from ._eng_dedup_batch_corr import finite_sort_order, finite_sort_orders, masked_abs_spearman_presorted

        _others = _TypedList()
        _orders = _TypedList()
        _kept_cols = [np.ascontiguousarray(_eng_arrs[_kept_col]) for _kept_col in _masked]
        for _col in _kept_cols:
            _others.append(_col)
        for _order in finite_sort_orders(_kept_cols):  # a kept column is sorted once for the whole fit, not once per candidate; the columns not sorted yet are sorted together on the pool
            _orders.append(_order)
        _arr_cc = np.ascontiguousarray(_arr_c)
        _rho = np.empty(len(_masked), dtype=np.float64)  # Intentional float64: |rho| statistic vector
        _order_c = cand_order() if cand_order is not None else None
        if _order_c is None:
            _order_c = finite_sort_order(_arr_cc, cache=False)
        masked_abs_spearman_presorted(_arr_cc, np.ascontiguousarray(_fin_c), _order_c, _others, _orders, float(_REL_TOL), _rho)
        _masked_hit = {_kept_col for _kept_col, _r in zip(_masked, _rho) if _r >= 0.99}
    for _kept_col, _collides in _decided:
        if _collides is None:
            _collides = _kept_col in _masked_hit
        if _collides:
            _colliding_kept.append(_kept_col)


def _collect_kept_engineered_rows(_eng_keep, _eng_row_of, _active_mask, _row_to_kc):
    """Collect the rank-matrix rows of the engineered columns that are kept."""
    for _kc in _eng_keep:
        _r = _eng_row_of.get(_kc)
        if _r is not None:
            _active_mask[_r] = True
            _row_to_kc[_r] = _kc


def _masked_corr_prefilter(_active_mask, _ranks_c, _rank_rows, _row_to_kc, _fast_kept_set, _colliding_kept):
    """Pre-filter engineered-duplicate candidates by masked one-vs-many absolute rank correlation."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import one_vs_many_abs_corr_masked

    if _active_mask.any():
        _fast_corrs = one_vs_many_abs_corr_masked(
            _ranks_c,
            _rank_rows.matrix[: _rank_rows.used],
            _active_mask,
            np.asarray(_rank_rows.means[: _rank_rows.used], dtype=_crit_np_dtype()),
            np.asarray(_rank_rows.ss[: _rank_rows.used], dtype=_crit_np_dtype()),
        )
        for _r, _kc in _row_to_kc.items():
            _fast_kept_set.add(_kc)
            if _fast_corrs[_r] >= 0.99:
                _colliding_kept.append(_kc)

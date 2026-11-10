"""Per-pair memory of the batched usability ``|corr|`` table, keyed by the CONTENT of the operand rows.

``dispatch_batch_pair_usability_corr`` already remembers the table of a whole call (same target, same operand matrix, same pair arrays). The dominant-pair scan of a conjunction fit asks twice for
overlapping pair sets over the same operands: the second request (the first sweep's retry over a re-filtered pool) has other pair arrays, and in operand-chunked groups other row orders, so the whole-call
key never matches and every pair was scored again (100000 rows x 1000 columns: 513591 pairs in 19.5 s, then 330868 pairs in 13.4 s).

A ``(pair, form)`` value is a pure function of the target, the two operand rows and the form id: the warp kernel gives every (pair, form) its own warp, and the float width is decided per pair
from its two operands (a non-finite value or a large offset in either operand selects the double kernel), so the value does not depend on which other pairs were in the call. Here each operand row is
identified by its content signature and each ordered pair by the two row ids, so a later call over any subset or reordering of the same rows reads the stored values and scores only the pairs it has
not seen. Only values produced by the warp kernel are stored (the CPU kernel is double precision and its values differ in the last digits).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np

#: Pairs one table may hold (a 1000-column fit asks for 513591 pairs x 5 forms; the float32 table of that is 10 MB).
_MAX_PAIRS_PER_TABLE = 4_000_000
#: Tables kept (one per target and form list).
_MAX_TABLES = 2

_LOCK = threading.Lock()
_UIDS: "dict[tuple, int]" = {}
_TABLES: "dict[tuple, dict]" = {}


def _row_ids(operand_matrix: np.ndarray) -> np.ndarray:
    """Small integer id of every row of ``operand_matrix`` by its content signature (equal rows get equal ids across calls)."""
    from ._fe_resident_operands import operand_signature

    sigs = [operand_signature(np.ascontiguousarray(operand_matrix[i])) for i in range(int(operand_matrix.shape[0]))]
    with _LOCK:
        return np.fromiter((_UIDS.setdefault(s, len(_UIDS)) for s in sigs), dtype=np.int64, count=len(sigs))


def _table_key(y: np.ndarray, form_ids: np.ndarray) -> tuple:
    """The key of the target and form list a table belongs to."""
    from ._fe_resident_operands import operand_signature

    return (operand_signature(np.ascontiguousarray(y)), tuple(int(f) for f in form_ids))


def clear() -> None:
    """Forget every stored table and row id (tests; a fit does not need it: the key carries the content)."""
    with _LOCK:
        _UIDS.clear()
        _TABLES.clear()


def serve(
    y: np.ndarray,
    operand_matrix: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    form_ids: np.ndarray,
    compute: "Callable[[np.ndarray, np.ndarray, Optional[str]], tuple[np.ndarray, str]]",
) -> "tuple[np.ndarray, str]":
    """``(table, backend)`` of the pairs, from the stored values where known and ``compute(pair_a, pair_b, force_backend)`` for the rest.

    ``compute`` scores the given pairs over ``operand_matrix`` and returns ``(table, backend)``; its result is stored when the backend is the warp kernel."""
    tkey = _table_key(y, form_ids)
    ids = _row_ids(operand_matrix)
    codes = (ids[pair_a.astype(np.int64, copy=False)] << 32) | ids[pair_b.astype(np.int64, copy=False)]
    with _LOCK:
        stored = _TABLES.get(tkey)
    if stored is None:
        table, backend = compute(pair_a, pair_b, None)
        if backend == "cuda_warp":
            _store(tkey, codes, table)
        return table, backend
    s_codes, s_values, s_backend = stored["codes"], stored["values"], stored["backend"]
    pos = np.searchsorted(s_codes, codes)
    pos[pos >= s_codes.shape[0]] = 0
    hit = s_codes[pos] == codes
    if bool(hit.all()):
        return s_values[pos].copy(), s_backend
    miss = ~hit
    new_table, backend = compute(pair_a[miss], pair_b[miss], s_backend)
    out = np.empty((codes.shape[0], s_values.shape[1]), dtype=new_table.dtype)
    out[hit] = s_values[pos[hit]]
    out[miss] = new_table
    if backend == s_backend:
        _store(tkey, codes[miss], new_table)
    return out, (s_backend if backend == s_backend else backend)


def _store(tkey: tuple, codes: np.ndarray, values: np.ndarray) -> None:
    """Merge ``values`` of ``codes`` into the table of ``tkey`` (the first value of a repeated code is kept)."""
    uniq, first = np.unique(codes, return_index=True)
    values = values[first]
    with _LOCK:
        cur: Optional[dict] = _TABLES.get(tkey)
        if cur is not None:
            merged = np.concatenate([cur["codes"], uniq])
            vals = np.concatenate([cur["values"], values.astype(cur["values"].dtype, copy=False)])
            merged_u, idx = np.unique(merged, return_index=True)  # the stored value wins over an equal new code
            uniq, values = merged_u, vals[idx]
        if uniq.shape[0] > _MAX_PAIRS_PER_TABLE:
            return
        if tkey not in _TABLES and len(_TABLES) >= _MAX_TABLES:
            _TABLES.pop(next(iter(_TABLES)))  # evict-ok: memo; a miss re-scores the pairs
        _TABLES[tkey] = {"codes": uniq, "values": values, "backend": "cuda_warp"}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Prefetch of the raw x raw pairs
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#: Pair-form evaluations (pairs x forms x scored rows) from which the prefetch is worth a thread: below it the dominant-pair scan costs well under a second.
_PREFETCH_MIN_WORK = 2_000_000_000
#: Operands per block of the prefetch: a block pair scores up to this many squared pairs in one call.
_PREFETCH_BLOCK = 160
_PREFETCH: "dict[int, threading.Thread]" = {}


def _prefetch_run(names: list, columns: Callable[[str], np.ndarray], y_batch: np.ndarray, stride: int) -> None:
    """Score every ordered pair of the raw operands (in column order) in blocks and leave the values in the table (the dispatcher stores them)."""
    from .batch_pair_usability_corr_gpu import ALL_PAIR_FORM_IDS, dispatch_batch_pair_usability_corr

    p = len(names)
    rows = [None] * p

    def row(i: int) -> np.ndarray:
        """The operand's row exactly as the scan builds it: the crit-dtype column, subsampled with the target's stride."""
        if rows[i] is None:
            v = columns(names[i])
            rows[i] = np.ascontiguousarray(v[::stride] if stride > 1 else v)
        return rows[i]

    blk = _PREFETCH_BLOCK
    for ci in range(0, p, blk):
        for cj in range(ci, p, blk):
            ia = list(range(ci, min(p, ci + blk)))
            ib = list(range(cj, min(p, cj + blk)))
            used = sorted(set(ia) | set(ib))
            pos = {i: k for k, i in enumerate(used)}
            pairs = [(a, b) for a in ia for b in ib if a < b]
            if not pairs:
                continue
            matrix = np.vstack([row(i) for i in used])
            pa = np.array([pos[a] for a, _ in pairs], dtype=np.int64)
            pb = np.array([pos[b] for _, b in pairs], dtype=np.int64)
            dispatch_batch_pair_usability_corr(y_batch, matrix, pa, pb, form_ids=ALL_PAIR_FORM_IDS)
    for i in range(p):
        rows[i] = None


def start_prefetch(estimator: Any, X: Any) -> None:
    """Start a daemon thread that scores the raw x raw pairs of ``X`` for the later dominant-pair scan; a no-op without a GPU, without the continuous target, on a frame that is not pandas or
    when the work is small. Failures only mean the scan scores the pairs itself."""
    try:
        from ._fe_usability_signal import _corr_stride, _crit_np_dtype
        from ._gpu_policy import gpu_globally_disabled
        from .batch_pair_usability_corr_gpu import _CUDA_AVAIL, ALL_PAIR_FORM_IDS

        y = getattr(estimator, "_fe_prewarp_y_continuous_", None)
        if y is None or not _CUDA_AVAIL or gpu_globally_disabled() or not hasattr(X, "columns") or len(X) != len(y):
            return
        import pandas as pd

        names = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        stride = int(_corr_stride(len(y)))
        scored = len(y) // max(1, stride)
        if len(names) * (len(names) - 1) // 2 * len(ALL_PAIR_FORM_IDS) * scored < _PREFETCH_MIN_WORK:
            return
        dtype = _crit_np_dtype()
        y_arr = np.asarray(y, dtype=dtype).reshape(-1)
        y_batch = np.asarray(y_arr[::stride] if stride > 1 else y_arr, dtype=dtype)

        def columns(name: str) -> np.ndarray:
            """One raw column as the scan's operand: the crit dtype, one-dimensional."""
            return np.asarray(X[name], dtype=dtype).ravel()

        def target() -> None:
            """Thread body: errors are logged and dropped (the scan then scores the pairs it needs)."""
            try:
                _prefetch_run(names, columns, y_batch, stride)
            except Exception as exc:  # the prefetch is an optimisation only
                import logging

                logging.getLogger(__name__).warning("usability pair prefetch stopped (%s: %s); the scan scores its own pairs", type(exc).__name__, exc)

        thread = threading.Thread(target=target, name="usability-pair-prefetch", daemon=True)
        _PREFETCH[id(estimator)] = thread
        thread.start()
    except Exception as exc:  # never break a fit for an optimisation
        import logging

        logging.getLogger(__name__).debug("usability pair prefetch not started: %s", exc)


def join_prefetch(estimator: Any) -> None:
    """Wait for the prefetch thread of ``estimator`` (started by :func:`start_prefetch`), if any: the scan that follows then reads the pairs from the table instead of scoring them twice."""
    thread = _PREFETCH.pop(id(estimator), None)
    if thread is not None:
        thread.join()


__all__ = ["clear", "join_prefetch", "serve", "start_prefetch"]

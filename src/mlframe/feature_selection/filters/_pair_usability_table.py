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
from typing import Callable, Optional

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


__all__ = ["clear", "serve"]

"""Bounded resident store for the binned candidate pool of the greedy CMI construction.

At 500000 rows x ~3900 candidates the int64 codes are 15 GB, and even the int16 ones (3.9 GB) do not fit a 4 GB card; the whole pool resident left no room for any other stage, which fell to the host path
(a 6800 s fit). The pool here keeps the codes in the narrowest unsigned dtype (one byte for up to 250 bins) and, once the device share is used up, in host memory; reading a candidate widens it back to the
int64 device array every consumer expects, so a block of candidates costs one small upload and cast each instead of keeping the pool resident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

#: The most device memory the narrow codes of one pool may take, whatever the room is.
_DEVICE_POOL_BYTES = 512 * 1024 * 1024


class CandidateCodePool(dict):
    """``name -> int64 device codes``, stored narrow (device while the budget lasts, host after) and widened on read; ``bool(pool)`` is true once a candidate is stored. Built on the cupy module ``cp`` with at most ``device_budget`` bytes of narrow codes on the device."""

    def __init__(self, cp: Any, device_budget: int) -> None:
        super().__init__()
        self._cp = cp
        self._device_left = int(device_budget)

    def put(self, name: str, codes: Any, narrow_dtype: Any) -> None:
        """Store the device ``codes`` of candidate ``name`` as ``narrow_dtype`` (chosen by the caller from the bin count: no value-range read-back)."""
        cp = self._cp
        narrow = codes.astype(narrow_dtype, copy=False)
        if int(narrow.nbytes) <= self._device_left:
            self._device_left -= int(narrow.nbytes)
            super().__setitem__(name, cp.array(narrow, copy=True))
        else:
            super().__setitem__(name, cp.asnumpy(narrow))

    def _widen(self, stored: Any) -> Any:
        """The int64 device array of one stored narrow column."""
        cp = self._cp
        dev = stored if isinstance(stored, cp.ndarray) else cp.asarray(stored)
        return dev.astype(cp.int64)

    def __getitem__(self, name: str) -> Any:
        """Widened device codes of ``name`` (``KeyError`` when absent)."""
        return self._widen(super().__getitem__(name))

    def get(self, name: str, default: Any = None) -> Any:
        """Widened device codes of ``name``, or ``default``."""
        stored = super().get(name)
        return default if stored is None else self._widen(stored)


def bin_candidate_pool(
    cp: Any, names: Sequence[str], columns: Sequence[np.ndarray], nbins: int, qbin_dtype: Any, block_codes: Callable[..., "list | None"], fingerprints: Callable[..., "list[int]"], budget_rows_bytes: int
) -> "tuple[CandidateCodePool, dict[str, int]] | None":
    """Bin the candidate ``columns`` in blocks, fingerprint each block while its int64 codes are alive and keep only the narrow codes in a :class:`CandidateCodePool`.

    ``block_codes(columns, nbins, qbin_dtype)`` is ``resident_qbin_codes_block``, ``fingerprints(cp, columns)`` is ``pool_fingerprints``; ``budget_rows_bytes`` bounds the int64 bytes of one block. ``None`` when
    a block cannot be binned on the device (the caller keeps its own fallback)."""
    from ._device_sort_safe import device_room

    n = int(columns[0].shape[0])
    per_col = max(1, 8 * n)
    step = max(2, int(max(per_col, int(budget_rows_bytes)) // per_col))
    room = int(device_room(cp))
    narrow_dtype = np.uint8 if int(nbins) + 1 <= 255 else np.int32  # codes are 0..nbins (NaN code nbins + 1 at most)
    pool = CandidateCodePool(cp, min(_DEVICE_POOL_BYTES, room // 8))
    fps: "dict[str, int]" = {}
    for c0 in range(0, len(columns), step):
        part_cols = list(columns[c0 : c0 + step])
        part_names = list(names[c0 : c0 + step])
        codes = block_codes(part_cols, nbins, qbin_dtype)
        if codes is None:
            return None
        for nm, fp in zip(part_names, fingerprints(cp, codes)):
            fps[nm] = int(fp)
        for nm, c in zip(part_names, codes):
            pool.put(nm, c, narrow_dtype)
        del codes
    return pool, fps

"""Integer-coded pair coverage for the pairwise combo enumerator.

The greedy pass in :mod:`enumerator` asks, for every candidate combo, how many of its axis-value pairs are still uncovered.
Done with Python sets that is ~146k tuples per candidate across ~540 axes. Here each pair of axis values gets a fixed integer
id, a combo becomes a vector of value indices, and one step's candidates are scored in a single numba ``prange`` pass.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numba import njit, prange


class PairSpace:
    """Fixed integer ids for every pair ``(axis_i=value_a, axis_j=value_b)`` with ``i < j``."""

    def __init__(self, axes: dict[str, tuple[Any, ...]], to_key: Callable[[Any], Any]) -> None:
        self.names: list[str] = list(axes)
        self.values: list[tuple[Any, ...]] = [tuple(v) for v in axes.values()]
        self.keys: list[list[Any]] = [[to_key(x) for x in vals] for vals in self.values]
        for name, keys in zip(self.names, self.keys):
            if len(set(map(repr, keys))) != len(keys):
                raise ValueError(f"axis {name!r} has duplicate values; pair ids would not be unique")
        n_axes = len(self.names)
        self.nv = np.array([len(v) for v in self.values], dtype=np.int64)
        self.offset = np.zeros((n_axes, n_axes), dtype=np.int64)
        starts: list[int] = []
        block_i: list[int] = []
        block_j: list[int] = []
        total = 0
        for i in range(n_axes):
            for j in range(i + 1, n_axes):
                self.offset[i, j] = total
                starts.append(total)
                block_i.append(i)
                block_j.append(j)
                total += int(self.nv[i]) * int(self.nv[j])
        self.size = total
        self._starts = np.array(starts, dtype=np.int64)
        self._block_i = block_i
        self._block_j = block_j

    def codes(self, keyed_values: list[Any]) -> np.ndarray:
        """Value index per axis for one combo; ``-1`` when a value is outside the axis (its pairs are not in the universe)."""
        out = np.empty(len(self.names), dtype=np.int64)
        for i, v in enumerate(keyed_values):
            try:
                out[i] = self.keys[i].index(v)
            except ValueError:
                out[i] = -1
        return out

    def decode(self, pid: int) -> tuple[str, Any, str, Any]:
        """The ``(name_i, value_i, name_j, value_j)`` tuple of a pair id, with the raw axis values."""
        block = int(np.searchsorted(self._starts, pid, side="right")) - 1
        i, j = self._block_i[block], self._block_j[block]
        rem = pid - int(self._starts[block])
        ci, cj = divmod(rem, int(self.nv[j]))
        return self.names[i], self.values[i][ci], self.names[j], self.values[j][cj]


@njit(parallel=True)
def count_uncovered(codes: np.ndarray, nv: np.ndarray, offset: np.ndarray, uncovered: np.ndarray) -> np.ndarray:
    """For each candidate row of *codes*, how many of its in-universe pairs are still uncovered (candidates in parallel)."""
    n_cand, n_axes = codes.shape
    out = np.zeros(n_cand, dtype=np.int64)
    for k in prange(n_cand):
        count = 0
        for i in range(n_axes):
            ci = codes[k, i]
            if ci < 0:
                continue
            for j in range(i + 1, n_axes):
                cj = codes[k, j]
                if cj < 0:
                    continue
                if uncovered[offset[i, j] + ci * nv[j] + cj]:
                    count += 1
        out[k] = count
    return out


@njit
def mark_covered(codes: np.ndarray, nv: np.ndarray, offset: np.ndarray, covered: np.ndarray) -> None:
    """Set every in-universe pair of one combo as covered."""
    n_axes = codes.shape[0]
    for i in range(n_axes):
        ci = codes[i]
        if ci < 0:
            continue
        for j in range(i + 1, n_axes):
            cj = codes[j]
            if cj < 0:
                continue
            covered[offset[i, j] + ci * nv[j] + cj] = True

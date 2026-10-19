"""Family-wise test of a pair's weaker operand against the information its stronger operand already carries.

The order-2 maxT floor guards a pair against being the best of many chance pairs, but it compares the pair's joint MI with a floor, and a pair that contains an operand which clears that floor alone
(a selected composite) clears it with ANY partner. The evidence for the pair is then what the partner adds, ``CMI(partner; y | strong)``, and that has to be judged as the best of the many partners
that were tried: with a hundred noise columns, one of them clears a per-column permutation null by chance.

For a strong operand ``z`` this computes, once, the conditional MI of every candidate column given ``z`` and the null distribution of the MAXIMUM of those over all candidates (the target permuted
within the strata of ``z``, which keeps the ``z``-``y`` association and breaks every partner-``y`` one). A partner clears the test when its conditional MI exceeds the ``q`` quantile of that maximum.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

logger = logging.getLogger(__name__)

#: Permutations of the maximum and the quantile read from it (the same q as the pool's order-2 maxT floor).
N_PERMUTATIONS = 40
QUANTILE = 0.95
#: Rows above this are strided away for the test, all columns together (decision margin is wide; see MLFRAME_PAIR_NULL_MAX_ROWS).
MAX_ROWS = 250_000


@njit(cache=True, parallel=True)
def _conditional_mi_columns(codes: np.ndarray, widths: np.ndarray, y: np.ndarray, z: np.ndarray, ky: int, kz: int, out: np.ndarray) -> None:
    """``out[j] = CMI(codes[j]; y | z)`` in nats (plug-in) for every row ``j`` of the ``(p, n)`` code matrix."""
    p, n = codes.shape
    inv_n = 1.0 / n
    for j in prange(p):
        kw = widths[j]
        cnt = np.zeros((kz, kw, ky), dtype=np.int32)
        for i in range(n):
            cnt[z[i], codes[j, i], y[i]] += 1
        n_zy = np.zeros((kz, ky), dtype=np.int64)
        n_z = np.zeros(kz, dtype=np.int64)
        for a in range(kz):
            for b in range(kw):
                for c in range(ky):
                    v = cnt[a, b, c]
                    n_zy[a, c] += v
                    n_z[a] += v
        cmi = 0.0
        for a in range(kz):
            for b in range(kw):
                n_zw = 0
                for c in range(ky):
                    n_zw += cnt[a, b, c]
                if n_zw == 0:
                    continue
                for c in range(ky):
                    v = cnt[a, b, c]
                    if v > 0:
                        cmi += v * inv_n * np.log(v * n_z[a] / (n_zw * n_zy[a, c]))
        out[j] = cmi


class ConditionalPartnerGate:
    """Per-strong-operand family-wise test of the partners of a pair; the null of each strong operand is computed once and reused for every pair that contains it.

    ``data`` is the ``(n, p)`` code matrix, ``classes_y`` the target codes and ``candidates`` the column indices a partner may be.
    """

    def __init__(self, data: np.ndarray, classes_y: np.ndarray, candidates: "list[int]", seed: int = 0):
        self._data = data
        self._y = np.ascontiguousarray(classes_y, dtype=np.int64).ravel()
        self._cand = [int(c) for c in candidates]
        self._pos = {c: k for k, c in enumerate(self._cand)}
        self._seed = int(seed)
        self._cache: "dict[int, Optional[tuple[np.ndarray, float]]]" = {}

    def clears(self, weak_idx: int, strong_idx: int) -> bool:
        """Whether column ``weak_idx`` carries information about the target beyond column ``strong_idx`` that stands out among all the candidate partners.

        ``True`` when the test cannot be run (a constant target, an unknown column): no evidence is available, so the pair keeps the verdict it had.
        """
        strong_idx, weak_idx = int(strong_idx), int(weak_idx)
        if strong_idx not in self._cache:
            self._cache[strong_idx] = self._compute(strong_idx)
        result = self._cache[strong_idx]
        k = self._pos.get(weak_idx)
        if result is None or k is None:
            return True
        observed, floor = result
        return bool(observed[k] > floor)

    def _compute(self, strong_idx: int) -> "Optional[tuple[np.ndarray, float]]":
        """``(conditional MI of every candidate given the strong operand, the q quantile of the permutation maximum)``, or ``None`` for a degenerate target."""
        y = self._y
        n = int(y.shape[0])
        if n == 0 or int(y.max()) == int(y.min()):
            return None
        step = max(1, n // MAX_ROWS)
        rows = slice(None, None, step)
        y = np.ascontiguousarray(y[rows])
        z = np.asarray(self._data[rows, strong_idx]).astype(np.int64)
        codes = np.ascontiguousarray(np.asarray(self._data[rows][:, self._cand]).T)
        codes = codes.astype(np.int64) - codes.min(axis=1, keepdims=True).astype(np.int64)
        widths = (codes.max(axis=1) + 1).astype(np.int64)
        y = y - y.min()
        z = z - z.min()
        ky, kz = int(y.max()) + 1, int(z.max()) + 1
        observed = np.empty(len(self._cand), dtype=np.float64)
        with parallel_kernel_entry():
            _conditional_mi_columns(codes, widths, y, z, ky, kz, observed)
        order = np.argsort(z, kind="stable")
        bounds = np.flatnonzero(np.diff(z[order])) + 1
        strata = np.split(order, bounds)
        rng = np.random.default_rng(self._seed + 7919 * strong_idx)
        maxima = np.empty(N_PERMUTATIONS, dtype=np.float64)
        scratch = np.empty(len(self._cand), dtype=np.float64)
        for r in range(N_PERMUTATIONS):
            y_perm = y.copy()
            for idx in strata:
                y_perm[idx] = y[rng.permutation(idx)]
            with parallel_kernel_entry():
                _conditional_mi_columns(codes, widths, y_perm, z, ky, kz, scratch)
            maxima[r] = scratch.max()
        return observed, float(np.quantile(maxima, QUANTILE))

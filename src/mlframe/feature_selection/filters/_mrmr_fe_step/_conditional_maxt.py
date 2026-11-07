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


@njit(cache=True, parallel=True)
def _gather_transposed(data: np.ndarray, step: int, cand: np.ndarray, out: np.ndarray) -> None:
    """``out[j, i] = data[i * step, cand[j]]`` for the ``(candidates, rows)`` output, in 64 x 64 tiles, parallel over the candidate tiles: the fancy column gather and the transpose of numpy
    (two strided passes, 4 s at 250000 rows x 600 candidates) in one."""
    p, n = out.shape
    tile = 64
    for j0 in prange((p + tile - 1) // tile):
        j_lo = j0 * tile
        j_hi = min(j_lo + tile, p)
        for i0 in range(0, n, tile):
            i_hi = min(i0 + tile, n)
            for j in range(j_lo, j_hi):
                c = cand[j]
                for i in range(i0, i_hi):
                    out[j, i] = data[i * step, c]


def _candidate_code_matrix(gathered: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(shifted codes (candidates, rows), widths)`` of the ``(rows, candidates)`` gathered codes: each candidate's smallest code becomes 0, the matrix is int16 when every width fits (the kernels
    read it once per permutation, narrow codes cut that traffic 4x; the counts are the same integers) and int64 otherwise.

    The transpose is the parallel tiled copy of the family gate (numpy's strided ``.T`` copy was 2.4 s of the 2.8 s this took at 250000 rows x 600 candidates)."""
    from .._family_maxt_gate import _transpose_into

    n_rows, n_cand = int(gathered.shape[0]), int(gathered.shape[1])
    if gathered.dtype.kind not in "iu" or gathered.dtype.itemsize > 4:
        codes = np.ascontiguousarray(gathered.T)
    else:
        codes = np.empty((n_cand, n_rows), dtype=gathered.dtype)
        with parallel_kernel_entry():
            _transpose_into(np.ascontiguousarray(gathered), codes, 0)
    return _shift_codes(codes)


@njit(cache=True, parallel=True)
def _row_range(codes: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    """Smallest and largest code of every row of ``codes`` (parallel over the rows)."""
    for j in prange(codes.shape[0]):
        lo = codes[j, 0]
        hi = codes[j, 0]
        for i in range(1, codes.shape[1]):
            v = codes[j, i]
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
        low[j] = lo
        high[j] = hi


@njit(cache=True, parallel=True)
def _shift_rows_int16(codes: np.ndarray, low: np.ndarray, out: np.ndarray) -> None:
    """``out[j, i] = codes[j, i] - low[j]`` as int16 (parallel over the rows; every width fits int16 by the caller's check)."""
    for j in prange(codes.shape[0]):
        lo = low[j]
        for i in range(codes.shape[1]):
            out[j, i] = codes[j, i] - lo


def _shift_codes(codes: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(codes shifted so every row starts at 0, widths)``: int16 when every width fits, int64 otherwise."""
    n_cand = int(codes.shape[0])
    low = np.empty(n_cand, dtype=np.int64)
    high = np.empty(n_cand, dtype=np.int64)
    if codes.dtype.kind in "iu" and codes.dtype.itemsize <= 4:
        with parallel_kernel_entry():
            _row_range(codes, low, high)
    else:
        low[:] = codes.min(axis=1)
        high[:] = codes.max(axis=1)
    widths = (high - low) + 1
    if int(widths.max()) <= 32767:
        shifted = np.empty(codes.shape, dtype=np.int16)
        if codes.dtype.kind in "iu" and codes.dtype.itemsize <= 4:
            with parallel_kernel_entry():
                _shift_rows_int16(codes, low, shifted)
        else:
            for j0 in range(0, n_cand, 64):
                shifted[j0 : j0 + 64] = codes[j0 : j0 + 64].astype(np.int64) - low[j0 : j0 + 64, None]
        return shifted, widths
    return codes.astype(np.int64) - low[:, None], widths


#: Candidate-matrix cells (columns x rows) from which the null runs on the device: below it the upload costs about what the host kernel does.
DEVICE_MIN_CELLS = 20_000_000


def _device_conditional_mi(codes, widths, targets, z, ky, kz) -> "Optional[np.ndarray]":
    """The ``(targets, candidates)`` conditional MI on the device for a candidate matrix big enough to pay for the upload (and when the device is allowed), else ``None``."""
    from .._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled() or codes.shape[0] * codes.shape[1] < DEVICE_MIN_CELLS:
        return None
    from ._conditional_maxt_gpu import conditional_mi_targets_device

    return conditional_mi_targets_device(codes, widths, targets, z, ky, kz)


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
        self._prep: "Optional[tuple]" = None  # filled by ``_prepared``
        self._prep_built = False

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

    def _prepared(self) -> "Optional[tuple]":
        """``(row slice, shifted target, (candidates, rows) int64 code matrix, widths, target width)`` shared by every strong operand, built once (each call used to gather, transpose and widen the same
        matrix again); ``None`` for a degenerate target."""
        if self._prep_built:
            return self._prep
        y = self._y
        n = int(y.shape[0])
        if n == 0 or int(y.max()) == int(y.min()):
            self._prep_built = True
            self._prep = None
            return None
        step = max(1, n // MAX_ROWS)
        rows = slice(None, None, step)
        y = np.ascontiguousarray(y[rows])
        data = np.asarray(self._data)
        if data.dtype.kind in "iu" and data.dtype.itemsize <= 4 and data.ndim == 2:
            gathered_t = np.empty((len(self._cand), (n + step - 1) // step), dtype=data.dtype)
            with parallel_kernel_entry():
                _gather_transposed(data, step, np.asarray(self._cand, dtype=np.int64), gathered_t)
            codes, widths = _shift_codes(gathered_t)
        else:
            codes, widths = _candidate_code_matrix(np.asarray(data[rows][:, self._cand]))
        y = y - y.min()
        self._prep = (rows, y, codes, widths, int(y.max()) + 1)
        self._prep_built = True
        return self._prep

    def _compute(self, strong_idx: int) -> "Optional[tuple[np.ndarray, float]]":
        """``(conditional MI of every candidate given the strong operand, the q quantile of the permutation maximum)``, or ``None`` for a degenerate target."""
        prepared = self._prepared()
        if prepared is None:
            return None
        rows, y, codes, widths, ky = prepared
        z = np.asarray(self._data[rows, strong_idx]).astype(np.int64)
        z = z - z.min()
        kz = int(z.max()) + 1
        observed = np.empty(len(self._cand), dtype=np.float64)  # Intentional float64: plug-in MI statistic vector (exact permutation ties)
        order = np.argsort(z, kind="stable")
        bounds = np.flatnonzero(np.diff(z[order])) + 1
        strata = np.split(order, bounds)
        rng = np.random.default_rng(self._seed + 7919 * strong_idx)
        # the permuted targets are drawn first, in the order the loop drew them, so the stream is the same whichever kernel counts them
        targets = np.empty((N_PERMUTATIONS + 1, y.shape[0]), dtype=y.dtype)
        targets[0] = y
        for r in range(N_PERMUTATIONS):
            y_perm = targets[r + 1]
            y_perm[:] = y
            for idx in strata:
                y_perm[idx] = y[rng.permutation(idx)]
        on_device = _device_conditional_mi(codes, widths, targets, z, ky, kz)
        if on_device is not None:
            observed[:] = on_device[0]
            return observed, float(np.quantile(on_device[1:].max(axis=1), QUANTILE))
        with parallel_kernel_entry():
            _conditional_mi_columns(codes, widths, y, z, ky, kz, observed)
        maxima = np.empty(N_PERMUTATIONS, dtype=np.float64)  # Intentional float64: null maxima vector of the maxT family (exact permutation ties)
        scratch = np.empty(len(self._cand), dtype=np.float64)  # Intentional float64: scratch buffer for the same statistic
        for r in range(N_PERMUTATIONS):
            with parallel_kernel_entry():
                _conditional_mi_columns(codes, widths, targets[r + 1], z, ky, kz, scratch)
            maxima[r] = scratch.max()
        return observed, float(np.quantile(maxima, QUANTILE))

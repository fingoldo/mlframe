"""Host (numba) twin of the batched non-separability gate of the pure-form retention (``_fe_pure_form_retention_gpu_resident._batched_gate_scalars``).

For every candidate pair the gate regresses the candidate form on the 12-column additive basis of its two raw operands (centered), and needs four scalars: the form's standard deviation, the
standard deviation of the OLS residual, the residual sum of squares and the residual dot the centered relevance target. They come from the normal equations (a trace-scaled ridge of 1e-10 keeps
every Gram solvable), the same expressions as the device path: ``ss_resid = fc.fc - beta.rhs``, ``num = fc.y - beta.(X^T y)``. One ``prange`` over the pairs builds the 12 x 12 Gram, the two
right-hand sides and solves it by Gaussian elimination with partial pivoting, so no BLAS thread pool is involved and every run returns the same values. The per-candidate reference of the default path
(``LinearRegression`` on the standardised basis) stays untouched; this kernel is the baseline the device batch is timed and checked against and the batched host path of the STRICT route when the
host is the faster side.
"""

from __future__ import annotations

import numpy as np
from mlframe.utils.array_shape import length_of
from numba import njit, prange


@njit(cache=True, nogil=True)
def _solve_spd12(g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the 12 x 12 system ``g x = b`` by Gaussian elimination with partial pivoting (``g`` and ``b`` are overwritten)."""
    n = 12
    for c in range(n):
        piv = c
        best = abs(g[c, c])
        for r in range(c + 1, n):
            if abs(g[r, c]) > best:
                best = abs(g[r, c])
                piv = r
        if piv != c:
            for k in range(n):
                t = g[c, k]
                g[c, k] = g[piv, k]
                g[piv, k] = t
            t = b[c]
            b[c] = b[piv]
            b[piv] = t
        d = g[c, c]
        for r in range(c + 1, n):
            f = g[r, c] / d
            if f != 0.0:
                for k in range(c, n):
                    g[r, k] -= f * g[c, k]
                b[r] -= f * b[c]
    x = np.zeros(n)
    for r in range(n - 1, -1, -1):
        s = b[r]
        for k in range(r + 1, n):
            s -= g[r, k] * x[k]
        x[r] = s / g[r, r]
    return x


@njit(cache=True, parallel=True, nogil=True)
def pair_gate_scalars_njit(v: np.ndarray, bc: np.ndarray, ia: np.ndarray, ib: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """``(P, 4)`` gate scalars ``(form std, residual std, residual SS, residual . y)`` of the P columns of ``v`` (n, P).

    ``bc`` is the (U, n, 6) centered additive basis of the distinct operands, ``ia`` / ``ib`` the operand slots of each pair and ``yc`` the centered relevance target (n,)."""
    n, p_count = v.shape
    out = np.empty((p_count, 4))  # Intentional float64: gate scalars compared with thresholds at full precision
    for p in prange(p_count):
        g = np.zeros((12, 12))
        rhs = np.zeros(12)
        xy = np.zeros(12)
        x = np.empty(12)
        s = 0.0
        for i in range(n):
            s += v[i, p]
        fm = s / n
        sfc = 0.0
        fyr = 0.0
        a_blk = bc[ia[p]]
        b_blk = bc[ib[p]]
        for i in range(n):
            fc = v[i, p] - fm
            for k in range(6):
                x[k] = a_blk[i, k]
                x[6 + k] = b_blk[i, k]
            for a in range(12):
                rhs[a] += x[a] * fc
                xy[a] += x[a] * yc[i]
                for b in range(a, 12):
                    g[a, b] += x[a] * x[b]
            sfc += fc * fc
            fyr += fc * yc[i]
        tr = 0.0
        for a in range(12):
            tr += g[a, a]
            for b in range(a):
                g[a, b] = g[b, a]
        ridge = 1e-10 * tr / 12.0 + 1e-30
        for a in range(12):
            g[a, a] += ridge
        rhs0 = rhs.copy()
        beta = _solve_spd12(g, rhs)
        ssr = sfc
        num = fyr
        for a in range(12):
            ssr -= beta[a] * rhs0[a]
            num -= beta[a] * xy[a]
        out[p, 0] = np.sqrt(sfc / n)
        out[p, 1] = np.sqrt(max(ssr, 0.0) / n)
        out[p, 2] = ssr
        out[p, 3] = num
    return out


def host_gate_scalars(form_values: "list[np.ndarray]", base_columns: "list[np.ndarray]", src_pairs: list, base_names: list, rel_y: np.ndarray) -> "np.ndarray | None":
    """``(P, 4)`` gate scalars on the host with :func:`pair_gate_scalars_njit`, or ``None`` when an operand of a pair is not in ``base_names``.

    The additive basis of each distinct operand is the host's ``_single_operand_basis`` (standardise, square, cube, signed sqrt, signed log1p, reciprocal magnitude), centered here."""
    from ._fe_pure_form_retention import _single_operand_basis

    name_to_idx = {nm: i for i, nm in enumerate(base_names)}
    ia_l, ib_l = [], []
    for a, b in src_pairs:
        if name_to_idx.get(a) is None or name_to_idx.get(b) is None:
            return None
        ia_l.append(name_to_idx[a])
        ib_l.append(name_to_idx[b])
    used = sorted(set(ia_l) | set(ib_l))
    slot = {u: k for k, u in enumerate(used)}
    n = int(length_of(rel_y))
    bc = np.empty((len(used), n, 6))  # Intentional float64: the Gram feeds a ridge solve
    for u, k in slot.items():
        cols = np.stack(_single_operand_basis(np.asarray(base_columns[u], dtype=np.float64)), axis=1)  # Intentional float64: see above
        bc[k] = cols - cols.mean(axis=0, keepdims=True)
    v = np.ascontiguousarray(np.stack([np.asarray(f, dtype=np.float64) for f in form_values], axis=1))  # Intentional float64: see above
    y = np.asarray(rel_y, dtype=np.float64)  # Intentional float64: see above
    ia = np.fromiter((slot[i] for i in ia_l), dtype=np.int64, count=len(ia_l))
    ib = np.fromiter((slot[i] for i in ib_l), dtype=np.int64, count=len(ib_l))
    return np.asarray(pair_gate_scalars_njit(v, bc, ia, ib, np.ascontiguousarray(y - y.mean())))


__all__ = ["pair_gate_scalars_njit", "host_gate_scalars"]

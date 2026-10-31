"""The Miller-Madow-corrected information estimators the RelaxMRMR score is built from.

They live apart from the score so the serial formula and the parallel pair loop call the same code rather than keeping two copies of an
entropy-bias correction that has to agree to the last bit.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _mi_mm_njit(a: np.ndarray, b: np.ndarray, K_a: int, K_b: int) -> float:
    """Miller-Madow-corrected plug-in I(A; B) on integer codes, unclamped."""

    n = a.shape[0]

    if n <= 0:

        return 0.0

    # Intentional float64: njit joint-table accumulator; numba cannot call _crit_np_dtype, and the MI sums need exact f64 ties.
    joint = np.zeros((K_a, K_b), dtype=np.float64)

    for i in range(n):

        joint[a[i], b[i]] += 1.0

    Pa = joint.sum(axis=1)

    Pb = joint.sum(axis=0)

    n_f = float(n)

    mi = 0.0

    m_ab = 0

    for i in range(K_a):

        for j in range(K_b):

            v = joint[i, j]

            if v > 0.0:

                m_ab += 1

                mi += (v / n_f) * math.log(v * n_f / (Pa[i] * Pb[j]))

    m_a = 0

    for i in range(K_a):

        if Pa[i] > 0.0:

            m_a += 1

    m_b = 0

    for j in range(K_b):

        if Pb[j] > 0.0:

            m_b += 1

    return mi - (m_ab - m_a - m_b + 1) / (2.0 * n_f)


@njit(nogil=True, cache=True)
def _count_positive_1d(P: np.ndarray) -> int:
    """Number of strictly positive cells in a 1-D table."""
    m = 0
    for k in range(P.shape[0]):
        if P[k] > 0.0:
            m += 1
    return m


@njit(nogil=True, cache=True)
def _count_positive_2d(P: np.ndarray) -> int:
    """Number of strictly positive cells in a 2-D table."""
    m = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0.0:
                m += 1
    return m


@njit(nogil=True, cache=True)
def _cmi_joint_tables_njit(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, K_x: int, K_y: int, K_z: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Joint (x, y, z) count table, its z, xz and yz marginals, and the number of occupied joint cells."""

    n = x.shape[0]

    # Intentional float64: njit joint-table accumulator; numba cannot call _crit_np_dtype, and the MI sums need exact f64 ties.
    joint = np.zeros((K_x, K_y, K_z), dtype=np.float64)

    for i in range(n):

        joint[x[i], y[i], z[i]] += 1.0

    # Intentional float64: njit joint-table accumulator; numba cannot call _crit_np_dtype, and the MI sums need exact f64 ties.
    Pz = np.zeros(K_z, dtype=np.float64)

    # Intentional float64: njit joint-table accumulator; numba cannot call _crit_np_dtype, and the MI sums need exact f64 ties.
    Pxz = np.zeros((K_x, K_z), dtype=np.float64)

    # Intentional float64: njit joint-table accumulator; numba cannot call _crit_np_dtype, and the MI sums need exact f64 ties.
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)

    m_xyz = 0

    for i in range(K_x):

        for j in range(K_y):

            for k in range(K_z):

                v = joint[i, j, k]

                if v > 0.0:

                    m_xyz += 1

                Pz[k] += v

                Pxz[i, k] += v

                Pyz[j, k] += v

    return joint, Pz, Pxz, Pyz, m_xyz


@njit(nogil=True, cache=True)
def _cmi_mm_njit(x: np.ndarray, y: np.ndarray, z: np.ndarray, K_x: int, K_y: int, K_z: int) -> float:
    """Miller-Madow-corrected plug-in I(X; Y | Z) on integer codes, unclamped."""

    n = x.shape[0]

    if n <= 0:

        return 0.0

    joint, Pz, Pxz, Pyz, m_xyz = _cmi_joint_tables_njit(x, y, z, K_x, K_y, K_z)

    n_f = float(n)

    cmi = 0.0

    for i in range(K_x):

        for j in range(K_y):

            for k in range(K_z):

                v = joint[i, j, k]

                if v > 0.0:

                    cmi += (v / n_f) * math.log((v * Pz[k]) / (Pxz[i, k] * Pyz[j, k]))

    m_z: int = _count_positive_1d(Pz)

    m_xz: int = _count_positive_2d(Pxz)

    m_yz: int = _count_positive_2d(Pyz)

    return float(cmi - (m_xyz - m_xz - m_yz + m_z) / (2.0 * n_f))


@njit(nogil=True, cache=True)
def _composite_codes_njit(z1: np.ndarray, z2: np.ndarray, K_z2: int) -> np.ndarray:
    """Integer code of the pair (Z_1, Z_2)."""

    n = z1.shape[0]

    out = np.empty(n, dtype=np.int64)

    for i in range(n):

        out[i] = int(z1[i]) * K_z2 + int(z2[i])

    return out

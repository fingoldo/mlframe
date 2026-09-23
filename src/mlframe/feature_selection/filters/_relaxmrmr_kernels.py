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
def _cmi_mm_njit(x: np.ndarray, y: np.ndarray, z: np.ndarray, K_x: int, K_y: int, K_z: int) -> float:
    """Miller-Madow-corrected plug-in I(X; Y | Z) on integer codes, unclamped."""

    n = x.shape[0]

    if n <= 0:

        return 0.0

    joint = np.zeros((K_x, K_y, K_z), dtype=np.float64)

    for i in range(n):

        joint[x[i], y[i], z[i]] += 1.0

    Pz = np.zeros(K_z, dtype=np.float64)

    Pxz = np.zeros((K_x, K_z), dtype=np.float64)

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

    n_f = float(n)

    cmi = 0.0

    for i in range(K_x):

        for j in range(K_y):

            for k in range(K_z):

                v = joint[i, j, k]

                if v > 0.0:

                    cmi += (v / n_f) * math.log((v * Pz[k]) / (Pxz[i, k] * Pyz[j, k]))

    m_z = 0

    for k in range(K_z):

        if Pz[k] > 0.0:

            m_z += 1

    m_xz = 0

    for i in range(K_x):

        for k in range(K_z):

            if Pxz[i, k] > 0.0:

                m_xz += 1

    m_yz = 0

    for j in range(K_y):

        for k in range(K_z):

            if Pyz[j, k] > 0.0:

                m_yz += 1

    return cmi - (m_xyz - m_xz - m_yz + m_z) / (2.0 * n_f)


@njit(nogil=True, cache=True)
def _composite_codes_njit(z1: np.ndarray, z2: np.ndarray, K_z2: int) -> np.ndarray:
    """Integer code of the pair (Z_1, Z_2)."""

    n = z1.shape[0]

    out = np.empty(n, dtype=np.int64)

    for i in range(n):

        out[i] = int(z1[i]) * K_z2 + int(z2[i])

    return out

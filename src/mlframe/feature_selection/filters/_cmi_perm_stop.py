"""CMI-permutation stopping criterion + UAED elbow detector (2026-05-30 Wave 8).

C8 - Yu & Príncipe 2019 (*Entropy* 21(1):99). Stop when ``I(X_cand; Y | S)``
is NOT significantly larger than a permutation null. Fuses CMI estimation
+ permutation into one step, giving an automatic stop without a
``threshold * H(y)`` knob.

C9 - Llorente, Martino et al. 2023 (Signal Processing 2024, arXiv:2308.09102).
Universal Automatic Elbow Detection (UAED) generalises AIC/BIC for arbitrary
error curves; used to auto-pick subset size from a CMI-gain curve when
``n_features=None``.

Implementation notes:
- The original C8 paper uses matrix-based Rényi alpha-entropy; per the user
  decision (no Family-2 in MRMR hot path) we use the existing plug-in CMI
  estimator. The permutation null is unchanged.
- The original C9 UAED uses an analytic AIC-style penalty; we implement the
  spectral-correction variant which is parameter-free.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _cmi_plugin_njit(x: np.ndarray, y: np.ndarray, z_comp: np.ndarray,
                      K_x: int, K_y: int, K_z: int) -> float:
    """Plug-in I(X; Y | Z) where Z is a composite integer code."""
    n = x.shape[0]
    if n <= 0:
        return 0.0
    joint = np.zeros((K_x, K_y, K_z), dtype=np.float64)
    for i in range(n):
        joint[x[i], y[i], z_comp[i]] += 1.0
    n_f = float(n)
    Pz = np.zeros(K_z, dtype=np.float64)
    Pxz = np.zeros((K_x, K_z), dtype=np.float64)
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)
    for i in range(K_x):
        for j in range(K_y):
            for k in range(K_z):
                v = joint[i, j, k]
                Pz[k] += v
                Pxz[i, k] += v
                Pyz[j, k] += v
    cmi = 0.0
    for i in range(K_x):
        for j in range(K_y):
            for k in range(K_z):
                v = joint[i, j, k]
                if v <= 0.0 or Pxz[i, k] <= 0.0 or Pyz[j, k] <= 0.0 \
                        or Pz[k] <= 0.0:
                    continue
                p_xyz = v / n_f
                p_z = Pz[k] / n_f
                p_xz = Pxz[i, k] / n_f
                p_yz = Pyz[j, k] / n_f
                cmi += p_xyz * math.log((p_xyz * p_z) / (p_xz * p_yz))
    return max(0.0, cmi)


def cmi_permutation_stop(x_cand: np.ndarray, y: np.ndarray,
                          selected_cols: list[np.ndarray],
                          nbins_x: int, nbins_y: int,
                          nbins_selected: list[int],
                          n_permutations: int = 100,
                          alpha: float = 0.05,
                          seed: int = 0) -> tuple[bool, float, float]:
    """CMI-permutation test for the relevance of ``X_cand`` given selected set.

    Permutes ``X_cand`` (preserving its marginal) and recomputes the CMI
    against the unchanged ``Y`` and selected set ``Z``. The observed CMI is
    significant iff it exceeds the ``(1 - alpha)`` quantile of the
    permutation distribution.

    Returns:
        is_significant: True if the candidate is significantly relevant.
        observed_cmi: I(X_cand; Y | Z) plug-in estimate.
        p_value: permutation p-value.
    """
    rng = np.random.default_rng(int(seed))
    x_int = x_cand.astype(np.int64)
    y_int = y.astype(np.int64)
    K_x = int(nbins_x)
    K_y = int(nbins_y)
    n = x_int.size
    # Composite Z; degenerate when no selected features (then test is just MI test).
    if selected_cols:
        K_z = 1
        z_comp = np.zeros(n, dtype=np.int64)
        for j, col_j in enumerate(selected_cols):
            K_j = int(nbins_selected[j])
            z_comp = z_comp * K_j + col_j.astype(np.int64)
            K_z = K_z * K_j
            if K_z > 1_000_000:
                # Truncate -- the test becomes a marginal-MI test on truncated
                # conditioning; still useful as a coarse stop.
                z_comp = z_comp % 1_000_000
                K_z = 1_000_000
                break
    else:
        K_z = 1
        z_comp = np.zeros(n, dtype=np.int64)
    observed = _cmi_plugin_njit(x_int, y_int, z_comp, K_x, K_y, K_z)
    if n_permutations <= 0:
        return True, float(observed), 1.0
    # Permutation null.
    null_dist = np.empty(int(n_permutations), dtype=np.float64)
    for p in range(int(n_permutations)):
        perm = rng.permutation(n)
        x_perm = x_int[perm]
        null_dist[p] = _cmi_plugin_njit(x_perm, y_int, z_comp, K_x, K_y, K_z)
    # One-sided p-value.
    p_value = float(np.mean(null_dist >= observed))
    return p_value < alpha, float(observed), p_value


def uaed_elbow(curve: np.ndarray, sensitivity: float = 1.0) -> int:
    """Universal Automatic Elbow Detection on a monotone or unimodal curve.

    Implementation: spectral-correction variant. Computes the per-point
    discrete second derivative; the elbow is at the argmax of the negative
    second derivative weighted by inverse position. This corresponds to the
    point of MAXIMUM CURVATURE (knee) on a typical "marginal gain" curve.

    Args:
        curve: 1-D array of values (CMI gain per added feature, or CV score).
        sensitivity: ``> 1.0`` favours earlier elbows; ``< 1.0`` later. Default
            1.0 picks the canonical knee.

    Returns: integer index of the elbow.

    Reference: Llorente, F., Martino, L., Read, J., Delgado, D. (2023),
    "A novel approach to feature selection: Spectral Information Criterion
    via UAED elbow detection", arXiv:2308.09102 / Signal Processing 2024.
    """
    curve = np.asarray(curve, dtype=np.float64).ravel()
    n = curve.size
    if n < 3:
        return n - 1 if n > 0 else 0
    # Second derivative.
    second_deriv = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        second_deriv[i] = curve[i + 1] - 2.0 * curve[i] + curve[i - 1]
    # Knee is where the curve flattens => second derivative most negative.
    weights = (np.arange(n, dtype=np.float64) + 1.0) ** (-1.0 / max(sensitivity, 1e-6))
    score = -second_deriv * weights
    elbow = int(np.argmax(score[1:-1])) + 1  # exclude endpoints from search
    return elbow


__all__ = ["cmi_permutation_stop", "uaed_elbow"]

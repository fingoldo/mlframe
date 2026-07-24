"""Conditional Permutation Test — Berrett, Wang, Barber, Samworth 2020.

Permutes ``X`` CONDITIONAL on ``Z``, preserving the ``X | Z`` distribution.
The resulting permutation null gives valid p-values for the conditional
independence test ``H_0: X ⊥ Y | Z`` under ARBITRARY confounding.

Why this beats Besag-Clifford for MRMR: Besag-Clifford permutes the
candidate column unconditionally, which inflates Type-I error when the
candidate is correlated with already-selected features ``Z`` -- exactly the
regime MRMR's redundancy control is designed for. CPT is the principled
fix.

Algorithm (discrete X, Z case):
  1. Partition observations by their Z-stratum.
  2. WITHIN each stratum, permute X values independently.
  3. Compute the test statistic on the (permuted X, Y, Z) sample.
  4. Repeat B times to build the conditional permutation null.

For continuous X (we discretise via MRMR's binning), the within-stratum
permutation degenerates when strata have <= 1 element. SCREEN_CONFIRM_B-3 fix
this docstring previously (and incorrectly) claimed the
implementation "falls back to nearest-stratum borrowing" in that case -- the
actual code simply EXCLUDES singleton-Z-stratum rows from every permutation
replicate (``x_perm`` keeps the real observed value at those rows in ALL draws;
see ``stratum_arrs``'s ``len(idx_list) > 1`` filter below). No nearest-stratum
borrowing is implemented. This matters because the D10 caller
(``evaluation.py``'s conditional permutation test) composes ``z_comp`` from
EVERY currently-selected feature, so the conditioning cardinality -- and the
singleton-stratum fraction -- grows every round of a single fit: later rounds
permute a progressively smaller fraction of ``x``, understating the true null
variance and biasing the test toward under-powered ("no signal") verdicts as
more features are selected. True nearest-stratum borrowing (Berrett et al.'s
actual local-permutation walk) is NOT yet implemented; treat this test's power
as degrading, not constant, across a fit with a growing conditioning set.

Reference: Berrett, T.B., Wang, Y., Barber, R.F., Samworth, R.J. (2020),
"The Conditional Permutation Test for Independence While Controlling for
Confounders", *J. R. Statist. Soc. B* 82(1):175-197.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def conditional_permutation_test(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    nbins_x: int, nbins_y: int, nbins_z: int,
    statistic_fn=None,
    n_permutations: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Test ``H_0: X ⊥ Y | Z`` via within-stratum permutation.

    Args:
        x, y, z: 1-D integer-encoded arrays.
        nbins_x / y / z: cardinality of each variable.
        statistic_fn: callable ``(x, y, z) -> float``; default is plug-in
            ``I(X; Y | Z)``.
        n_permutations: number of conditional permutations.
        seed: RNG seed.

    Returns:
        (observed_statistic, p_value)
    """
    from ._cmi_perm_stop import _cmi_plugin_njit
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x).astype(np.int64).ravel()
    y = np.asarray(y).astype(np.int64).ravel()
    z = np.asarray(z).astype(np.int64).ravel()
    if statistic_fn is None:
        def _default_stat(_x, _y, _z):
            """Plug-in conditional MI estimator used when the caller doesn't supply a custom statistic."""
            return float(_cmi_plugin_njit(_x, _y, _z, int(nbins_x), int(nbins_y), int(nbins_z)))

        statistic_fn = _default_stat
    observed = float(statistic_fn(x, y, z))
    if n_permutations <= 0:
        return observed, 1.0
    # Group indices by z-stratum for fast within-stratum permutation.
    strata: dict[int, list] = {}
    for idx, zv in enumerate(z):
        strata.setdefault(int(zv), []).append(idx)
    # Materialise the per-stratum index arrays ONCE (constant across permutations)
    # and drop singleton strata up front: pre-fix this rebuilt each ``arr`` from a
    # Python list via ``np.asarray`` on EVERY permutation (B * n_strata calls --
    # ~48% of wall at B=200), all redundant work. Bit-identical: ``rng.permutation``
    # sees the SAME int64 arrays in the SAME dict-iteration order, so the RNG draw
    # sequence and the resulting permutations are unchanged.
    stratum_arrs = [np.asarray(idx_list, dtype=np.int64) for idx_list in strata.values() if len(idx_list) > 1]
    # SCREEN_CONFIRM_B-3 fix: surface the excluded (singleton-stratum) row fraction so a progressively
    # degenerate test (see the module docstring) is visible/monitorable instead of silently understating
    # power as the conditioning set grows.
    _n_permutable = sum(arr.shape[0] for arr in stratum_arrs)
    _n_total = int(z.shape[0]) if hasattr(z, "shape") else len(z)
    if _n_total > 0:
        _excluded_frac = 1.0 - (_n_permutable / _n_total)
        if _excluded_frac > 0.5:
            logger.warning(
                "mrmr: conditional_permutation_test excluded %.0f%% of rows from every permutation "
                "replicate (singleton Z-strata, %d distinct strata over %d rows) -- the conditional "
                "permutation null is degenerate for most of the sample; treat the p-value as under-powered.",
                100.0 * _excluded_frac, len(strata), _n_total,
            )
    null_dist = np.empty(int(n_permutations), dtype=np.float64)
    for p in range(int(n_permutations)):
        x_perm = x.copy()
        for arr in stratum_arrs:
            shuffled = rng.permutation(arr)
            x_perm[arr] = x[shuffled]
        null_dist[p] = float(statistic_fn(x_perm, y, z))
    # (1 + #{null >= observed}) / (B + 1) continuity correction (Phipson & Smyth 2010): the observed
    # statistic is itself one realisation under the null, so a Monte-Carlo permutation p-value can
    # never be exactly 0. The naive ``mean(null >= observed)`` can return 0 and overstate significance.
    n_exceed = int(np.count_nonzero(null_dist >= observed))
    p_value = (1.0 + n_exceed) / (int(n_permutations) + 1.0)
    return observed, p_value


__all__ = ["conditional_permutation_test"]

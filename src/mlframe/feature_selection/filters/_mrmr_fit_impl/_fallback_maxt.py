"""Family-wise (maxT) significance floor for the empty-support fallback rescue.

The rescue scans several raw candidates and accepts the ones that sit above their permutation null. Testing each candidate at alpha on its
own lets the chance of at least one noise column passing grow with the number of candidates scanned. This module instead builds the null
distribution of the MAXIMUM plug-in MI over the whole scanned set under label permutations, and accepts a candidate only when its observed
MI exceeds the (1 - alpha) quantile of that maximum. The family-wise error rate is then alpha regardless of how many candidates are scanned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._mdlp_honest_bycol_njit import seeded_permutation
from .._parity_fe import _mi_codes_njit

FALLBACK_MAXT_PERMUTATIONS = 999
FAMILY_ALPHA = 0.01


def fallback_maxt_floor(
    data: np.ndarray,
    nbins: np.ndarray,
    target_col: int,
    cols_idx: Sequence[int],
    alpha: float,
    n_perm: int = FALLBACK_MAXT_PERMUTATIONS,
    seed: int = 0,
    honest_codes_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed plug-in MI of each candidate column and the family-wise threshold over the set.

    ``data`` holds integer codes (one column per feature plus the target), ``nbins`` the number of codes per column, and ``cols_idx`` the
    column indices of the candidates. Returns ``(observed, null_max)``: ``observed[j]`` is the MI of ``cols_idx[j]`` with the target, and
    ``null_max`` holds the maximum MI over all candidates for each of the ``n_perm`` label permutations; the decision is the exact test in
    ``_family_maxt_gate.exact_pvalue_pass``. ``alpha`` is kept for signature compatibility and is applied by the caller.
    """
    y = np.ascontiguousarray(data[:, target_col], dtype=np.int64)
    n_y = int(nbins[target_col])
    cand = [np.ascontiguousarray(data[:, c], dtype=np.int64) for c in cols_idx]
    n_x = [int(nbins[c]) for c in cols_idx]
    # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    observed = np.array([_mi_codes_njit(cand[j], n_x[j], y, n_y) for j in range(len(cand))], dtype=np.float64)
    spec = getattr(honest_codes_fn, "spec", None) if honest_codes_fn is not None else None
    if spec is not None:
        # Plain 'mdlp': the numba honest chain re-bins every column on each permuted target (prange over permutations), the same null the
        # categorise-per-permutation loop below builds, with the same permutation sequence for a given seed.
        from .._honest_mdlp_null import honest_plain_null_maxima

        null_max = honest_plain_null_maxima(spec, data, nbins, target_col, list(cols_idx), int(n_perm), seed)
        return observed, null_max
    rng = np.random.default_rng(seed)
    null_max = np.empty(int(n_perm), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    if honest_codes_fn is not None:
        # Supervised binning: re-bin every column on each permuted RAW target (the production categorisation), so the null keeps the
        # selection bias of the cut points that the observed codes carry.
        base_seed = int(rng.integers(0, 2**31 - 1))  # the stream of the batched null (``permuted_labels_seeded``)
        for b in range(int(n_perm)):
            row_perm = seeded_permutation(y.shape[0], base_seed + b)
            codes_p, nb_p = honest_codes_fn(row_perm)
            yp = np.ascontiguousarray(y[row_perm])
            n_yp = int(yp.max()) + 1
            null_max[b] = max(
                [0.0]
                + [
                    _mi_codes_njit(np.ascontiguousarray(np.asarray(codes_p[:, c]).astype(np.int64)), int(nb_p[c]), yp, n_yp)
                    for c in cols_idx
                    if c < codes_p.shape[1]
                ]
            )
        return observed, null_max
    for b in range(int(n_perm)):
        yp = y[rng.permutation(y.shape[0])]
        null_max[b] = max(_mi_codes_njit(cand[j], n_x[j], yp, n_y) for j in range(len(cand)))
    return observed, null_max


__all__ = ["FALLBACK_MAXT_PERMUTATIONS", "fallback_maxt_floor"]

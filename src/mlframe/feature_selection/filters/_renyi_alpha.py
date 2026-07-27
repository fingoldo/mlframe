"""Matrix-based Rényi alpha-entropy mutual-information estimator (2026-07).

Yu, Giraldo, Jenssen, Príncipe 2020 (IEEE TPAMI, "Multivariate Extension of
Matrix-based Rényi's alpha-order Entropy Functional",
https://arxiv.org/abs/1808.07912). Estimates joint/conditional entropy from
the eigen-spectrum of a normalized RBF Gram matrix - no histogram, no
plug-in discretization bias, and it generalizes to multivariate conditioning
sets by simple elementwise (Hadamard) products of per-variable Gram
matrices, which is what makes the conditional-MI form below tractable.

For a sample of ``n`` draws of a (possibly multivariate) variable ``X``,
build the RBF Gram matrix ``K`` (``K[i,j] = exp(-||x_i - x_j||^2 / 2*sigma^2)``),
normalize to a trace-1 PSD "density matrix" ``A = K / tr(K)``, then the
alpha-order Rényi entropy is::

    S_alpha(A) = 1/(1-alpha) * log2(sum_i lambda_i(A)^alpha)

Joint entropy of ``(X, Z)`` uses the Hadamard product ``K_x ⊙ K_z``
(re-normalized to trace 1) as its Gram matrix - this is the multivariate
extension the paper derives, and it is why conditioning on several already-
selected variables is just multiplying in more per-variable Gram matrices
rather than growing a higher-dimensional joint density estimate.

Mutual information and conditional MI follow the same chain-rule identities
as Shannon entropy::

    I_alpha(X; Y)    = S(A_x) + S(A_y) - S(A_xy)
    I_alpha(X; Y | Z) = S(A_xz) + S(A_yz) - S(A_z) - S(A_xyz)

``alpha`` close to 1 (default 1.01, following the paper's own experiments)
makes the matrix-based Rényi entropy converge to a Shannon-like quantity
without the numerical singularity at exactly alpha=1.

Cost: O(n^2) Gram matrices + O(n^3) eigendecomposition. The paper's own
motivating case is small-n (n < 500) where plug-in MI's discretization bias
is worst; ``max_n`` subsamples (uniformly, without replacement) above that
so a caller accidentally passing a large column doesn't hang - this mirrors
``_ksg.py``'s GPU-threshold gate, a compute-cost guard, not a numerical
requirement (the estimator itself has no upper bound on n).

Opt-in via ``estimator='renyi_alpha'`` in ``_mi_dispatch.py``'s
``score_pair_mi`` - same maturity tier as ``mine``/``infonet``/``fastmi``:
available for benchmarking and ad-hoc scoring, not (yet) looped into MRMR's
per-candidate greedy scan.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA = 1.01
_DEFAULT_MAX_N = 1500


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce a 1-D array to a (n, 1) column; pass a (n, d) array through unchanged."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _silverman_sigma(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb RBF bandwidth, averaged across columns of a (possibly multivariate) block."""
    n = x.shape[0]
    std = float(np.std(x, axis=0).mean())
    if std <= 0.0 or n <= 1:
        return 1.0
    d = x.shape[1]
    return float(std * (n ** (-1.0 / (d + 4))))


def _rbf_gram(x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """RBF Gram matrix for a (n, d) block; ``sigma=None`` picks Silverman's bandwidth."""
    x = _as_2d(x)
    if sigma is None:
        sigma = _silverman_sigma(x)
    sigma = max(float(sigma), 1e-12)
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.asarray(np.exp(-d2 / (2.0 * sigma * sigma)))


def _hadamard_gram(*grams: np.ndarray) -> np.ndarray:
    """Elementwise product of several Gram matrices - the multivariate joint Gram matrix (Yu et al. 2020, Def. 3)."""
    out = grams[0].copy()
    for g in grams[1:]:
        out *= g
    return np.asarray(out)


def _renyi_entropy_from_gram(K: np.ndarray, alpha: float = _DEFAULT_ALPHA) -> float:
    """Alpha-order matrix-based Rényi entropy (in bits) of a Gram matrix ``K`` (trace-normalized internally)."""
    if alpha == 1.0:
        # alpha=1 is the module's own documented mathematical singularity
        # (division by 1-alpha below) - every public entry point defaults to alpha=1.01 specifically to
        # avoid it, but nothing stopped a caller from passing alpha=1.0 explicitly (estimator_kwargs=
        # {'alpha': 1.0} through score_pair_mi), which would have raised ZeroDivisionError deep inside a
        # dispatcher call with no caller-facing explanation. Fail loudly and specifically instead.
        raise ValueError(
            "_renyi_entropy_from_gram: alpha=1.0 is the mathematical singularity of the Renyi entropy "
            "(division by 1-alpha); use a value close to but not equal to 1 (module default: 1.01)."
        )
    # A NaN/Inf anywhere in the input propagates through the squared-distance computation into the Gram
    # matrix, and LAPACK then fails to converge and raises the opaque 'Eigenvalues did not converge'
    # from deep inside the estimator. The caller's real need is to DETECT the poisoned column, so report
    # it the way every other non-finite result is reported - as a non-finite MI - rather than as a crash
    # in a linear-algebra routine the caller never invoked.
    if not np.isfinite(K).all():
        return float("nan")
    tr = float(np.trace(K))
    if tr <= 0.0:
        return 0.0
    A = K / tr
    eigvals = np.linalg.eigvalsh(A)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return 0.0
    s = float(np.sum(eigvals**alpha))
    if s <= 0.0:
        return 0.0
    return float(np.log2(s) / (1.0 - alpha))


def _maybe_subsample(arrays: Sequence[np.ndarray], max_n: int, random_state: int) -> list:
    """Uniformly subsample the same row indices out of every array in ``arrays`` when ``n > max_n``; pass through unchanged otherwise."""
    n = arrays[0].shape[0]
    if n <= max_n:
        return list(arrays)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_n, replace=False)
    return [a[idx] for a in arrays]


def renyi_alpha_mi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = _DEFAULT_ALPHA,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    max_n: int = _DEFAULT_MAX_N,
    random_state: int = 0,
) -> float:
    """Matrix-based Rényi alpha-order mutual information ``I_alpha(X; Y)`` in bits, clamped to >= 0."""
    x2, y2 = _maybe_subsample([_as_2d(x), _as_2d(y)], max_n, random_state)
    Kx = _rbf_gram(x2, sigma_x)
    Ky = _rbf_gram(y2, sigma_y)
    Sx = _renyi_entropy_from_gram(Kx, alpha)
    Sy = _renyi_entropy_from_gram(Ky, alpha)
    Sxy = _renyi_entropy_from_gram(_hadamard_gram(Kx, Ky), alpha)
    # max() must not be the thing that decides a poisoned input is fine: max(0.0, nan) returns 0.0, because
    # every comparison against NaN is False. The clamp exists to absorb the estimator's small negative noise,
    # so keep it for finite values and let a non-finite estimate through unchanged - that is the only signal a
    # caller has that the column carried NaN/Inf.
    _mi = Sx + Sy - Sxy
    return float(_mi) if not np.isfinite(_mi) else max(0.0, _mi)


def renyi_alpha_cmi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    alpha: float = _DEFAULT_ALPHA,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    sigma_z: Optional[float] = None,
    max_n: int = _DEFAULT_MAX_N,
    random_state: int = 0,
) -> float:
    """Matrix-based Rényi alpha-order conditional MI ``I_alpha(X; Y | Z)`` in bits, clamped to >= 0.

    ``z`` may be multivariate (n, k) - the conditioning set is folded into one Gram matrix via the
    Hadamard-product extension, so conditioning on several already-selected MRMR variables at once
    is a single extra elementwise product, not a growing joint-density estimate.

    UNITS: returns BITS (log2-based), exactly like ``renyi_alpha_mi``. MRMR's estimator contract is NATS, so
    any dispatcher wiring this into a nats CMI path MUST convert at the boundary (``value * math.log(2.0)``),
    the same conversion ``_mi_dispatch`` already applies to ``renyi_alpha_mi``. Do NOT feed this raw into a
    nats path - it would inflate the CMI by ``1/ln(2)``.
    """
    x2, y2, z2 = _maybe_subsample([_as_2d(x), _as_2d(y), _as_2d(z)], max_n, random_state)
    Kx = _rbf_gram(x2, sigma_x)
    Ky = _rbf_gram(y2, sigma_y)
    Kz = _rbf_gram(z2, sigma_z)
    Kxz = _hadamard_gram(Kx, Kz)
    Kyz = _hadamard_gram(Ky, Kz)
    Kxyz = _hadamard_gram(Kx, Ky, Kz)
    Sxz = _renyi_entropy_from_gram(Kxz, alpha)
    Syz = _renyi_entropy_from_gram(Kyz, alpha)
    Sz = _renyi_entropy_from_gram(Kz, alpha)
    Sxyz = _renyi_entropy_from_gram(Kxyz, alpha)
    # max() must not be the thing that decides a poisoned input is fine: max(0.0, nan) returns 0.0, because
    # every comparison against NaN is False. The clamp exists to absorb the estimator's small negative noise,
    # so keep it for finite values and let a non-finite estimate through unchanged - that is the only signal a
    # caller has that the column carried NaN/Inf.
    _cmi = Sxz + Syz - Sz - Sxyz
    return float(_cmi) if not np.isfinite(_cmi) else max(0.0, _cmi)


__all__ = ["renyi_alpha_mi", "renyi_alpha_cmi"]

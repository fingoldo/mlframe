"""Non-polynomial basis families for pair feature engineering.

Closes the scope gap of orthogonal polynomials (Hermite/Legendre/Chebyshev/Laguerre cannot capture periodic, threshold, or rational patterns):

* **Fourier** -- ``sum_k a_k sin(2*pi*k*z) + b_k cos(2*pi*k*z)``. Periodic targets (``y ~ sin(2*pi*x_a*x_b)``).
* **RBF** -- ``sum_k w_k * exp(-((z - c_k) / sigma)^2)``. Centres ``c_k`` fixed at train-fold quantiles; bandwidth ``sigma`` by Silverman's rule. Captures local bumps, soft thresholds.
* **Sigmoid** -- ``sum_k w_k * sigma(s * (z - tau_k))``. Thresholds ``tau_k`` at train-fold quantiles; slope ``s`` set so a 10%-data-spread covers ``2 / s``. Captures sharp thresholds / cumulative-distribution targets.
* **Pade** -- rational ``(a_0 + a_1*z + a_p*z^p) / (1 + b_1*z + b_q*z^q)``. Captures targets with poles (``y ~ x_a / x_b``). Research-grade -- denominator stability not constrained; expect OPTUNA/CMA to find high-MI fits when stable, fail gracefully otherwise.

Each family ships ``fit(x)``, ``apply(x, params)``, ``eval_njit(z, c)``, ``coef_size(degree)``, and ``canonical_seeds(degree)`` matching the contracts used in ``hermite_fe.py``. They register into the ``EXTRA_BASES`` dict that the dispatcher in ``hermite_fe.py`` auto-merges into ``_POLY_BASES`` at import time.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

def _fourier_fit(x: np.ndarray):
    """Min-max normalise x to [0, 1] -- one period covers the full data range. Returns (z, params)."""
    lo = float(np.min(x))
    hi = float(np.max(x))
    raw_span = hi - lo
    # Degenerate (constant / near-constant) column: when the span is negligible relative to the
    # column's own scale, a 1e-12 floor maps x to a z whose dynamic range is dominated by a single
    # outlier (e.g. one value 1e-9 off 63 identical ones -> z spanning [0, ~8]); sin(2*pi*k*z) then
    # produces a high-frequency garbage feature off rounding noise. Map such a column to z=0 (a finite
    # constant feature) rather than amplifying noise into structure.
    scale = abs(lo) + abs(hi) + 1.0
    if raw_span <= 1e-9 * scale:
        return np.zeros_like(x, dtype=np.float64), dict(lo=lo, span=0.0, degenerate=True)
    z = (x - lo) / raw_span
    return z, dict(lo=lo, span=raw_span, degenerate=False)


def _fourier_apply(x: np.ndarray, params: dict) -> np.ndarray:
    if params.get("degenerate"):
        return np.zeros_like(x, dtype=np.float64)
    return np.asarray((x - params["lo"]) / params["span"])


@njit(fastmath=True, cache=True)
def _fourier_eval_njit(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``sum_k a_k * sin(2*pi*k*z) + b_k * cos(2*pi*k*z)`` for k = 1..K. ``c`` packs ``[a_1, b_1, a_2, b_2, ..., a_K, b_K]`` -- length 2K."""
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    K = c.shape[0] // 2
    if K == 0:
        return out
    two_pi = 2.0 * math.pi
    # Angle-addition (Chebyshev) recurrence: compute sin/cos of the base angle
    # 2*pi*z ONCE per sample, then step the harmonics k=2..K via
    #   sin(k*a) = sin((k-1)*a)*cos(a) + cos((k-1)*a)*sin(a)
    #   cos(k*a) = cos((k-1)*a)*cos(a) - sin((k-1)*a)*sin(a)
    # replacing 2*K transcendental calls per sample with 2 + O(K) mults.
    # Output matches the direct math.sin/cos form to ~1 ULP (~1e-15).
    for i in range(n):
        ang1 = two_pi * z[i]
        s1 = math.sin(ang1)
        c1 = math.cos(ang1)
        sk = s1
        ck = c1
        s = c[0] * sk + c[1] * ck
        for k in range(2, K + 1):
            sk, ck = sk * c1 + ck * s1, ck * c1 - sk * s1
            s += c[2 * (k - 1)] * sk + c[2 * (k - 1) + 1] * ck
        out[i] = s
    return out


def _fourier_coef_size(degree: int) -> int:
    """Fourier ``degree`` = max harmonic K -> 2K coefficients."""
    return 2 * max(1, degree)


def _fourier_canonical_seeds(degree: int) -> list:
    """Single-frequency identities: pure sin(2pi*k*z) at each k."""
    seeds = []
    K = max(1, degree)
    for k in range(1, K + 1):
        s = np.zeros(2 * K, dtype=np.float64)
        s[2 * (k - 1)] = 1.0  # sin coefficient
        seeds.append(s)
        c = np.zeros(2 * K, dtype=np.float64)
        c[2 * (k - 1) + 1] = 1.0  # cos coefficient
        seeds.append(c)
    return seeds


# ---------------------------------------------------------------------------
# RBF basis (radial basis functions, Gaussian kernel)
# ---------------------------------------------------------------------------

def _rbf_fit(x: np.ndarray):
    """Fit RBF centres at quantiles 0.1..0.9 and Silverman bandwidth. 9 fixed centres -> 9 weight coefficients during search."""
    quantiles = np.linspace(0.1, 0.9, 9)
    centres = np.quantile(x, quantiles).astype(np.float64)
    std = float(np.std(x) + 1e-12)
    n = len(x)
    bandwidth = float(1.06 * std * (n ** (-1.0 / 5.0))) + 1e-12
    return x.astype(np.float64), dict(centres=centres, bandwidth=bandwidth)


def _rbf_apply(x: np.ndarray, params: dict) -> np.ndarray:
    """RBF doesn't transform x for eval -- centres + bandwidth come from params. Just ensure float64 contiguous."""
    return np.ascontiguousarray(x, dtype=np.float64)


@njit(fastmath=True, cache=True)
def _rbf_eval_kernel_njit(z: np.ndarray, c: np.ndarray, centres: np.ndarray, bandwidth: float) -> np.ndarray:
    """``sum_k c_k * exp(-((z - centres_k) / bandwidth)^2)``."""
    n = z.shape[0]
    K = centres.shape[0]
    out = np.zeros(n, dtype=np.float64)
    inv_bw_sq = 1.0 / (bandwidth * bandwidth)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        zi = z[i]
        s = 0.0
        for k in range(min(K, nc)):
            d = zi - centres[k]
            s += c[k] * math.exp(-d * d * inv_bw_sq)
        out[i] = s
    return out


# RBF eval signature differs (needs centres, bandwidth) -- the hermite_fe registry passes only (z, c). We close over centres / bandwidth via a closure factory.

def _rbf_make_eval(params: dict):
    centres = params["centres"]
    bandwidth = params["bandwidth"]
    def _eval(z, c):
        return _rbf_eval_kernel_njit(z, c, centres, bandwidth)
    return _eval


def _rbf_coef_size(degree: int) -> int:
    """RBF degree maps to number of centres used. Up to 9 centres at quantiles 0.1..0.9; degree saturates at 9."""
    return min(max(1, degree + 1), 9)


def _rbf_canonical_seeds(degree: int) -> list:
    """Each canonical seed activates a single centre."""
    seeds = []
    K = _rbf_coef_size(degree)
    for k in range(K):
        s = np.zeros(K, dtype=np.float64)
        s[k] = 1.0
        seeds.append(s)
    # Constant offset (all centres equally weighted -- approximates mean of data).
    seeds.append(np.ones(K, dtype=np.float64) / K)
    return seeds


# ---------------------------------------------------------------------------
# Sigmoid basis (cumulative-style thresholds)
# ---------------------------------------------------------------------------

def _sigmoid_fit(x: np.ndarray):
    """Fit thresholds at quantiles 0.1..0.9 and slope to span the 10%-90% interquantile range with sharpness 4."""
    quantiles = np.linspace(0.1, 0.9, 9)
    thresholds = np.quantile(x, quantiles).astype(np.float64)
    iqr = float(np.quantile(x, 0.9) - np.quantile(x, 0.1)) + 1e-12
    slope = 4.0 / (iqr / 9.0)  # 4 -> ~95% of sigmoid in iqr/9 width
    return x.astype(np.float64), dict(thresholds=thresholds, slope=slope)


def _sigmoid_apply(x: np.ndarray, params: dict) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


@njit(fastmath=True, cache=True)
def _sigmoid_eval_kernel_njit(z: np.ndarray, c: np.ndarray, thresholds: np.ndarray, slope: float) -> np.ndarray:
    n = z.shape[0]
    K = thresholds.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        zi = z[i]
        s = 0.0
        for k in range(min(K, nc)):
            arg = slope * (zi - thresholds[k])
            # Numerically stable sigmoid.
            if arg >= 0.0:
                e = math.exp(-arg)
                sig = 1.0 / (1.0 + e)
            else:
                e = math.exp(arg)
                sig = e / (1.0 + e)
            s += c[k] * sig
        out[i] = s
    return out


def _sigmoid_make_eval(params: dict):
    thresholds = params["thresholds"]
    slope = params["slope"]
    def _eval(z, c):
        return _sigmoid_eval_kernel_njit(z, c, thresholds, slope)
    return _eval


def _sigmoid_coef_size(degree: int) -> int:
    return min(max(1, degree + 1), 9)


def _sigmoid_canonical_seeds(degree: int) -> list:
    seeds = []
    K = _sigmoid_coef_size(degree)
    for k in range(K):
        s = np.zeros(K, dtype=np.float64)
        s[k] = 1.0
        seeds.append(s)
    # Cumulative ramp -- monotone increase across all thresholds.
    seeds.append(np.linspace(0.0, 1.0, K).astype(np.float64))
    return seeds


# ---------------------------------------------------------------------------
# Pade basis (rational functions)
# ---------------------------------------------------------------------------

def _pade_fit(x: np.ndarray):
    """Standardize input -- Pade is sensitive to scale. Reference domain is roughly [-3, 3] post z-score."""
    mean = float(np.mean(x))
    raw_std = float(np.std(x))
    # Degenerate (constant / near-constant) column: an additive 1e-12 std floor does NOT dominate a
    # tiny-but-nonzero std (e.g. std ~ 1e-10 from one outlier), so z = (x - mean) / std blows the
    # outlier to z ~ 8 while the bulk sits near 0 -- the rational Horner eval then produces a garbage
    # feature dominated by that single point. Treat a column whose std is negligible relative to its
    # own scale as constant and map it to z=0.
    scale = abs(mean) + 1.0
    if raw_std <= 1e-9 * scale:
        return np.zeros_like(x, dtype=np.float64), dict(mean=mean, std=0.0, degenerate=True)
    return (x - mean) / raw_std, dict(mean=mean, std=raw_std, degenerate=False)


def _pade_apply(x: np.ndarray, params: dict) -> np.ndarray:
    if params.get("degenerate"):
        return np.zeros_like(x, dtype=np.float64)
    return np.asarray(((x - params["mean"]) / params["std"]).astype(np.float64))


@njit(fastmath=True, cache=True)
def _pade_eval_njit(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Rational ``(a_0 + a_1*z + ... + a_p*z^p) / (1 + b_1*z + ... + b_q*z^q)``.

    ``c`` packs ``[a_0, a_1, ..., a_{p}, b_1, ..., b_{q}]`` for ``deg == p == q`` (equal num/den degree). Length = 2*p + 1.

    Denominator is clamped to avoid poles: if ``|den| < 1e-3`` the output element is set to 0.0 (we're searching for STABLE rationals; the optimizer will steer away from coefficient sets that produce near-singular denominators because they yield low MI on typical targets).
    """
    n = z.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc < 2:
        return out
    p = (nc - 1) // 2  # numerator degree
    # coefs[0..p] = numerator a_0..a_p, coefs[p+1..2p] = denominator b_1..b_p
    for i in range(n):
        zi = z[i]
        # Horner numerator
        num = c[p]
        for k in range(p - 1, -1, -1):
            num = num * zi + c[k]
        # Horner denominator (b_0 = 1 implicit)
        den = c[2 * p]
        for k in range(2 * p - 1, p, -1):
            den = den * zi + c[k]
        den = den * zi + 1.0  # implicit b_0 = 1
        if abs(den) < 1e-3:
            out[i] = 0.0
        else:
            out[i] = num / den
    return out


def _pade_coef_size(degree: int) -> int:
    """For Pade ``[degree/degree]``: ``degree+1`` numerator + ``degree`` denominator coefficients = ``2*degree + 1``."""
    return 2 * max(1, degree) + 1


def _pade_canonical_seeds(degree: int) -> list:
    """Pure polynomial seed (denominator = 1) and identity ``z``."""
    p = max(1, degree)
    seeds = []
    # Pure z -> a_0=0, a_1=1, b_*=0
    s = np.zeros(2 * p + 1, dtype=np.float64)
    if p >= 1:
        s[1] = 1.0
        seeds.append(s)
    # Pure z^2 -> a_2=1
    if p >= 2:
        s2 = np.zeros(2 * p + 1, dtype=np.float64)
        s2[2] = 1.0
        seeds.append(s2)
    # Reciprocal: 1 / (1 + z) -> a_0=1, b_1=1
    s3 = np.zeros(2 * p + 1, dtype=np.float64)
    s3[0] = 1.0
    s3[p + 1] = 1.0  # b_1
    seeds.append(s3)
    return seeds


# ---------------------------------------------------------------------------
# Registry of extra (non-polynomial) bases. hermite_fe imports this and merges into its top-level ``_POLY_BASES`` dict.
# ---------------------------------------------------------------------------

EXTRA_BASES = {
    "fourier": dict(
        fit=_fourier_fit,
        apply=_fourier_apply,
        eval_njit=_fourier_eval_njit,
        eval=_fourier_eval_njit,  # numpy and njit share the same function
        coef_size_func=_fourier_coef_size,
        canonical_seeds_func=_fourier_canonical_seeds,
        dist_note="periodic on [0, 1]",
        kind="non-polynomial",
    ),
    "rbf": dict(
        fit=_rbf_fit,
        apply=_rbf_apply,
        eval_njit_factory=_rbf_make_eval,  # needs centres/bandwidth from params
        coef_size_func=_rbf_coef_size,
        canonical_seeds_func=_rbf_canonical_seeds,
        dist_note="local bumps via Gaussian kernel at quantile centres",
        kind="non-polynomial",
    ),
    "sigmoid": dict(
        fit=_sigmoid_fit,
        apply=_sigmoid_apply,
        eval_njit_factory=_sigmoid_make_eval,
        coef_size_func=_sigmoid_coef_size,
        canonical_seeds_func=_sigmoid_canonical_seeds,
        dist_note="sharp thresholds via sigmoid bumps at quantiles",
        kind="non-polynomial",
    ),
    "pade": dict(
        fit=_pade_fit,
        apply=_pade_apply,
        eval_njit=_pade_eval_njit,
        eval=_pade_eval_njit,
        coef_size_func=_pade_coef_size,
        canonical_seeds_func=_pade_canonical_seeds,
        dist_note="rational p/q with p=q (handles ratios with poles)",
        kind="non-polynomial",
    ),
}

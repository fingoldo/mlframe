"""B2 core: distribution zoo generators, target / feature candidate sets, safe judge on FRESH data (research agent B2, 2026-11-24).

Everything is CPU, deterministic given the rng. No existing file is imported or edited; wave-1 logic (tt_a1 LL + sign-flip null, tt_a2 Pearson info + permutation null,
tt_a4 clipped-inverse + Duan-smearing judge) is re-implemented here, vectorised, so that n up to 1e5 and p up to 200 stay cheap.
"""
from __future__ import annotations

import logging
import functools
import os
import warnings

for _k, _v in (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("NUMBA_NUM_THREADS", "2")):
    os.environ.setdefault(_k, _v)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy import stats
from scipy.special import gamma as _G, expit

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
np.seterr(all="ignore")

LOG3 = float(np.log(3.0))
LOG2 = float(np.log(2.0))
WINF = np.array([1.0, 0.6, 0.4])
WINF = WINF / np.linalg.norm(WINF)

# ======================================================================================================================================================================
# TARGET ZOO: y = family(u, noise), u = standardised linear index of the 3 informative features (sd 1), p-3 features are pure noise
# ======================================================================================================================================================================
FAMILIES = ("normal", "lognormal", "gamma", "weibull", "pareto", "neg_skew", "bimodal", "bounded01", "bounded0c", "poisson", "negbin", "zero_infl", "ordinal", "contam")
VARIANTS = ("homo", "mult")


def _std_noise(kind: str, rng, n: int) -> np.ndarray:
    """Mean-0 sd-1 noise with the shape of the family (for additive / homoscedastic variants)."""
    if kind == "lognormal":
        s = 0.5
        return np.asarray((np.exp(s * rng.standard_normal(n)) - np.exp(s * s / 2)) / np.sqrt((np.exp(s * s) - 1) * np.exp(s * s)))
    if kind == "gamma":
        return np.asarray((rng.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0))
    if kind == "weibull":
        k = 1.5
        m = _G(1 + 1 / k)
        return np.asarray((rng.weibull(k, n) - m) / np.sqrt(_G(1 + 2 / k) - m * m))
    if kind == "pareto":
        a = 2.5
        return np.asarray((rng.random(n) ** (-1 / a) - a / (a - 1)) / np.sqrt(a / ((a - 1) ** 2 * (a - 2))))
    return np.asarray(rng.standard_normal(n))


def _unit_noise(kind: str, rng, n: int) -> np.ndarray:
    """Positive noise with mean 1 (multiplicative variants)."""
    if kind == "lognormal":
        return np.asarray(np.exp(0.5 * rng.standard_normal(n) - 0.125))
    if kind == "gamma":
        return np.asarray(rng.gamma(2.0, 0.5, n))
    if kind == "weibull":
        return np.asarray(rng.weibull(1.5, n) / _G(1 + 1 / 1.5))
    if kind == "pareto":
        a = 2.5
        return np.asarray(rng.random(n) ** (-1 / a) * (a - 1) / a)
    raise ValueError(kind)


def gen_target(family: str, variant: str, n: int, p: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((n, p))
    u = X[:, :3] @ WINF  # sd 1
    g = rng.standard_normal(n)
    h = variant == "homo"
    if family == "normal":
        y = 2 * u + g if h else (6 + 2 * u) * (1 + 0.2 * np.clip(g, -3, 3))
    elif family in ("lognormal", "gamma", "weibull", "pareto"):
        slope = {"lognormal": 0.8, "gamma": 0.7, "weibull": 0.7, "pareto": 0.5}[family]
        mu = np.exp(slope * u)
        if family == "lognormal" and not h:
            y = np.exp(slope * u + 0.5 * g)
        elif h:
            y = mu + {"lognormal": 0.6, "gamma": 0.5, "weibull": 0.5, "pareto": 0.4}[family] * _std_noise(family, rng, n)
        else:
            y = mu * _unit_noise(family, rng, n)
    elif family == "neg_skew":
        y = 4 - np.exp(0.6 * u) + 0.3 * g if h else 4 - np.exp(0.6 * u + 0.4 * g)
    elif family == "bimodal":
        b = rng.integers(0, 2, n)
        y = u + 3 * b + 0.5 * g if h else (4 + u + 3 * b) * (1 + 0.2 * np.clip(g, -3, 3))
    elif family == "bounded01":
        mu = expit(1.2 * u)
        y = np.clip(mu + 0.08 * g, 0.001, 0.999) if h else np.clip(rng.beta(mu * 8, (1 - mu) * 8), 1e-6, 1 - 1e-6)
    elif family == "bounded0c":
        y = np.clip(4 + 2 * u + 2 * g, 0, 10) if h else 10 * np.clip(rng.beta(expit(0.9 * u - 0.3) * 5, (1 - expit(0.9 * u - 0.3)) * 5), 1e-6, 1 - 1e-6)
    elif family == "poisson":
        mu = np.exp(0.5 * u + 1.0)
        y = np.maximum(0.0, np.round(mu + 1.0 * g)) if h else rng.poisson(mu).astype(float)
    elif family == "negbin":
        mu = np.exp(0.5 * u + 1.0)
        y = np.maximum(0.0, np.round(mu + 2.0 * g)) if h else rng.poisson(rng.gamma(2.0, mu / 2.0)).astype(float)
    elif family == "zero_infl":
        z = rng.random(n) < 0.35
        y = np.where(z, 0.0, 3 + u + 0.7 * g) if h else np.where(z, 0.0, np.exp(0.7 * u + 0.5 * g))
    elif family == "ordinal":
        if h:
            y = 1.0 + np.searchsorted(np.array([-1, -0.3, 0.3, 1.0]), u + 1.0 * g)
        else:
            y = 1.0 + np.searchsorted(np.array([0.6, 1.0, 1.6, 2.8]), np.exp(0.6 * u + 0.5 * g))
        y = y.astype(float)
    elif family == "contam":
        y = 2 * u + g if h else np.exp(0.8 * u + 0.5 * g)
        c = rng.random(n) < 0.01
        y = np.where(c, y + 20.0 * rng.choice([-1.0, 1.0], n), y) if h else np.where(c, y * 30.0, y)
    else:
        raise ValueError(family)
    return X, np.asarray(y, float)


# ======================================================================================================================================================================
# FEATURE ZOO
# ======================================================================================================================================================================
XDISTS = ("normal", "lognormal", "t3", "uniform", "bounded", "discrete", "zero_infl", "bimodal", "mixed_skew")
LINKS = ("none", "log", "sqrt", "square", "recip", "abs")
_DISC_P = np.array([0.35, 0.25, 0.2, 0.12, 0.08])


def gen_x(dist: str, n: int, rng) -> np.ndarray:
    if dist == "normal":
        return np.asarray(rng.standard_normal(n))
    if dist == "lognormal":
        return np.asarray(np.exp(rng.standard_normal(n)))
    if dist == "t3":
        return np.asarray(rng.standard_t(3, n))
    if dist == "uniform":
        return np.asarray(rng.uniform(0.25, 4.0, n))
    if dist == "bounded":
        return np.asarray(np.clip(rng.beta(3, 2, n), 1e-4, None))
    if dist == "discrete":
        return np.asarray(rng.choice(np.arange(1.0, 6.0), n, p=_DISC_P))
    if dist == "zero_infl":
        return np.where(rng.random(n) < 0.3, 0.0, np.exp(0.7 * rng.standard_normal(n)))
    if dist == "bimodal":
        return np.asarray(np.where(rng.random(n) < 0.5, -2.0, 2.0) + 0.6 * rng.standard_normal(n))
    if dist == "mixed_skew":
        return np.asarray(np.exp(0.8 * rng.standard_normal(n)) - np.exp(0.32))
    raise ValueError(dist)


@functools.cache
def _ref(dist: str):
    """The median and the reference sample of one distribution, drawn once."""
    x = gen_x(dist, 200_000, np.random.default_rng(12345))
    return (float(np.median(x)), x)


def link_valid(dist: str, link: str) -> bool:
    x = _ref(dist)[1]
    if link in ("log", "recip"):
        return bool(x.min() > 0)
    if link == "sqrt":
        return bool(x.min() >= 0)
    return True


def link_fn(link: str, x: np.ndarray, c: float) -> np.ndarray:
    if link == "none":
        return x
    if link == "log":
        return np.asarray(np.log(x))
    if link == "sqrt":
        return np.asarray(np.sqrt(x))
    if link == "square":
        return (x - c) ** 2
    if link == "recip":
        return 1.0 / x
    if link == "abs":
        return np.asarray(np.abs(x - c))
    raise ValueError(link)


@functools.cache
def _link_moments(dist: str, link: str) -> "tuple[float, float]":
    """Mean and clipped standard deviation of the link over the reference sample of ``dist``."""
    c, xr = _ref(dist)
    gr = link_fn(link, xr, c)
    return (float(np.mean(gr)), float(np.std(np.clip(gr, *np.percentile(gr, [0.01, 99.99])))))


def g_std(dist: str, link: str, x: np.ndarray) -> np.ndarray:
    """Link standardised by population mean / sd (reference sample), so that effect sizes are comparable across fixtures; heavy tails are clipped at the
    reference 0.01 / 99.99 percentiles for the SCALE only."""
    c, _xr = _ref(dist)
    m, s = _link_moments(dist, link)
    return np.asarray((link_fn(link, x, c) - m) / s)


def gen_feature_fixture(dist: str, link: str, n: int, rng, amp: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x (feature under test), z (second, linear feature), y = amp * g_std(x) + 0.5 z + 0.7 e."""
    x = gen_x(dist, n, rng)
    z = rng.standard_normal(n)
    e = rng.standard_normal(n)
    return x, z, amp * g_std(dist, link, x) + 0.5 * z + 0.7 * e


# ======================================================================================================================================================================
# TARGET CANDIDATES
# ======================================================================================================================================================================
def yj_fwd(y, lam):
    out = np.empty_like(y)
    pos = y >= 0
    out[pos] = np.log1p(y[pos]) if abs(lam) < 1e-8 else ((y[pos] + 1) ** lam - 1) / lam
    ng = ~pos
    out[ng] = -np.log1p(-y[ng]) if abs(lam - 2) < 1e-8 else -(((1 - y[ng]) ** (2 - lam)) - 1) / (2 - lam)
    return out


def yj_inv(z, lam):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = np.expm1(z[pos]) if abs(lam) < 1e-8 else np.maximum(z[pos] * lam + 1, 1e-9) ** (1 / lam) - 1
    ng = ~pos
    out[ng] = -np.expm1(-z[ng]) if abs(lam - 2) < 1e-8 else 1 - np.maximum(1 - (2 - lam) * z[ng], 1e-9) ** (1 / (2 - lam))
    return out


def build_candidates(y: np.ndarray, rng, extended: bool = False, floor: float = 1e-6, yj_n: int = 4000) -> dict:
    """name -> (fwd, inv, logjac). Domain-gated on y. ``extended`` adds log1p / sqrt for y >= 0 (not in the wave-1 recommendation)."""
    sd = float(np.std(y)) + 1e-12
    mn, mx = float(y.min()), float(y.max())
    sg = np.sign
    c = {
        "identity": (lambda t: t, lambda z: z, lambda t: np.zeros_like(t)),
        "cbrt": (np.cbrt, lambda z: z**3, lambda t: -LOG3 - (2 / 3) * np.log(np.maximum(np.abs(t), floor))),
        "signed_sqrt": (lambda t: sg(t) * np.sqrt(np.abs(t)), lambda z: sg(z) * z**2, lambda t: -LOG2 - 0.5 * np.log(np.maximum(np.abs(t), floor))),
        "signed_square": (lambda t: sg(t) * t**2, lambda z: sg(z) * np.sqrt(np.abs(z)), lambda t: LOG2 + np.log(np.maximum(np.abs(t), floor))),
        "asinh": (lambda t: np.arcsinh(t / sd), lambda z: np.sinh(z) * sd, lambda t: -np.log(sd) - 0.5 * np.log1p((t / sd) ** 2)),
    }
    if mn > 0:
        c["log"] = (np.log, np.exp, lambda t: -np.log(t))
        c["reciprocal"] = (lambda t: -1.0 / t, lambda z: -1.0 / np.minimum(z, -1e-9), lambda t: -2 * np.log(t))
    if mn > 0 and mx < 1:
        eps = 1e-9
        c["logit"] = (lambda t: np.log(np.clip(t, eps, 1 - eps) / (1 - np.clip(t, eps, 1 - eps))), lambda z: expit(z), lambda t: -np.log(np.clip(t, eps, 1)) - np.log(np.clip(1 - t, eps, 1)))
    if extended and mn >= 0:
        c["log1p"] = (np.log1p, np.expm1, lambda t: -np.log1p(t))
        c["sqrt"] = (lambda t: np.sqrt(np.maximum(t, 0)), lambda z: np.maximum(z, 0) ** 2, lambda t: -LOG2 - 0.5 * np.log(np.maximum(t, floor)))
    sub = y[:: max(1, len(y) // yj_n)]  # deterministic stride subsample: the same lambda in the criterion, the null and the judge
    try:
        lam = float(np.clip(stats.yeojohnson(sub)[1], -1.0, 3.0))
    except Exception as exc:
        logging.getLogger(__name__).debug("tt_b2_core: ignored %s: %s", type(exc).__name__, exc)
        lam = 1.0
    c["yeo_johnson"] = (
        lambda t, l=lam: yj_fwd(t, l),
        lambda z, l=lam: yj_inv(z, l),
        lambda t, l=lam: (l - 1) * np.sign(t) * np.log1p(np.abs(t)),
    )
    c["yeo_johnson"] += (lam,)  # 4th slot: lambda (kept for the |lambda-1|<0.1 collapse)
    return c


# ======================================================================================================================================================================
# JUDGE on FRESH data
# ======================================================================================================================================================================
def _ridge_multi(D, Z, alpha=1.0):
    """Ridge (alpha on all but the intercept) for many targets at once; D has an intercept column first. Returns coef (p+1, K)."""
    G = D.T @ D
    pen = np.full(D.shape[1], alpha)
    pen[0] = 0.0
    G = G + np.diag(pen)
    return np.linalg.solve(G, D.T @ Z)


def inv_margin(name: str, cand: tuple, margin: float) -> float:
    """Clip margin beyond the training range of f(y). Zero for Yeo-Johnson with lambda outside [0, 2]: its image is bounded on one side and the inverse explodes at the
    boundary (a4 noted the failure; its retry only fires when the inverse is not finite, but 1e9 is finite)."""
    if name == "yeo_johnson" and len(cand) > 3 and not (0.0 <= cand[3] <= 2.0):
        return 0.0
    return margin


def judge_target(Xtr, ytr, Xte, yte, cands: dict, seed: int = 0, margin: float = 0.05, n_smear: int = 64) -> dict:
    """Ridge(1.0) head on f(y) fitted on 75 % of the training rows, smearing residual quantiles from the other 25 %, scored on the FRESH test set (original scale).
    Same clipping / smearing protocol as tt_a4_oracle.judge (clip to the training range of f(y) +5 %, retry without margin if the inverse is not finite).
    Returns name -> dict(rmse_smear, rmse_clip, mae_clip, mae_smear)."""
    n = len(ytr)
    idx = np.random.default_rng(10_000 + seed).permutation(n)
    a, b = idx[: int(0.75 * n)], idx[int(0.75 * n):]
    names = list(cands)
    Z = np.empty((n, len(names)))
    for j, k in enumerate(names):
        with np.errstate(all="ignore"):
            Z[:, j] = cands[k][0](ytr)
    Da = np.c_[np.ones(len(a)), Xtr[a]]
    ok = np.isfinite(Z[a]).all(0)
    Zf = np.where(np.isfinite(Z), Z, 0.0)
    W = _ridge_multi(Da, Zf[a])
    Db = np.c_[np.ones(len(b)), Xtr[b]]
    Dt = np.c_[np.ones(len(Xte)), Xte]
    Pb = Db @ W
    Pt = Dt @ W
    out = {}
    qs = (np.arange(n_smear) + 0.5) / n_smear
    for j, k in enumerate(names):
        if not ok[j]:
            out[k] = dict(rmse_smear=np.inf, rmse_clip=np.inf, mae_clip=np.inf, mae_smear=np.inf)
            continue
        inv = cands[k][1]
        za = Z[a, j]
        R = float(za.max() - za.min())
        q = np.quantile(Zf[b, j] - Pb[:, j], qs)
        zt = Pt[:, j]
        res = None
        for mg in (inv_margin(k, cands[k], margin), 0.0):
            lo, hi = float(za.min()) - mg * R, float(za.max()) + mg * R
            if k == "reciprocal":  # -1/y image is (-inf, 0): keep the clipped argument away from 0 (predicted y at most ~2 x the training maximum), as in tt_a1.judge
                hi = min(hi, float(za.max()) / 2)
            with np.errstate(all="ignore"):
                med = inv(np.clip(zt, lo, hi))
                sm = np.mean(inv(np.clip(zt[:, None] + q[None, :], lo, hi)), axis=1)
            if np.all(np.isfinite(med)) and np.all(np.isfinite(sm)):
                res = (med, sm)
                break
        if res is None:
            out[k] = dict(rmse_smear=np.inf, rmse_clip=np.inf, mae_clip=np.inf, mae_smear=np.inf)
            continue
        med, sm = res
        out[k] = dict(
            rmse_smear=float(np.sqrt(np.mean((sm - yte) ** 2))), rmse_clip=float(np.sqrt(np.mean((med - yte) ** 2))),
            mae_clip=float(np.mean(np.abs(med - yte))), mae_smear=float(np.mean(np.abs(sm - yte))),
        )
    return out


def outcome(ratio_pick: float, ratio_best: float, helps_tol: float = 0.98, hit_tol: float = 0.02, harm_tol: float = 1.02, picked_id: bool = False) -> str:
    """HIT / WRONG / MISS (a transform helps) or TN / FP / HARM (nothing helps). ratios are metric(candidate)/metric(identity), best over non-identity candidates."""
    helps = ratio_best <= helps_tol
    if helps:
        if picked_id:
            return "MISS"
        return "HIT" if ratio_pick <= ratio_best + hit_tol else "WRONG"
    if picked_id:
        return "TN"
    return "HARM" if ratio_pick > harm_tol else "FP"

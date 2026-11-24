"""Safe judge and oracle for target-transform search (agent 4, 2026-11-23).

API (all CPU, deterministic given seed):

* ``FIXTURES``: names of the fixtures; ``make(kind, n, p, seed) -> (X, y)``.
* ``MODELS``: ``ridge``, ``hgb``, ``lgbm`` (if installed), ``ridge_spline``.
* ``judge(X, y, model, names=None, seed=0) -> dict[name -> dict]``: for every applicable candidate transform ``f`` fits ``model`` on ``f(y)`` (60 % of rows),
  gets the smearing residuals on a calibration block (15 %) and scores on the untouched test block (25 %), on the ORIGINAL scale. Fields per candidate:
  ``rmse_smear`` (conditional-mean prediction: ``mean_k inv(clip(zhat + r_k))``, Duan smearing generalised to any monotone f; the metric for a user who needs the expected value / totals),
  ``rmse_clip`` (conditional-median style prediction ``inv(clip(zhat))``), ``mae_clip`` (median prediction under MAE, the metric a transformed model really optimises, the one for a user who needs a
  typical value), ``mae_smear``, ``rmse_unsafe`` (plain ``inv(zhat)`` without clipping, the old judge; can be inf/nan).
* ``oracle(fixture, model, seed=0, n=12000, metric='rmse_smear') -> list[(name, score_dict)]`` sorted best first; ``oracle_table(fixtures, models, seeds)`` averages over seeds.

Clipping: the model output on the transformed scale is clipped to ``[zmin - 0.05 R, zmax + 0.05 R]`` of the training ``f(y)`` (R the range) before inversion, so exp / cube / Yeo-Johnson
inverses cannot overflow and trees' extrapolation (none) and ridge's extrapolation (large) are bounded.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

warnings.simplefilter("ignore")

FIXTURES = (
    "lognormal", "hetero_lognormal", "cube", "linear", "hetero_linear", "exp_linear", "zero_inflated", "counts", "neg_skew", "bounded01", "heavy_tail",
)
METRICS = ("rmse_smear", "rmse_clip", "mae_clip", "mae_smear")


def make(kind: str, n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Features (standard normal) and the target of the given kind."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    e = rng.standard_normal(n)
    s = X[:, 0] * 0.9 + X[:, 1] * 0.6
    if kind == "lognormal":
        y = np.exp(s + 0.4 * e)
    elif kind == "hetero_lognormal":
        y = np.exp(s + 0.4 * (1 + np.abs(X[:, 0])) * e)
    elif kind == "cube":
        y = (X[:, 0] + 0.7 * X[:, 1] + 0.3 * e) ** 3
    elif kind == "linear":
        y = X[:, 0] + 0.7 * X[:, 1] + 0.3 * e
    elif kind == "hetero_linear":
        y = X[:, 0] + 0.7 * X[:, 1] + 0.3 * (1 + np.abs(X[:, 0])) * e
    elif kind == "exp_linear":
        y = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]) + 0.2 * e
    elif kind == "zero_inflated":
        y = np.where(rng.random(n) < 0.3, 0.0, np.exp(0.7 * s / 0.9 + 0.5 * e))
    elif kind == "counts":
        y = rng.poisson(np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.7)).astype(float)
    elif kind == "neg_skew":
        y = 2.0 - np.exp(0.7 * X[:, 0] + 0.4 * X[:, 1] + 0.3 * e)
    elif kind == "bounded01":
        y = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 0] + 0.8 * X[:, 1] + 0.5 * e)))
    elif kind == "heavy_tail":
        y = X[:, 0] + 0.7 * X[:, 1] + 0.3 * rng.standard_t(2.5, n)
    else:
        raise ValueError(kind)
    return X, y


def _clipfun(lo: float, hi: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda z: np.clip(z, lo, hi)


def candidate_set(ytr: np.ndarray) -> dict[str, tuple[Callable, Callable]]:
    """(forward, inverse) pairs applicable to the training target; inverses assume an already clipped argument."""
    from sklearn.preprocessing import PowerTransformer

    out: dict[str, tuple[Callable, Callable]] = {
        "identity": (lambda t: t, lambda z: z),
        "cbrt": (np.cbrt, lambda z: z**3),
        "signed_sqrt": (lambda t: np.sign(t) * np.sqrt(np.abs(t)), lambda z: np.sign(z) * z**2),
    }
    sd = float(np.std(ytr)) + 1e-12
    out["asinh"] = (lambda t: np.arcsinh(t / sd), lambda z: np.sinh(z) * sd)
    mn, mx = float(ytr.min()), float(ytr.max())
    if mn > 0.0:
        out["log"] = (np.log, np.exp)
        from scipy.stats import boxcox

        _, lam = boxcox(ytr)
        if abs(lam) < 1e-6:
            lam = 1e-6
        out["box_cox"] = (lambda t: (np.power(np.maximum(t, 1e-12), lam) - 1.0) / lam, lambda z: np.power(np.maximum(lam * z + 1.0, 1e-12), 1.0 / lam))
    if mn >= 0.0:
        out["log1p"] = (np.log1p, np.expm1)
        out["sqrt"] = (np.sqrt, lambda z: np.maximum(z, 0.0) ** 2)
    if mn > 0.0 and mx < 1.0:
        eps = 1e-9
        out["logit"] = (lambda t: np.log(np.clip(t, eps, 1 - eps) / (1 - np.clip(t, eps, 1 - eps))), lambda z: 1.0 / (1.0 + np.exp(-z)))
    pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(ytr.reshape(-1, 1))
    out["yeo_johnson"] = (lambda t: pt.transform(np.asarray(t).reshape(-1, 1)).ravel(), lambda z: pt.inverse_transform(np.asarray(z).reshape(-1, 1)).reshape(np.shape(z)))
    return out


def models() -> dict[str, Callable[[], Any]]:
    """Regressors with a fixed small budget, one thread each."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    out: dict[str, Callable[[], Any]] = {
        "ridge": lambda: Ridge(1.0),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=100),
        "ridge_spline": lambda: make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(1.0)),
    }
    try:
        from lightgbm import LGBMRegressor

        out["lgbm"] = lambda: LGBMRegressor(n_estimators=150, learning_rate=0.1, verbose=-1, n_jobs=1)
    except Exception:  # pragma: no cover
        pass
    return out


MODELS = tuple(models())


def judge(X: np.ndarray, y: np.ndarray, model: str, names: list[str] | None = None, seed: int = 0, margin: float = 0.05, n_smear: int = 64) -> dict[str, dict[str, float]]:
    """Score every applicable candidate transform for ``model`` on the test block, original scale (see module doc)."""
    n = len(y)
    idx = np.random.default_rng(10_000 + seed).permutation(n)
    a, b, t = idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.75 * n)], idx[int(0.75 * n):]
    cands = candidate_set(y[a])
    make_model = models()[model]
    res: dict[str, dict[str, float]] = {}
    for name, (f, inv) in cands.items():
        if names is not None and name not in names:
            continue
        z = f(y)
        m = make_model().fit(X[a], z[a])
        za = z[a]
        R = float(za.max() - za.min())
        _clipfun(float(za.min()) - margin * R, float(za.max()) + margin * R)
        zb = np.asarray(m.predict(X[b]))
        resid = z[b] - zb
        q = np.quantile(resid, (np.arange(n_smear) + 0.5) / n_smear)
        zt = np.asarray(m.predict(X[t]))
        with np.errstate(all="ignore"):
            unsafe = inv(zt)
            for mg in (margin, 0.0):  # a bounded-image inverse (Yeo-Johnson with lambda<0) is invalid beyond the training range: retry without the margin
                lo, hi = float(za.min()) - mg * R, float(za.max()) + mg * R
                med = inv(np.clip(zt, lo, hi))
                smear = np.mean(inv(np.clip(zt[:, None] + q[None, :], lo, hi)), axis=1)
                if np.all(np.isfinite(med)) and np.all(np.isfinite(smear)):
                    break
        yt = y[t]

        def rmse(p: np.ndarray) -> float:
            return float(np.sqrt(np.mean((p - yt) ** 2))) if np.all(np.isfinite(p)) else float("inf")

        res[name] = {
            "rmse_smear": rmse(smear), "rmse_clip": rmse(med), "mae_clip": float(np.mean(np.abs(med - yt))), "mae_smear": float(np.mean(np.abs(smear - yt))),
            "rmse_unsafe": rmse(unsafe),
        }
    return res


def oracle(fixture: str, model: str, seed: int = 0, n: int = 12000, metric: str = "rmse_smear") -> list[tuple[str, dict[str, float]]]:
    """Candidates for (fixture, model, seed) sorted best first by ``metric`` ('rmse_smear' conditional mean; 'mae_clip' conditional median)."""
    X, y = make(fixture, n, 8, seed)
    r = judge(X, y, model, seed=seed)
    return sorted(r.items(), key=lambda kv: kv[1][metric])


def oracle_table(fixtures=FIXTURES, model_names=MODELS, seeds=range(7), n: int = 12000, metric: str = "rmse_smear") -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    """Per (fixture, model): candidates ranked by mean metric over seeds, as (name, mean RMSE-like metric, mean ratio to identity)."""
    out = {}
    for fx in fixtures:
        for mo in model_names:
            acc: dict[str, list[float]] = {}
            for s in seeds:
                for name, d in oracle(fx, mo, s, n, metric):
                    acc.setdefault(name, []).append(d[metric] / oracle_identity(fx, mo, s, n, metric))
            out[(fx, mo)] = sorted(((k, float(np.mean(v)), float(np.mean(v))) for k, v in acc.items()), key=lambda r: r[1])
    return out


def oracle_identity(fixture: str, model: str, seed: int, n: int, metric: str) -> float:
    """Identity score (helper for ratios)."""
    X, y = make(fixture, n, 8, seed)
    return judge(X, y, model, names=["identity"], seed=seed)["identity"][metric]

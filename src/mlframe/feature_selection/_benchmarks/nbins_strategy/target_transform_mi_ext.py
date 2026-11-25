"""Extension of ``target_transform_mi.py``: does the MI under equal-width bins pick a target transform that lowers the error on the original scale? (2026-11-23, PLAN_target_transform_in_mrmr step 1)

Adds to the first experiment: 7 seeds, a heteroscedastic variant of two targets, a fitted Yeo-Johnson transform, and LightGBM and CatBoost next to ridge and ``HistGradientBoostingRegressor``. For every
(target kind, seed) every candidate ``f`` gets the sum over features of ``MI(x_j; f(y))`` with equal-width bins (the proposer) and the RMSE on the ORIGINAL scale of each model fitted on ``f(y)`` and inverted
(the judge). The summary says, per model, how often the transform the MI proposes is the one with the lowest RMSE, and the mean RMSE of the proposed transform against the identity.

Usage: ``python target_transform_mi_ext.py [--seeds 7] [--rows 20000]``. Results: ``audits/2026-11-23/PLAN_target_transform_in_mrmr.md``.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any, Callable

import numpy as np

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

Transform = tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], bool]


def make(kind: str, n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Features and a target of the given kind: ``lognormal`` (log linearises), ``cube`` (cbrt linearises), ``linear``; ``hetero_*`` scale the noise with the signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    base = kind.replace("hetero_", "")
    noise = rng.standard_normal(n)
    if base == "lognormal":
        scale = 0.4 * (1.0 + np.abs(X[:, 0])) if kind.startswith("hetero_") else 0.4
        y = np.exp(0.9 * X[:, 0] + 0.6 * X[:, 1] + scale * noise)
    elif base == "cube":
        y = (X[:, 0] + 0.7 * X[:, 1] + 0.3 * noise) ** 3
    else:
        scale = 0.3 * (1.0 + np.abs(X[:, 0])) if kind.startswith("hetero_") else 0.3
        y = X[:, 0] + 0.7 * X[:, 1] + scale * noise
    return X, y


def equal_width_codes(v: np.ndarray, k: int) -> np.ndarray:
    """Equal-width bin codes of ``v`` (``k`` bins)."""
    edges = np.linspace(v.min(), v.max(), k + 1)[1:-1]
    return np.asarray(np.searchsorted(edges, v, side="right"))


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in MI (nats) of two integer code vectors."""
    joint = np.zeros((int(a.max()) + 1, int(b.max()) + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    m = joint > 0
    return float((joint[m] * np.log(joint[m] / (pa @ pb)[m])).sum())


def candidates(y_train: np.ndarray) -> dict[str, Transform]:
    """The candidate transforms; ``log`` only for a positive target, Yeo-Johnson with the lambda fitted on the training rows."""
    from sklearn.preprocessing import PowerTransformer

    out: dict[str, Transform] = {
        "identity": (lambda t: t, lambda z: z, True),
        "cbrt": (np.cbrt, lambda z: z**3, True),
        "signed_sqrt": (lambda t: np.sign(t) * np.sqrt(np.abs(t)), lambda z: np.sign(z) * z**2, True),
    }
    if float(y_train.min()) > 0.0:
        out["log"] = (np.log, np.exp, True)
    pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(y_train.reshape(-1, 1))
    out["yeo_johnson"] = (lambda t: pt.transform(np.asarray(t).reshape(-1, 1)).ravel(), lambda z: pt.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel(), True)
    return out


def models() -> dict[str, Callable[[], Any]]:
    """The regressors: ridge, sklearn histogram GBM, LightGBM and CatBoost (a small fixed budget each)."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    return {
        "ridge": lambda: Ridge(1.0),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=100),
        "lgbm": lambda: LGBMRegressor(n_estimators=150, learning_rate=0.1, verbose=-1, n_jobs=4),
        "catboost": lambda: CatBoostRegressor(iterations=200, depth=6, verbose=0, thread_count=4),
    }


def main() -> None:
    """Run the grid and print the per-case lines and the summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args()
    kinds = ("lognormal", "cube", "linear", "hetero_lognormal", "hetero_linear")
    zoo = models()
    picked_is_best = {m: 0 for m in zoo}
    ratio_picked: dict[str, list[float]] = {m: [] for m in zoo}
    cases = 0
    for kind in kinds:
        for seed in range(a.seeds):
            X, y = make(kind, a.rows, 8, seed)
            n = len(y)
            tr = np.arange(n) < int(n * 0.7)
            te = ~tr
            cand = candidates(y[tr])
            mi_score = {name: sum(mutual_information(equal_width_codes(X[:, j], 10), equal_width_codes(f(y), 10)) for j in range(X.shape[1])) for name, (f, _inv, _ok) in cand.items()}
            picked = max(mi_score, key=lambda k: mi_score[k])
            rmse: dict[str, dict[str, float]] = {m: {} for m in zoo}
            for name, (f, inv, _ok) in cand.items():
                z = f(y)
                for mname, make_model in zoo.items():
                    model = make_model().fit(X[tr], z[tr])
                    pred = inv(np.asarray(model.predict(X[te])))
                    rmse[mname][name] = float(np.sqrt(np.mean((pred - y[te]) ** 2)))
            cases += 1
            line = " ".join(f"{m}:best={min(rmse[m], key=lambda k: rmse[m][k])},ratio={rmse[m][picked] / rmse[m]['identity']:.3f}" for m in zoo)
            print(f"CASE {kind} seed {seed} MI picks {picked} | {line}", flush=True)
            for mname in zoo:
                if min(rmse[mname], key=lambda k: rmse[mname][k]) == picked:
                    picked_is_best[mname] += 1
                ratio_picked[mname].append(rmse[mname][picked] / rmse[mname]["identity"])
    print(f"SUMMARY cases {cases}")
    for mname in zoo:
        r = np.asarray(ratio_picked[mname])
        ok = np.isfinite(r)  # an inverse that overflows (Yeo-Johnson on a heavy tail) gives a non-finite RMSE: counted, not averaged
        print(
            f"SUMMARY {mname}: the MI-picked transform is the best one in {picked_is_best[mname]}/{cases}; non-finite {int((~ok).sum())}; "
            f"RMSE(picked)/RMSE(identity) mean {r[ok].mean():.3f} median {np.median(r[ok]):.3f} worst {r[ok].max():.3f} best {r[ok].min():.3f}"
        )


if __name__ == "__main__":
    main()

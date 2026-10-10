"""Downstream error of the offset-product features: 5-fold CV MAE and RMSE of ridge and HistGradientBoosting with the preset / 1-parameter / 2-parameter feature added.

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp10_downstream 0 1 2`` (target indices) -> ``ds_<first idx>.json`` in the scratch folder.
Features are chosen inside each training fold. Each record holds ``mae`` and ``rmse`` dicts keyed ``<model>|<feature set>`` (mean over folds); ``agg10`` turns them into relative
improvement over the raw-columns baseline. R^2 is not a decision metric and is not computed (records from before 2026-10-10 hold R^2 under ``r2`` and are legacy).
"""

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import KFold

from ..common._paths import scratch_dir
from ..common.downstream import METRICS, REFERENCE_METRICS, errors_by_feature_set
from .core import clean, family_eval, get_presets, preset_matrix_best, qbin, rank01, unary_mat
from .core2 import family_eval2

UN, BI = get_presets("minimal")
nu = len(UN)
TN = [
    "(x-.3)(z-.7)",
    "(x-.5)(z-.5)",
    "log(2x)(z-.4)",
    "(e^x-1.8)(sqrt(z+.1)-.5)",
    "mix 3xz+.5z+1.5x",
    "mix xz+2z-x",
    "(x-.5)z",
    "log(2x)sin(z/3)",
    "log(x)sin(z) noshift",
    "x^2+z add",
]


def truth(i, x, z):
    """Truth feature ``i`` of the ten downstream targets."""
    return [
        (x - 0.3) * (z - 0.7),
        (x - 0.5) * (z - 0.5),
        np.log(2 * x) * (z - 0.4),
        (np.exp(x) - 1.8) * (np.sqrt(z + 0.1) - 0.5),
        3 * x * z + 0.5 * z + 1.5 * x,
        x * z + 2 * z - x,
        (x - 0.5) * z,
        np.log(2 * x) * np.sin(z / 3),
        np.log(x) * np.sin(z),
        x**2 + z,
    ][i]


def pair_uv(UX, UZ, p):
    """The (u, v) operand pair of pair index ``p`` (role, unary, unary)."""
    role, rem = divmod(p, nu * nu)
    iu, iv = divmod(rem, nu)
    return (UX[iu], UZ[iv]) if role == 0 else (UZ[iu], UX[iv])


def run(i, n, seed):
    """One 5-fold run: per-fold feature selection, then MAE and RMSE per model and feature set (mean over folds)."""
    rng = np.random.default_rng(9000 + 31 * seed + i + n)
    x, z = rng.random(n), rng.random(n)
    tr = truth(i, x, z)
    y = tr + rng.standard_normal(n) * tr.std()
    base = np.column_stack([x, z, rng.random(n), rng.random(n)])
    UX, UZ = unary_mat(UN, x), unary_mat(UN, z)
    acc = {}
    for trn, ten in KFold(5, shuffle=True, random_state=seed).split(x):
        yb = qbin(y[trn], 10)
        yr = rank01(y[trn])
        UXt, UZt = np.ascontiguousarray(UX[:, trn]), np.ascontiguousarray(UZ[:, trn])
        feats = {}
        # preset best
        _, ids = preset_matrix_best(UXt, UZt, BI, yb)
        iu, iv, bn = ids
        feats["preset"] = clean(BI[bn](UX[iu], UZ[iv]))
        # 1-param: better of closed-form ols and grid
        best = (-1, None)
        for mode in (3, 4):
            mis, ts = family_eval(UXt, UZt, yb, yr, mode, 10, 10, False, 8, 0.0, True, False, False, False, 9)
            j = int(mis.argmax())
            if mis[j] > best[0]:
                best = (mis[j], (j, ts[j]))
        j, t = best[1]
        u, v = pair_uv(UX, UZ, j)
        feats["p1"] = (u + t) * v
        best = (-1, None)
        for mode, G in ((6, 9), (7, 7)):
            mis, ss, tt = family_eval2(UXt, UZt, yb, yr, mode, 10, 10, False, False, False, G)
            j = int(mis.argmax())
            if mis[j] > best[0]:
                best = (mis[j], (j, ss[j], tt[j]))
        j, s, t = best[1]
        u, v = pair_uv(UX, UZ, j)
        feats["p2"] = (u + s) * (v + t)
        sets = {
            "raw": base,
            "+preset": np.column_stack([base, feats["preset"]]),
            "+p1": np.column_stack([base, feats["p1"]]),
            "+p2": np.column_stack([base, feats["p2"]]),
            "+p1+p2": np.column_stack([base, feats["p1"], feats["p2"]]),
        }
        errs = errors_by_feature_set({fs: (X[trn], X[ten]) for fs, X in sets.items()}, y[trn], y[ten])
        for key, e in errs.items():
            for metric in (*METRICS, *REFERENCE_METRICS):
                acc.setdefault(metric, {}).setdefault(key, []).append(e[metric])
    return {metric: {k: float(np.mean(v)) for k, v in d.items()} for metric, d in acc.items()}


def main(argv=None) -> None:
    """Run the downstream study for the given target indices; ``--quick`` shrinks it to one tiny seed for a smoke test."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+", type=int, help="indices into the target list")
    ap.add_argument("--quick", action="store_true", help="n=1500, one seed (smoke test, not for results)")
    args = ap.parse_args(argv)
    plan = ((1500, 1),) if args.quick else ((5000, 3), (30000, 2))
    out = []
    path = scratch_dir("stat_study") / f"ds_{args.targets[0]}.json"
    for i in args.targets:
        for n, seeds in plan:
            for sd in range(seeds):
                t0 = time.time()
                out.append(dict(target=TN[i], n=n, seed=sd, **run(i, n, sd)))
                print(TN[i], n, sd, f"{time.time() - t0:.0f}s", flush=True)
                path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()

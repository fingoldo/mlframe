"""Evaluation harness for wave-2 operators: fit half / held-out half protocol, held-out MI versus the best existing candidate, 5-fold CV MAE / RMSE of ridge and HGB on the held-out half
with recipes frozen from the fit half, and the replay proof (JSON round trip + DataFrame-by-name + row-permutation equivariance)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..brainstorm.h import MI, clean, existing, mi_pair, ybins
from ..common.downstream import METRICS, fit_errors, make_models, rel_improvement

__all__ = ["cv_relative", "leak_ablation", "prep", "top_cols", "replay_proof", "heldout_mi"]


def prep(fa: np.ndarray, fb: np.ndarray, q: float = 0.005) -> np.ndarray:
    """Make an engineered column safe for a linear model: non-finite to 0, clipped to the fit-half ``[q, 1 - q]`` quantiles (the clip bounds belong in the recipe)."""
    fa, fb = clean(fa), clean(fb)
    lo, hi = np.quantile(fa, [q, 1 - q])
    return np.clip(fb, lo, hi)


def top_cols(XA: np.ndarray, yb: tuple, cols: list, cap: int = 6) -> list:
    """The ``cap`` columns with the largest univariate train MI (mirrors the repo's max-pair-columns limit)."""
    if len(cols) <= cap:
        return list(cols)
    sc = [mi_pair(XA[:, c], XA[:, c], yb[0], yb[0], yb[2])[0] for c in cols]
    return [cols[i] for i in np.argsort(sc)[::-1][:cap]]


def heldout_mi(fa, fb, yb) -> tuple:
    """(train MI, held-out MI) of a feature with decile edges from the fit half."""
    return MI(fa, fb, yb)


def cv_relative(sets: dict, y: np.ndarray, seed: int = 0, folds: int = 5, models: dict | None = None) -> dict:
    """5-fold CV on the held-out half. ``sets`` maps a name to a feature matrix (recipes already frozen, no selection inside the folds). Returns
    ``{'<metric>|<model>|<set>': value}`` with metric in mae / rmse (relative improvement over the ``raw`` set of the same model, positive = better) and ``r2`` (absolute difference, reference only).
    """
    models = models or make_models()
    kf = KFold(folds, shuffle=True, random_state=seed)
    err = {}
    for name, X in sets.items():
        for mn, model in models.items():
            acc = [fit_errors(model, X[a], y[a], X[b], y[b]) for a, b in kf.split(X)]
            err[(mn, name)] = {m: float(np.mean([e[m] for e in acc])) for m in (*METRICS, "r2")}
    out = {}
    for (mn, name), e in err.items():
        b = err[(mn, "raw")]
        for m in METRICS:
            out[f"{m}|{mn}|{name}"] = rel_improvement(e[m], b[m])
        out[f"r2|{mn}|{name}"] = e["r2"] - b["r2"]
    return out


def leak_ablation(sets_ab: dict, ya: np.ndarray, yb_: np.ndarray, models: dict | None = None) -> dict:
    """Train on the fit half with each feature set, test on the held-out half; ``sets_ab = {name: (XA, XB)}``. Relative MAE / RMSE improvement over ``raw`` per model."""
    models = models or make_models()
    err = {(mn, nm): fit_errors(model, a, ya, b, yb_) for nm, (a, b) in sets_ab.items() for mn, model in models.items()}
    return {f"{m}|{mn}|{nm}": rel_improvement(e[m], err[(mn, "raw")][m]) for (mn, nm), e in err.items() for m in METRICS}


def replay_proof(recipe: dict, replay_fn, XB: np.ndarray, ref: np.ndarray, rng) -> dict:
    """Prove ``replay_fn(recipe, X)`` is a pure function of X: (1) equals the fit-time column ``ref`` computed by the fitting code path, (2) survives a JSON round trip,
    (3) identical through a DataFrame addressed by column name, (4) row-permutation equivariant, (5) single-row batches agree with the full batch. Returns booleans.
    """
    base = replay_fn(recipe, XB)
    rj = json.loads(json.dumps(recipe))
    ncol = XB.shape[1]
    names = [f"c{j}" for j in range(ncol)]
    rn = dict(recipe, src=[names[j] for j in recipe["src"]])
    df = pd.DataFrame(XB, columns=names)
    perm = rng.permutation(len(XB))
    idx = rng.choice(len(XB), 5, replace=False)
    return {
        "matches_fit_path": bool(np.allclose(base, ref, rtol=0, atol=1e-12)),
        "json_roundtrip": bool(np.array_equal(replay_fn(rj, XB), base)),
        "dataframe_by_name": bool(np.array_equal(replay_fn(rn, df), base)),
        "row_perm_equivariant": bool(np.array_equal(replay_fn(recipe, XB[perm]), base[perm])),
        "single_row_batches": bool(all(np.array_equal(replay_fn(recipe, XB[[i]]), base[[i]]) for i in idx)),
    }


def existing_best(XA, XB, ya, yb_, cols):
    """Best existing candidate (raw columns + 1734-combo pair preset) chosen on the fit half: returns (yb tuple, (train MI, held-out MI, train col, held-out col, name))."""
    yb = ybins(ya, yb_)
    return yb, existing(XA, XB, yb, cols, True)

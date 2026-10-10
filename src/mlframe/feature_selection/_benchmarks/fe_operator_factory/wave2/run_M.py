"""Operator M as a PAIR-SELECTION step: screen all column pairs on y versus on the out-of-fold additive-model residual, then engineer the top pair with the existing pair preset
(and with the closed-form offset product) and compare the held-out MI and the 5-fold CV MAE / RMSE of ridge and HGB.

Screens: ``preset_y`` / ``preset_r`` (best of 216 preset combos by MI, as in ``brainstorm.ops_M``), ``cell_y`` / ``cell_r`` (single-pass 6 x 6 quantile-cell ANOVA statistic, df-corrected).
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.run_M [--n 5000] [--seeds 8] [--seed0 0] [--hard] [--tag eval] [--no-downstream]``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from ..brainstorm.h import MI, existing_pair, ybins
from ..brainstorm.ops_M import resid_oof, screen
from ..common._paths import results_dir
from .harness import cv_relative, prep
from .targets import gen_M
from .warp_service import oof_fit, replay

SCREENS = ("preset_y", "preset_r", "cell_y", "cell_r")
# (screen, engineered feature) arms that get the 5-fold CV; the picks and held-out MI of every screen are recorded for all arms
CV_ARMS = {("preset_y", "preset"), ("cell_r", "preset"), ("cell_r", "ols2"), ("cell_r", "preset_on_resid")}
__all__ = ["cell_screen", "all_pair_scores", "offset_ols2"]


def _qcodes(x: np.ndarray, k: int) -> np.ndarray:
    """Equal-frequency codes in ``[0, k)`` from the stable rank."""
    r = np.empty(len(x), np.int64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return np.minimum(r * k // len(x), k - 1)


def cell_screen(Q: np.ndarray, t: np.ndarray, pairs: list, k: int = 6) -> np.ndarray:
    """Single-pass pair statistic: share of the variance of ``t`` explained by the k x k quantile-cell means, minus the null expectation ``(k*k - 1) / n``; ``Q`` holds the column codes."""
    tc = t - t.mean()
    v = float((tc**2).sum())
    n = len(t)
    out = np.empty(len(pairs))
    for a, (i, j) in enumerate(pairs):
        c = Q[:, i] * k + Q[:, j]
        s = np.bincount(c, tc, k * k)
        cnt = np.maximum(np.bincount(c, minlength=k * k), 1)
        out[a] = float((s**2 / cnt).sum() / v) - (k * k - 1) / n
    return out


def all_pair_scores(X: np.ndarray, y: np.ndarray, r: np.ndarray, pairs: list) -> dict:
    """Pair scores of the four screens (y and residual, preset-MI and cell-ANOVA)."""
    Q = np.column_stack([_qcodes(X[:, j], 6) for j in range(X.shape[1])])
    return {"preset_y": screen(X, y, pairs), "preset_r": screen(X, r, pairs), "cell_y": cell_screen(Q, y, pairs), "cell_r": cell_screen(Q, r, pairs)}


def offset_ols2(u_a: np.ndarray, v_a: np.ndarray, y_a: np.ndarray) -> tuple:
    """Closed-form offset product ``(u + s)(v + t)`` from the 4x4 OLS ``rank(y) ~ a + b u + c v + d uv`` (``s = c/d``, ``t = b/d``); returns (s, t) or (0, 0) if d ~ 0."""
    ry = np.argsort(np.argsort(y_a)) / len(y_a)
    A = np.column_stack([np.ones_like(u_a), u_a, v_a, u_a * v_a])
    b = np.linalg.lstsq(A, ry, rcond=None)[0]
    if abs(b[3]) < 1e-9:
        return 0.0, 0.0
    return float(b[2] / b[3]), float(b[1] / b[3])


def _engineer(XA, XB, ya, yb, pair, resid_target=None):
    """Best preset combo of ``pair`` (train MI) and the ols2 offset product; returns dict of (train col, held-out col)."""
    i, j = pair
    tgt = ya if resid_target is None else resid_target
    ybt = yb if resid_target is None else ybins(tgt, tgt)
    pr = existing_pair(XA[:, i], XA[:, j], XB[:, i], XB[:, j], ybt)
    s, t = offset_ols2(XA[:, i], XA[:, j], tgt)
    return {"preset": (pr[2], pr[3]), "ols2": ((XA[:, i] + s) * (XA[:, j] + t), (XB[:, i] + s) * (XB[:, j] + t))}


def one_case(kind: str, n: int, seed: int, hard: bool, downstream: bool) -> dict:
    """One (kind, seed) row: ranks and top picks of each screen, then the downstream effect of engineering each screen's top-1 pair."""
    rng = np.random.default_rng(20_000 + seed)
    X, y, true_pairs = gen_M(rng, n, kind, hard)
    p = X.shape[1]
    pairs = list(itertools.combinations(range(p), 2))
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    t0 = time.time()
    r = resid_oof(XA, ya, np.random.default_rng(seed))
    t_resid = time.time() - t0
    t0 = time.time()
    sc = all_pair_scores(XA, ya, r, pairs)
    t_scr = time.time() - t0
    row = dict(kind=kind, n=n, seed=seed, hard=hard, t_resid=t_resid, t_screens=t_scr, true_pairs=[list(t) for t in true_pairs], n_fit=h)
    yb = ybins(ya, yB)
    perm = np.random.default_rng(seed + 7).permutation(h)
    Q = np.column_stack([_qcodes(XA[:, j], 6) for j in range(p)])
    row["null_cell_r_max"] = float(cell_screen(Q, r[perm], pairs).max())
    for nm, s in sc.items():
        order = np.argsort(-s)
        row[f"{nm}|top"] = list(pairs[int(order[0])])
        row[f"{nm}|top_score"] = float(s[order[0]])
        row[f"{nm}|rank_true"] = [int(np.where(order == pairs.index(tp))[0][0]) + 1 for tp in true_pairs]
        row[f"{nm}|hit_top1"] = bool(true_pairs and tuple(pairs[int(order[0])]) in true_pairs)
        row[f"{nm}|hit_top3"] = [bool(pairs.index(tp) in order[:3]) for tp in true_pairs]
    if kind == "W" and downstream:
        sets = {"raw": XB}
        feats = {}
        for nm in SCREENS:
            pair = tuple(row[f"{nm}|top"])
            for fn, (fa, fb) in _engineer(XA, XB, ya, yb, pair).items():
                row[f"mi_te|{nm}|{fn}"] = MI(fa, fb, yb)[1]
                if (nm, fn) in CV_ARMS:
                    sets[f"{nm}|{fn}"] = np.column_stack([XB, prep(fa, fb)])
                    feats[f"{nm}|{fn}"] = prep(fa, fb)
            if (nm, "preset_on_resid") in CV_ARMS:
                fa, fb = _engineer(XA, XB, ya, yb, pair, resid_target=r)["preset"]
                sets[f"{nm}|preset_on_resid"] = np.column_stack([XB, prep(fa, fb)])
                feats[f"{nm}|preset_on_resid"] = prep(fa, fb)
        for tp in true_pairs[:1]:
            for fn, (fa, fb) in _engineer(XA, XB, ya, yb, tp).items():
                sets[f"truepair|{fn}"] = np.column_stack([XB, prep(fa, fb)])
                feats[f"truepair|{fn}"] = prep(fa, fb)
        row.update(cv_relative(sets, yB, seed=seed))
        warps = []
        for j in range(p):
            fa_j, rec_j = oof_fit("warp1d", XA, ya, [j], seed=seed, nb=20)
            warps.append(prep(fa_j, replay(rec_j, XB)))
        base = np.column_stack([XB, *warps])
        sets_ab = {"raw": base, **{k: np.column_stack([base, v]) for k, v in feats.items()}}
        row.update({"ab|" + k: v for k, v in cv_relative(sets_ab, yB, seed=seed).items()})
    return row


def main(argv=None) -> None:
    """CLI entry: W / N / 0 kinds over the seeds; one JSON row per (kind, seed) appended to ``wave2/results/rows_M_<n>_<hard|ideal>_<tag>.jsonl``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=["W", "N", "0"])
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--no-downstream", action="store_true")
    a = ap.parse_args(argv)
    out = results_dir("wave2") / f"rows_M_{a.n}_{'hard' if a.hard else 'ideal'}_{a.tag}.jsonl"
    with out.open("a") as fh:
        for kind in a.kinds:
            for s in range(a.seed0, a.seed0 + a.seeds):
                t0 = time.time()
                row = one_case(kind, a.n, s, a.hard, not a.no_downstream)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"M_{kind} n={a.n} seed={s} top y={row['preset_y|top']} r={row['preset_r|top']} cell_r={row['cell_r|top']} true={row['true_pairs']} {time.time() - t0:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()

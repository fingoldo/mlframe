"""Run operator B / C / G over its W, N, 0 (+ hard H, HN) cases: held-out MI vs the best existing candidate, 5-fold CV MAE / RMSE (ridge, HGB), replay proof, cost; one JSON row per (case, seed).

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.run_ops <B|C|G> <n> [--cases W N 0 H HN] [--seeds 8] [--seed0 0] [--tag eval] [--no-downstream]``.
``--tag calib`` uses other seeds and is only used to calibrate the acceptance constant (see ``aggregate``). Output: ``wave2/results/rows_<op>_<n>_<tag>.jsonl``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from ..brainstorm.h import clean, mi_pair, ybins
from ..common._paths import results_dir
from ..common.downstream import make_models
from .harness import cv_relative, existing_best, heldout_mi, leak_ablation, prep, replay_proof, top_cols
from .rowstats import fit_rowstat, replay_rowstat
from .targets import CASES
from .warp_service import oof_fit, replay


def _train_mi(f: np.ndarray, yb: tuple) -> float:
    """In-sample MI of a (out-of-fold) training feature against the fit-half target codes."""
    return mi_pair(clean(f), clean(f), yb[0], yb[0], yb[2])[0]


def fit_B(XA, ya, cols, yb, seed, m=20.0):
    """Operator B: over all pairs of ``cols`` and K in (6, 10) pick the cross-fitted cell table with the best train MI; returns (train column, recipe, info)."""
    best = None
    for i, j in itertools.combinations(cols, 2):
        for K in (6, 10):
            f, rec = oof_fit("cell2d", XA, ya, [i, j], seed=seed, K=K, m=m)
            m = _train_mi(f, yb)
            if best is None or m > best[0]:
                best = (m, f, rec, {"pair": [i, j], "K": K})
    return best[1], best[2], best[3]


def fit_C(XA, ya, cols, yb, seed):
    """Operator C: per column a cross-fitted 20-bin warp; keep the column with the best train MI; returns (train column, recipe, info)."""
    best = None
    for j in cols:
        f, rec = oof_fit("warp1d", XA, ya, [j], seed=seed, nb=20)
        m = _train_mi(f, yb)
        if best is None or m > best[0]:
            best = (m, f, rec, {"col": j})
    return best[1], best[2], best[3]


def fit_G(XA, ya, cols, yb, seed):
    """Operator G: learned (statistic, subset); returns (train column, recipe, info)."""
    rec = fit_rowstat(XA, ya, cols, seed=seed)
    return replay_rowstat(rec, XA), rec, {"stat": rec["stat"], "cols": rec["src"]}


OPS = {"B": (fit_B, replay), "C": (fit_C, replay), "G": (fit_G, replay_rowstat)}


def one_case(op: str, case: str, n: int, seed: int, downstream: bool = True, hgb_iter: int = 150, cell_m: float = 20.0) -> dict:
    """One (case, seed) evaluation row."""
    gen, cols0 = CASES[op][case]
    rng = np.random.default_rng(10_000 + seed)
    X, y, truth = gen(rng, n)
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    yb = ybins(ya, yB)
    cols = top_cols(XA, yb, cols0)
    t0 = time.time()
    _, ex = existing_best(XA, XB, ya, yB, cols)
    t_ex = time.time() - t0
    fit, rep = OPS[op]
    t0 = time.time()
    kw = {"m": cell_m} if op == "B" else {}
    fa, rec, info = fit(XA, ya, cols0 if op == "G" else cols, yb, seed, **kw)
    t_new = time.time() - t0
    fb = rep(rec, XB)
    row = dict(
        op=op, case=case, n=n, seed=seed, cell_m=cell_m if op == "B" else None, ex_name=ex[4], ex_tr=ex[0], ex_te=ex[1], t_new=t_new, t_existing=t_ex, info=info
    )
    row["new_tr"], row["new_te"] = heldout_mi(fa, fb, yb)
    row["gain"] = row["new_te"] - row["ex_te"]
    row["gain_n"] = row["gain"] * h
    row["n_fit"] = h
    if op != "G":
        fb_avg = rep(rec, XB, mode="foldavg")
        row["newfa_te"] = heldout_mi(fa, fb_avg, yb)[1]
    if truth is not None:
        row["truth_te"] = heldout_mi(truth[:h], truth[h:], yb)[1]
    ref = rep(rec, XB)
    row["replay"] = replay_proof(rec, rep, XB, ref, rng)
    if downstream:
        sets = {"raw": XB, "ex": np.column_stack([XB, prep(ex[2], ex[3])]), "new": np.column_stack([XB, prep(fa, fb)])}
        if op != "G":
            sets["newfa"] = np.column_stack([XB, prep(fa, fb_avg)])
        if truth is not None:
            sets["truth"] = np.column_stack([XB, prep(truth[:h], truth[h:])])
        models = make_models(hgb_iter)
        row.update(cv_relative(sets, yB, seed=seed, models=models))
        if op != "G":
            fin = rep(rec, XA)
            ab = {"raw": (XA, XB), "oof": (np.column_stack([XA, prep(fa, fa)]), sets["new"]), "insample": (np.column_stack([XA, prep(fin, fin)]), sets["new"])}
            row.update({f"leak|{k}": v for k, v in leak_ablation(ab, ya, yB, models).items()})
    return row


def main(argv=None) -> None:
    """CLI entry: run the requested cases and append one JSON row per (case, seed) to the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=sorted(OPS))
    ap.add_argument("n", type=int)
    ap.add_argument("--cases", nargs="+", default=["W", "N", "0", "H", "HN"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--no-downstream", action="store_true")
    ap.add_argument("--hgb-iter", type=int, default=150)
    ap.add_argument("--cell-m", type=float, default=20.0, help="operator B shrinkage pseudo-count")
    a = ap.parse_args(argv)
    out = results_dir("wave2") / f"rows_{a.op}_{a.n}_{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for case in a.cases:
            for s in range(a.seed0, a.seed0 + a.seeds):
                t0 = time.time()
                row = one_case(a.op, case, a.n, s, not a.no_downstream, a.hgb_iter, a.cell_m)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"{a.op}_{case} n={a.n} seed={s} MIex={row['ex_te']:.4f} MInew={row['new_te']:.4f} gain*n={row['gain_n']:.1f} {time.time() - t0:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()

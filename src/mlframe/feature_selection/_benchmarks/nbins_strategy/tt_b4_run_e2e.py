"""End-to-end value of the recommended transforms (research B4, direction (3)).

Per (p, seed): a correlated panel (tt_b4_panel.make_panel), n train rows, n fresh test rows. Feature sets:
  none        raw columns only
  blind       raw + EVERY valid candidate transform of EVERY column (the cost of adding everything)
  onepass     raw + transforms accepted by recommend_feature_transforms(selected=None) (rule gated, replication on)
  onepass_sel same with selected = the true informative columns (a good MRMR support)
  stepwise    raw + transforms accepted by recommend_stepwise ("add"); stepwise_rep = the same columns REPLACED (ridge only)
  oracle      brute force: for each column independently the candidate with the best held-out ridge R2 (75/25 split of the train rows), replaced, refit on all train rows
Models: ridge (standardised, lambda 1), additive cubic-spline GAM (6 quantile knots, ridge), HistGradientBoosting (200 iterations, 15 leaves). Held-out R2 and RMSE on the test rows.
Usage: python tt_b4_run_e2e.py --p 20 --n 20000 --seeds 3 --models ridge,gam,hgb --out FILE.jsonl [--skip blind_gam,blind_hgb]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_models import MODELS, r2_rmse, replacement_table
from tt_b4_panel import make_panel
from tt_b4_reference import CAND_DEFAULT, Recipes, all_candidate_matrix, recommend_feature_transforms, recommend_stepwise
from tt_b4_run_detect import config_for


def add_cols(Xtr, Xte, chosen):
    rec = Recipes(Xtr, chosen)
    a, b = rec.transform_cols(Xtr), rec.transform_cols(Xte)
    if not a:
        return Xtr, Xte
    return np.c_[Xtr, np.column_stack([a[j] for j in a])], np.c_[Xte, np.column_stack([b[j] for j in b])]


def replace_cols(Xtr, Xte, chosen):
    rec = Recipes(Xtr, chosen)
    a, b = rec.transform_cols(Xtr), rec.transform_cols(Xte)
    A, B = Xtr.copy(), Xte.copy()
    for j in a:
        A[:, j] = a[j]; B[:, j] = b[j]
    return A, B


def run_one(p, seed, n, models, skip):
    cfg = config_for(p, seed)
    Xall, yall, info = make_panel(2 * n, p, seed, **cfg)
    X, y, Xt, yt = Xall[:n], yall[:n], Xall[n:], yall[n:]
    inf = list(info["inf_cols"])
    sets = {}
    secs = {}
    sets["none"] = (X, Xt)
    t = time.perf_counter()
    Bm, meta = all_candidate_matrix(X, X)
    Bt, _ = all_candidate_matrix(X, Xt)
    sets["blind"] = (np.c_[X, Bm], np.c_[Xt, Bt])
    secs["blind"] = time.perf_counter() - t
    t = time.perf_counter(); r1 = recommend_feature_transforms(X, y, n_null=50, seed=seed, rule="gated"); secs["onepass"] = time.perf_counter() - t
    ch1 = {int(j): r1.table["best"][j] for j in np.flatnonzero(r1.table["accepted"])}
    sets["onepass"] = add_cols(X, Xt, ch1)
    t = time.perf_counter(); r2 = recommend_feature_transforms(X, y, selected=inf, n_null=50, seed=seed, rule="gated"); secs["onepass_sel"] = time.perf_counter() - t
    ch2 = {int(j): r2.table["best"][j] for j in np.flatnonzero(r2.table["accepted"])}
    sets["onepass_sel"] = add_cols(X, Xt, ch2)
    t = time.perf_counter(); r3 = recommend_stepwise(X, y, n_null=50, seed=seed); secs["stepwise"] = time.perf_counter() - t
    ch3 = {j: f for j, f in r3.table["chosen"].items() if f != "id"}
    sets["stepwise"] = add_cols(X, Xt, ch3)
    sets["stepwise_rep"] = replace_cols(X, Xt, ch3)
    t = time.perf_counter()
    nf = int(n * 0.75)
    base, tab = replacement_table(X[:nf], y[:nf], X[nf:], y[nf:], cols=None, cands=CAND_DEFAULT)
    ch_or = {}
    for j in range(p):
        row = np.where(np.isnan(tab[j]), -np.inf, tab[j])
        k = int(np.argmax(row))
        if CAND_DEFAULT[k] != "id":
            ch_or[j] = CAND_DEFAULT[k]
    sets["oracle"] = replace_cols(X, Xt, ch_or)
    secs["oracle"] = time.perf_counter() - t
    out = {"p": p, "seed": seed, "n": n, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "inf": [(int(c), info["link"][c]) for c in inf],
           "chosen": {"onepass": {int(k): v for k, v in ch1.items()}, "onepass_sel": {int(k): v for k, v in ch2.items()}, "stepwise": {int(k): v for k, v in ch3.items()},
                      "oracle": {int(k): v for k, v in ch_or.items()}},
           "seconds": secs, "ncols": {k: int(v[0].shape[1]) for k, v in sets.items()}, "res": {}}
    for name, (A, B) in sets.items():
        out["res"][name] = {}
        for m in models:
            if f"{name}_{m}" in skip or (name == "stepwise_rep" and m != "ridge"):
                continue
            t = time.perf_counter()
            pred = MODELS[m](A, y, B) if m != "hgb" else MODELS[m](A, y, B, seed=seed)
            r2, rmse = r2_rmse(yt, pred)
            out["res"][name][m] = {"r2": r2, "rmse": rmse, "seconds": time.perf_counter() - t}
        print(f"  seed {seed} {name:12s} cols {A.shape[1]:5d} " + " ".join(f"{m}={out['res'][name][m]['r2']:.4f}" for m in out["res"][name]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--models", default="ridge,gam,hgb")
    ap.add_argument("--skip", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = a.models.split(",")
    skip = set(a.skip.split(",")) if a.skip else set()
    with open(a.out, "a", encoding="utf-8") as f:
        for s in range(a.seed0, a.seed0 + a.seeds):
            t = time.perf_counter()
            o = run_one(a.p, s, a.n, models, skip)
            f.write(json.dumps(o) + "\n"); f.flush()
            print(f"p={a.p} n={a.n} seed={s} done in {time.perf_counter() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()

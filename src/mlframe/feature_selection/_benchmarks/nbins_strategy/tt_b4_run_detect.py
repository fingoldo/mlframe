"""Detection of the right unary transform on correlated panels (research B4, direction (2)).

For each (p, seed): a panel with 3-6 nonlinear informative columns (known links), 1 linear informative column, correlated partner noise columns (rho 0.3-0.9),
mixed heavy-tailed / discrete noise. The JUDGE is the brute-force table on a fresh test set: test R2 of ridge on all raw columns with ONE column replaced by ONE candidate.
Methods (all see only the training rows):
  M1   one pass, selected=None (marginal), rules column / maxT / pooled / gated, each with replication on the held-out half
  M2   one pass, selected = the true informative columns as raw linear columns (what a good MRMR support would supply)
  M3   stepwise (relevance gate -> transform decision -> design grows -> backfit)
Scoring per informative nonlinear column j: G* = best judge gain over identity; realized gain of the method's pick; hit (>= 0.8 G*), partial (0.3-0.8), miss.
Linear informative columns: any accepted transform is useless. Partner / noise columns: any accepted transform is a wrong credit.
Usage: python tt_b4_run_detect.py --p 20 --seeds 6 --out FILE.jsonl
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
from tt_b4_models import replacement_table
from tt_b4_panel import make_panel
from tt_b4_reference import CAND_DEFAULT, CAND_MRMR, recommend_feature_transforms, recommend_stepwise

RULES = ("column", "maxT", "pooled", "gated")


def config_for(p, seed):
    k_nl = (4, 5, 3, 6, 4, 5, 3, 6)[seed % 8]
    return dict(k_nl=k_nl, n_lin=1 if seed % 2 == 0 else 2, heavy_informative=bool(seed % 2), discrete_informative=bool(seed % 3 == 0),
                lin_marginals=("gauss", "lognormal") if seed % 2 else ("gauss",))


def run_one(p, seed, n, r2=0.7, cand_set="default"):
    CANDS = CAND_MRMR if cand_set == "mrmr" else CAND_DEFAULT
    cfg = config_for(p, seed)
    cfg["r2"] = r2
    Xall, yall, info = make_panel(2 * n, p, seed, **cfg)
    X, y, Xt, yt = Xall[:n], yall[:n], Xall[n:], yall[n:]
    inf = info["inf_cols"]
    role = info["role"]
    out = {"p": p, "seed": seed, "n": n, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           "inf": [(int(c), info["link"][c], info["marginal"][c]) for c in inf], "partners": [(c, info["partner_of"][c], info["rho"][c]) for c in range(p) if role[c] == "partner"]}
    res = {}
    t = time.perf_counter(); r1 = recommend_feature_transforms(X, y, candidates=CANDS, n_null=50, seed=seed, rule="gated", replicate_rules=RULES); t1 = time.perf_counter() - t
    tb = r1.table
    res["M1"] = {"seconds": t1, "accepted": {rl: {int(j): tb["best"][j] for j in np.flatnonzero(tb["accepted_by_rule"][rl])} for rl in RULES}}
    t = time.perf_counter(); r2 = recommend_feature_transforms(X, y, selected=list(inf), candidates=CANDS, n_null=50, seed=seed, rule="gated", replicate_rules=RULES); t2 = time.perf_counter() - t
    tb2 = r2.table
    res["M2"] = {"seconds": t2, "accepted": {rl: {int(j): tb2["best"][j] for j in np.flatnonzero(tb2["accepted_by_rule"][rl])} for rl in RULES}}
    t = time.perf_counter(); r3 = recommend_stepwise(X, y, candidates=CANDS, n_null=50, seed=seed); t3 = time.perf_counter() - t
    ch = r3.table["chosen"]
    res["M3"] = {"seconds": t3, "chosen": {int(j): f for j, f in ch.items()}, "accepted": {"stepwise": {int(j): f for j, f in ch.items() if f != "id"}}}
    # judge: columns to evaluate
    need = set(inf) | {c for c in range(p) if role[c] == "partner"}
    for m in ("M1", "M2"):
        for rl in RULES:
            need |= set(res[m]["accepted"][rl])
    need |= set(res["M3"]["chosen"])
    rng = np.random.default_rng(seed)
    if p <= 100:
        cols = list(range(p))
    else:
        extra = [int(c) for c in rng.choice([c for c in range(p) if c not in need], 40, replace=False)]
        cols = sorted(need | set(extra))
    t = time.perf_counter()
    base, tab = replacement_table(X, y, Xt, yt, cols=cols, cands=CANDS)
    out["judge_seconds"] = time.perf_counter() - t
    out["base_r2"] = base
    out["judge"] = {int(c): [None if np.isnan(v) else float(v - base) for v in tab[a]] for a, c in enumerate(cols)}
    out["methods"] = res
    out["cands"] = list(CANDS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--r2", type=float, default=0.7)
    ap.add_argument("--cands", default="default", choices=["default", "mrmr"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "a", encoding="utf-8") as f:
        for s in range(a.seed0, a.seed0 + a.seeds):
            t = time.perf_counter()
            o = run_one(a.p, s, a.n, a.r2, a.cands)
            f.write(json.dumps(o) + "\n"); f.flush()
            print(f"p={a.p} seed={s} done in {time.perf_counter() - t:.1f} s (M1 {o['methods']['M1']['seconds']:.1f}, M2 {o['methods']['M2']['seconds']:.1f}, "
                  f"M3 {o['methods']['M3']['seconds']:.1f}, judge {o['judge_seconds']:.1f})", flush=True)


if __name__ == "__main__":
    main()

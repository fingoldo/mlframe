"""B2 driver, TARGET side: run wave-1 criterion (and B2 fixes) over (family, variant, n, p, seed) cells; JSONL per run.

python tt_b2_run_target.py --out FILE [--fam a,b] [--var homo,mult] [--n 5000] [--p 8] [--seeds 7] [--B 99] [--shard 0/2] [--fixes f1,f2]
Each line: cell keys + picks per criterion + judge ratios (rmse_smear, mae_clip per candidate, relative scores in raw units) + diagnostics.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np

import tt_b2_core as C
import tt_b2_crit as K
import tt_b2_fix as FX


def run_cell(fam: str, var: str, n: int, p: int, seed: int, B: int, fixes: tuple, n_te: int = 50000) -> dict:
    ss = zlib.crc32(f"{fam}|{var}|{n}|{p}|{seed}".encode())
    rng = np.random.default_rng(ss)
    X, y = C.gen_target(fam, var, n, p, np.random.default_rng(ss + 1))
    Xt, yt = C.gen_target(fam, var, n_te, p, np.random.default_rng(ss + 2))
    t0 = time.perf_counter()
    res = K.target_decide(X, y, rng, B=B)
    extra = FX.apply_fixes(res, X, y, rng, B, fixes) if fixes else {}
    t_crit = time.perf_counter() - t0
    cj = dict(res["cands"])
    base_names = list(cj)
    if "E" in fixes and float(y.min()) >= 0:
        ce = C.build_candidates(y, rng, extended=True)
        cj.update({k: ce[k] for k in ("log1p", "sqrt")})
    jd = C.judge_target(X, y, Xt, yt, cj, seed=seed)
    am, nu = K.atom_mass(y)
    out = dict(
        fam=fam, var=var, n=n, p=p, seed=seed, picks={k: res[k] for k in ("LL", "LLnull", "W1")}, best=res["best"], gain=res["gain"], tau=res["tau"], pval=res["pval"], lam=res["lam"],
        g2_ratio=res.get("g2_ratio"), atom=am, n_unique=nu, ymin=float(y.min()), ymax=float(y.max()), t_crit=t_crit,
        base_names=base_names, rmse={k: v["rmse_smear"] for k, v in jd.items()}, mae={k: v["mae_clip"] for k, v in jd.items()}, rmse_med={k: v["rmse_clip"] for k, v in jd.items()},
    )
    out["picks"].update(extra.get("picks", {}))
    out["diag"] = extra.get("diag", {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fam", default=",".join(C.FAMILIES))
    ap.add_argument("--var", default="homo,mult")
    ap.add_argument("--cfgs", default="", help="fam:var,fam:var overrides the fam x var product")
    ap.add_argument("--n", default="5000")
    ap.add_argument("--p", default="8")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--B", type=int, default=99)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fixes", default="")
    a = ap.parse_args()
    si, sm = map(int, a.shard.split("/"))
    fixes = tuple(f for f in a.fixes.split(",") if f)
    fv = [tuple(x.split(":")) for x in a.cfgs.split(",")] if a.cfgs else [(f, v) for f in a.fam.split(",") for v in a.var.split(",")]
    cells = [(f, v, int(n), int(p), s) for f, v in fv for n in a.n.split(",") for p in a.p.split(",") for s in range(a.seed0, a.seed0 + a.seeds)]
    cells = [c for i, c in enumerate(cells) if i % sm == si]
    t0 = time.perf_counter()
    with open(a.out, "a", encoding="utf-8") as fh:
        for i, c in enumerate(cells):
            r = run_cell(*c, B=a.B, fixes=fixes)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"[{si}/{sm}] {i + 1}/{len(cells)} {c} picks={r['picks']} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", a.out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

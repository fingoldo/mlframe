"""B2 driver, FEATURE side (single feature under test + one linear covariate z): wave-1 criterion (A2 Pearson info of f(x) with the residual of y, permutation null)
versus a fresh-data judge (ridge on [z, f(x)], RMSE and MAE on 50000 fresh rows).

python tt_b2_run_feat.py --out FILE [--dist a,b] [--link l1,l2] [--n 5000] [--amp 1.0] [--seeds 10] [--B 100] [--shard 0/2] [--extended 0]
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np

import tt_b2_core as C
import tt_b2_crit as K


def judge_feature(x, z, y, xt, zt, yt, extended=False):
    Ctr = K.feat_candidates(x, extended)
    Cte = K.feat_candidates(xt, extended, ref=x)
    out = {}
    D0 = np.c_[np.ones(len(y)), z]
    for k in Ctr:
        D = np.c_[D0, Ctr[k]]
        Dt = np.c_[np.ones(len(yt)), zt, Cte[k]]
        W = C._ridge_multi(D, y[:, None])[:, 0]
        e = yt - Dt @ W
        out[k] = (float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))))
    return out


def run_cell(dist, link, n, amp, seed, B, extended, n_te=50000):
    ss = zlib.crc32(f"{dist}|{link}|{n}|{amp}|{seed}".encode())
    rng = np.random.default_rng(ss)
    x, z, y = C.gen_feature_fixture(dist, link, n, np.random.default_rng(ss + 1), amp)
    xt, zt, yt = C.gen_feature_fixture(dist, link, n_te, np.random.default_rng(ss + 2), amp)
    d = K.feat_decide(x, z[:, None], y, rng, B=B, extended=extended)
    j = judge_feature(x, z, y, xt, zt, yt, extended)
    return dict(dist=dist, link=link, n=n, amp=amp, seed=seed, pick=d["pick"], best=d["best"], gain=d["gain"], tau=d["tau"], pval=d["pval"],
                rmse={k: v[0] for k, v in j.items()}, mae={k: v[1] for k, v in j.items()}, ext=extended)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist", default=",".join(C.XDISTS))
    ap.add_argument("--link", default=",".join(C.LINKS))
    ap.add_argument("--pairs", default="", help="dist:link,dist:link overrides the dist x link product")
    ap.add_argument("--n", default="5000")
    ap.add_argument("--amp", default="1.0")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--extended", type=int, default=0)
    a = ap.parse_args()
    si, sm = map(int, a.shard.split("/"))
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")] if a.pairs else [(d, l) for d in a.dist.split(",") for l in a.link.split(",") if C.link_valid(d, l)]
    cells = [(d, l, int(n), float(am), s) for d, l in pairs for n in a.n.split(",") for am in a.amp.split(",") for s in range(a.seeds)]
    cells = [c for i, c in enumerate(cells) if i % sm == si]
    t0 = time.perf_counter()
    with open(a.out, "a", encoding="utf-8") as fh:
        for i, c in enumerate(cells):
            r = run_cell(*c, B=a.B, extended=bool(a.extended))
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{si}/{sm}] {i + 1}/{len(cells)} {c} pick={r['pick']} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", a.out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

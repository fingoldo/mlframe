"""B2 type-I calibration on TRUE-NULL targets (identity is the exact model), many seeds, no judge: acceptance rates of the wave-1 stages and of the CV selector versus n.

python tt_b2_calib.py --out FILE [--cfg normal_homo,pos_linear,normal_mult] [--n 300,1000,5000] [--seeds 100] [--B 49]
Configs: normal_homo (y = 2u + g), pos_linear (y = 10 + 2u + g, so log / reciprocal are available and wrong), normal_mult ((6+2u)(1+0.2 g), heteroscedastic).
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


def gen(cfg, n, p, rng):
    if cfg == "normal_homo":
        return C.gen_target("normal", "homo", n, p, rng)
    if cfg == "normal_mult":
        return C.gen_target("normal", "mult", n, p, rng)
    if cfg == "pos_linear":
        X, y = C.gen_target("normal", "homo", n, p, rng)
        return X, y + 10.0
    raise ValueError(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="normal_homo,pos_linear,normal_mult")
    ap.add_argument("--n", default="300,1000,5000")
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--B", type=int, default=49)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sm = map(int, a.shard.split("/"))
    cells = [(c, int(n), s) for c in a.cfg.split(",") for n in a.n.split(",") for s in range(a.seeds)]
    cells = [x for i, x in enumerate(cells) if i % sm == si]
    t0 = time.perf_counter()
    with open(a.out, "a", encoding="utf-8") as fh:
        for i, (cfg, n, s) in enumerate(cells):
            ss = zlib.crc32(f"calib|{cfg}|{n}|{a.p}|{s}".encode())
            rng = np.random.default_rng(ss)
            X, y = gen(cfg, n, a.p, np.random.default_rng(ss + 1))
            res = K.target_decide(X, y, rng, B=a.B)
            cv = FX.cv_decide(res["des"], y, res["cands"], rng)
            cm = FX.cv_decide(res["des"], y, res["cands"], rng, kind="abs")
            fh.write(json.dumps(dict(cfg=cfg, n=n, p=a.p, seed=s, LL=res["LL"], LLnull=res["LLnull"], W1=res["W1"], CV=cv["pick"], CVm=cm["pick"], cv_p=cv["pval"], pval=res["pval"], B=a.B)) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{si}/{sm}] {i + 1}/{len(cells)} {cfg} n={n} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", a.out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""B2 driver, many features: p features, 3 informative (x0 lognormal with log link, x1 normal with square link, x2 normal linear control), p-3 noise features of mixed
distributions. Wave-1 A2 criterion on every column (residual = y - OLS on all other columns), decided by four rules: per-feature null (max over candidates only), global
max-T over features x candidates, studentised global max-T, Bonferroni on per-feature permutation p-values.

python tt_b2_run_p.py --out FILE [--p 8,50,200] [--n 5000] [--seeds 7] [--B 100] [--amp 1.0] [--shard 0/2]
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np

import tt_b2_core as C
import tt_b2_crit as K

NOISE_CYCLE = ("normal", "lognormal", "uniform", "t3", "discrete")


def gen_multi(n: int, p: int, rng, amp: float):
    cols = [C.gen_x("lognormal", n, rng), C.gen_x("normal", n, rng), C.gen_x("normal", n, rng)]
    dists = ["lognormal", "normal", "normal"]
    for j in range(3, p):
        d = NOISE_CYCLE[(j - 3) % len(NOISE_CYCLE)]
        cols.append(C.gen_x(d, n, rng))
        dists.append(d)
    X = np.column_stack(cols)
    e = rng.standard_normal(n)
    y = amp * (C.g_std("lognormal", "log", X[:, 0]) + C.g_std("normal", "square", X[:, 1])) + 0.5 * X[:, 2] + 0.7 * e
    return X, y, dists


def true_design(X: np.ndarray) -> np.ndarray:
    """Context for the per-feature judge: the other informative columns enter with their TRUE links."""
    Xc = X.copy()
    Xc[:, 0] = np.log(X[:, 0])
    Xc[:, 1] = X[:, 1] ** 2
    return Xc


def judge_informative(X, y, Xt, yt, j, extended=False):
    """RMSE / MAE of ridge on [true-link context of the other informative columns, all noise columns, f(x_j)] for every candidate f of feature j."""
    Xc, Xct = true_design(X), true_design(Xt)
    keep = [c for c in range(X.shape[1]) if c != j]
    Ctr = K.feat_candidates(X[:, j], extended)
    Cte = K.feat_candidates(Xt[:, j], extended, ref=X[:, j])
    [c for c in keep if c < 3]
    keep = [c for c in keep]
    # other informative columns in their true form; for j's own slot swap in the candidate
    base, based = Xc[:, keep], Xct[:, keep]
    out = {}
    for k in Ctr:
        D = np.c_[np.ones(len(y)), base, Ctr[k]]
        Dt = np.c_[np.ones(len(yt)), based, Cte[k]]
        W = C._ridge_multi(D, y[:, None])[:, 0]
        e = yt - Dt @ W
        out[k] = (float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))))
    return out


def run_cell(n, p, seed, B, amp, extended, n_te=20000):
    ss = zlib.crc32(f"multi|{n}|{p}|{amp}|{seed}".encode())
    rng = np.random.default_rng(ss)
    X, y, dists = gen_multi(n, p, np.random.default_rng(ss + 1), amp)
    Xt, yt, _ = gen_multi(n_te, p, np.random.default_rng(ss + 2), amp)
    t0 = time.perf_counter()
    m = K.feat_many(X, y, B, rng, extended)
    acc = K.feat_accept(m)
    t_crit = time.perf_counter() - t0
    out = dict(n=n, p=p, seed=seed, amp=amp, t_crit=t_crit, best=m["best"], T=m["T"].tolist(), pj=acc["pj"].tolist(), ext=extended)
    for rule in ("perfeat", "maxT", "maxT_stud", "bonf"):
        out[rule] = acc[rule].astype(int).tolist()
    out["judge"] = {}
    for j in range(3):
        jd = judge_informative(X, y, Xt, yt, j, extended)
        out["judge"][str(j)] = dict(rmse={k: v[0] for k, v in jd.items()}, mae={k: v[1] for k, v in jd.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", default="8,50,200")
    ap.add_argument("--n", default="5000")
    ap.add_argument("--amp", default="1.0")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--extended", type=int, default=0)
    a = ap.parse_args()
    si, sm = map(int, a.shard.split("/"))
    cells = [(int(n), int(p), s, float(am)) for n in a.n.split(",") for p in a.p.split(",") for am in a.amp.split(",") for s in range(a.seed0, a.seed0 + a.seeds)]
    cells = [c for i, c in enumerate(cells) if i % sm == si]
    t0 = time.perf_counter()
    with open(a.out, "a", encoding="utf-8") as fh:
        for i, (n, p, s, am) in enumerate(cells):
            r = run_cell(n, p, s, a.B, am, bool(a.extended))
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{si}/{sm}] {i + 1}/{len(cells)} n={n} p={p} s={s} t_crit={r['t_crit']:.1f}s elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", a.out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

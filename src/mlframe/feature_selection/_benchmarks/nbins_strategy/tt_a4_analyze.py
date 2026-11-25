"""Summarise tt_a4_results.json: oracle best per (fixture, model), proposer correctness, regret, false-positive cost, unsafe-inverse distortion."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

PROP = ("identity", "cbrt", "signed_sqrt", "log", "yeo_johnson")
here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(here, "tt_a4_results.json")))
fixtures = sorted({r["fixture"] for r in R})
models = list(R[0]["judge"])


def sc(r, m, c, k="rmse_smear"):
    return r["judge"][m][c][k]


print("== oracle: top transform by mean ratio to identity (rmse_smear), per fixture x model; [mae_clip winner]")
for fx in fixtures:
    for m in models:
        rs = [r for r in R if r["fixture"] == fx]
        for k in ("rmse_smear", "mae_clip"):
            acc = defaultdict(list)
            for r in rs:
                idn = sc(r, m, "identity", k)
                for c, d in r["judge"][m].items():
                    acc[c].append(d[k] / idn)
            top = sorted(((np.mean(v), c) for c, v in acc.items() if len(v) == len(rs)), key=lambda t: t[0])[:3]
            print(f"{fx:17s} {m:12s} {k:10s} " + " ".join(f"{c}={v:.3f}" for v, c in top))

print("\n== proposer (MI equal-width) vs safe judge, rmse_smear, among its own 5 candidates")
print("fixture model: picked-is-best(own set)/n ; picked-is-within-2%-of-best ; mean ratio picked/identity ; mean regret picked/best(all cands)")
tot = defaultdict(lambda: [0, 0, 0, []])
for fx in fixtures:
    for m in models:
        rs = [r for r in R if r["fixture"] == fx]
        best = near = 0
        ratios, regrets = [], []
        for r in rs:
            own = {c: sc(r, m, c) for c in PROP if c in r["judge"][m]}
            p = r["picked"]
            b = min(own, key=own.get)
            best += p == b
            near += own[p] <= 1.02 * own[b]
            ratios.append(own[p] / own["identity"])
            allb = min(d["rmse_smear"] for d in r["judge"][m].values())
            regrets.append(own[p] / allb)
        picks = defaultdict(int)
        for r in rs:
            picks[r["picked"]] += 1
        print(f"{fx:17s} {m:12s} {best}/{len(rs)} {near}/{len(rs)} ratio {np.mean(ratios):.3f} regret {np.mean(regrets):.3f} picks {dict(picks)}")
        t = tot[m]
        t[0] += best
        t[1] += near
        t[2] += len(rs)
        t[3] += ratios
print("\n== totals per model")
for m, t in tot.items():
    rr = np.array(t[3])
    print(f"{m:12s} best {t[0]}/{t[2]} within2% {t[1]}/{t[2]} ratio mean {rr.mean():.3f} median {np.median(rr):.3f} worst {rr.max():.3f} share worse-than-identity(>1.02) {np.mean(rr > 1.02):.2f}")

print("\n== unsafe-inverse distortion for the proposer's pick (old judge vs safe judge)")
for m in models:
    nonf = tot_n = 0
    flips = 0
    for r in R:
        own_u = {c: sc(r, m, c, "rmse_unsafe") for c in PROP if c in r["judge"][m]}
        own_s = {c: sc(r, m, c) for c in PROP if c in r["judge"][m]}
        p = r["picked"]
        tot_n += 1
        nonf += not np.isfinite(own_u[p])
        flips += (min(own_u, key=lambda k: (own_u[k] if np.isfinite(own_u[k]) else 1e300)) != min(own_s, key=own_s.get))
    print(f"{m:12s} unsafe non-finite for pick {nonf}/{tot_n}; unsafe-best differs from safe-best {flips}/{tot_n}")
print("\n== mean retransformation effect: rmse_clip / rmse_smear of log on lognormal-type fixtures (>1 means smearing helps RMSE)")
for fx in ("lognormal", "hetero_lognormal", "exp_linear", "counts"):
    for m in models:
        for c in ("log", "log1p"):
            v = [r["judge"][m][c]["rmse_clip"] / r["judge"][m][c]["rmse_smear"] for r in R if r["fixture"] == fx and c in r["judge"][m]]
            if v:
                print(f"{fx:17s} {m:12s} {c:6s} {np.mean(v):.3f}")

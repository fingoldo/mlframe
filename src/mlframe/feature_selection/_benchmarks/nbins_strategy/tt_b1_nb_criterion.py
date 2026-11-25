"""Head-matched criteria and the cross table criterion x judge (research agent B1, wave 2).

The best transform of a feature depends on the head: logistic regression wants the log-likelihood-ratio to be linear in f(x), LDA wants a mean shift at common variance, Gaussian naive Bayes models the class
variance itself. This script adds the classification analogue of the wave-1 Jacobian-corrected Gaussian log-likelihood: the class-conditional Gaussian log-likelihood of f(x) plus the log-Jacobian of f,
  ccll_q = sum_i [ log N(f(x_i); m_c, v_c) + log|f'(x_i)| ]   (per-class variance: QDA / Gaussian naive Bayes head)
  ccll_l = same with the pooled variance                      (LDA head)
(non-monotone candidates square / abs fold two preimages onto one value: the Jacobian carries the factor 1/2). Gain over identity, permutation null of the class labels (95th percentile of the
maximum over candidates, 40 draws), same decision rule as tt_b1_cls_feature_transform.
It re-reads the JSON of tt_b1_cls_feature_transform.py (the data are regenerated from kind / K / seed), so every criterion can be scored against the LR, LDA and GNB judge columns stored there.

Usage: python tt_b1_nb_criterion.py file1.json [file2.json ...]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import collections
import json
import sys
import warnings

import numpy as np

import tt_b1_cls_feature_transform as T

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

TINY = 1e-9
GAIN, NONE, TOL, HARM = 0.010, 0.003, 0.005, 0.005


def logjac(name, x, med, iqr):
    ax = np.abs(x)
    if name == "id":
        return np.zeros_like(x)
    if name == "log":
        return -np.log(np.maximum(x, TINY))
    if name == "recip":
        return -2.0 * np.log(np.maximum(x, TINY))
    if name == "sqrt":
        return -np.log(2.0) - 0.5 * np.log(np.maximum(x, TINY))
    if name == "log1p":
        return -np.log1p(x)
    if name == "logit":
        return -np.log(np.maximum(x * (1.0 - x), TINY))
    if name == "square":
        return np.log(np.maximum(np.abs(x - med), TINY))
    if name == "abs":
        return np.full_like(x, -np.log(2.0))
    if name == "tanh":
        u = (x - med) / iqr
        return np.log(np.maximum(1.0 - np.tanh(u) ** 2, TINY)) - np.log(iqr)
    if name == "cbrt":
        return -np.log(3.0) - (2.0 / 3.0) * np.log(np.maximum(ax, 1e-6))
    if name == "asinh":
        u = (x - med) / iqr
        return -0.5 * np.log1p(u * u) - np.log(iqr)
    if name == "sqr_raw":
        return np.log(np.maximum(ax, TINY))
    raise ValueError(name)


def ccll(Fraw, J, y, K):
    """Per-row class-conditional Gaussian log-likelihood with Jacobian; returns (quadratic, linear) variants for every column."""
    n = len(y)
    Y = T.onehot(y, K)
    nk = Y.sum(0)
    M = (Y.T @ Fraw) / nk[:, None]
    V = np.maximum((Y.T @ (Fraw * Fraw)) / nk[:, None] - M * M, 1e-300)
    q = -0.5 * (nk[:, None] * (np.log(2 * np.pi * V) + 1.0)).sum(0) / n
    vp = (nk[:, None] * V).sum(0) / n
    l = -0.5 * (np.log(2 * np.pi * vp) + 1.0)
    return q + J.mean(0), l + J.mean(0)


def decide(obs, null, names):
    oth = [k for k in names if k != "id"]
    pick = max(oth, key=lambda k: obs[k])
    return pick, bool(obs[pick] - obs["id"] > max(np.quantile(null, 0.95), 0.0))


def main():
    rows = []
    for f in sys.argv[1:]:
        rows += json.load(open(f))
    out = []
    for r in rows:
        kind, K, seed = r["kind"], r["K"], r["seed"]
        n = 20000
        x, Z, y = T.make(kind, K, n, seed)
        h = n // 2
        C = T.cands(x, raw_sq=(kind == "square_off"))
        names = r["names"]
        xt = x[:h]
        med = float(np.median(x))
        iqr = float(np.subtract(*np.percentile(x, [75, 25]))) + 1e-12
        Fraw = np.c_[[C[k][:h] for k in names]].T
        J = np.c_[[logjac(k, xt, med, iqr) for k in names]].T
        ytr = y[:h]
        oq, ol = ccll(Fraw, J, ytr, K)
        rng = np.random.default_rng(seed + 777)
        iid = names.index("id")
        oth = [j for j in range(len(names)) if j != iid]
        nq, nl = [], []
        for _ in range(40):
            q, l = ccll(Fraw, J, ytr[rng.permutation(h)], K)
            nq.append(float(np.max(q[oth] - q[iid])))
            nl.append(float(np.max(l[oth] - l[iid])))
        res = {"kind": kind, "K": K, "seed": seed}
        res["ccll_q"] = decide(dict(zip(names, oq)), nq, names)
        res["ccll_l"] = decide(dict(zip(names, ol)), nl, names)
        for c in T.CRIT:
            d = T.decisions(r)[("null_perm", c)]
            res[c] = (d[0], d[1])
            d2 = T.decisions(r)[("null_bs", c)]
            res[c + "_bs"] = (d2[0], d2[1])
        res["judge"] = {"lr": r["judge_lr"], "lda": r["judge_lda"], "gnb": r["judge_gnb"]}
        out.append(res)
    crits = ["lrt", "lrt_bs", "eta", "eta_bs", "resid", "resid_bs", "mi_ew", "bd", "ccll_q", "ccll_l"]
    kinds = list(dict.fromkeys(o["kind"] for o in out))
    for jn in ("lr", "lda", "gnb"):
        print(f"\n===== judge = {jn.upper()} (held-out log loss of the head on [z, f(x)]); tiers: gain > {GAIN}, none < {NONE}, ambiguous between; counts of correct decisions per fixture =====")
        print(f"{'fixture':16s} {'K':>2s} {'g/a/n':>7s} {'best':>10s} " + " ".join(f"{c:>8s}" for c in crits))
        pooled = {c: [0, 0, 0, 0, 0] for c in crits}  # hits, gain cases, FP, none cases, harmful
        for kind in kinds:
            for K in sorted({o["K"] for o in out if o["kind"] == kind}):
                rs = [o for o in out if o["kind"] == kind and o["K"] == K]
                tiers, bests = [], []
                for o in rs:
                    jl = o["judge"][jn]
                    b = min(jl, key=lambda q: jl[q][0])
                    g = jl["id"][0] - jl[b][0]
                    tiers.append("gain" if g > GAIN else ("none" if g < NONE else "ambig"))
                    bests.append(b)
                tc = collections.Counter(tiers)
                line = f"{kind:16s} {K:>2d} {tc['gain']}/{tc['ambig']}/{tc['none']:<3d} {collections.Counter(bests).most_common(1)[0][0]:>10s} "
                for c in crits:
                    ok = 0
                    for o, t, b in zip(rs, tiers, bests):
                        jl = o["judge"][jn]
                        pick, acc = o[c]
                        harm = acc and jl[pick][0] - jl["id"][0] > HARM
                        if t == "gain":
                            good = acc and jl[pick][0] <= jl[b][0] + TOL
                            pooled[c][0] += good
                            pooled[c][1] += 1
                        elif t == "none":
                            good = not acc
                            pooled[c][2] += acc
                            pooled[c][3] += 1
                        else:
                            good = not harm
                        pooled[c][4] += harm
                        ok += good
                    line += f"{ok:>8d} "
                print(line)
        print("pooled over all fixtures: " + "; ".join(f"{c}: hits {v[0]}/{v[1]}, FP {v[2]}/{v[3]}, harmful {v[4]}" for c, v in pooled.items()))


if __name__ == "__main__":
    main()

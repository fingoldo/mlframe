"""Target-side analogue for classification (research agent B1, wave 2): does a monotone transform of the CONTINUOUS score that defines the classes matter?

A latent score s = 0.9 log(x1) + 0.5 z1 + 0.35 z2 + 0.5 noise (x1 lognormal) is replaced by t = g(s) for monotone g (identity, exp, cube, cbrt, sigmoid, -exp(-s)) and cut into K = 3 classes in four ways:
  Q  quantile terciles of t (rank cut)            -> labels must be IDENTICAL for every monotone g
  T  fixed cut values given on the s scale and mapped through g (same events {s <= c})   -> labels must be IDENTICAL
  E  three equal-width bins of t                  -> labels differ with g (this is the class definition, not a transform of anything the model sees)
  Z  mean +- 1 sd thresholds on t                 -> labels differ with g
For each (definition, g): label agreement with the g = identity labelling, smallest class share, held-out multinomial-logit macro AUC / log loss of [x1, z1, z2], and the feature-side decision
of the permutation-calibrated deviance gain (lrt, tt_b1_cls_feature_transform) for the candidate transforms of x1 (accepted?, pick).
Usage: python tt_b1_target_side.py [--seeds 7] [--rows 20000] [--nperm 40]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import collections
import warnings

import numpy as np

import tt_b1_cls_feature_transform as T

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

G = {
    "id": lambda s: s,
    "exp": np.exp,
    "cube": lambda s: s ** 3,
    "cbrt": np.cbrt,
    "sigmoid": lambda s: 1.0 / (1.0 + np.exp(-2.0 * s)),
    "negexp": lambda s: -np.exp(-s),
}
K = 3


def data(n, seed):
    rng = np.random.default_rng(777 + seed)
    x1 = np.exp(rng.standard_normal(n))
    Z = rng.standard_normal((n, 2))
    s = 0.9 * np.log(x1) + 0.5 * Z[:, 0] + 0.35 * Z[:, 1] + 0.5 * rng.standard_normal(n)
    return x1, Z, s


def labels(s, gname, scheme, cuts_s):
    g = G[gname]
    t = g(s)
    if scheme == "Q":
        q = np.quantile(t, [1 / 3, 2 / 3])
        return np.searchsorted(q, t, side="right")
    if scheme == "T":
        gc = g(np.asarray(cuts_s))
        return np.searchsorted(gc, t, side="right")
    if scheme == "E":
        e = np.linspace(t.min(), t.max(), K + 1)[1:-1]
        return np.searchsorted(e, t, side="right")
    if scheme == "Z":
        m, sd = t.mean(), t.std()
        return np.searchsorted([m - sd, m + sd], t, side="right")
    raise ValueError(scheme)


def decide(x1, Z, y, h, nperm, seed):
    """Permutation-calibrated deviance gain: (accepted, pick, gain) for transforms of x1 given the other features; classes with < 30 train rows make the case undefined."""
    ytr = y[:h]
    if np.bincount(ytr, minlength=K).min() < 30:
        return None
    C = T.cands(x1)
    names = list(C)
    Fm = np.c_[[C[k][:h] for k in names]].T
    Fs, _, _ = T.stdz(Fm)
    Zs, _, _ = T.stdz(Z[:h])
    Y = T.onehot(ytr, K)
    L0, ll0, _ = T.base_offset(Zs, Y)
    obs = T.crit_lrt(Fs, Y, L0, ll0)
    iid = names.index("id")
    oth = [j for j in range(len(names)) if j != iid]
    rng = np.random.default_rng(seed + 5)
    nul = []
    for _ in range(nperm):
        o = T.crit_lrt(Fs[rng.permutation(h)], Y, L0, ll0)
        nul.append(float(np.max(o[oth] - o[iid])))
    j = max(oth, key=lambda i: obs[i])
    return bool(obs[j] - obs[iid] > np.quantile(nul, 0.95)), names[j], float(obs[j] - obs[iid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--nperm", type=int, default=40)
    a = ap.parse_args()
    h = a.rows // 2
    rec = collections.defaultdict(list)
    ident = collections.Counter()
    for seed in range(a.seeds):
        x1, Z, s = data(a.rows, seed)
        cuts_s = np.quantile(s, [1 / 3, 2 / 3])
        base = {sc: labels(s, "id", sc, cuts_s) for sc in "QTEZ"}
        for sc in "QTEZ":
            for gname in G:
                y = labels(s, gname, sc, cuts_s)
                agree = float((y == base[sc]).mean())
                ident[(sc, gname)] += bool(np.array_equal(y, base[sc]))
                share = float(np.bincount(y, minlength=K).min() / len(y))
                if np.bincount(y[:h], minlength=K).min() < 30:
                    rec[(sc, gname)].append({"agree": agree, "share": share, "ll": np.nan, "auc": np.nan, "dec": None})
                    continue
                Ytr = T.onehot(y[:h], K)
                Xtr = np.c_[x1[:h], Z[:h]]
                Xte = np.c_[x1[h:], Z[h:]]
                mu, sd = Xtr.mean(0), Xtr.std(0)
                W, _ = T.fit_mnl(np.c_[np.ones(h), (Xtr - mu) / sd], Ytr)
                P = T.predict_proba(np.c_[np.ones(len(Xte)), (Xte - mu) / sd], W, K)
                ll, auc = T._metrics(P, y[h:], K)
                rec[(sc, gname)].append({"agree": agree, "share": share, "ll": ll, "auc": auc, "dec": decide(x1, Z, y, h, a.nperm, seed)})
    print(f"rows={a.rows}, seeds={a.seeds}, K={K}; x1 lognormal, the right feature transform is log; LR judge uses raw [x1, z1, z2]")
    print(f"{'def':3s} {'g':8s} {'identical labels vs id':>22s} {'agree':>7s} {'min share':>9s} {'LR logloss':>10s} {'LR AUC':>7s}  lrt accepted/picks")
    for sc in "QTEZ":
        for gname in G:
            r = rec[(sc, gname)]
            decs = [x["dec"] for x in r if x["dec"] is not None]
            acc = sum(d[0] for d in decs)
            picks = collections.Counter(d[1] if d[0] else "-" for d in decs).most_common(3)
            print(f"{sc:3s} {gname:8s} {ident[(sc, gname)]:>20d}/{a.seeds} {np.mean([x['agree'] for x in r]):7.3f} {np.mean([x['share'] for x in r]):9.3f} "
                  f"{np.nanmean([x['ll'] for x in r]) if decs else float('nan'):10.4f} {np.nanmean([x['auc'] for x in r]) if decs else float('nan'):7.4f}  {acc}/{len(decs)} {picks}")


if __name__ == "__main__":
    main()

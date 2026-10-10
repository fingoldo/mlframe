"""Operator M: pair-interaction screen on the residual of an out-of-fold additive model versus the same screen on y; W / N / 0 targets over 8 numeric columns.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.ops_M [--seeds S] [--n N]``."""

import argparse
import itertools
import json

import numpy as np

from ..common._paths import scratch_dir
from .h import UN, _best, clean, kfold_idx

SUB = ["identity", "sqr", "sqrt", "log", "reciproc", "sin"]


def U(x):
    """Six-unary transform stack of one column."""
    with np.errstate(all="ignore"):
        return np.array([clean(UN[k](x)) for k in SUB])


def backfit(X, y, nb_=15, sweeps=3):
    """Additive model by binned backfitting; returns the in-sample fitted values."""
    n, p = X.shape
    F = np.zeros((n, p))
    e = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, nb_ + 1)[1:-1])) for j in range(p)]
    I = [np.searchsorted(e[j], X[:, j]) for j in range(p)]
    mu = y.mean()
    for _ in range(sweeps):
        for j in range(p):
            r = y - mu - F.sum(1) + F[:, j]
            s = np.bincount(I[j], r)
            c = np.bincount(I[j])
            t = s / np.maximum(c, 1)
            t -= t.mean()
            F[:, j] = t[I[j]]
    return mu + F.sum(1)


def resid_oof(X, y, rng, k=5):
    """Out-of-fold residual of the additive model (each fold predicted from the others)."""
    r = np.zeros(len(y))
    for idx in kfold_idx(len(y), k, rng):
        m = np.ones(len(y), bool)
        m[idx] = False
        # fit additive on m, predict idx via bin lookup
        n, p = X[m].shape
        nb_ = 15
        Xa = X[m]
        e = [np.unique(np.quantile(Xa[:, j], np.linspace(0, 1, nb_ + 1)[1:-1])) for j in range(p)]
        I = [np.searchsorted(e[j], Xa[:, j]) for j in range(p)]
        F = np.zeros((n, p))
        mu = y[m].mean()
        tabs = [None] * p
        for _ in range(3):
            for j in range(p):
                rr = y[m] - mu - F.sum(1) + F[:, j]
                t = np.bincount(I[j], rr, len(e[j]) + 1) / np.maximum(np.bincount(I[j], minlength=len(e[j]) + 1), 1)
                t -= t.mean()
                F[:, j] = t[I[j]]
                tabs[j] = t
        pred = mu + sum(tabs[j][np.searchsorted(e[j], X[idx, j])] for j in range(p))
        r[idx] = y[idx] - pred
    return r


def screen(X, t, pairs):
    """Best preset-pair MI of ``t`` for every pair in ``pairs``."""
    yb = np.searchsorted(np.unique(np.quantile(t, np.linspace(0, 1, 11)[1:-1])), t, side="right").astype(np.int64)
    ny = yb.max() + 1
    Us = [U(X[:, j]) for j in range(X.shape[1])]
    return np.array([_best(Us[i], Us[j], yb, ny)[0] for i, j in pairs])


def gen(rng, n, kind):
    """Synthetic data: additive part plus a log(2 x4) sin(3 x5) interaction for kind W, none for N, pure noise y for 0."""
    X = rng.random((n, 8))
    add = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 * 2 + 1.5 * X[:, 2] - np.abs(X[:, 3] - 0.5) * 3
    inter = (np.log(2 * X[:, 4]) * np.sin(3 * X[:, 5]) * 2.5) if kind == "W" else 0 * X[:, 0]
    s = add + inter
    y = rng.standard_normal(n) if kind == "0" else s + 0.2 * s.std() * rng.standard_normal(n)
    return X, y


def main(argv=None) -> None:
    """W / N / 0 runs of the residual interaction screen; ``--seeds`` and ``--n`` shrink it for a smoke test (defaults are the protocol values)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=8000)
    args = ap.parse_args(argv)
    pairs = list(itertools.combinations(range(8), 2))
    tp = pairs.index((4, 5))
    out = (scratch_dir("brainstorm") / "results_M.jsonl").open("a")
    for kind in ("W", "N", "0"):
        R = []
        for s in range(args.seeds):
            rng = np.random.default_rng(2000 + s)
            X, y = gen(rng, args.n, kind)
            r = resid_oof(X, y, rng)
            sy, sr = screen(X, y, pairs), screen(X, r, pairs)
            perm = rng.permutation(len(r))
            sn = screen(X, r[perm], pairs)

            def rk(sc):
                """Rank of the true pair among the screened pairs (1 = best)."""
                return int((sc > sc[tp]).sum()) + 1

            R.append(
                dict(
                    rank_y=rk(sy),
                    rank_r=rk(sr),
                    top_y=sy.max(),
                    top_r=sr.max(),
                    true_y=sy[tp],
                    true_r=sr[tp],
                    null_top=sn.max(),
                    top_r_pair=str(pairs[int(sr.argmax())]),
                    top_y_pair=str(pairs[int(sy.argmax())]),
                )
            )
        m = {k: float(np.mean([r[k] for r in R])) for k in R[0] if not k.endswith("pair")}
        print(
            kind, {k: round(v, 4) for k, v in m.items()}, "top_y pairs", [r["top_y_pair"] for r in R], "top_r pairs", [r["top_r_pair"] for r in R], flush=True
        )
        out.write(json.dumps(dict(kind=kind, mean=m)) + "\n")
    out.close()


if __name__ == "__main__":
    main()

"""Brainstorm harness: held-out binned MI helpers, the existing-candidate baseline (17 unary x 17 unary x 6 binary = 1734 combos), out-of-fold helpers and ``run_case`` (held-out MI plus ridge / HGB MAE and RMSE)."""

import argparse
import itertools
import json
import time
import warnings

import numba as nb
import numpy as np

warnings.filterwarnings("ignore")
from ..common.downstream import METRICS, REFERENCE_METRICS, errors_by_feature_set, make_models, relative_table

NB = 10


@nb.njit(cache=True)
def _mi1(f, y, e, ny):
    """Plug-in MI (nats) of feature values binned by the interior edges ``e`` against integer classes ``y``."""
    c = np.zeros((len(e) + 1, ny))
    n = len(f)
    for i in range(n):
        c[np.searchsorted(e, f[i], side="right"), y[i]] += 1.0
    r = c.sum(1)
    k = c.sum(0)
    m = 0.0
    for i in range(c.shape[0]):
        for j in range(ny):
            if c[i, j] > 0:
                m += c[i, j] / n * np.log(c[i, j] * n / (r[i] * k[j]))
    return m


@nb.njit(cache=True)
def mi_pair(ftr, fte, ytr, yte, ny):
    """Decile edges from the train feature; returns (train MI, test MI) with those edges."""
    s = np.sort(ftr)
    n = len(ftr)
    e = np.empty(9)
    for i in range(9):
        e[i] = s[(i + 1) * n // 10]
    return _mi1(ftr, ytr, e, ny), _mi1(fte, yte, e, ny)


def clean(f):
    """NaN to 0 and clip to +-1e12 so downstream binning never sees non-finite values."""
    f = np.nan_to_num(np.asarray(f, float), nan=0.0, posinf=1e12, neginf=-1e12)
    return np.clip(f, -1e12, 1e12)


def ybins(ytr, yte):
    """Decile codes of the target, edges from the train part; returns (train codes, test codes, number of classes)."""
    e = np.unique(np.quantile(ytr, np.linspace(0, 1, NB + 1)[1:-1]))
    return np.searchsorted(e, ytr, side="right").astype(np.int64), np.searchsorted(e, yte, side="right").astype(np.int64), len(e) + 1


def MI(ftr, fte, yb):
    """Held-out MI helper: ``mi_pair`` on cleaned train / test features and the ``ybins`` tuple."""
    return mi_pair(clean(ftr), clean(fte), yb[0], yb[1], yb[2])


def _slog(x):
    """Signed-safe log: ``log|x|`` with 0 mapped to 0."""
    a = np.abs(x)
    return np.where(a > 0, np.log(np.where(a > 0, a, 1)), 0.0)


def _sp(x, p):
    """Signed power helper (kept for parity with the original harness)."""
    x = np.where(x == 0, 1e-9, x)
    return np.power(np.abs(x), p) * np.sign(x) if p == -1 else np.power(np.abs(x), p)


UN = {
    "identity": lambda x: x,
    "neg": lambda x: -x,
    "abs": np.abs,
    "sqr": lambda x: x * x,
    "reciproc": lambda x: 1 / np.where(x == 0, 1e-9, x),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "log": _slog,
    "sin": np.sin,
    "sign": np.sign,
    "rint": np.rint,
    "qubed": lambda x: x**3,
    "invsq": lambda x: 1 / np.where(x == 0, 1e-9, x * x),
    "invcub": lambda x: 1 / np.where(x == 0, 1e-9, x**3),
    "cbrt": np.cbrt,
    "invcbrt": lambda x: 1 / np.where(x == 0, 1e-9, np.cbrt(x)),
    "invsqrt": lambda x: 1 / np.where(x == 0, 1e-9, np.sqrt(np.abs(x))),
    "exp": lambda x: np.exp(np.clip(x, -50, 50)),
}
BI = {"mul": np.multiply, "add": np.add, "sub": np.subtract, "div": lambda a, b: a / np.where(b == 0, 1e-9, b), "max": np.maximum, "min": np.minimum}


@nb.njit(cache=True)
def _best(UA, UB, y, ny):
    """Best train-MI combo over all unary x unary x binary pairs (njit); returns (MI, i, j, binary index)."""
    nu, n = UA.shape
    step = max(1, n // 1000)
    best = -1.0
    bi = bj = bb = 0
    f = np.empty(n)
    sub = np.empty((n + step - 1) // step)
    for i in range(nu):
        for j in range(nu):
            for b in range(6):
                for k in range(n):
                    a = UA[i, k]
                    c = UB[j, k]
                    if b == 0:
                        v = a * c
                    elif b == 1:
                        v = a + c
                    elif b == 2:
                        v = a - c
                    elif b == 3:
                        v = a / (c if c != 0 else 1e-9)
                    elif b == 4:
                        v = max(a, c)
                    else:
                        v = min(a, c)
                    if v > 1e12:
                        v = 1e12
                    elif v < -1e12:
                        v = -1e12
                    elif v != v:
                        v = 0.0
                    f[k] = v
                for k in range(len(sub)):
                    sub[k] = f[k * step]
                s = np.sort(sub)
                e = np.empty(9)
                for q in range(9):
                    e[q] = s[(q + 1) * len(s) // 10]
                m = _mi1(f, y, e, ny)
                if m > best:
                    best = m
                    bi = i
                    bj = j
                    bb = b
    return best, bi, bj, bb


def existing_pair(atr, btr, ate, bte, yb):
    """best of 17x17x6=1734 preset combos on train MI; returns (mi_tr, mi_te, ftr, fte, name)"""
    with np.errstate(all="ignore"):
        Ua = {k: clean(f(atr)) for k, f in UN.items()}
        Ub = {k: clean(f(btr)) for k, f in UN.items()}
        Ta = {k: clean(f(ate)) for k, f in UN.items()}
        Tb = {k: clean(f(bte)) for k, f in UN.items()}
        names = list(UN)
        bnames = list(BI)
        UA = np.array([Ua[k] for k in names])
        UB = np.array([Ub[k] for k in names])
        m, i, j, b = _best(UA, UB, yb[0], yb[2])
        best = (m, names[i], names[j], bnames[b])
        m, i, j, bn = best
        ftr = clean(BI[bn](Ua[i], Ub[j]))
        fte = clean(BI[bn](Ta[i], Tb[j]))
    a, b = MI(ftr, fte, yb)
    return a, b, ftr, fte, f"{bn}({i},{j})"


def existing(Xtr, Xte, yb, cols, pairs=True):
    """Best existing candidate: raw columns and (optionally) every pair preset combo, by train MI; returns (train MI, test MI, train feature, test feature, name)."""
    best = (-1,)
    for c in cols:
        a, b = MI(Xtr[:, c], Xte[:, c], yb)
        if a > best[0]:
            best = (a, b, Xtr[:, c], Xte[:, c], f"raw{c}")
    for i, j in itertools.combinations(cols, 2) if pairs else []:
        r = existing_pair(Xtr[:, i], Xtr[:, j], Xte[:, i], Xte[:, j], yb)
        if r[0] > best[0]:
            best = r
    return best


def run_case(name, gen, new_fn, cols, seeds=8, n=12000, out=None, do_hgb=True, pairs=True):
    """Run one operator on one synthetic target: held-out MI of the best existing candidate vs the new feature, and ridge / HGB MAE and RMSE with each feature set added.

    The split is the first half (fit: operator parameters, best existing candidate) vs the second half (score). The downstream error uses the same split; relative
    improvement is versus the raw-columns-only model of the same family. Wins counts seeds where the new feature's held-out MI beats the best existing candidate.
    """
    rows = []
    models = {k: v for k, v in make_models().items() if do_hgb or k != "hgb"}
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        X, y, truth = gen(rng, n)
        h = n // 2
        Xtr, Xte, ytr, yte = X[:h], X[h:], y[:h], y[h:]
        yb = ybins(ytr, yte)
        ex = existing(Xtr, Xte, yb, cols, pairs)
        t0 = time.time()
        ftr, fte, info = new_fn(Xtr, ytr, Xte, rng)
        tn = time.time() - t0
        mn = MI(ftr, fte, yb)
        mt = MI(truth[:h], truth[h:], yb) if truth is not None else (np.nan, np.nan)
        sets = {
            "raw": (Xtr, Xte),
            "ex": (np.column_stack([Xtr, ex[2]]), np.column_stack([Xte, ex[3]])),
            "new": (np.column_stack([Xtr, clean(ftr)]), np.column_stack([Xte, clean(fte)])),
        }
        if truth is not None:
            sets["truth"] = (np.column_stack([Xtr, clean(truth[:h])]), np.column_stack([Xte, clean(truth[h:])]))
        rel = relative_table(errors_by_feature_set(sets, ytr, yte, models))
        r = dict(ex_tr=ex[0], ex_te=ex[1], new_tr=mn[0], new_te=mn[1], tr_tr=mt[0], tr_te=mt[1], t_new=tn)
        for model in models:
            for fs in ("ex", "new", "truth"):
                for metric in (*METRICS, *REFERENCE_METRICS):
                    r[f"rel_{metric}|{model}|{fs}"] = rel.get(f"{model}|{fs}", {}).get(metric, np.nan)
        rows.append(r)
    keys = rows[0].keys()
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    sd = {k: float(np.nanstd([r[k] for r in rows])) for k in keys}
    win = int(sum(r["new_te"] > r["ex_te"] for r in rows))
    rel_txt = " ".join(
        f"{model}[ex/new/truth] MAE {mean[f'rel_mae|{model}|ex']:+.3f}/{mean[f'rel_mae|{model}|new']:+.3f}/{mean[f'rel_mae|{model}|truth']:+.3f}"
        f" RMSE {mean[f'rel_rmse|{model}|ex']:+.3f}/{mean[f'rel_rmse|{model}|new']:+.3f}/{mean[f'rel_rmse|{model}|truth']:+.3f}"
        f" (ref. R2 delta {mean[f'rel_r2|{model}|ex']:+.3f}/{mean[f'rel_r2|{model}|new']:+.3f}/{mean[f'rel_r2|{model}|truth']:+.3f})"
        for model in models
    )
    print(
        f"{name:34s} MIte ex={mean['ex_te']:.4f}({mean['ex_tr']:.4f}tr) new={mean['new_te']:.4f}({mean['new_tr']:.4f}tr) truth={mean['tr_te']:.4f} sd_new={sd['new_te']:.4f} wins={win}/{seeds} | "
        f"rel.improvement vs raw: {rel_txt} | t_new={mean['t_new']:.2f}s",
        flush=True,
    )
    if out is not None:
        out.write(json.dumps(dict(name=name, mean=mean, sd=sd, wins=win)) + "\n")
        out.flush()
    return mean


def kfold_idx(n, k, rng):
    """Random disjoint index folds ``p[i::k]`` of a permutation."""
    p = rng.permutation(n)
    return [p[i::k] for i in range(k)]


def oof(fit_pred, Xtr, ytr, Xte, k=5, rng=None):
    """fit_pred(Xa,ya,Xb)->pred ; returns OOF train feature and full-fit test feature"""
    rng = rng or np.random.default_rng(0)
    ftr = np.zeros(len(ytr))
    for idx in kfold_idx(len(ytr), k, rng):
        m = np.ones(len(ytr), bool)
        m[idx] = False
        ftr[idx] = fit_pred(Xtr[m], ytr[m], Xtr[idx])
    return ftr, fit_pred(Xtr, ytr, Xte)


def case_args(argv=None, seeds=8, n=12000):
    """Parse ``<case names> [--seeds S] [--n N]`` shared by the operator scripts; the defaults are the protocol values behind the committed results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="case names, e.g. B_W B_N B_0")
    ap.add_argument("--seeds", type=int, default=seeds)
    ap.add_argument("--n", type=int, default=n)
    return ap.parse_args(argv)

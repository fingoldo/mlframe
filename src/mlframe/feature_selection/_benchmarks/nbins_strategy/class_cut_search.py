"""Classification analogue of the target-transform search (PLAN-04): for a continuous or ordinal quantity, can the data tell WHERE to cut it into classes?

A latent score ``t`` is cut at several quantiles into a binary target ``1[t > q]``. For every cut the script prints the symmetric uncertainty ``SU = 2 MI / (H(x) + H(y))`` summed over the columns (the cut-comparable
normalisation: raw MI is capped by ``H(y)`` and peaks at the balanced cut for that reason alone) and the held-out AUC of a logistic regression and of a gradient-boosting classifier. The question the plan asks:
does the argmax of the normalised MI over the cuts pick the cut at which the features are most useful? Three fixtures: ``uniform`` (one linear signal), ``tail`` (a column that matters only above the 90th percentile)
and ``noise`` (no signal, the null of the curve).

Usage: ``python class_cut_search.py [--seeds 5] [--rows 20000]``.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Optional

import numpy as np

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

CUTS = (0.1, 0.25, 0.5, 0.75, 0.9)


def make(kind: str, n: int, p: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Features and the latent score of the given kind."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    noise = rng.standard_normal(n)
    if kind == "uniform":
        t = X[:, 0] + 0.5 * X[:, 1] + 0.7 * noise
    elif kind == "tail":
        t = X[:, 0] + 0.5 * X[:, 1] + 0.7 * noise + 3.0 * X[:, 2] * (X[:, 0] > 1.0)
    else:
        t = noise
    return X, t


def _codes(v: np.ndarray, k: int = 10) -> np.ndarray:
    """Equal-frequency codes of a column."""
    edges = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)[1:-1]))
    return np.searchsorted(edges, v, side="right").astype(np.int64)


def _su(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric uncertainty of two integer code vectors."""
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    j = np.bincount(a * kb + b, minlength=ka * kb).astype(np.float64).reshape(ka, kb)
    j /= j.sum()
    pa, pb = j.sum(1), j.sum(0)
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    m = j > 0
    mi = float((j[m] * np.log(j[m] / np.outer(pa, pb)[m])).sum())
    return 2.0 * mi / (ha + hb) if ha + hb > 0 else 0.0


def main(argv: Optional[list] = None) -> None:
    """Run the three fixtures over the cuts and print the curves and whether the argmaxes agree."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args(argv)
    for kind in ("uniform", "tail", "noise"):
        agree_lr = agree_gb = 0
        for seed in range(a.seeds):
            X, t = make(kind, a.rows, 8, seed)
            n = len(t)
            tr = np.arange(n) < n // 2
            te = ~tr
            cols = [_codes(X[:, j]) for j in range(X.shape[1])]
            rows = []
            for q in CUTS:
                y = (t > np.quantile(t, q)).astype(np.int64)
                su = sum(_su(c, y) for c in cols)
                lr = roc_auc_score(y[te], LogisticRegression(max_iter=500).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
                gb = roc_auc_score(y[te], HistGradientBoostingClassifier(max_iter=100).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
                rows.append((q, su, lr, gb))
            best_su = max(rows, key=lambda r: r[1])[0]
            best_lr = max(rows, key=lambda r: r[2])[0]
            best_gb = max(rows, key=lambda r: r[3])[0]
            agree_lr += int(best_su == best_lr)
            agree_gb += int(best_su == best_gb)
            curve = " ".join(f"q{r[0]:.2f}:SU={r[1]:.3f}/lr={r[2]:.3f}/gb={r[3]:.3f}" for r in rows)
            print(f"CURVE {kind} seed {seed} argmax SU {best_su} lr {best_lr} gb {best_gb} | {curve}", flush=True)
        print(f"AGREE {kind}: argmax(SU) equals argmax(held-out AUC) for logistic regression in {agree_lr}/{a.seeds}, for gradient boosting in {agree_gb}/{a.seeds}", flush=True)


if __name__ == "__main__":
    main()

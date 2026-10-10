"""Operators B, C and G on classification targets (binary, 4-class ordinal, 4-class permuted) built from the same W / N / 0 constructions.

The class probability is driven by the standardised noisy signal ``z`` of the regression generator (``targets.CASES``): binary ``P(1) = sigmoid(2 z)``; 4-class logits ``1.2 z (k - 1.5)``
(``4c``, class index monotone in the signal) and the same with the class labels permuted by ``(2, 0, 3, 1)`` (``4cp``, class index NOT monotone in the signal). For case ``0`` the signal is
pure noise, so the classes are independent of the columns. The operators are fitted on the class index as a numeric target (the only coding the wave-2 service supports; one-vs-rest warps
were not built), so ``4cp`` shows what that coding costs when the index carries no order.

Decision metrics: held-out MI of the engineered column with the class labels (decile edges from the fit half), and, with the column added to the raw ones (frozen from the fit half) in 5-fold
stratified CV on the held-out half, the relative improvement over raw columns of log-loss (decision) and Brier score, plus the absolute AUC difference (binary: AUC; 4-class: one-vs-rest macro AUC) for logistic regression and
HistGradientBoostingClassifier. Accuracy is not used.

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.classif run <n> --labels bin 4c 4cp --ops B C G --cases W N 0 --seeds 5`` then ``... summary <n>``.
Output: ``wave2/results/rows_cls_<n>.jsonl`` and ``classif.txt``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..brainstorm.h import existing
from ..common._paths import results_dir
from .harness import heldout_mi, prep, top_cols
from .stress import accept, fit_op, ms, read_rows, replay_op, write_rows
from .targets import CASES

__all__ = ["make_labels", "cv_class", "one_run", "summarize", "main"]

PERM = np.array([2, 0, 3, 1])


def make_labels(rng, z: np.ndarray, kind: str) -> np.ndarray:
    """Class labels from the standardised signal ``z``: ``bin`` Bernoulli(sigmoid(2 z)); ``4c`` softmax with logits 1.2 z (k - 1.5); ``4cp`` = ``4c`` relabelled by ``PERM``."""
    if kind == "bin":
        return (rng.random(len(z)) < 1.0 / (1.0 + np.exp(-2.0 * z))).astype(np.int64)
    lg = 1.2 * z[:, None] * (np.arange(4) - 1.5)[None, :]
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    c = (rng.random(len(z))[:, None] > np.cumsum(p, 1)).sum(1).clip(0, 3)
    return PERM[c].astype(np.int64) if kind == "4cp" else c.astype(np.int64)


def _models(hgb_iter: int) -> dict:
    """Fresh ``{'logreg': ..., 'hgb': ...}`` classifiers."""
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "hgb": HistGradientBoostingClassifier(max_iter=hgb_iter, early_stopping=False, random_state=0),
    }


def _scores(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    """Log-loss, Brier score and AUC (binary AUC or one-vs-rest macro AUC) of class probabilities ``p`` (columns ordered as classes 0..k-1)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    p = p / p.sum(1, keepdims=True)
    oh = np.eye(k)[y]
    auc = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(k)))
    return {"logloss": float(log_loss(y, p, labels=list(range(k)))), "brier": float(((p - oh) ** 2).sum(1).mean()), "auc": float(auc)}


def cv_class(sets: dict, y: np.ndarray, k: int, seed: int, hgb_iter: int = 100) -> dict:
    """Stratified 5-fold CV on the held-out half. Keys ``logloss|<model>|<set>`` and ``brier|...`` = relative improvement over the ``raw`` set (positive = better), ``auc|...`` = absolute difference."""
    err = {}
    for name, X in sets.items():
        for mn in ("logreg", "hgb"):
            sc = []
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                m = _models(hgb_iter)[mn].fit(X[a], y[a])
                pr = np.zeros((len(b), k))
                pr[:, m.classes_] = m.predict_proba(X[b])
                sc.append(_scores(y[b], pr, k))
            err[(mn, name)] = {kk: float(np.mean([s[kk] for s in sc])) for kk in sc[0]}
    out = {}
    for (mn, name), e in err.items():
        b = err[(mn, "raw")]
        out[f"logloss|{mn}|{name}"] = (b["logloss"] - e["logloss"]) / b["logloss"]
        out[f"brier|{mn}|{name}"] = (b["brier"] - e["brier"]) / b["brier"]
        out[f"auc|{mn}|{name}"] = e["auc"] - b["auc"]
    return out


def one_run(op: str, case: str, label: str, n: int, seed: int, hgb_iter: int = 100) -> dict:
    """One (operator, case, label type, seed) row: held-out MI against the best existing candidate and the classification metrics of raw / existing / new / truth feature sets."""
    gen, cols0 = CASES[op][case]
    rng = np.random.default_rng(40_000 + seed)
    X, ycont, truth = gen(rng, n)
    y = make_labels(rng, (ycont - ycont.mean()) / ycont.std(), label)
    k = 2 if label == "bin" else 4
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    yb = (ya, yB, k)
    cols = top_cols(XA, yb, cols0)
    t0 = time.perf_counter()
    ex = existing(XA, XB, yb, cols, True)
    t_ex = time.perf_counter() - t0
    t0 = time.perf_counter()
    fa, rec, info = fit_op(op, XA, ya.astype(float), cols0 if op == "G" else cols, yb, seed, **({"m": 3.0} if op == "B" else {}))
    t_fit = time.perf_counter() - t0
    fb = replay_op(op, rec, XB)
    tr, te = heldout_mi(fa, fb, yb)
    row = {
        "op": op,
        "case": case,
        "label": label,
        "n": n,
        "seed": seed,
        "n_fit": h,
        "ex_name": ex[4],
        "ex_te": ex[1],
        "new_te": te,
        "gain_n": (te - ex[1]) * h,
        "info": info,
    }
    row.update(t_fit=t_fit, t_existing=t_ex, truth_te=heldout_mi(truth[:h], truth[h:], yb)[1] if truth is not None else None)
    row.update(accept(te - ex[1], ex[1], h))
    sets = {"raw": XB, "ex": np.column_stack([XB, prep(ex[2], ex[3])]), "new": np.column_stack([XB, prep(fa, fb)])}
    if truth is not None:
        sets["truth"] = np.column_stack([XB, prep(truth[:h], truth[h:])])
    row.update(cv_class(sets, yB, k, seed, hgb_iter))
    return row


def summarize(n: int) -> str:
    """Per (label, operator, case): held-out MI (existing / new / truth), acceptance, and the relative log-loss improvement (plus Brier and AUC difference) of the ``ex`` and ``new`` sets."""
    rows = list({(r["op"], r["case"], r["label"], r["seed"]): r for r in read_rows(f"rows_cls_{n}.jsonl")}.values())
    out = [
        f"# classification, n={n}; logloss / brier = relative improvement over raw columns (positive = better), auc = absolute difference; mean+-sd over seeds"
    ]
    for lab in ("bin", "4c", "4cp"):
        for op in "BCG":
            for case in ("W", "N", "0"):
                rs = [r for r in rows if r["label"] == lab and r["op"] == op and r["case"] == case]
                if not rs:
                    continue
                out.append(
                    f"[{lab}] {op}_{case} seeds={len(rs)} MI ex={ms([r['ex_te'] for r in rs])} new={ms([r['new_te'] for r in rs])} truth={ms([r['truth_te'] for r in rs])} "
                    f"gain_n={np.mean([r['gain_n'] for r in rs]):.1f} acc c40={sum(r['acc_c40'] for r in rs)}/{len(rs)} both={sum(r['acc_both'] for r in rs)}/{len(rs)}"
                )
                for mod in ("logreg", "hgb"):
                    out.append(
                        f"      {mod:6s} "
                        + " | ".join(
                            f"{s}: LL {ms([r[f'logloss|{mod}|{s}'] for r in rs])} Br {ms([r[f'brier|{mod}|{s}'] for r in rs])} AUC {ms([r[f'auc|{mod}|{s}'] for r in rs])}"
                            for s in ("ex", "new")
                        )
                    )
    return "\n".join(out)


def main(argv=None) -> None:
    """CLI: ``run`` appends rows, ``summary`` writes ``classif.txt``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("n", type=int)
    ap.add_argument("--labels", nargs="+", default=["bin", "4c", "4cp"])
    ap.add_argument("--ops", nargs="+", default=["B", "C", "G"])
    ap.add_argument("--cases", nargs="+", default=["W", "N", "0"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--hgb-iter", type=int, default=100)
    a = ap.parse_args(argv)
    if a.mode == "summary":
        txt = summarize(a.n)
        (results_dir("wave2") / "classif.txt").write_text(txt)
        print(txt)
        return
    for lab in a.labels:
        for op in a.ops:
            for case in a.cases:
                t0 = time.time()
                for s in range(a.seed0, a.seed0 + a.seeds):
                    write_rows(f"rows_cls_{a.n}.jsonl", [one_run(op, case, lab, a.n, s, a.hgb_iter)])
                print(f"cls {lab} {op}_{case} n={a.n} seeds={a.seeds} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

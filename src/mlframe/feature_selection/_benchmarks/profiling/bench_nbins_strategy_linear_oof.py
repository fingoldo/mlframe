"""End-to-end comparison of ``nbins_strategy`` values on a synthetic binary target: MRMR is fitted inside every training fold, a logistic regression is trained on the columns it selects (raw and engineered), and the
out-of-fold metrics are reported next to the selected columns.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.bench_nbins_strategy_linear_oof [--rows 20000] [--cols 50] [--folds 3] [--strategies a,b,c] [--out FILE.json]``.

The data are those of ``rss_scale_fit.py`` (``BENCH_TARGET=binary``): standard-normal columns, ``signal = x0*x1 + 0.7*x2 - 0.5*x3**2 + 0.3*x4``, ``y = (signal + 0.5 * noise > median(signal))``. A linear model
sees only ``x2`` and ``x4`` of the five generating columns in raw form; ``x0*x1`` and ``x3**2`` have to come from the engineered columns, so the AUC of a strategy measures how much of the signal its selected
set carries in a form a linear model can use. Two reference rows frame it: every raw column, and the four generating terms.

Protocol. The selection is made on the training rows of each fold only (a selection made on all rows would leak the test fold into the metrics), the model is fitted on the training rows and predicts the held-out
rows. Two summaries per strategy: ``pooled`` (the metrics of the one out-of-fold prediction vector over all rows) and ``fold mean +- std`` (the metric of each fold, then the mean and the standard deviation).
``fold sets`` is the number of different selected sets across the folds (1 = the same columns in every fold). The MRMR settings are the all-families ones of ``family_scale.all_families_kwargs``.
Nothing else heavy may run beside it: the wall of a fold is part of the table.

HISTORY (2026-11-22, 20000 x 50, 3 folds, one run): see the tracker row MRMR-NBINS-STRATEGY-LINEAR-OOF-2026-11-22.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)

DEFAULT_STRATEGIES = "mdlp,mdlp_validated,optimal_joint,mah,sturges,freedman_diaconis,quantile_spacing,uniform"
METRICS = ("auc", "logloss", "acc", "brier")


def make_data(rows: int, cols: int) -> "tuple[pd.DataFrame, pd.Series]":
    """The binary-target frame of ``rss_scale_fit.py``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((rows, cols), dtype=np.float32), columns=[f"x{i}" for i in range(cols)])
    signal = X["x0"].to_numpy() * X["x1"].to_numpy() + 0.7 * X["x2"].to_numpy() - 0.5 * X["x3"].to_numpy() ** 2 + 0.3 * X["x4"].to_numpy()
    noisy = signal + 0.5 * rng.standard_normal(rows, dtype=np.float32)
    return X, pd.Series((noisy > np.median(signal)).astype(np.int8), name="y")


def _linear():
    """The linear model: standardised logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    """AUC, log loss, accuracy and Brier score of the probabilities ``p``."""
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y_true, p)), "logloss": float(log_loss(y_true, p)), "acc": float(accuracy_score(y_true, p > 0.5)), "brier": float(brier_score_loss(y_true, p))}


def summarise(y: np.ndarray, oof: np.ndarray, folds: list) -> dict:
    """Pooled metrics of the out-of-fold vector and the mean / std of the per-fold metrics."""
    out = {"pooled": metrics(y, oof)}
    per_fold = [metrics(y[te], oof[te]) for _tr, te in folds]
    out["fold_mean"] = {m: float(np.mean([f[m] for f in per_fold])) for m in METRICS}
    out["fold_std"] = {m: float(np.std([f[m] for f in per_fold])) for m in METRICS}
    return out


def reference_rows(X: pd.DataFrame, y: pd.Series, folds: list) -> list:
    """The two reference rows: all raw columns, and the generating terms ``x2, x4, x3^2, x0*x1``."""
    def raw_all(d):
        return d.to_numpy(dtype=np.float64)

    def terms(d):
        return np.column_stack([d["x2"], d["x4"], d["x3"] ** 2, d["x0"] * d["x1"]]).astype(np.float64)

    rows = []
    for label, fn in (("ref: all raw columns", raw_all), ("ref: x2, x4, x3^2, x0*x1 (generating terms)", terms)):
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            oof[te] = _linear().fit(fn(X.iloc[tr]), y.iloc[tr]).predict_proba(fn(X.iloc[te]))[:, 1]
        rows.append({"strategy": label, "features": "-", "wall_s": 0.0, "fold_sets": 0, **summarise(y.to_numpy(), oof, folds)})
    return rows


def strategy_row(strategy: str, X: pd.DataFrame, y: pd.Series, folds: list) -> dict:
    """One strategy: MRMR inside each fold, the linear model on its selected columns, the out-of-fold summaries."""
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = F.all_families_kwargs()
    kwargs["nbins_strategy"] = strategy
    oof = np.full(len(y), np.nan)
    chosen, walls = [], []
    for tr, te in folds:
        t0 = time.perf_counter()
        prior = float(y.iloc[tr].mean())
        try:
            m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kwargs).fit(X.iloc[tr], y.iloc[tr])
            a, b = m.transform(X.iloc[tr]), m.transform(X.iloc[te])
            names = [str(c) for c in (a.columns if hasattr(a, "columns") else range(a.shape[1]))]
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
            a, b = a[:, ok], b[:, ok]
        except Exception as exc:  # a failed fold scores as the prior: it shows in the table instead of aborting the run
            print(f"FOLDFAIL {strategy} {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            names, a, b = [], np.empty((len(tr), 0)), np.empty((len(te), 0))
        walls.append(time.perf_counter() - t0)
        chosen.append(names)
        oof[te] = _linear().fit(a, y.iloc[tr]).predict_proba(b)[:, 1] if a.shape[1] else prior
    row = {"strategy": strategy, "features": " | ".join(chosen[0]) if chosen[0] else "(none)", "wall_s": float(np.mean(walls)), "fold_sets": len({tuple(sorted(c)) for c in chosen}), "n_features": [len(c) for c in chosen]}
    row.update(summarise(y.to_numpy(), oof, folds))
    return row


def render(rows: list) -> str:
    """The table: pooled AUC / logloss, fold mean +- std of the AUC, accuracy, Brier, wall, fold sets, the first fold's columns."""
    lines = ["strategy | pooled AUC | fold AUC mean +- std | pooled logloss | pooled acc | pooled Brier | wall s/fold | fold sets | selected columns (fold 1)"]
    for r in rows:
        lines.append(
            f"{r['strategy']} | {r['pooled']['auc']:.4f} | {r['fold_mean']['auc']:.4f} +- {r['fold_std']['auc']:.4f} | {r['pooled']['logloss']:.4f} | {r['pooled']['acc']:.4f} | {r['pooled']['brier']:.4f} | "
            f"{r['wall_s']:.1f} | {r['fold_sets']} | {r['features']}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the references and every strategy, print the table, optionally write the JSON."""
    from sklearn.model_selection import StratifiedKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--cols", type=int, default=50)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    X, y = make_data(args.rows, args.cols)
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=0).split(X, y))
    rows = reference_rows(X, y, folds)
    for st in args.strategies.split(","):
        rows.append(strategy_row(st, X, y, folds))
        print(f"ROW {json.dumps(rows[-1])}", flush=True)
    print(render(rows), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()

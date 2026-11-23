"""End-to-end comparison of ``nbins_strategy`` values on a synthetic target (binary or regression): MRMR is fitted inside every training fold, a linear model is trained on the columns it selects (raw and
engineered), and the out-of-fold metrics are reported next to what the selection recovered of the generating columns.

Usage: ``python -m mlframe.feature_selection._benchmarks.nbins_strategy.linear_oof [--rows 20000] [--cols 50] [--folds 3] [--targets binary,regression] [--seeds 0,1] [--strategies a,b,c] [--out FILE.json]``.

DATA. Standard-normal columns, ``signal = x0*x1 + 0.7*x2 - 0.5*x3**2 + 0.3*x4``, ``noisy = signal + 0.5 * noise``; ``binary``: ``y = noisy > median(signal)``, ``regression``: ``y = noisy``. A linear model sees
only ``x2`` and ``x4`` of the five generating columns in raw form; ``x0*x1`` and ``x3**2`` must come from engineered columns, so the score of a strategy measures how much of the signal its selected set
carries in a form a linear model can use. Reference rows: every raw column, and the four generating terms.

PROTOCOL. The selection is made on the training rows of each fold only (a selection made on all rows would leak the test fold into the metrics), the linear model (logistic regression / ridge, standardised) is fitted
on the training rows and predicts the held-out rows. Per (target, strategy): ``pooled`` metrics (of the one out-of-fold prediction vector over all rows), ``fold mean +- std`` of the same metric, and the mean over
the data seeds. Binary metrics: AUC, log loss, accuracy, Brier. Regression metrics: R2, RMSE, MAE.

TRUTH RECOVERY (the criterion of ``gt_sweep.py`` at the level a selection that contains engineered columns can be judged): every selected feature name is parsed for the source columns it uses (``x<i>``).
``sources_recall`` is the share of the five generating columns (x0..x4) used by at least one selected feature; ``noise_source_share`` the share of selected features that use any noise column (x5..) as an INPUT, at any weight: an UPPER BOUND of the noise in the selection, not a count of noise signal. A noise column is often only a reference or a dither (an ordinal-pattern feature places x3 and x4 against a random column, which reveals their magnitudes; a SIR direction lists its candidate inputs), and the Fleuret gate admits such a feature because it has significant information about the signal columns given each selected one (tracker NOISE-IN-SELECTION, 2026-11-23). A loading-weighted share is the honest version (open);
``raw_precision`` / ``raw_recall`` look at the RAW selected columns only against {x0..x4}. ``fold sets`` is the number of different selected sets across the folds (1 = the same in every fold).
The MRMR settings are the all-families ones of ``family_scale.all_families_kwargs``. Nothing else heavy may run beside it: the wall of a fold is part of the table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Optional
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)

DEFAULT_STRATEGIES = "mdlp,mdlp_validated,optimal_joint,mah,sturges,freedman_diaconis,quantile_spacing,uniform"
TRUE_SOURCES = frozenset(range(5))
BINARY_METRICS = ("auc", "logloss", "acc", "brier")
REGRESSION_METRICS = ("r2", "rmse", "mae")
_SOURCE = re.compile(r"x(\d+)")
_RAW = re.compile(r"^x(\d+)$")


def make_data(rows: int, cols: int, target: str, seed: int) -> "tuple[pd.DataFrame, pd.Series]":
    """The frame of ``rss_scale_fit.py`` for the ``binary`` / ``regression`` target kinds, drawn from ``seed``."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((rows, cols), dtype=np.float32), columns=[f"x{i}" for i in range(cols)])
    signal = X["x0"].to_numpy() * X["x1"].to_numpy() + 0.7 * X["x2"].to_numpy() - 0.5 * X["x3"].to_numpy() ** 2 + 0.3 * X["x4"].to_numpy()
    noisy = signal + 0.5 * rng.standard_normal(rows, dtype=np.float32)
    if target == "binary":
        return X, pd.Series((noisy > np.median(signal)).astype(np.int8), name="y")
    if target == "regression":
        return X, pd.Series(noisy.astype(np.float32), name="y")
    raise ValueError(f"unknown target {target!r}; expected 'binary' or 'regression'")


def _linear(target: str):
    """The linear model: standardised logistic regression (binary) or ridge (regression)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000) if target == "binary" else Ridge(alpha=1.0))


def _predict(model, X: np.ndarray, target: str) -> np.ndarray:
    """Probability of class 1 (binary) or the predicted value (regression)."""
    return np.asarray(model.predict_proba(X)[:, 1] if target == "binary" else model.predict(X))


def metrics(target: str, y_true: np.ndarray, p: np.ndarray) -> dict:
    """The metrics of one prediction vector: AUC, log loss, accuracy, Brier (binary) or R2, RMSE, MAE (regression)."""
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

    if target == "binary":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return {"auc": float(roc_auc_score(y_true, p)), "logloss": float(log_loss(y_true, p)), "acc": float(accuracy_score(y_true, p > 0.5)), "brier": float(brier_score_loss(y_true, p))}
    return {"r2": float(r2_score(y_true, p)), "rmse": float(np.sqrt(mean_squared_error(y_true, p))), "mae": float(mean_absolute_error(y_true, p))}


def summarise(target: str, y: np.ndarray, oof: np.ndarray, folds: list) -> dict:
    """Pooled metrics of the out-of-fold vector and the mean / std of the per-fold metrics."""
    names = BINARY_METRICS if target == "binary" else REGRESSION_METRICS
    per_fold = [metrics(target, y[te], oof[te]) for _tr, te in folds]
    return {
        "pooled": metrics(target, y, oof),
        "fold_mean": {m: float(np.mean([f[m] for f in per_fold])) for m in names},
        "fold_std": {m: float(np.std([f[m] for f in per_fold])) for m in names},
    }


def sources_of(name: str) -> frozenset:
    """The source columns (``x<i>``) a selected feature name is built from."""
    return frozenset(int(i) for i in _SOURCE.findall(name))


def truth_recovery(selected: list) -> dict:
    """What the selected names recover of the generating columns: sources recall, the share of features that use a noise column, and the raw-only precision / recall."""
    if not selected:
        return {"sources_recall": 0.0, "noise_source_share": 0.0, "raw_precision": 0.0, "raw_recall": 0.0, "n_selected": 0, "n_engineered": 0}
    used: "set[str]" = set()
    noisy = 0
    raw = []
    for name in selected:
        src = sources_of(name)
        used |= src
        noisy += bool(src - TRUE_SOURCES)
        raw_match = _RAW.match(name)
        if raw_match:
            raw.append(int(raw_match.group(1)))
    hit = sum(1 for r in raw if r in TRUE_SOURCES)
    return {
        "sources_recall": len(used & TRUE_SOURCES) / len(TRUE_SOURCES),
        "noise_source_share": noisy / len(selected),
        "raw_precision": hit / len(raw) if raw else 0.0,
        "raw_recall": hit / len(TRUE_SOURCES),
        "n_selected": len(selected),
        "n_engineered": len(selected) - len(raw),
    }


def reference_rows(X: pd.DataFrame, y: pd.Series, folds: list, target: str) -> list:
    """The two reference rows: all raw columns, and the generating terms ``x2, x4, x3^2, x0*x1``."""
    def raw_all(d):
        return d.to_numpy(dtype=np.float64)

    def terms(d):
        return np.column_stack([d["x2"], d["x4"], d["x3"] ** 2, d["x0"] * d["x1"]]).astype(np.float64)

    rows = []
    for label, fn in (("ref: all raw columns", raw_all), ("ref: x2, x4, x3^2, x0*x1 (generating terms)", terms)):
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            oof[te] = _predict(_linear(target).fit(fn(X.iloc[tr]), y.iloc[tr]), fn(X.iloc[te]), target)
        rows.append({"strategy": label, "features": "-", "wall_s": 0.0, "fold_sets": 0, **summarise(target, y.to_numpy(), oof, folds)})
    return rows


def strategy_row(strategy: str, X: pd.DataFrame, y: pd.Series, folds: list, target: str, extra: Optional[dict] = None) -> dict:
    """One strategy: MRMR inside each fold, the linear model on its selected columns, the out-of-fold summaries and the truth recovery (mean over the folds)."""
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = F.all_families_kwargs()
    kwargs["nbins_strategy"] = strategy
    kwargs.update(extra or {})
    oof = np.full(len(y), np.nan)
    chosen, walls, recoveries = [], [], []
    for tr, te in folds:
        t0 = time.perf_counter()
        prior = float(y.iloc[tr].mean())
        try:
            m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kwargs).fit(X.iloc[tr], y.iloc[tr])
            a, b = m.transform(X.iloc[tr]), m.transform(X.iloc[te])
            names = [str(c) for c in (a.columns if hasattr(a, "columns") else range(a.shape[1]))]
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
            a, b, names = a[:, ok], b[:, ok], [nm for nm, k in zip(names, ok) if k]
        except Exception as exc:  # a failed fold scores as the prior: it shows in the table instead of aborting the run
            print(f"FOLDFAIL {strategy} {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            names, a, b = [], np.empty((len(tr), 0)), np.empty((len(te), 0))
        walls.append(time.perf_counter() - t0)
        chosen.append(names)
        recoveries.append(truth_recovery(names))
        oof[te] = _predict(_linear(target).fit(a, y.iloc[tr]), b, target) if a.shape[1] else prior
    row = {"strategy": strategy, "features": " | ".join(chosen[0]) if chosen[0] else "(none)", "wall_s": float(np.mean(walls)), "fold_sets": len({tuple(sorted(c)) for c in chosen})}
    row.update({k: float(np.mean([r[k] for r in recoveries])) for k in recoveries[0]})
    row.update(summarise(target, y.to_numpy(), oof, folds))
    return row


def render(rows: list, target: str) -> str:
    """One table per target: the pooled primary metric, its fold mean +- std, the secondary pooled metrics, the truth recovery, the wall and the first fold's columns."""
    if target == "binary":
        head = "strategy | pooled AUC | fold AUC mean +- std | pooled logloss | pooled acc"
        lead = lambda r: f"{r['pooled']['auc']:.4f} | {r['fold_mean']['auc']:.4f} +- {r['fold_std']['auc']:.4f} | {r['pooled']['logloss']:.4f} | {r['pooled']['acc']:.4f}"  # noqa: E731
    else:
        head = "strategy | pooled R2 | fold R2 mean +- std | pooled RMSE | pooled MAE"
        lead = lambda r: f"{r['pooled']['r2']:.4f} | {r['fold_mean']['r2']:.4f} +- {r['fold_std']['r2']:.4f} | {r['pooled']['rmse']:.4f} | {r['pooled']['mae']:.4f}"  # noqa: E731
    lines = [f"[{target}] {head} | sources recall | noise-source share (upper bound) | raw precision | raw recall | n selected (engineered) | wall s/fold | fold sets | selected columns (fold 1)"]
    for r in rows:
        truth = (
            f"{r['sources_recall']:.2f} | {r['noise_source_share']:.2f} | {r['raw_precision']:.2f} | {r['raw_recall']:.2f} | {r['n_selected']:.1f} ({r['n_engineered']:.1f})"
            if "sources_recall" in r
            else "- | - | - | - | -"
        )
        lines.append(f"{r['strategy']} | {lead(r)} | {truth} | {r['wall_s']:.1f} | {r['fold_sets']} | {r['features']}")
    return "\n".join(lines)


def main() -> None:
    """Run the references and every strategy for each target and data seed, print one table per target, optionally write the JSON."""
    from sklearn.model_selection import KFold, StratifiedKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--cols", type=int, default=50)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--targets", default="binary,regression")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    ap.add_argument("--mrmr", default="", help="extra MRMR keyword arguments as JSON, e.g. '{\"redundancy_policy\": \"emit_both\"}'")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    results: list = []
    for target in args.targets.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            X, y = make_data(args.rows, args.cols, target, seed)
            splitter = StratifiedKFold(args.folds, shuffle=True, random_state=seed) if target == "binary" else KFold(args.folds, shuffle=True, random_state=seed)
            folds = list(splitter.split(X, y))
            rows = reference_rows(X, y, folds, target)
            for st in args.strategies.split(","):
                rows.append(strategy_row(st, X, y, folds, target, json.loads(args.mrmr) if args.mrmr else None))
                print(f"ROW {json.dumps({'target': target, 'seed': seed, **rows[-1]})}", flush=True)
            for r in rows:
                results.append({"target": target, "seed": seed, **r})
            print(render(rows, target), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()

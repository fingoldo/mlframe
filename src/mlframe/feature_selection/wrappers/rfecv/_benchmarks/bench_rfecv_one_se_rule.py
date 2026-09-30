"""Multi-dataset bench: RFECV ``one_se_*`` tolerance band -- across-fold std (legacy) vs standard error std/sqrt(k).

Each (dataset, split-seed, estimator) runs ONE RFECV search; every rule is then replayed on the same ``cv_results_`` via
``select_optimal_nfeatures_`` (the search path does not depend on the rule), so all rules are paired on the identical curve and identical
held-out split.  Held-out OOS = a fresh 1/3 split never seen by RFECV; the kept subset is refit with a fresh estimator on the RFECV train split.

Run (background, JSON written to ``_results/``):
    python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_rfecv_one_se_rule [--seeds 5] [--jobs 3] [--fast]
Summarise:
    python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_rfecv_one_se_rule --summarize <results.json>[,<more.json>]

Verdict (13 datasets x 6 seeds x catboost/lightgbm/linear = 234 paired units; california_housing was unreachable offline and replaced by make_friedman1;
results in ``_results/rfecv_one_se_rule_20260930_065629.json`` + ``..._070439.json``): the SE band became the default.
  paired vs one_se_max_foldstd, dOOS mean +- se | W/T/L (tie = |d| <= 1e-4) | d n_kept:
    one_se_max (SE)  +0.00040 +- 0.00021 | 47/151/36 | -7.36     <- adopted (per estimator: catboost +0.00026, lightgbm +0.00062, linear +0.00031)
    one_se_max_2se   -0.00010 +- 0.00012 | 10/207/17 | +1.67     (REJECTED: no gain, keeps more)
    one_se_min (SE)  -0.00280 +- 0.00124 | 102/58/74 | -26.24    (REJECTED as default: loses OOS; remains selectable)
    argmax           -0.00094 +- 0.00080 | 93/79/62  | -20.62    (REJECTED as default: loses OOS)
  mean n_kept: foldstd 56.3, SE 49.0, 2SE 58.0, one_se_min 30.1, argmax 35.7.
  stability (mean pairwise Jaccard over seeds): foldstd 0.728, SE 0.641, 2SE 0.753, one_se_min 0.553, argmax 0.561 -- the SE band trades stability for size.
  Per-dataset dOOS is inside noise everywhere; largest feature cuts: clf_p_gg_n -40.8, clf_manynoise -22.9, reg_lowsnr -7.6.
The legacy band is ``n_features_selection_rule='one_se_max_foldstd'``.  The older single-synthetic CatBoost bench is
``mlframe/feature_selection/_benchmarks/bench_rfecv_one_se_rule.py``.

Rules: one_se_max_foldstd (legacy default), one_se_max (SE band, std/sqrt(k)), one_se_max_2se (2 x SE), one_se_min (SE), argmax.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RULES = ("one_se_max_foldstd", "one_se_max", "one_se_max_2se", "one_se_min", "argmax")
BASELINE = "one_se_max_foldstd"
ESTIMATORS = ("catboost", "lightgbm", "linear")
RESULTS_DIR = Path(__file__).parent / "_results"


def _synthetic(kind: str):
    from sklearn.datasets import make_classification, make_regression

    if kind == "clf_lowsnr":
        X, y = make_classification(1500, 40, n_informative=6, n_redundant=2, class_sep=0.5, flip_y=0.05, random_state=0, shuffle=False)
        task = "binary"
    elif kind == "clf_highsnr":
        X, y = make_classification(2000, 40, n_informative=10, n_redundant=0, class_sep=1.5, random_state=0, shuffle=False)
        task = "binary"
    elif kind == "clf_redundant":
        X, y = make_classification(2000, 40, n_informative=8, n_redundant=15, class_sep=1.0, random_state=0, shuffle=False)
        task = "binary"
    elif kind == "clf_manynoise":
        X, y = make_classification(2000, 150, n_informative=8, n_redundant=0, class_sep=1.0, random_state=0, shuffle=False)
        task = "binary"
    elif kind == "clf_p_gg_n":
        X, y = make_classification(150, 300, n_informative=8, n_redundant=0, class_sep=1.5, random_state=0, shuffle=False)
        task = "binary"
    elif kind == "clf_multiclass":
        X, y = make_classification(2000, 40, n_informative=10, n_redundant=4, n_classes=3, n_clusters_per_class=1, class_sep=1.0, random_state=0, shuffle=False)
        task = "multiclass"
    elif kind == "reg_lowsnr":
        X, y = make_regression(1500, 40, n_informative=6, noise=60.0, random_state=0, shuffle=False)
        task = "regression"
    elif kind == "reg_manynoise":
        X, y = make_regression(2000, 120, n_informative=10, noise=20.0, effective_rank=15, tail_strength=0.5, random_state=0, shuffle=False)
        task = "regression"
    elif kind == "reg_friedman":
        from sklearn.datasets import make_friedman1

        X, y = make_friedman1(1500, n_features=30, noise=1.5, random_state=0)
        task = "regression"
    else:
        raise KeyError(kind)
    return X, y, task


def load_dataset(name: str):
    """Return ``(X_df, y, task)`` with a FIXED generation seed so feature identity is stable across split seeds (Jaccard stability)."""
    from sklearn import datasets as skd

    if name.startswith(("clf_", "reg_")):
        X, y, task = _synthetic(name)
    elif name == "breast_cancer":
        X, y = skd.load_breast_cancer(return_X_y=True)
        task = "binary"
    elif name == "wine":
        X, y = skd.load_wine(return_X_y=True)
        task = "multiclass"
    elif name == "diabetes":
        X, y = skd.load_diabetes(return_X_y=True)
        task = "regression"
    elif name == "digits_subset":
        X, y = skd.load_digits(return_X_y=True)
        m = y < 5
        X, y = X[m], y[m]
        keep = X.std(axis=0) > 0
        X = X[:, keep]
        task = "multiclass"
    elif name == "california":
        X, y = skd.fetch_california_housing(return_X_y=True)
        idx = np.random.RandomState(0).choice(len(y), 3000, replace=False)
        X, y = X[idx], y[idx]
        task = "regression"
    else:
        raise KeyError(name)
    X = np.asarray(X, dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), np.asarray(y), task


DATASETS = (
    "clf_lowsnr", "clf_highsnr", "clf_redundant", "clf_manynoise", "clf_p_gg_n", "clf_multiclass",
    "reg_lowsnr", "reg_manynoise", "reg_friedman", "breast_cancer", "wine", "diabetes", "digits_subset",
)
# 'california' (fetch_california_housing) needs a download and is unavailable offline (403 through the sandbox proxy); pass --datasets to include it.


def make_estimator(kind: str, task: str, seed: int):
    if kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        kw = dict(iterations=60, depth=4, learning_rate=0.15, verbose=0, random_seed=seed, thread_count=1, allow_writing_files=False)
        return CatBoostRegressor(**kw) if task == "regression" else CatBoostClassifier(**kw)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        kw = dict(n_estimators=60, learning_rate=0.1, num_leaves=15, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=1)
        return LGBMRegressor(**kw) if task == "regression" else LGBMClassifier(**kw)
    from sklearn.linear_model import LogisticRegression, Ridge

    return Ridge(alpha=1.0) if task == "regression" else LogisticRegression(C=0.3, max_iter=500)


def oos_score(est, Xte, yte, task: str) -> float:
    from sklearn.metrics import r2_score, roc_auc_score

    if task == "regression":
        return float(r2_score(yte, est.predict(Xte)))
    proba = est.predict_proba(Xte)
    if task == "binary":
        return float(roc_auc_score(yte, proba[:, 1]))
    return float(roc_auc_score(yte, proba, multi_class="ovr"))


def _replay(sel, rule: str):
    """Re-pick the subset size under ``rule`` on the already-fitted RFECV's curve; return the boolean support."""
    from mlframe.feature_selection.wrappers.rfecv import _stability_select as ss

    orig = ss.band_half_width
    real_rule = rule
    if rule == "one_se_max_2se":
        real_rule = "one_se_max"
        ss.band_half_width = lambda std, k, band="se": 2.0 * orig(std, k, band)
    try:
        sel.n_features_selection_rule = real_rule
        cv = sel.cv_results_
        sel.select_optimal_nfeatures_(
            checked_nfeatures=cv["nfeatures"], cv_mean_perf=cv["cv_mean_perf"], cv_std_perf=cv["cv_std_perf"],
            feature_cost=sel.feature_cost, smooth_perf=sel.smooth_perf,
        )
    finally:
        ss.band_half_width = orig
    return np.asarray(sel.support_, dtype=bool)


def run_one(dataset: str, seed: int, est_kind: str, max_refits: int = 12) -> list:
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.wrappers import RFECV

    X, y, task = load_dataset(dataset)
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=strat)
    t0 = time.process_time()
    sel = RFECV(estimator=make_estimator(est_kind, task, seed), n_features_selection_rule=BASELINE, cv=3, random_state=seed, max_refits=max_refits, verbose=0)
    sel.fit(Xtr, ytr)
    fit_s = time.process_time() - t0
    rows = []
    for rule in RULES:
        t1 = time.process_time()
        mask = _replay(sel, rule)
        kept = [c for c, m in zip(X.columns, mask) if m] or list(X.columns)
        est = make_estimator(est_kind, task, seed)
        est.fit(Xtr[kept], ytr)
        rows.append({
            "dataset": dataset, "seed": seed, "est": est_kind, "rule": rule, "oos": oos_score(est, Xte[kept], yte, task),
            "n_kept": len(kept), "kept": kept, "p": X.shape[1], "fit_s": fit_s, "select_s": time.process_time() - t1,
            "curve_nf": list(map(int, sel.cv_results_["nfeatures"])),
        })
    return rows


def _job(args):
    try:
        return run_one(*args)
    except Exception as exc:  # keep the sweep resilient; failures are reported, never silently dropped
        return [{"dataset": args[0], "seed": args[1], "est": args[2], "error": f"{type(exc).__name__}: {exc}"}]


def summarize(rows: list) -> str:
    df = pd.DataFrame([r for r in rows if "error" not in r])
    errs = [r for r in rows if "error" in r]
    out = []
    key = ["dataset", "seed", "est"]
    piv_oos = df.pivot_table(index=key, columns="rule", values="oos")
    piv_n = df.pivot_table(index=key, columns="rule", values="n_kept")
    out.append(f"units (dataset x seed x est): {len(piv_oos)}   failures: {len(errs)}")
    out.append("\nAGGREGATE per rule (mean over units): oos, n_kept")
    out.append(pd.DataFrame({"oos": piv_oos.mean(), "n_kept": piv_n.mean()}).round(4).to_string())

    def paired(base_rule, rule, scope_df_idx=None):
        d = (piv_oos[rule] - piv_oos[base_rule]).dropna()
        dn = (piv_n[rule] - piv_n[base_rule]).dropna()
        if scope_df_idx is not None:
            d, dn = d[scope_df_idx(d.index)], dn[scope_df_idx(dn.index)]
        # tie = |dOOS| < 1e-4 (well under seed noise); per-metric scale differs (AUC vs R2) but all are O(1)
        tol = 1e-4
        return len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan, int((d > tol).sum()), int((d.abs() <= tol).sum()), int((d < -tol).sum()), dn.mean()

    out.append(f"\nPAIRED vs {BASELINE}: dOOS mean +- se | win/tie/loss | d n_kept (negative = fewer features)")
    for rule in RULES:
        if rule == BASELINE:
            continue
        n, m, se, w, t, l, dn = paired(BASELINE, rule)
        out.append(f"  {rule:14s} n={n} dOOS={m:+.5f} +- {se:.5f}  W/T/L={w}/{t}/{l}  dN={dn:+.2f}")
    out.append("\nPAIRED per estimator: one_se_max(SE) vs foldstd")
    for est in ESTIMATORS:
        n, m, se, w, t, l, dn = paired(BASELINE, "one_se_max", lambda idx, est=est: idx.get_level_values("est") == est)
        out.append(f"  {est:9s} n={n} dOOS={m:+.5f} +- {se:.5f}  W/T/L={w}/{t}/{l}  dN={dn:+.2f}")
    out.append("\nPAIRED per dataset: one_se_max(SE) vs foldstd (dOOS mean, dN mean)")
    for ds in sorted(df.dataset.unique()):
        n, m, se, w, t, l, dn = paired(BASELINE, "one_se_max", lambda idx, ds=ds: idx.get_level_values("dataset") == ds)
        out.append(f"  {ds:14s} n={n} dOOS={m:+.5f} +- {se:.5f}  W/T/L={w}/{t}/{l}  dN={dn:+.2f}")
    # stability: mean pairwise Jaccard across split seeds per (dataset, est, rule)
    out.append("\nSTABILITY: mean pairwise Jaccard of kept sets across seeds (mean over dataset x est)")
    stab = {}
    for rule in RULES:
        vals = []
        for (_, _), g in df[df.rule == rule].groupby(["dataset", "est"]):
            sets = [set(k) for k in g.kept]
            js = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
            if js:
                vals.append(np.mean(js))
        stab[rule] = np.mean(vals)
    out.append(pd.Series(stab).round(4).to_string())
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="3 datasets, 2 seeds, 1 estimator (smoke)")
    ap.add_argument("--datasets", type=str, default=None, help="comma-separated subset of datasets")
    ap.add_argument("--summarize", type=str, default=None)
    a = ap.parse_args()
    if a.summarize:
        rows = [r for f in a.summarize.split(",") for r in json.loads(Path(f).read_text())]
        print(summarize(rows))
        return
    from joblib import Parallel, delayed

    datasets, seeds, ests = (DATASETS[:1] + DATASETS[8:9], range(2), ("lightgbm",)) if a.fast else (DATASETS, range(a.seeds), ESTIMATORS)
    if a.datasets:
        datasets = tuple(a.datasets.split(","))
    jobs = list(itertools.product(datasets, seeds, ests))
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_job)(j) for j in jobs)
    rows = [r for rs in res for r in rs]
    out = RESULTS_DIR / f"rfecv_one_se_rule_{time.strftime('%Y%m%d_%H%M%S')}{'_fast' if a.fast else ''}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, sort_keys=True))
    print("wrote", out, f"wall={time.time() - t0:.0f}s")
    print(summarize(rows))


if __name__ == "__main__":
    main()

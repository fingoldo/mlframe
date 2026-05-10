"""PR-4 methods benchmark on synthetic data.

Compares baseline RFECV vs:
- RFECV + stability_selection
- RFECV + multi-estimator voting
- RFECV + stability_selection + multi-estimator combined

Metrics: recall of true informative features, n_features_, wall-clock time,
selection stability (Jaccard across 3 random_states).

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_pr4_methods
"""
from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV


@dataclass
class BenchResult:
    method: str
    seed: int
    n_features_total: int
    n_features_selected: int
    informative_recall: float
    fit_seconds: float
    selected_idx: list


def _make_problem(n=600, p=40, n_inform=8, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=n_inform,
        n_redundant=0, random_state=seed, shuffle=False, class_sep=2.0,
    )
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), y, set(range(n_inform))


def _selected_idx(rfecv, X) -> list:
    cols = list(X.columns)
    feat_in = list(rfecv.feature_names_in_)
    if len(rfecv.support_) == 0:
        return []
    if isinstance(rfecv.support_[0], (bool, np.bool_)):
        sel_names = {feat_in[i] for i, s in enumerate(rfecv.support_) if s}
    else:
        sel_names = {feat_in[i] for i in rfecv.support_}
    return [cols.index(n) for n in sel_names if n in cols]


def _run_one(method_name: str, factory, X, y, informative_idx, seed) -> BenchResult:
    rfecv = factory(seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X, y)
    elapsed = time.perf_counter() - t0
    sel = _selected_idx(rfecv, X)
    recall = len(set(sel) & informative_idx) / max(1, len(informative_idx))
    return BenchResult(
        method=method_name, seed=seed,
        n_features_total=X.shape[1], n_features_selected=len(sel),
        informative_recall=recall, fit_seconds=elapsed,
        selected_idx=sorted(sel),
    )


def main():
    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    rows: list[dict] = []
    print("# PR-4 methods bench on synthetic (n=600, p=40, 8 informative, class_sep=2.0)")
    print()

    methods = {
        "baseline": lambda seed: RFECV(
            estimator=LogisticRegression(max_iter=400, random_state=seed),
            cv=3, max_refits=8, verbose=0, random_state=seed,
        ),
        "stability_selection_lr": lambda seed: RFECV(
            estimator=LogisticRegression(max_iter=400, random_state=seed),
            stability_selection=True, stability_n_bootstrap=30,
            stability_threshold=0.5,
            verbose=0, random_state=seed,
        ),
        "multi_estimator_lr_rf": lambda seed: RFECV(
            estimators=[
                LogisticRegression(max_iter=400, random_state=seed),
                RandomForestClassifier(n_estimators=20, random_state=seed, n_jobs=1),
            ],
            cv=3, max_refits=8, verbose=0, random_state=seed,
        ),
        "stability_plus_multi": lambda seed: RFECV(
            estimators=[
                LogisticRegression(max_iter=400, random_state=seed),
                RandomForestClassifier(n_estimators=20, random_state=seed, n_jobs=1),
            ],
            stability_selection=True, stability_n_bootstrap=30,
            stability_threshold=0.5,
            verbose=0, random_state=seed,
        ),
    }

    X0, y0, informative_idx = _make_problem(seed=0)
    print(f"informative={sorted(informative_idx)}")
    print()

    for method_name, factory in methods.items():
        for seed in seeds:
            X, y, _ = _make_problem(seed=seed)
            res = _run_one(method_name, factory, X, y, informative_idx, seed)
            rows.append(asdict(res))
            print(
                f"  {method_name:25s} seed={seed} "
                f"n_sel={res.n_features_selected:3d} recall={res.informative_recall:.2f} "
                f"t={res.fit_seconds:5.1f}s"
            )

    # Aggregate
    print()
    print("=== SUMMARY (mean across seeds) ===")
    df = pd.DataFrame(rows)
    summary = df.groupby("method").agg(
        n_sel_mean=("n_features_selected", "mean"),
        recall_mean=("informative_recall", "mean"),
        recall_std=("informative_recall", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).round(3)

    # Selection stability: mean pairwise Jaccard of selected_idx across seeds
    def _jac(a, b):
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    stab = {}
    for method_name in methods.keys():
        sub = df[df["method"] == method_name]
        sels = sub["selected_idx"].tolist()
        if len(sels) < 2:
            stab[method_name] = float("nan")
            continue
        pairs = [_jac(sels[i], sels[j]) for i in range(len(sels)) for j in range(i + 1, len(sels))]
        stab[method_name] = float(np.mean(pairs))
    summary["stability_jaccard"] = pd.Series(stab)

    print(summary.to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pr4_methods_{timestamp}.json"
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": rows,
        "summary": summary.reset_index().to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[result] {out_path}")


if __name__ == "__main__":
    main()

"""Standalone head-to-head benchmark: mlframe RFECV vs sklearn.feature_selection.RFECV.

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_rfecv_vs_sklearn

What it does:
    - Builds 5-7 synthetic problems (varied informative/redundant/noise mix,
      classification + regression).
    - Runs both selectors with 3 estimators (LogisticRegression, RandomForest,
      CatBoost - if installed).
    - For each (problem, estimator), records:
        * CV score on the selected subset (under a held-out CV)
        * Recall of true informative features (synthetic only)
        * Wall-clock time per fit
        * Selection stability (Jaccard) across 3 random seeds
    - Aggregates into JSON written to ``_benchmarks/_results/h2h_<timestamp>.json``
      and PNG comparison plots per metric.

This script is reproducible (fixed random seeds) but NOT part of CI - the CI
guard-rail at ``mlframe/tests/feature_selection/test_wrappers_h2h_sklearn.py``
catches regressions; this script produces the human-readable report.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV as SkRFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from mlframe.feature_selection.wrappers import RFECV as OurRFECV

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ----------------------------------------------------------------------------
# Problem fixtures (deterministic for fixed random_state)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    name: str
    task: str           # "classification" | "regression"
    n: int
    p: int
    informative: int
    redundant: int = 0
    noise: float = 0.0
    class_sep: float = 1.5

    def make(self, random_state: int) -> tuple[pd.DataFrame, np.ndarray]:
        if self.task == "classification":
            X, y = make_classification(
                n_samples=self.n, n_features=self.p,
                n_informative=self.informative, n_redundant=self.redundant,
                n_repeated=0, n_classes=2, n_clusters_per_class=1,
                class_sep=self.class_sep, random_state=random_state,
            )
        else:
            X, y = make_regression(
                n_samples=self.n, n_features=self.p,
                n_informative=self.informative,
                noise=self.noise, random_state=random_state,
            )
        cols = [f"f{i}" for i in range(self.p)]
        return pd.DataFrame(X, columns=cols), y

    @property
    def informative_idx(self) -> set[int]:
        # sklearn.make_classification puts informative cols first
        return set(range(self.informative))


PROBLEMS: list[Problem] = [
    Problem("clf_easy_small",       "classification", n=400, p=20,  informative=4,  redundant=0,  class_sep=2.0),
    Problem("clf_redundant",        "classification", n=400, p=25,  informative=5,  redundant=10, class_sep=1.5),
    Problem("clf_noisy_wide",       "classification", n=300, p=60,  informative=5,  redundant=0,  class_sep=1.5),
    Problem("clf_hard",             "classification", n=600, p=30,  informative=8,  redundant=4,  class_sep=1.0),
    Problem("reg_easy",             "regression",     n=400, p=20,  informative=5,  noise=0.5),
    Problem("reg_correlated_noisy", "regression",     n=300, p=25,  informative=4,  noise=2.0),
]


# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------
@dataclass
class RunResult:
    problem: str
    estimator: str
    selector: str          # "ours" | "sklearn"
    seed: int
    n_features_total: int
    n_features_selected: int
    selected_idx: list[int]
    informative_recall: float
    cv_score_on_subset: float
    fit_seconds: float


# ----------------------------------------------------------------------------
# Estimator factories
# ----------------------------------------------------------------------------
def estimator_factories(problem: Problem) -> dict[str, Callable[[], object]]:
    facs: dict[str, Callable[[], object]] = {}
    if problem.task == "classification":
        facs["logreg"] = lambda: LogisticRegression(max_iter=500, random_state=0)
        facs["rf"]     = lambda: RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=1)
        if HAS_CATBOOST:
            facs["cb"] = lambda: CatBoostClassifier(iterations=80, depth=4, verbose=0, random_seed=0, allow_writing_files=False)
    else:
        facs["ridge"] = lambda: Ridge(random_state=0)
        facs["rf"]    = lambda: RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1)
        if HAS_CATBOOST:
            facs["cb"] = lambda: CatBoostRegressor(iterations=80, depth=4, verbose=0, random_seed=0, allow_writing_files=False)
    return facs


def cv_for_problem(problem: Problem, seed: int):
    if problem.task == "classification":
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return KFold(n_splits=3, shuffle=True, random_state=seed)


# ----------------------------------------------------------------------------
# Single (problem, estimator, selector, seed) run
# ----------------------------------------------------------------------------
def _selected_idx_from_ours(rfecv: OurRFECV, X: pd.DataFrame) -> list[int]:
    feat_in = list(rfecv.feature_names_in_)
    if len(rfecv.support_) == 0:
        return []
    if isinstance(rfecv.support_[0], (bool, np.bool_)):
        sel_names = [feat_in[i] for i, s in enumerate(rfecv.support_) if s]
    else:
        sel_names = [feat_in[i] for i in rfecv.support_]
    cols = list(X.columns)
    return [cols.index(n) for n in sel_names if n in cols]


def run_one(problem: Problem, estimator_name: str, estimator_factory: Callable, seed: int) -> dict[str, RunResult]:
    X, y = problem.make(seed)
    informative = problem.informative_idx
    cv_scoring = cv_for_problem(problem, seed=10_000 + seed)  # different folds for evaluation

    # ----- ours
    ours_t0 = time.perf_counter()
    ours = OurRFECV(
        estimator=estimator_factory(),
        cv=cv_for_problem(problem, seed=seed),
        max_refits=10,
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours.fit(X, y)
    ours_t = time.perf_counter() - ours_t0
    ours_idx = _selected_idx_from_ours(ours, X)
    cols_ours = [X.columns[i] for i in ours_idx]
    if cols_ours:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours_score = float(np.mean(cross_val_score(estimator_factory(), X[cols_ours], y, cv=cv_scoring)))
    else:
        ours_score = float("nan")
    ours_recall = len(set(ours_idx) & informative) / max(1, len(informative))

    # ----- sklearn
    sk_t0 = time.perf_counter()
    try:
        sk = SkRFECV(
            estimator=estimator_factory(),
            cv=cv_for_problem(problem, seed=seed),
            step=1,
            min_features_to_select=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk.fit(X, y)
        sk_t = time.perf_counter() - sk_t0
        sk_idx = [i for i, s in enumerate(sk.support_) if s]
        cols_sk = [X.columns[i] for i in sk_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk_score = float(np.mean(cross_val_score(estimator_factory(), X[cols_sk], y, cv=cv_scoring))) if cols_sk else float("nan")
        sk_recall = len(set(sk_idx) & informative) / max(1, len(informative))
    except Exception as exc:
        # Some estimators (e.g. CatBoost) don't expose coef_/feature_importances_
        # in the way sklearn's RFECV expects. Record as NaN.
        sk_t = time.perf_counter() - sk_t0
        sk_idx, sk_score, sk_recall = [], float("nan"), float("nan")
        print(f"  [sklearn h2h skipped: {type(exc).__name__}: {exc}]")

    return {
        "ours": RunResult(
            problem=problem.name, estimator=estimator_name, selector="ours",
            seed=seed, n_features_total=problem.p, n_features_selected=len(ours_idx),
            selected_idx=ours_idx, informative_recall=ours_recall,
            cv_score_on_subset=ours_score, fit_seconds=ours_t,
        ),
        "sklearn": RunResult(
            problem=problem.name, estimator=estimator_name, selector="sklearn",
            seed=seed, n_features_total=problem.p, n_features_selected=len(sk_idx),
            selected_idx=sk_idx, informative_recall=sk_recall,
            cv_score_on_subset=sk_score, fit_seconds=sk_t,
        ),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(seeds: tuple[int, ...] = (0, 1, 2), out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"# h2h benchmark: mlframe RFECV vs sklearn.RFECV")
    print(f"# {len(PROBLEMS)} problems x ~3 estimators x {len(seeds)} seeds")
    print(f"# CatBoost available: {HAS_CATBOOST}")
    print()

    for problem in PROBLEMS:
        print(f"== problem {problem.name} ({problem.task} n={problem.n} p={problem.p} informative={problem.informative}) ==")
        for est_name, est_factory in estimator_factories(problem).items():
            ours_results, sk_results = [], []
            for seed in seeds:
                pair = run_one(problem, est_name, est_factory, seed)
                ours_results.append(pair["ours"])
                sk_results.append(pair["sklearn"])
                print(
                    f"  est={est_name:6s} seed={seed} ours[n={pair['ours'].n_features_selected:3d} "
                    f"score={pair['ours'].cv_score_on_subset:.4f} t={pair['ours'].fit_seconds:5.1f}s] "
                    f"sk[n={pair['sklearn'].n_features_selected:3d} "
                    f"score={pair['sklearn'].cv_score_on_subset:.4f} t={pair['sklearn'].fit_seconds:5.1f}s]"
                )
            # Jaccard stability across the 3 seeds for each selector
            for selector_name, runs in [("ours", ours_results), ("sklearn", sk_results)]:
                supports = [set(r.selected_idx) for r in runs]
                pairwise_jaccard = [
                    jaccard(supports[i], supports[j])
                    for i in range(len(supports)) for j in range(i + 1, len(supports))
                ]
                stability = float(np.mean(pairwise_jaccard)) if pairwise_jaccard else 1.0
                for r in runs:
                    rec = asdict(r)
                    rec["stability_jaccard"] = stability
                    rows.append(rec)
            print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"h2h_{timestamp}.json"
    summary = _aggregate(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "problems": [asdict(p) for p in PROBLEMS],
        "seeds": list(seeds),
        "has_catboost": HAS_CATBOOST,
        "rows": rows,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[result] {out_path}")
    print()
    print("=== SUMMARY (mean across seeds, per problem x estimator) ===")
    print(_render_summary_table(summary))

    try:
        plot_path = out_dir / f"h2h_{timestamp}.png"
        _plot_summary(summary, plot_path)
        print(f"[plot]   {plot_path}")
    except Exception as exc:
        print(f"[plot]   skipped: {type(exc).__name__}: {exc}")

    return out_path


def _aggregate(rows: list[dict]) -> dict:
    """Collapse per-seed rows into per-(problem, estimator, selector) means."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    grp = df.groupby(["problem", "estimator", "selector"]).agg(
        n_features_selected=("n_features_selected", "mean"),
        cv_score=("cv_score_on_subset", "mean"),
        recall=("informative_recall", "mean"),
        fit_seconds=("fit_seconds", "mean"),
        stability=("stability_jaccard", "first"),
    ).reset_index()
    return grp.to_dict(orient="records")


def _render_summary_table(summary: list[dict]) -> str:
    if not summary:
        return "  (no rows)"
    df = pd.DataFrame(summary)
    pivot = df.pivot_table(
        index=["problem", "estimator"], columns="selector",
        values=["cv_score", "recall", "fit_seconds", "n_features_selected", "stability"],
    )
    return pivot.round(3).to_string()


def _plot_summary(summary: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summary)
    if df.empty:
        return

    metrics = [
        ("cv_score", "CV score on selected subset (higher better)"),
        ("recall", "Recall of true informative features (higher better)"),
        ("n_features_selected", "Selected #features (lower better, when same recall)"),
        ("stability", "Selection stability across seeds, Jaccard (higher better)"),
        ("fit_seconds", "Wall-clock fit time, seconds (lower better)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3 * len(metrics)))
    for ax, (m, title) in zip(axes, metrics):
        pivot = df.pivot_table(index=["problem", "estimator"], columns="selector", values=m)
        pivot.plot.bar(ax=ax, alpha=0.85, width=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

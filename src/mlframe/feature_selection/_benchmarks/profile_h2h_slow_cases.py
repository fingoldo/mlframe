"""cProfile pass on the h2h-bench cases where ours was slowest:

- reg_easy ridge: ours 4.4s vs sklearn 0.22s (20x slower)
- clf_easy logreg: ours 5.1s vs sklearn 0.52s (10x slower)

These are tiny problems (n=400, p=20) where the user's estimator fit is
sub-millisecond. Sklearn's RFECV does step-by-step elimination (O(p) cheap
fits per CV fold = ~60 fits total). Our RFECV uses MBH which fits an
internal CatBoost surrogate to predict scores - that overhead alone
dominates when the outer estimator is trivially cheap.

The diagnostic question: what fraction of our wall-clock is in
(a) the user's estimator fit (irreducible)
(b) MBH's internal CatBoost surrogate
(c) CV split + voting + bookkeeping
(d) other overhead

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.profile_h2h_slow_cases
"""
from __future__ import annotations

import cProfile
import io
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from mlframe.feature_selection.wrappers import RFECV

RESULTS = Path(__file__).parent / "_results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _profile(label: str, fn) -> tuple[pstats.Stats, float]:
    profiler = cProfile.Profile()
    t0 = perf_counter()
    profiler.enable()
    fn()
    profiler.disable()
    t1 = perf_counter()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stream = io.StringIO()
    stats.stream = stream
    stats.print_stats(40)
    out_path = RESULTS / f"profile_h2h_slow_{label}.txt"
    out_path.write_text(stream.getvalue(), encoding="utf-8")
    print(f"\n=== {label}  wall={t1 - t0:.3f}s  -> {out_path}")
    print(stream.getvalue())
    return stats, t1 - t0


def _reg_easy_ridge():
    """reg_easy / Ridge: n=400, p=20, informative=5, noise=0.5."""
    X, y = make_regression(
        n_samples=400, n_features=20, n_informative=5,
        noise=0.5, random_state=0,
    )
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    rfecv = RFECV(
        estimator=Ridge(random_state=0),
        cv=cv, max_refits=20, verbose=0, random_state=0,
    )
    rfecv.fit(Xdf, y)


def _clf_easy_logreg():
    """clf_easy_small / LogReg: n=400, p=20, informative=4, class_sep=2.0."""
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=4, n_redundant=0,
        n_repeated=0, n_classes=2, n_clusters_per_class=1,
        class_sep=2.0, random_state=0,
    )
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=500, random_state=0),
        cv=cv, max_refits=20, verbose=0, random_state=0,
    )
    rfecv.fit(Xdf, y)


def main() -> int:
    print("# cProfile pass on h2h-bench slow cases (linear/Ridge on tiny problems)")
    _profile("reg_easy_ridge", _reg_easy_ridge)
    _profile("clf_easy_logreg", _clf_easy_logreg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

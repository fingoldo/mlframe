"""cProfile pass for PR-12 / PR-13 newly added RFECV features.

Per ``feedback_profile_new_features``: every new non-trivial feature must
be profiled with cProfile; numba.njit / parallel / cuda / cupy applied
where it materially helps.

This script profiles:

- Conditional Permutation Importance (PR-12 ``importance_getter='conditional_permutation'``)
- Truncated SFFS final-pass swap (PR-13 ``swap_top_k=K``)

Trivial features skipped with justification:
- __sklearn_tags__ - one-time call returning a dataclass; not on any hot path.
- cv auto-detect - one isinstance + monotonic-check at fit entry; ~us.
- cv_results_df_ property - one ``pd.DataFrame(dict)`` call on access; ~ms.
- resume-from-checkpoint pickle (PR-12) - one pickle.dump / os.replace per
  outer iter; bounded by IO, not CPU; not actionable.

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.profile_new_features
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from mlframe.feature_selection.wrappers import RFECV, get_feature_importances


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
    stats.print_stats(30)
    out_path = RESULTS / f"profile_pr1213_{label}.txt"
    out_path.write_text(stream.getvalue(), encoding="utf-8")
    print(f"\n=== {label}  wall={t1 - t0:.3f}s  -> {out_path}")
    print(stream.getvalue())
    return stats, t1 - t0


# ----------------------------------------------------------------------------
# Conditional Permutation Importance (PR-12)
# ----------------------------------------------------------------------------
def _cpi_medium():
    rng = np.random.default_rng(0)
    X, y = make_classification(
        n_samples=600, n_features=40, n_informative=10,
        n_redundant=5, random_state=0, shuffle=False, class_sep=2.0,
    )
    cols = [f"f{i}" for i in range(40)]
    Xdf = pd.DataFrame(X, columns=cols)
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xdf, y)
    _ = get_feature_importances(
        model=model, current_features=cols,
        importance_getter="conditional_permutation",
        data=Xdf, target=y,
    )


def _cpi_regression():
    X, y = make_regression(
        n_samples=400, n_features=30, n_informative=8,
        random_state=0, shuffle=False, noise=1.0,
    )
    cols = [f"f{i}" for i in range(30)]
    Xdf = pd.DataFrame(X, columns=cols)
    model = Ridge().fit(Xdf, y)
    _ = get_feature_importances(
        model=model, current_features=cols,
        importance_getter="conditional_permutation",
        data=Xdf, target=y,
    )


# ----------------------------------------------------------------------------
# Truncated SFFS swap (PR-13)
# ----------------------------------------------------------------------------
def _sffs_swap_medium():
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=8,
        n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
    )
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=300, random_state=0),
        cv=3, max_refits=8, verbose=0, random_state=0,
        swap_top_k=5,
    )
    rfecv.fit(Xdf, y)


def main() -> int:
    print("# cProfile pass on PR-12/PR-13 new features")
    _profile("cpi_classification", _cpi_medium)
    _profile("cpi_regression", _cpi_regression)
    _profile("sffs_swap", _sffs_swap_medium)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bench: adaptive coarse-to-fine elimination pace (``dichotomic_step='auto'``) vs legacy fixed midpoint (``'midpoint'``).

The RFECV ``ExhaustiveDichotomic`` search probes feature-count N's via a bisection schedule. ``step='auto'`` strides by
``max(1, floor(frac * n_remaining))`` while the unevaluated N-pool is large and the CV curve is flat, then refines to step=1
near the knee. This is a SPEED lever: the FINAL selected set + held-out score must stay near-equivalent to the legacy
midpoint search.

Metrics per (scenario, seed): wall time both modes, refit-iteration count (len(cv_results_['mean_test_score'])), Jaccard of
the selected support, and the held-out (test-split) score delta. Run isolated on synthetic frames with the host env vars:

    CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
        python src/mlframe/feature_selection/wrappers/_benchmarks/bench_dichotomic_adaptive_step.py

Decision rule (CLAUDE.md variant-default policy): flip ``dichotomic_step='auto'`` default only on a MAJORITY of
scenarios x seeds with materially-lower wall AND near-equivalent selection (Jaccard>=0.9 + held-out delta within noise).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers.rfecv import RFECV
from mlframe.feature_selection.wrappers._enums import OptimumSearch

SCENARIOS = [
    ("narrow_p30", dict(n_features=30, n_informative=8, n_redundant=6)),
    ("mid_p80", dict(n_features=80, n_informative=12, n_redundant=10)),
    ("wide_p200", dict(n_features=200, n_informative=15, n_redundant=20)),
    ("wide_p400", dict(n_features=400, n_informative=20, n_redundant=30)),
    ("wide_p600", dict(n_features=600, n_informative=20, n_redundant=40)),
]
SEEDS = [0, 1, 2]
N_SAMPLES = 1500


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _run(X_tr, y_tr, step, seed, p):
    est = LogisticRegression(max_iter=300, random_state=seed)
    sel = RFECV(
        estimator=est,
        top_predictors_search_method=OptimumSearch.ExhaustiveDichotomic,
        dichotomic_step=step,
        dichotomic_epsilon=0.0,  # isolate the step schedule from the random-kick
        cv=3,
        max_noimproving_iters=10,
        random_state=seed,
        verbose=0,
        leave_progressbars=False,
    )
    t0 = time.perf_counter()
    sel.fit(X_tr, y_tr)
    wall = time.perf_counter() - t0
    n_iters = len(sel.cv_results_.get("nfeatures", [])) if hasattr(sel, "cv_results_") else 0
    return sel, wall, n_iters


def main():
    rows = []
    for sc_name, sc_kw in SCENARIOS:
        for seed in SEEDS:
            X, y = make_classification(
                n_samples=N_SAMPLES, n_classes=2, random_state=seed,
                n_clusters_per_class=2, **sc_kw,
            )
            import pandas as pd
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
            p = sc_kw["n_features"]

            sel_a, wall_a, it_a = _run(X_tr, y_tr, "auto", seed, p)
            sel_m, wall_m, it_m = _run(X_tr, y_tr, "midpoint", seed, p)

            sup_a = list(np.asarray(X.columns)[sel_a.support_])
            sup_m = list(np.asarray(X.columns)[sel_m.support_])
            jac = _jaccard(sup_a, sup_m)

            # held-out delta: fit a fresh model on each support, score on the test split.
            def _hold(sup):
                if not sup:
                    return float("nan")
                m = LogisticRegression(max_iter=300, random_state=seed)
                m.fit(X_tr[sup], y_tr)
                return accuracy_score(y_te, m.predict(X_te[sup]))

            ho_a, ho_m = _hold(sup_a), _hold(sup_m)
            rows.append(dict(
                scenario=sc_name, seed=seed, p=p,
                wall_auto=wall_a, wall_mid=wall_m,
                speedup=wall_m / wall_a if wall_a else float("nan"),
                iters_auto=it_a, iters_mid=it_m,
                n_auto=len(sup_a), n_mid=len(sup_m),
                jaccard=jac, ho_auto=ho_a, ho_mid=ho_m, ho_delta=ho_a - ho_m,
            ))
            print(f"{sc_name:12s} seed={seed} p={p:4d} "
                  f"wall auto={wall_a:6.2f}s mid={wall_m:6.2f}s spd={rows[-1]['speedup']:4.2f}x "
                  f"iters {it_a:3d}/{it_m:3d} N {len(sup_a):3d}/{len(sup_m):3d} "
                  f"J={jac:.2f} ho_d={ho_a - ho_m:+.4f}")

    # Aggregate verdict.
    import statistics as st
    wins = [r for r in rows if r["speedup"] >= 1.10 and r["jaccard"] >= 0.9 and abs(r["ho_delta"]) <= 0.02]
    print("\n=== SUMMARY ===")
    print(f"scenarios x seeds: {len(rows)}")
    print(f"median speedup: {st.median(r['speedup'] for r in rows):.2f}x")
    print(f"median jaccard: {st.median(r['jaccard'] for r in rows):.2f}")
    print(f"median |ho_delta|: {st.median(abs(r['ho_delta']) for r in rows):.4f}")
    print(f"median iters auto/mid: {st.median(r['iters_auto'] for r in rows):.0f}/{st.median(r['iters_mid'] for r in rows):.0f}")
    print(f"wins (spd>=1.10 & J>=0.9 & |ho_delta|<=0.02): {len(wins)}/{len(rows)}")
    return rows


if __name__ == "__main__":
    main()

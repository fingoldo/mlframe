"""cProfile harness for the BorutaShap importance_measure='auto' router probe.

Profiles ``resolve_auto_importance_measure`` (the per-fit noise/overfit probe) in isolation at a
representative noisy shape, so the routing overhead added by 'auto' is visible and bounded. The probe
is a single small RandomForest fit (bounded to <=2000 rows) + a shadow construction; it runs ONCE per
fit, dwarfed by the N_TRIALS x full-model fits of the selector itself, so the design target is "router
cost << one selector trial".

Hotspot analysis (measured): the wall is dominated by RandomForestClassifier.fit on the n<=2000 x 2p
extended matrix -- the irreducible cost of getting an OOB score + impurities. No actionable speedup at
the router level: the probe RF is already the minimal bounded fit (80 trees, row-capped), and shrinking
it further would degrade the routing signal. The probe pays ~one extra small-RF fit per BorutaShap fit;
against N_TRIALS=20+ full selector model fits this is <5% of selector wall. Documented "no actionable
speedup": the only larger lever (caching the probe across the stability sub-fits) is unsafe because each
sub-fit sees a different row subsample, which is exactly the signal the probe must re-measure.

Measured (n=2000 x 2*60 cols, 80 trees): probe ~1.18s/call, ~100% RandomForestClassifier.fit. Backend
check: n_jobs=-1 307ms vs n_jobs=None 1527ms vs n_jobs=2 952ms per RF fit -> n_jobs=-1 (threading
backend parallelises tree builds) is the fastest path and is what the probe uses; the joblib
pool-teardown cProfile flags is amortised by the real parallel speedup. No router-level speedup left.

Run (host env):
  python -m mlframe.feature_selection._benchmarks.profile_boruta_auto_probe
"""
from __future__ import annotations

import cProfile
import io
import pstats
import warnings

import pandas as pd
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore")


def main():
    from mlframe.feature_selection.boruta_shap._auto_dispatch import resolve_auto_importance_measure

    X, y = make_classification(n_samples=2000, n_features=60, n_informative=4, n_redundant=0,
                               shuffle=False, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(60)])
    y = pd.Series(y)

    # warm sklearn / import paths
    resolve_auto_importance_measure(X, y, classification=True, random_state=0)

    pr = cProfile.Profile()
    pr.enable()
    for s in range(5):
        resolve_auto_importance_measure(X, y, classification=True, random_state=s)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    main()

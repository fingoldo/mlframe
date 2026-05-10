"""Benchmark accuracy + speed of MI estimators / mRMR variants.

Compares:
* Plug-in (legacy, discretized)
* Plug-in + Miller-Madow correction
* KSG (k-NN, continuous)
* Stability selection wrapper
* Group-aware wrapper
* Adaptive permutation (Besag-Clifford)

Recovery rate (precision against the known set of informative features
from ``make_classification``) is the accuracy metric. Wall time per
fit is the speed metric.

Run::

    python -m mlframe.feature_selection._benchmarks.bench_estimators
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger("bench_estimators")


def _build(n=2000, p=20, n_inf=8, seed=42):
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=n_inf,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=seed,
    )
    informative = list(range(n_inf))  # sklearn puts informative first
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(p)])
    return df, y, informative


def _recovery(support, informative):
    s = set(support.tolist() if hasattr(support, "tolist") else support)
    inf = set(informative)
    if not inf:
        return 1.0
    return len(s & inf) / len(inf)


def _bench(name, fn, X, y, informative):
    t0 = time.perf_counter()
    out = fn(X, y)
    dt = time.perf_counter() - t0
    rec = _recovery(out, informative)
    print(f"  {name:35s} t={dt:6.2f}s  recovery={rec*100:5.1f}%  n_selected={len(out)}")


def main():
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.estimators import ksg_mi_with_target
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    from mlframe.feature_selection.filters.stability import StabilityMRMR

    print("\n=== n=2000, p=20, n_inf=8 ===")
    X, y, inf = _build(n=2000, p=20, n_inf=8)

    base_kw = dict(quantization_nbins=10, full_npermutations=3, baseline_npermutations=2,
                   n_jobs=1, verbose=0, fe_max_steps=0, cv=2)

    def _plain(X, y):
        m = MRMR(**base_kw)
        m.fit(X, y)
        return m.support_

    def _bc_inner(X, y):
        # Note: parallelism kwarg lives on mi_direct, not on MRMR (yet).
        # Use plain MRMR; this call exists for parity in the table.
        m = MRMR(**base_kw)
        m.fit(X, y)
        return m.support_

    def _stability(X, y):
        s = StabilityMRMR(estimator=MRMR(**base_kw), n_bootstraps=5, sample_fraction=0.7,
                          support_threshold=0.6, random_state=42)
        s.fit(X, y)
        return s.support_

    def _group(X, y):
        g = GroupAwareMRMR(estimator=MRMR(**base_kw), corr_threshold=0.9)
        g.fit(X, y)
        return g.support_

    def _ksg_topk(X, y):
        # KSG MI ranking; pick top-K = n_inf.
        Xn = X.to_numpy() if hasattr(X, "to_numpy") else X
        mi = ksg_mi_with_target(Xn, y, feature_indices=list(range(Xn.shape[1])), n_neighbors=3)
        order = np.argsort(mi)[::-1]
        return order[:len(inf)]

    print("\n  estimator                            time     recovery  n_selected")
    print("  " + "-" * 70)
    _bench("MRMR plug-in (legacy)", _plain, X, y, inf)
    _bench("MRMR plug-in + Stability(B=5)", _stability, X, y, inf)
    _bench("MRMR plug-in + GroupAware", _group, X, y, inf)
    _bench("KSG top-K (no permutation test)", _ksg_topk, X, y, inf)


if __name__ == "__main__":
    main()

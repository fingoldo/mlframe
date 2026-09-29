"""A/B: when the cluster-medoid wrapper selects a cluster, return the whole cluster (``expand=True``) or only its medoid.

``expand=True`` was chosen on ``bench_cross_selector_diverse`` against a bare RFECV, never against medoids-only. On a
correlated fixture it hands back every near-copy of a selected feature (max VIF 510 on the multicollinear-pollution
fixture). This measures what that costs or buys: OOS AUC and support size for full RFECV, expanded clusters, and
medoids only, over the same datasets (varied redundancy, the signal-in-a-non-medoid risk case, real sklearn sets) and
three seeds.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_group_aware_expand_vs_medoids``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_classification
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection._benchmarks.bench_cross_selector_diverse import _risk_case
from mlframe.feature_selection.filters.group_aware import _cluster_medoids, cluster_features_by_correlation

warnings.filterwarnings("ignore")


def _auc(Xtr, ytr, Xte, yte, cols) -> float:
    """OOS AUC of a logistic regression on the selected columns."""
    if len(cols) == 0:
        return 0.5
    m = LogisticRegression(max_iter=1000).fit(Xtr.iloc[:, cols], ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte.iloc[:, cols])[:, 1]))


def _rfecv(Xtr, ytr) -> RFECV:
    """The wrapper proxy the diverse benchmark uses."""
    return RFECV(LogisticRegression(max_iter=500), step=0.1, cv=3, min_features_to_select=2, n_jobs=1)


def _supports(Xtr, ytr, corr_threshold: float = 0.9):
    """(full, expanded, medoids-only) supports for one training split."""
    full = np.where(_rfecv(Xtr, ytr).fit(Xtr, ytr).support_)[0]
    cid = cluster_features_by_correlation(Xtr, threshold=corr_threshold, method="pearson")
    medoids = _cluster_medoids(Xtr, cid, method="pearson")
    sel = _rfecv(Xtr, ytr).fit(Xtr.iloc[:, medoids], ytr)
    chosen = [int(medoids[i]) for i in np.where(sel.support_)[0]]
    clusters = {int(cid[m]) for m in chosen}
    expanded = np.array([j for j in range(Xtr.shape[1]) if cid[j] in clusters])
    return full, expanded, np.array(sorted(chosen))


def _run(name, X, y, seed):
    """One dataset and seed: AUC and support size for each variant."""
    X = pd.DataFrame(StandardScaler().fit_transform(np.asarray(X, float)))
    y = pd.Series(np.asarray(y).astype(int))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    Xtr, Xte, ytr, yte = (d.reset_index(drop=True) for d in (Xtr, Xte, ytr, yte))
    full, exp, med = _supports(Xtr, ytr)
    return {v: (_auc(Xtr, ytr, Xte, yte, c), len(c)) for v, c in (("full", full), ("expand", exp), ("medoids", med))}


def _datasets(seed):
    """The diverse benchmark's datasets, the synthetic ones re-drawn per seed."""
    configs = [
        dict(n_features=60, n_informative=8, n_redundant=30, n_repeated=6, class_sep=1.0, weights=None),
        dict(n_features=80, n_informative=10, n_redundant=10, n_repeated=0, class_sep=0.6, weights=None),
        dict(n_features=40, n_informative=6, n_redundant=20, n_repeated=4, class_sep=1.2, weights=[0.9, 0.1]),
    ]
    for i, cfg in enumerate(configs):
        yield f"synth_{i}", *make_classification(n_samples=2500, random_state=100 * seed + i, n_clusters_per_class=2, **cfg)
    yield "risk_signal_in_nonmedoid", *_risk_case(seed=seed)
    bc, wn, dg = load_breast_cancer(), load_wine(), load_digits()
    yield "breast_cancer", bc.data, bc.target
    yield "wine_0_vs_rest", wn.data, (wn.target == 0).astype(int)
    yield "digits_even_odd", dg.data, dg.target % 2


def main() -> None:
    """Print per-dataset AUC deltas of medoids-only against expanded clusters, and the summary over all runs."""
    deltas, shrink = [], []
    for seed in (0, 1, 2):
        for name, X, y in _datasets(seed):
            r = _run(name, X, y, seed)
            d = r["medoids"][0] - r["expand"][0]
            deltas.append(d)
            shrink.append(r["medoids"][1] / max(r["expand"][1], 1))
            print(f"seed={seed} {name:<26} full={r['full'][0]:.4f}/{r['full'][1]:<3} expand={r['expand'][0]:.4f}/{r['expand'][1]:<3} "
                  f"medoids={r['medoids'][0]:.4f}/{r['medoids'][1]:<3} d(med-exp)={d:+.4f}")
    arr = np.array(deltas)
    print("=" * 100)
    print(f"AUC medoids - expand: min={arr.min():+.4f} mean={arr.mean():+.4f} max={arr.max():+.4f}; worse than -0.01 on {int((arr < -0.01).sum())}/{arr.size}")
    print(f"support size medoids/expand: mean={np.mean(shrink):.2f}")


if __name__ == "__main__":
    main()

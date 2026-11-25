"""Is every selected feature needed given ALL the others? Out-of-fold logloss of a gradient-boosting model on the selected set with and without each feature (binary target, 20000 x 50 of ``linear_oof.make_data``).

``loss(without) - loss(with)`` near zero (t about 0) means the feature is redundant given the rest of the selection. A pairwise conditional-MI test is NOT a substitute: a feature that is a finer view of the
same underlying column keeps a significant CMI against its 8-bin version (artifact measured 2026-11-23), so conditioning on one coarse feature at a time proves nothing.

Usage: ``python selection_redundancy.py uniform,quantile``. Results of 2026-11-23: see ``audits/2026-11-23/_TRACKER.md`` (NOISE-IN-SELECTION).
"""

from __future__ import annotations

import sys, re, warnings, numpy as np
from mlframe.feature_selection._benchmarks.binning_label import binning_params
warnings.simplefilter("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.model_selection import StratifiedKFold
from mlframe.feature_selection._benchmarks.nbins_strategy.linear_oof import make_data
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
def oof_loss(A, y):
    L = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(A, y):
        p = H(max_iter=150, learning_rate=0.1, random_state=0).fit(A[tr], y[tr]).predict_proba(A[te])[:, 1].clip(1e-6, 1 - 1e-6)
        L[te] = -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    return L
for strategy in sys.argv[1].split(","):
    X, ys = make_data(20000, 50, "binary", 0); y = ys.to_numpy().astype(np.int64)
    kw = F.all_families_kwargs(); kw.update(binning_params(strategy))
    m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(X, ys)
    names = [str(c) for c in m.get_feature_names_out()]; T = np.nan_to_num(np.asarray(m.transform(X), dtype=np.float64))
    print("STRATEGY", strategy, "selected", len(names), flush=True)
    for r in m._engineered_recipes_:
        nm = str(r.name)
        if nm.startswith(("sir__", "opat__")): print("EXTRA", nm, {k: (np.round(v, 3).tolist() if hasattr(v, "__len__") else v) for k, v in (r.extra or {}).items()} if isinstance(r.extra, dict) else r.extra, flush=True)
    full = oof_loss(T, y)
    print("FULL logloss %.5f" % full.mean(), flush=True)
    for i, n in enumerate(names):
        ns = sorted({int(k) for k in re.findall(r"x(\d+)", n) if int(k) >= 5})
        without = oof_loss(np.delete(T, i, axis=1), y)
        d = without - full  # >0: removing the feature hurts
        print(("NOISE " if ns else "clean "), n[:55], "loss(without)-loss(with) = %+.5f  t=%.2f" % (d.mean(), d.mean() / (d.std() / np.sqrt(len(d)))), flush=True)

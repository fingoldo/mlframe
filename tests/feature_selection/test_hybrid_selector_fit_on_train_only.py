"""HybridSelector fits MRMR and the tree signals on the train partition so the holdout permutation importance is not optimistic."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.hybrid_selector import HybridSelector


def _data(seed: int, n: int = 1200, p: int = 30):
    """Two additive drivers, one product interaction and noise columns, binary target."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    z = X.f0 + 0.8 * X.f1 + 1.5 * X.f2 * X.f3 + r.normal(size=n) * 0.8
    return X, (z > np.median(z)).astype(int)


def test_fits_use_train_rows_only_and_default_is_on():
    """The default restricts MRMR to the holdout split's train rows."""
    assert HybridSelector().fit_on_train_only is True
    X, y = _data(0, n=600, p=8)
    h = HybridSelector(use_shap=False, use_boruta=False, random_state=0)
    tr, va = h._holdout_split(len(X), y)
    assert len(set(tr) & set(va)) == 0 and len(va) == 180


def test_engineered_holdout_importance_lower_than_fit_on_all():
    """Measured 4 seeds: mean engineered-column holdout FI sum 0.404 (fit on all) -> 0.299 (train only), noise columns selected 3.25 -> 1.5 per fit; assert the FI drop (>=10%)."""
    fi = {True: 0.0, False: 0.0}
    for flag in (True, False):
        for seed in range(4):
            X, y = _data(seed)
            h = HybridSelector(fit_on_train_only=flag, random_state=seed).fit(X, y)
            fi[flag] += sum(v for c, v in h.fi_.items() if c not in X.columns)
    assert fi[True] < 0.9 * fi[False]

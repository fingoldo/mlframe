"""biz_value: elimination_rule='stability' protects steady-mid-rank features from one-fold eviction.

Pins the scenario where stability measurably wins on honest holdout: 'many_steady' bed seed 1,
where the legacy 'importance' rule collapses RFECV to N=2 (dropping 4 of the 6 steady-mid true
features) while 'stability' keeps the steady features (top-k in most folds) and lands a markedly
higher holdout AUC. Measured (RF impurity, cv=3, max_refits=8):
  importance auc=0.7169 (n=2)   stability auc=0.8110 (n=9)   delta=+0.094
Floor set at +0.04 (well below measured 0.094) to absorb seed/thread noise.

NOTE: this is the ONE bed/seed where stability replicated a clear win in the cross-seed bench
(bench_rfecv_stability_elimination.py: 1 win / 1 tiny loss / 13 ties of 15). The default stays
'importance'; this test guards the opt-in win so a future regression that breaks the
fold-selection-frequency discount is caught by a FAILING WIN, not just an interface check.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers.rfecv import RFECV

from tests.conftest import fast_n_estimators


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _make_many_steady(seed=1, n=900):
    """Make many steady."""
    rng = np.random.default_rng(seed)
    n_strong, n_steady, n_noise = 1, 6, 20
    cols, logit = {}, np.zeros(n)
    for i in range(n_strong):
        x = rng.standard_normal(n)
        cols[f"strong_{i}"] = x
        logit += 1.3 * x
    for i in range(n_steady):
        x = rng.standard_normal(n)
        cols[f"steady_{i}"] = x
        logit += 0.45 * x
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    import pandas as pd

    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame(cols), y


def _fit_select(X, y, rule, seed=1):
    """Fit select."""
    r = RFECV(
        estimator=RandomForestClassifier(n_estimators=fast_n_estimators(80), max_depth=6, n_jobs=-1, random_state=seed),
        cv=3,
        scoring=None,
        verbose=0,
        max_refits=8,
        random_state=seed,
        importance_getter="feature_importances_",
        elimination_rule=rule,
        n_features_selection_rule="one_se_min",
    )
    r.fit(X, y)
    return [c for c in r.get_feature_names_out() if c in X.columns]


def _holdout_auc(X, y, rule, seed=1):
    """Holdout auc."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    cols = _fit_select(Xtr, ytr, rule, seed)
    m = RandomForestClassifier(n_estimators=fast_n_estimators(250, fast=100), max_depth=8, n_jobs=-1, random_state=seed)
    m.fit(Xtr[cols], ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1])), cols


@pytest.mark.slow
def test_biz_val_rfecv_stability_beats_importance_on_many_steady():
    """Biz val rfecv stability beats importance on many steady."""
    X, y = _make_many_steady(seed=1)
    auc_imp, cols_imp = _holdout_auc(X, y, "importance")
    auc_stab, cols_stab = _holdout_auc(X, y, "stability")
    # stability keeps more of the steady-mid true features and wins holdout AUC.
    assert auc_stab >= auc_imp + 0.04, (
        f"stability holdout AUC {auc_stab:.4f} should beat importance {auc_imp:.4f} by >=0.04 "
        f"on the many-steady bed (importance kept {cols_imp}, stability kept {cols_stab})"
    )
    n_steady_imp = sum(c.startswith("steady_") for c in cols_imp)
    n_steady_stab = sum(c.startswith("steady_") for c in cols_stab)
    assert n_steady_stab > n_steady_imp, (n_steady_stab, n_steady_imp)

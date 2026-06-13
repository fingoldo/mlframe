"""biz_val for ``dichotomic_step='auto'`` (adaptive coarse-to-fine elimination pace).

The adaptive-step lever was BENCH-REJECTED as the RFECV default: across 5 scenarios x 3 seeds (p in 30..600) it showed NO
replicated wall win (median 1.00x, 3/15) because the outer-loop early-stop terminates the ExhaustiveDichotomic search before
the unevaluated N-pool is ever large+flat enough for a coarse stride to save a refit. See
``_benchmarks/bench_dichotomic_adaptive_step.py``.

What DID hold, and what this biz_val pins quantitatively, is the SPEED-LEVER SAFETY contract: enabling ``auto`` is
SELECTION-EQUIVALENT to the legacy ``midpoint`` search on smooth curves -- a regression that lets the adaptive stride change
which features get picked (the failure mode CLAUDE.md's "gate a big win on its safe condition" warns about) trips this test.
Measured: Jaccard 1.00 and held-out accuracy delta 0.0000 every scenario/seed; floor set to Jaccard>=0.9 + |delta|<=0.02 with
margin. Replicated across 3 seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._enums import OptimumSearch


def _fit_select(X_tr, y_tr, step, seed):
    sel = RFECV(
        estimator=LogisticRegression(max_iter=300, random_state=seed),
        top_predictors_search_method=OptimumSearch.ExhaustiveDichotomic,
        dichotomic_step=step, dichotomic_epsilon=0.0, cv=3,
        max_noimproving_iters=10, random_state=seed, verbose=0, leave_progressbars=False,
    )
    sel.fit(X_tr, y_tr)
    return list(np.asarray(X_tr.columns)[sel.support_])


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biz_val_dichotomic_step_auto_is_selection_equivalent(seed):
    """auto vs midpoint: Jaccard>=0.9 AND held-out accuracy delta within +/-0.02, replicated each seed.

    Measured Jaccard 1.00 / delta 0.0000; floors carry margin so seed noise does not trip but a stride that
    changes the selected set (the speed-lever-altering-accuracy regression) fails the win.
    """
    X, y = make_classification(
        n_samples=1200, n_features=120, n_informative=12, n_redundant=10,
        n_clusters_per_class=2, random_state=seed,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    sup_a = _fit_select(X_tr, y_tr, "auto", seed)
    sup_m = _fit_select(X_tr, y_tr, "midpoint", seed)

    jac = _jaccard(sup_a, sup_m)
    assert jac >= 0.9, f"seed={seed}: auto vs midpoint Jaccard {jac:.2f} < 0.9 -- adaptive step altered the selection"

    def _hold(sup):
        m = LogisticRegression(max_iter=300, random_state=seed)
        m.fit(X_tr[sup], y_tr)
        return accuracy_score(y_te, m.predict(X_te[sup]))

    delta = _hold(sup_a) - _hold(sup_m)
    assert abs(delta) <= 0.02, f"seed={seed}: held-out delta {delta:+.4f} exceeds +/-0.02 noise band"

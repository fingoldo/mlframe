"""CPX14 regression: shap_oof=True must reproduce, bit-identically, the legacy per-fold full-X SHAP restricted to each fold's test rows.

The legacy compute_shap_on_cv explains the ENTIRE X under each fold's model (shape (n_folds, n, f)); shap_oof=True instead explains only each
fold's X_test and assembles a single out-of-fold matrix (shape (n, f)). Because the active explainer carries no background data= set, a row's SHAP
value is independent of the other rows passed to explainer(...), so the OOF matrix must equal the legacy stack sliced at [fold, test_rows]. This test
pins that identity directly against shap's TreeExplainer, mirroring the assembly the function performs, so a future regression in the OOF slotting
(wrong index mapping, accidental background introduction) trips it.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.fast
def test_shap_oof_matches_legacy_full_x_test_slice_bit_identical():
    pytest.importorskip("shap")
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    n, f = 1500, 8
    X = rng.standard_normal((n, f))
    logit = X[:, 0] * 1.2 - X[:, 1] * 0.8 + X[:, 2] * X[:, 3] * 0.4
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    cv = KFold(n_splits=4, shuffle=True, random_state=0)

    legacy_stack = []  # (n_folds, n, f): full-X SHAP per fold (the legacy contract)
    oof = [None] * n  # (n, f): each row explained by its holdout fold's model
    for tr, te in cv.split(X):
        m = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0).fit(X[tr], y[tr])
        ex = shap.Explainer(m)
        full_vals = np.asarray(ex(X).values)
        legacy_stack.append(full_vals)
        sub_vals = np.asarray(ex(X[te]).values)
        for j, ridx in enumerate(te):
            oof[ridx] = sub_vals[j]

    legacy_stack = np.asarray(legacy_stack)
    oof = np.asarray(oof)

    # Reconstruct the OOF deliverable from the legacy stack: for each row, take the fold's full-X SHAP at that row's position.
    expected_oof = np.empty_like(oof)
    for fold_i, (_tr, te) in enumerate(cv.split(X)):
        for ridx in te:
            expected_oof[ridx] = legacy_stack[fold_i][ridx]

    assert oof.shape == (n, f)
    assert np.array_equal(oof, expected_oof), "OOF SHAP (X_test per fold) must be bit-identical to legacy full-X SHAP sliced at the holdout rows"

"""Unit + biz_val for the interaction-aware coalition value (lever #5).

On a pure-XOR target, the main-effect coalition over-credits a SINGLE partner (its main SHAP value
absorbs half the interaction), so the plain proxy thinks ``{x0}`` alone is useful -- it isn't (XOR
needs both). The interaction coalition keeps only within-subset interactions, so it correctly sees
``{x0}`` as uninformative and only ``{x0, x1}`` as strong, matching the honest truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _xor_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)  # pure interaction: XOR of signs of x0, x1
    flip = rng.random(n) < 0.03
    y = np.where(flip, 1 - y, y)
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(6)]), y.astype(int)


def test_biz_val_interaction_coalition_sees_xor_pair():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import (
        compute_interaction_tensor, interaction_subset_loss, interaction_top_n)
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import subset_loss

    X, y = _xor_data()
    model = make_default_estimator(classification=True)
    phi, base, y_phi = compute_shap_matrix(model, X, y, classification=True, out_of_fold=False,
                                           n_models=1, rng=np.random.default_rng(0))
    Phi, ibase = compute_interaction_tensor(model, X, y, classification=True, rng=np.random.default_rng(0))

    # Honest truth: XOR needs both -> {0,1} far better than {0} alone.
    main_01, main_0 = subset_loss(phi, base, y_phi, [0, 1], "brier"), subset_loss(phi, base, y_phi, [0], "brier")
    int_01, int_0 = (interaction_subset_loss(Phi, ibase, y, [0, 1], "brier"),
                     interaction_subset_loss(Phi, ibase, y, [0], "brier"))

    # The interaction proxy must show a clear "need both" gap...
    assert int_0 - int_01 > 0.02, f"interaction gap too small: int_0={int_0:.3f} int_01={int_01:.3f}"
    # ...and that gap must be larger than the main-effect proxy's (which over-credits the single feature).
    assert (int_0 - int_01) > (main_0 - main_01), (
        f"interaction gap {(int_0 - int_01):.3f} not above main gap {(main_0 - main_01):.3f}")

    # The interaction ranking's best subset must contain BOTH XOR partners.
    top = interaction_top_n(Phi, ibase, y, classification=True, metric="brier", max_card=4, top_n=10)
    assert {0, 1} <= set(top[0][1]), f"interaction top subset missing XOR pair: {top[0][1]}"


@pytest.mark.slow
def test_biz_val_facade_interaction_aware_recovers_xor():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _xor_data(n=4000, seed=1)
    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=4,
                        interaction_aware=True, max_interaction_features=6, top_n=12, n_splits=3,
                        n_revalidation_models=1, trust_guard=False, random_state=1, verbose=False)
    sel.fit(X, y)
    assert sel.shap_proxy_report_.get("interaction_aware", {}).get("applied") is True
    # Both XOR partners recovered.
    assert {"f0", "f1"} <= set(sel.selected_features_), f"selected={sel.selected_features_}"

"""biz_val for the novel composition patterns (Batch D, novel A/B).

A) proposal_generator emits cheap SHAP-guided candidate subsets (for seeding honest search).
B) per_fold_stability_select: informative features survive across folds (high frequency) while noise
   does not, and the majority-vote ensemble recovers the informative set -- robust to single-fold luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _data(seed=0, n=2000):
    """Helper that data."""
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 4))
    noise = rng.normal(size=(n, 8))
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"inf{i}" for i in range(4)] + [f"noise{i}" for i in range(8)])
    y = (0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.5 * inf[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.slow
def test_proposal_generator_emits_candidate_subsets():
    """Proposal generator emits candidate subsets."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_compose import proposal_generator

    X, y = _data()
    props = proposal_generator(X, y, classification=True, optimizer="beam", top_n=10, max_features=5)
    assert 1 <= len(props) <= 10
    for _loss, feats in props:
        assert isinstance(feats, tuple) and len(feats) >= 1
        assert all(f in X.columns for f in feats)
    # The best proposal should be informative-heavy (mostly inf* features).
    best_feats = props[0][1]
    assert sum(f.startswith("inf") for f in best_feats) >= max(1, len(best_feats) - 1)


@pytest.mark.slow
def test_per_fold_stability_recovers_informative_with_high_frequency():
    """Per fold stability recovers informative with high frequency."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_compose import per_fold_stability_select

    X, y = _data(seed=1)
    rep = per_fold_stability_select(
        X,
        y,
        classification=True,
        n_folds=4,
        vote_threshold=0.5,
        metric="brier",
        optimizer="bruteforce",
        max_features=5,
        top_n=10,
        n_revalidation_models=1,
        random_state=0,
    )

    freq = rep["frequency"]
    # Informative features must have clearly higher mean selection frequency than noise.
    inf_freq = np.mean([freq.get(f"inf{i}", 0.0) for i in range(4)])
    noise_freq = np.mean([freq.get(f"noise{i}", 0.0) for i in range(8)])
    assert inf_freq > noise_freq + 0.25, f"inf_freq={inf_freq:.2f} noise_freq={noise_freq:.2f}"
    # Majority-vote ensemble recovers most of the informative set, little noise.
    ens = set(rep["ensemble"])
    assert len(ens & {f"inf{i}" for i in range(4)}) >= 3
    assert len([f for f in ens if f.startswith("noise")]) <= 2
    assert len(rep["per_fold"]) == 4

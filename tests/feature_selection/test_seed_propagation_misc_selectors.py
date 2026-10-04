"""The caller's seed must reach the fold splitters and default panel members of selectors that previously hardcoded random_state=0."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.hetero_vote import _default_panel
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_paired_parsimony import paired_one_se_pick


@pytest.mark.parametrize("classification", [True, False])
def test_default_panel_tree_member_follows_random_state(classification: bool) -> None:
    """The panel's random forest takes the caller's random_state (it was pinned to 0 regardless of the vote's seed)."""
    assert _default_panel(classification, random_state=17)["tree"].random_state == 17
    assert _default_panel(classification)["tree"].random_state == 0


def test_paired_parsimony_fold_split_follows_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """paired_one_se_pick shuffles its OOF folds with the supplied seed; None keeps the legacy seed 0."""
    import sklearn.model_selection as ms

    seen: list = []
    real = ms.StratifiedKFold

    class _Spy(real):
        """Record the random_state each StratifiedKFold is built with."""

        def __init__(self, *a, **kw):
            seen.append(kw["random_state"])
            super().__init__(*a, **kw)

    monkeypatch.setattr(ms, "StratifiedKFold", _Spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    ranked = [dict(features=(0, 1, 2), n_members=3, stable_score=0.1), dict(features=(0,), n_members=1, stable_score=0.1)]
    for seed in (None, 11):
        paired_one_se_pick(
            ranked, ranked[0], LogisticRegression(), X[:150], y[:150], X[150:], y[150:],
            classification=True, metric="brier", unit_to_members=None, seed=seed,
        )
    assert seen == [0, 11]

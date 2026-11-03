"""An unseen category carries no evidence under WoE = log(P(c|y=1)/P(c|y=0)): its encoding is 0, not the prior log-odds."""

from __future__ import annotations

import numpy as np

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def test_unseen_category_encodes_to_zero_on_an_imbalanced_target():
    """1% positives: the unseen fallback used to be log(0.01/0.99) = -4.6."""
    rng = np.random.default_rng(0)
    n = 4000
    cats = rng.choice(["a", "b", "c", "d"], size=n)
    y = (rng.random(n) < 0.01).astype(float)
    enc = LeakageSafeEncoder(method="woe")
    enc.fit_transform(cats, y)
    out = enc.transform(np.array(["never_seen", "a"]))
    assert out[0] == 0.0

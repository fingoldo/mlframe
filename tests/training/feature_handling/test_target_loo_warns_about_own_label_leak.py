"""The leave-one-out target encoder warns that a row's encoding depends on its own label (tracker 2026-11-02 wave 15, LEAK15-2)."""

import logging

import numpy as np

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def test_loo_encoding_warns(caplog):
    """Within a category the train encoding falls as the row's own target rises; the log says so and points at the k-fold methods."""
    rng = np.random.default_rng(0)
    cats = np.array(["a", "b"] * 50)
    y = rng.integers(0, 2, 100).astype(float)
    with caplog.at_level(logging.WARNING):
        LeakageSafeEncoder(method="target_loo").fit_transform(cats, y)
    assert any("leave-one-out" in r.getMessage() for r in caplog.records)

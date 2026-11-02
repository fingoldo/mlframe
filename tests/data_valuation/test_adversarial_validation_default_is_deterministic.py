"""Without an rng the data-valuation functions are deterministic, not re-seeded from entropy per call (tracker 2026-11-02 wave 8, RNG8-4)."""

import numpy as np

from mlframe.data_valuation._adversarial_validation import adversarial_validation


def test_two_calls_without_rng_agree():
    """Fold splits and the classifier seed come from a fixed default generator, so the AUC repeats."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(300, 4))
    X_test = rng.normal(loc=0.5, size=(300, 4))
    a = adversarial_validation(X_train, X_test, n_splits=3)
    b = adversarial_validation(X_train, X_test, n_splits=3)
    assert a["auc"] == b["auc"]

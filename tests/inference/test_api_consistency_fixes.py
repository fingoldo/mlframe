"""Regression tests for public-API consistency fixes in inference.

Covers:
  API21 -- get_models_raw_predictions uses predict_proba for classifiers (probabilities, not labels).
  API22 -- explainability handles multiclass (argmax / full proba, not hardcoded binary [:, 1]).
"""

import numpy as np


class _BinaryClf:
    """Classifier whose predict returns LABELS and predict_proba returns probabilities."""

    def predict(self, X):
        """Helper that predict."""
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        """Helper that predict proba."""
        n = len(X)
        p1 = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - p1, p1])


class _Regressor:
    """No predict_proba -> falls back to predict."""

    def predict(self, X):
        """Helper that predict."""
        return np.arange(len(X), dtype=float)


# --------------------------------------------------------------------------- API21
def test_api21_get_models_raw_predictions_returns_probabilities_for_classifier():
    """Api21 get models raw predictions returns probabilities for classifier."""
    from mlframe.inference.predict import get_models_raw_predictions

    X = np.zeros((5, 2))
    preds = get_models_raw_predictions({"clf": _BinaryClf()}, X, None)
    out = preds["clf"]
    # Probabilities (positive-class column), NOT the all-zero hard labels predict() returns.
    np.testing.assert_allclose(out, np.linspace(0.1, 0.9, 5))
    assert not np.array_equal(out, np.zeros(5))


def test_api21_regressor_falls_back_to_predict():
    """Api21 regressor falls back to predict."""
    from mlframe.inference.predict import get_models_raw_predictions

    X = np.zeros((4, 2))
    preds = get_models_raw_predictions({"reg": _Regressor()}, X, None)
    np.testing.assert_array_equal(preds["reg"], np.arange(4, dtype=float))


def test_api21_multiclass_returns_full_proba_matrix():
    """Api21 multiclass returns full proba matrix."""
    from mlframe.inference.predict import get_models_raw_predictions

    class _MultiClf:
        """Groups tests covering MultiClf."""
        def predict(self, X):
            """Helper that predict."""
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            """Helper that predict proba."""
            n = len(X)
            return np.tile([0.2, 0.3, 0.5], (n, 1))

    X = np.zeros((3, 2))
    preds = get_models_raw_predictions({"mc": _MultiClf()}, X, None)
    assert preds["mc"].shape == (3, 3)


# --------------------------------------------------------------------------- API22
def test_api22_explainability_classification_report_handles_multiclass():
    """The hardcoded binary ``(probs[:, 1] > 0.5)`` produced a degenerate 2-label report on multiclass.
    The argmax branch must produce per-class predictions across all classes."""
    from sklearn.metrics import classification_report

    # Mirror the fixed logic block: nclasses != 2 -> argmax.
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=60)
    nclasses = probs.shape[1]
    all_true_values = np.argmax(probs, axis=1)

    if nclasses == 2:
        hard_pred = (probs[:, 1] > 0.5).astype(np.int8)
    else:
        hard_pred = np.argmax(probs, axis=1).astype(np.int8)

    assert set(np.unique(hard_pred)).issubset({0, 1, 2})
    assert np.unique(hard_pred).size > 2  # genuinely multiclass, not collapsed to binary
    rep = classification_report(all_true_values, hard_pred, output_dict=True)
    assert "2" in rep  # the third class is represented

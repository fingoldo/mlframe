"""C1 — meta-test that training the same model on the same data with
the same random seed twice yields BIT-IDENTICAL predictions.

This is the property most non-trivial ML pipelines silently violate
once one introduces a global mutable cache (polars 1.x string cache,
torch's cuBLAS workspace, sklearn's RandomState pollution from a
sibling fit, ...). The user-facing symptom is "I trained twice and got
different scores" — almost always traced back to a hidden global, but
trivially gated by this test.

The test is parametrized across the model families that mlframe
publicly supports. Each combo:

  1. Builds a small fresh synthetic dataset (deterministic seed).
  2. Constructs the model from a config with explicit ``random_state``.
  3. Fits + predicts twice in the SAME process.
  4. Asserts ``np.array_equal`` on the predictions.

Models with no stochastic component (Ridge, Lasso, etc.) are still
covered — the test catches a future refactor that introduces
non-determinism behind one of them.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def regression_data():
    """Tiny deterministic regression dataset. Independent of conftest
    so the meta-test directory stays self-contained."""
    rng = np.random.default_rng(42)
    n_rows, n_features = 200, 8
    X = rng.standard_normal((n_rows, n_features))
    coef = rng.standard_normal(n_features)
    noise = 0.1 * rng.standard_normal(n_rows)
    y = X @ coef + noise
    return X, y


@pytest.fixture
def binary_data():
    """Builds seeded synthetic test data; returns ``(X, y)``."""
    rng = np.random.default_rng(42)
    n_rows, n_features = 200, 8
    X = rng.standard_normal((n_rows, n_features))
    coef = rng.standard_normal(n_features)
    logits = X @ coef
    y = (logits > 0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Linear models — closed-form fits, MUST be reproducible.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["ridge", "lasso", "elasticnet"])
def test_linear_regression_is_bit_reproducible(model_type, regression_data):
    """Ridge / Lasso / ElasticNet are closed-form / coordinate-descent
    with deterministic state. Two fits → bit-identical predictions."""
    from mlframe.training.configs import LinearModelConfig
    from mlframe.training.models import create_linear_model

    X, y = regression_data
    cfg = LinearModelConfig(model_type=model_type, alpha=1.0, random_state=42, max_iter=2000)

    m1 = create_linear_model(model_type, cfg, use_regression=True)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = create_linear_model(model_type, cfg, use_regression=True)
    m2.fit(X, y)
    p2 = m2.predict(X)

    np.testing.assert_array_equal(p1, p2, err_msg=(f"{model_type} predictions differ between two fits with identical seed — non-deterministic state pollution"))


# ---------------------------------------------------------------------------
# Stochastic linear models (SGD) — explicit seed must yield same fit.
# ---------------------------------------------------------------------------


def test_sgd_classification_is_bit_reproducible(binary_data):
    """``SGDClassifier`` IS stochastic; the contract is that a fixed
    ``random_state`` makes it deterministic. Catches a regression that
    breaks the seed plumbing."""
    from mlframe.training.configs import LinearModelConfig
    from mlframe.training.models import create_linear_model

    X, y = binary_data
    # SGD is for classification on binary data here.
    cfg = LinearModelConfig(model_type="sgd", random_state=42, max_iter=200, loss="log_loss")

    m1 = create_linear_model("sgd", cfg, use_regression=False)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = create_linear_model("sgd", cfg, use_regression=False)
    m2.fit(X, y)
    p2 = m2.predict(X)

    np.testing.assert_array_equal(p1, p2, err_msg=("SGDClassifier predictions differ between two fits with identical random_state — seed plumbing broken"))


# ---------------------------------------------------------------------------
# Decision-rule helpers — pure-function determinism.
# ---------------------------------------------------------------------------


def test_predict_from_probs_is_pure(regression_data):
    """``_predict_from_probs`` MUST be a pure function — calling twice
    on the same input gives ``np.array_equal`` output. Closes the
    invariant for the threshold-decision rule used by
    ``MultilabelDispatchConfig.per_label_thresholds``."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.helpers import _predict_from_probs

    rng = np.random.default_rng(42)
    probs = rng.random((50, 4))
    out_a = _predict_from_probs(
        probs.copy(),
        TargetTypes.MULTILABEL_CLASSIFICATION,
        threshold=0.5,
    )
    out_b = _predict_from_probs(
        probs.copy(),
        TargetTypes.MULTILABEL_CLASSIFICATION,
        threshold=0.5,
    )
    np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Pipeline-level: the polars-ds preprocessing pipeline used by all
# strategies must be reproducible.
# ---------------------------------------------------------------------------


def test_apply_preprocessing_extensions_is_reproducible():
    """``apply_preprocessing_extensions`` runs sklearn pipelines with
    user-supplied steps. Two invocations on the same input → identical
    output frames. Catches sklearn version drift introducing
    nondeterminism in scaler / PCA initialisation."""
    import pandas as pd
    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.pipeline import apply_preprocessing_extensions

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((100, 6)), columns=[f"f{i}" for i in range(6)])
    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        polynomial_degree=2,
        memory_safety_max_features=200,
        verbose_logging=False,
    )
    a, _, _, _ = apply_preprocessing_extensions(df.copy(), None, None, cfg, verbose=0)
    b, _, _, _ = apply_preprocessing_extensions(df.copy(), None, None, cfg, verbose=0)
    pd.testing.assert_frame_equal(a, b)

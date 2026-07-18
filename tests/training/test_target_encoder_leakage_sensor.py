"""A2-12: leakage sensor for the default category encoder.

The default target encoder in ``_get_pipeline_components`` MUST be an ordered / CV-style encoder (CatBoostEncoder), not a naive global-mean target encoder. A naive mean encoder leaks the row's own label into its encoding, producing wildly optimistic train-fold encodings on a high-cardinality categorical (each category seen once -> encoding == that row's label).

This sensor:
  1. Asserts the wired default is an ordered encoder (CatBoostEncoder), not the naive ``TargetEncoder``-with-no-CV.
  2. Quantifies the leak: a naive mean encoder's train encoding correlates ~1.0 with y on a one-row-per-category column, whereas the ordered default stays well below a safe threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core._setup_helpers import _get_pipeline_components


def test_a2_12_default_encoder_is_ordered() -> None:
    """The wired default for unspecified category_encoder must be CatBoostEncoder (ordered TS), not a naive mean encoder."""
    cfg = type("Cfg", (), {"category_encoder": None, "imputer": None, "scaler": None})()
    enc, _imp, _scl = _get_pipeline_components(cfg, cat_features=["c"], random_seed=42)
    assert enc is not None
    assert type(enc).__name__ == "CatBoostEncoder", f"default encoder must be ordered CatBoostEncoder; got {type(enc).__name__}"


def _high_card_frame(n: int = 400):
    """High card frame."""
    rng = np.random.default_rng(0)
    # One distinct category per row -> a naive mean encoder maps each category to exactly its own y.
    cats = pd.Series([f"c{i}" for i in range(n)], dtype="category")
    y = pd.Series(rng.integers(0, 2, n).astype(float))
    return pd.DataFrame({"c": cats}), y


def test_a2_12_naive_mean_encoder_leaks_ordered_does_not() -> None:
    """Naive global-mean target encoding leaks (train encoding ~= y); the ordered default does not."""
    ce = pytest.importorskip("category_encoders")
    X, y = _high_card_frame()

    # Naive (non-ordered) mean target encoder: full smoothing off, no CV -> direct mean per category.
    naive = ce.TargetEncoder(cols=["c"], smoothing=1e-9, min_samples_leaf=1)
    naive_enc = naive.fit_transform(X, y)["c"].to_numpy()
    naive_corr = abs(np.corrcoef(naive_enc, y.to_numpy())[0, 1])

    # Ordered default.
    cfg = type("Cfg", (), {"category_encoder": None, "imputer": None, "scaler": None})()
    ordered, _imp, _scl = _get_pipeline_components(cfg, cat_features=["c"], random_seed=42)
    ordered_enc = ordered.fit_transform(X, y)["c"].to_numpy()
    ordered_corr = abs(np.corrcoef(ordered_enc, y.to_numpy())[0, 1])

    assert naive_corr > 0.9, f"sanity: naive mean encoder should leak hard on one-row-per-category (got corr={naive_corr:.3f})"
    assert ordered_corr < 0.5, (
        f"ordered default leaks too much (train-fold optimism corr={ordered_corr:.3f} >= 0.5); "
        "the default category encoder may have been swapped to a non-ordered mean encoder."
    )

"""C3 — meta-test that bounded-input fits stay under a documented
memory budget.

Catches accidental allocator regressions: a refactor that silently
holds the train DataFrame in two formats simultaneously, a strategy
that builds the polars-to-pandas-to-numpy chain instead of
zero-copying, an inner sklearn step that returns a dense (N, K) array
when a sparse one would do.

The budget is intentionally LOOSE — we're catching ~5x regressions,
not micro-optimisations. The synthetic dataset is deliberately tiny
(200×8 floats ≈ 12 KB) so the budget reflects orchestration overhead,
not data size.

Uses ``tracemalloc`` for precision (no external deps; works on every
platform). Resets the tracker between fits to isolate per-fit overhead.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

# Per-model peak memory budget in MEGABYTES, measured during a
# fit + predict on the synthetic data below. Generous — these are
# guard rails for ~5x regressions, not perf benchmarks.
_BUDGETS_MB: dict[str, float] = {
    "ridge": 30.0,
    "lasso": 30.0,
    "elasticnet": 30.0,
    "huber": 30.0,
    "sgd": 30.0,
}


@pytest.fixture(scope="module")
def small_regression_data():
    """Builds seeded synthetic test data; returns ``(X, y)``."""
    rng = np.random.default_rng(123)
    n_rows, n_features = 200, 8
    X = rng.standard_normal((n_rows, n_features))
    y = X @ rng.standard_normal(n_features) + 0.1 * rng.standard_normal(n_rows)
    return X, y


@pytest.mark.parametrize("model_type,budget_mb", list(_BUDGETS_MB.items()))
def test_linear_fit_stays_under_memory_budget(model_type, budget_mb, small_regression_data):
    """Linear fit stays under memory budget."""
    from mlframe.training.configs import LinearModelConfig
    from mlframe.training.models import create_linear_model

    X, y = small_regression_data

    cfg_kwargs = {"model_type": model_type, "random_state": 42, "max_iter": 200}
    cfg = LinearModelConfig(**cfg_kwargs)

    # Force a clean baseline.
    gc.collect()
    tracemalloc.start()
    try:
        model = create_linear_model(model_type, cfg, use_regression=True)
        model.fit(X, y)
        _ = model.predict(X)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    assert peak_mb < budget_mb, f"{model_type}: peak {peak_mb:.2f} MB exceeds budget {budget_mb:.1f} MB during fit + predict on a 200×8 dataset"


def test_predict_from_probs_under_5mb_for_50k_x_10():
    """The decision-rule helper should be O(1) extra memory beyond
    output allocation. 50K×10 input → ~4 MB output; budget 5 MB."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.helpers import _predict_from_probs

    rng = np.random.default_rng(42)
    probs = rng.random((50_000, 10))

    gc.collect()
    tracemalloc.start()
    try:
        _ = _predict_from_probs(
            probs,
            TargetTypes.MULTILABEL_CLASSIFICATION,
            threshold=0.5,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    # 50K * 10 * int8 = 0.5 MB output; tracemalloc adds bookkeeping.
    # Loose 5 MB budget catches a ~10x over-allocation regression.
    assert peak_mb < 5.0, f"_predict_from_probs allocated {peak_mb:.2f} MB peak — a future refactor likely materialised an unwanted intermediate"

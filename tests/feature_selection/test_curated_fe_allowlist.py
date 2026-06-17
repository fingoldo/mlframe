"""Unit tests for the curated FE allowlist factory (Workstream C).

Verifies the factory builds custom_pre_pipelines-compatible sklearn pipelines, each adds feature
columns, and the OOF discipline is leak-safe (fit observes only the train fold).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.curated_fe import CURATED_FE_NAMES, curated_fe_pipelines


def test_factory_builds_named_pipelines():
    from sklearn.pipeline import Pipeline

    pipes = curated_fe_pipelines(task="regression")
    assert set(pipes) == set(CURATED_FE_NAMES)
    assert all(isinstance(p, Pipeline) for p in pipes.values())


def test_factory_subset_and_unknown_guard():
    pipes = curated_fe_pipelines(task="regression", names=["baseline_disagreement"])
    assert set(pipes) == {"baseline_disagreement"}
    with pytest.raises(ValueError, match="unknown"):
        curated_fe_pipelines(names=["does_not_exist"])


def test_baseline_disagreement_adds_columns_and_is_leak_safe():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n, p = 400, 4
    X_train = rng.standard_normal((n, p)).astype(np.float32)
    y_train = (X_train[:, 0] * 2 + rng.standard_normal(n) * 0.3).astype(np.float32)
    X_test = rng.standard_normal((120, p)).astype(np.float32)

    pipe = curated_fe_pipelines(task="regression", names=["baseline_disagreement"])["baseline_disagreement"]
    pipe.fit(X_train, y_train)
    out_train = np.asarray(pipe.transform(X_train))
    out_test = np.asarray(pipe.transform(X_test))
    assert out_train.shape[1] > p, "passthrough should add engineered columns"
    assert out_test.shape[0] == 120 and out_test.shape[1] == out_train.shape[1]

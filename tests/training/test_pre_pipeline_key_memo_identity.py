"""The single-slot key memo must not serve one pipeline's key for another that landed at the same id()."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import mlframe.training.pipeline._pipeline_cache as pc


def test_a_recycled_id_does_not_return_the_previous_pipelines_key(monkeypatch):
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2})
    y = np.arange(20) % 2
    # Every object reports the same id(): the worst case of CPython reusing a freed object's address.
    monkeypatch.setattr(pc, "id", lambda _o: 7, raising=False)
    key_a = pc._pre_pipeline_cache_key(X, None, Pipeline([("s", StandardScaler())]), y, "t")
    key_b = pc._pre_pipeline_cache_key(X, None, Pipeline([("s", MinMaxScaler())]), y, "t")
    assert key_a != key_b, "a differently-configured pipeline got the previous pipeline's cache key"


def test_the_same_objects_still_hit_the_memo(monkeypatch):
    X = pd.DataFrame({"a": np.arange(20.0)})
    y = np.arange(20) % 2
    pipe = Pipeline([("s", StandardScaler())])
    first = pc._pre_pipeline_cache_key(X, None, pipe, y, "t")
    calls = []
    monkeypatch.setattr(pc, "_pipeline_signature_for_cache", lambda p: calls.append(p) or "sig")
    assert pc._pre_pipeline_cache_key(X, None, pipe, y, "t") is first
    assert calls == [], "a repeat call on the same live objects must be served from the memo"

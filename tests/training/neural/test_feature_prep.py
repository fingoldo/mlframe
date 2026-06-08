"""Unit tests for the neural embedding/text feature preprocessor.

Pins that embedding ``List`` columns expand to numeric component columns (fed directly) and text columns become
dense HF transformer embeddings, so the tabular neural models (which have no native embedding/text layers) receive a
pure-numeric frame for every target type. The text path is exercised with a fake provider so the unit tests need no
``transformers`` install / model download; embedding expansion is tested for real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.neural.feature_prep import NeuralEmbeddingTextEncoder


class _FakeHFProvider:
    """Stand-in for HuggingFaceProvider: fixed-width deterministic vectors, no model load."""

    def __init__(self, dim: int = 6):
        self.embedding_dim = dim

    def transform(self, texts):
        # Deterministic per-text vector (length-based) so tests are reproducible without a real model.
        return np.array([[float(len(t)) + j for j in range(self.embedding_dim)] for t in texts], dtype=np.float32)


def _is_all_numeric(df: pd.DataFrame) -> bool:
    return all(dt.kind in ("f", "i", "u", "b") for dt in df.dtypes)


def test_embedding_column_expands_to_numeric_components():
    n, d = 30, 4
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "num_0": rng.normal(size=n).astype(np.float32),
        "emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)],
    })
    enc = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"])
    out = enc.fit_transform(X)
    assert "emb_0" not in out.columns
    assert [f"emb_0__e{j}" for j in range(d)] == [c for c in out.columns if c.startswith("emb_0__e")]
    assert _is_all_numeric(out)
    np.testing.assert_allclose(out["emb_0__e2"].to_numpy(), np.vstack(X["emb_0"].to_numpy())[:, 2], rtol=1e-6)


def test_text_column_hf_embeds_to_numeric():
    texts = ["red car fast", "blue car slow", "red bike", "green bike fast"] * 8
    X = pd.DataFrame({"num_0": np.arange(len(texts), dtype=np.float32), "text_0": texts})
    enc = NeuralEmbeddingTextEncoder(text_features=["text_0"])
    enc._provider = _FakeHFProvider(dim=6)  # inject fake; _get_provider returns the cached instance
    out = enc.fit_transform(X)
    assert "text_0" not in out.columns
    assert [f"text_0__h{j}" for j in range(6)] == [c for c in out.columns if c.startswith("text_0__h")]
    assert _is_all_numeric(out)


def test_missing_and_none_embedding_rows_zero_filled():
    n, d = 12, 3
    rng = np.random.default_rng(1)
    embs = [rng.normal(size=d).astype(np.float32) for _ in range(n)]
    embs[5] = None
    X = pd.DataFrame({"num_0": np.ones(n, dtype=np.float32), "emb_0": embs})
    enc = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"])
    out = enc.fit_transform(X)
    assert _is_all_numeric(out)
    assert np.allclose(out.loc[5, ["emb_0__e0", "emb_0__e1", "emb_0__e2"]].to_numpy().astype(float), 0.0)


def test_transform_on_unseen_frame_keeps_fixed_width():
    n, d = 20, 5
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)]})
    enc = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"]).fit(X)
    X2 = pd.DataFrame({"emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(7)]})
    out2 = enc.transform(X2)
    assert sum(c.startswith("emb_0__e") for c in out2.columns) == d


def test_no_object_columns_remain_mixed():
    n, d = 16, 2
    rng = np.random.default_rng(3)
    X = pd.DataFrame({
        "num_0": rng.normal(size=n).astype(np.float32),
        "emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)],
        "text_0": ["alpha beta", "beta gamma"] * (n // 2),
    })
    enc = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"], text_features=["text_0"])
    enc._provider = _FakeHFProvider(dim=3)
    out = enc.fit_transform(X)
    assert _is_all_numeric(out), f"object cols remain: {[c for c in out.columns if out[c].dtype.kind == 'O']}"


def test_provider_excluded_from_pickle():
    import pickle
    enc = NeuralEmbeddingTextEncoder(text_features=["text_0"])
    enc._provider = _FakeHFProvider(dim=4)
    enc.fit(pd.DataFrame({"text_0": ["a", "bb", "ccc"]}))
    restored = pickle.loads(pickle.dumps(enc))  # must not try to pickle the live provider
    assert not hasattr(restored, "_provider") or restored.__dict__.get("_provider") is None

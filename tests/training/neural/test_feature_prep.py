"""Unit tests for the neural embedding/text feature preprocessor.

Pins that embedding ``List`` columns expand to numeric component columns (fed directly) and text columns become dense
HF transformer embeddings, so the tabular neural models (which have no native embedding/text layers) receive a
pure-numeric frame for every target type. Text is exercised with the REAL default HuggingFace model
(``intfloat/multilingual-e5-small``) -- no mocking -- shared across tests via a module fixture (one model load).
Skips only when ``transformers`` / the model can't be fetched (offline CI).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.training.neural.feature_prep import DEFAULT_TEXT_MODEL, NeuralEmbeddingTextEncoder


@pytest.fixture(scope="module")
def hf_provider():
    pytest.importorskip("transformers")
    from mlframe.training.feature_handling.hf_provider import build_provider
    from mlframe.training.feature_handling.providers import EmbeddingProvider
    try:
        prov = build_provider(EmbeddingProvider(kind="huggingface", model=DEFAULT_TEXT_MODEL))
        prov.acquire()
    except Exception as e:  # pragma: no cover -- offline / model-fetch failure
        pytest.skip(f"HuggingFace model unavailable ({type(e).__name__}: {e})")
    return prov


def _is_all_numeric(df: pd.DataFrame) -> bool:
    return all(dt.kind in ("f", "i", "u", "b") for dt in df.dtypes)


# ---- embedding expansion (no model needed) -------------------------------------------------------------------------

def test_embedding_column_expands_to_numeric_components():
    n, d = 30, 4
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "num_0": rng.normal(size=n).astype(np.float32),
        "emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)],
    })
    out = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"]).fit_transform(X)
    assert "emb_0" not in out.columns
    assert [f"emb_0__e{j}" for j in range(d)] == [c for c in out.columns if c.startswith("emb_0__e")]
    assert _is_all_numeric(out)
    np.testing.assert_allclose(out["emb_0__e2"].to_numpy(), np.vstack(X["emb_0"].to_numpy())[:, 2], rtol=1e-6)


def test_missing_and_none_embedding_rows_zero_filled():
    n, d = 12, 3
    rng = np.random.default_rng(1)
    embs = [rng.normal(size=d).astype(np.float32) for _ in range(n)]
    embs[5] = None
    X = pd.DataFrame({"num_0": np.ones(n, dtype=np.float32), "emb_0": embs})
    out = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"]).fit_transform(X)
    assert _is_all_numeric(out)
    assert np.allclose(out.loc[5, ["emb_0__e0", "emb_0__e1", "emb_0__e2"]].to_numpy().astype(float), 0.0)


def test_transform_on_unseen_frame_keeps_fixed_width():
    n, d = 20, 5
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)]})
    enc = NeuralEmbeddingTextEncoder(embedding_features=["emb_0"]).fit(X)
    out2 = enc.transform(pd.DataFrame({"emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(7)]}))
    assert sum(c.startswith("emb_0__e") for c in out2.columns) == d


# ---- text via REAL HuggingFace model -------------------------------------------------------------------------------

def _encoder_with(provider, **kw):
    enc = NeuralEmbeddingTextEncoder(**kw)
    enc._provider = provider  # share the single loaded model; _get_provider returns the cached instance
    return enc


def test_text_column_hf_embeds_to_numeric(hf_provider):
    texts = ["red car fast", "blue car slow", "red bike", "green bike fast"] * 8
    X = pd.DataFrame({"num_0": np.arange(len(texts), dtype=np.float32), "text_0": texts})
    enc = _encoder_with(hf_provider, text_features=["text_0"])
    out = enc.fit_transform(X)
    assert "text_0" not in out.columns
    h_cols = [c for c in out.columns if c.startswith("text_0__h")]
    assert len(h_cols) == enc.text_embedding_dim_ == hf_provider.embedding_dim
    assert _is_all_numeric(out)
    # real embeddings are non-degenerate (not all-zero / all-equal)
    assert float(out[h_cols].to_numpy().std()) > 0.0


def test_no_object_columns_remain_mixed(hf_provider):
    n, d = 16, 2
    rng = np.random.default_rng(3)
    X = pd.DataFrame({
        "num_0": rng.normal(size=n).astype(np.float32),
        "emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)],
        "text_0": ["alpha beta", "beta gamma"] * (n // 2),
    })
    enc = _encoder_with(hf_provider, embedding_features=["emb_0"], text_features=["text_0"])
    out = enc.fit_transform(X)
    assert _is_all_numeric(out), f"object cols remain: {[c for c in out.columns if out[c].dtype.kind == 'O']}"


def test_provider_excluded_from_pickle(hf_provider):
    enc = _encoder_with(hf_provider, text_features=["text_0"])
    enc.fit(pd.DataFrame({"text_0": ["a", "bb", "ccc"]}))
    restored = pickle.loads(pickle.dumps(enc))  # must not try to pickle the live HF model
    assert restored.__dict__.get("_provider") is None
    assert restored.text_embedding_dim_ == enc.text_embedding_dim_

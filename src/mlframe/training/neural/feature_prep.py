"""Make embedding-vector and free-text columns consumable by the tabular neural models (MLP / ranker), which have no
native embedding or text input layers.

- **Embedding columns** (polars ``List(Float32)`` -> pandas object cells holding per-row float vectors) are fed
  DIRECTLY: each vector is expanded into its component float columns that flow straight into the MLP input.
- **Text columns** are turned into dense semantic vectors by a real HuggingFace tokenizer + pretrained transformer
  (the project's functional :class:`HuggingFaceProvider`: ``AutoTokenizer`` + ``AutoModel`` -> mean/cls/max pooled
  ``(N, hidden)`` embeddings). This is the right neural text representation -- NOT bag-of-words TF-IDF, which throws
  away word order and semantics and isn't learned.

This runs as a sklearn pipeline step BEFORE the network, so it is target-type-agnostic: every head (regression /
binary / multiclass / multilabel / learning-to-rank) sees a pure-numeric frame.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-small"


def _as_pandas(X, feature_names_in_: "list[str] | None" = None):
    """Neural pipeline runs on pandas; convert a polars frame defensively (the caller is expected to hand pandas).

    A raw ndarray reaching this step (e.g. an earlier sklearn Pipeline step eagerly materialised the frame to
    numpy for a NaN-intolerant estimator, per the mlframe eager-conversion convention) is reconstructed into a
    DataFrame using the column names captured at ``fit`` time (``feature_names_in_``, sklearn's own convention)
    -- same column ORDER as at fit, which sklearn Pipeline.transform already guarantees. Surfaced by fuzzing
    (2026-07-06): ``NeuralEmbeddingTextEncoder.transform`` needs real column names for its embedding/text lookup
    and cannot operate on a bare ndarray with no column identity.
    """
    if isinstance(X, np.ndarray) and feature_names_in_ is not None:
        return pd.DataFrame(X, columns=feature_names_in_)
    if not isinstance(X, pd.DataFrame) and hasattr(X, "to_pandas"):
        return X.to_pandas()
    return X


def _stack_embedding_column(series, dim: Optional[int]):
    """Stack an object column of per-row float vectors into an (N, D) float32 matrix.

    Missing rows are zero-filled; wrong-length rows are truncated / zero-padded to D. ``dim`` pins the fit-time width
    (so transform always emits the same column count); when None it is inferred from the first valid row.
    """
    vals = series.to_list() if hasattr(series, "to_list") else list(series)
    if dim is None:
        dim = 0
        for v in vals:
            if v is not None and hasattr(v, "__len__") and len(v) > 0:
                dim = len(v)
                break
    out = np.zeros((len(vals), dim), dtype=np.float32)
    if dim == 0:
        return out, dim
    for i, v in enumerate(vals):
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).ravel()
        if arr.shape[0] == 0:
            continue
        n = min(arr.shape[0], dim)
        out[i, :n] = arr[:n]
    return out, dim


class NeuralEmbeddingTextEncoder(BaseEstimator, TransformerMixin):
    """Expand embedding-list columns and HF-embed text columns into numeric columns for the tabular neural models.

    ``transform`` drops the original embedding/text columns and appends ``{col}__e{j}`` (embedding components) and
    ``{col}__h{j}`` (HF text-embedding dims) numeric columns. Columns absent from a given frame and all-missing
    embeddings are handled without error. The live HF model is excluded from pickling (re-acquired lazily on demand).
    """

    def __init__(
        self,
        embedding_features: Optional[Sequence[str]] = None,
        text_features: Optional[Sequence[str]] = None,
        text_model: str = DEFAULT_TEXT_MODEL,
    ):
        self.embedding_features = list(embedding_features) if embedding_features else []
        self.text_features = list(text_features) if text_features else []
        self.text_model = text_model

    def _get_provider(self):
        """Lazily build + acquire the frozen HF embedding provider. Cached on the instance, excluded from pickle."""
        prov = getattr(self, "_provider", None)
        if prov is None:
            from mlframe.training.feature_handling.hf_provider import build_provider
            from mlframe.training.feature_handling.providers import EmbeddingProvider
            prov = build_provider(EmbeddingProvider(kind="huggingface", model=self.text_model))
            prov.acquire()
            self._provider = prov
        return prov

    def fit(self, X, y=None):
        """Learn each pre-computed embedding column's width and, for raw text columns, resolve the pretrained provider's fixed embedding dimension (no actual training; the provider is frozen)."""
        X = _as_pandas(X)
        self.feature_names_in_ = list(X.columns)
        cols = set(X.columns)
        self.embedding_dims_: dict = {}
        for c in self.embedding_features:
            if c in cols:
                _, dim = _stack_embedding_column(X[c], None)
                self.embedding_dims_[c] = dim
        self.text_cols_ = [c for c in self.text_features if c in cols]
        self.text_embedding_dim_: Optional[int] = None
        if self.text_cols_:
            # Pretrained -> no training; just resolve the embedding width so the emitted column count is fixed.
            self.text_embedding_dim_ = int(self._get_provider().embedding_dim)
        return self

    def transform(self, X):
        """Expand pre-computed embedding columns into ``<col>__e<j>`` numeric columns and raw text columns into pretrained-provider embedding vectors, per the widths/provider resolved in ``fit``."""
        X = _as_pandas(X, feature_names_in_=getattr(self, "feature_names_in_", None))
        cols = set(X.columns)
        new_blocks: dict = {}
        for c, dim in getattr(self, "embedding_dims_", {}).items():
            if c not in cols:
                continue
            mat, _ = _stack_embedding_column(X[c], dim)
            for j in range(mat.shape[1]):
                new_blocks[f"{c}__e{j}"] = mat[:, j]
        text_cols = [c for c in getattr(self, "text_cols_", []) if c in cols]
        if text_cols:
            provider = self._get_provider()
            for c in text_cols:
                texts = X[c].fillna("").astype(str).tolist()
                vecs = np.asarray(provider.transform(texts), dtype=np.float32)
                for j in range(vecs.shape[1]):
                    new_blocks[f"{c}__h{j}"] = vecs[:, j]
        drop = [c for c in (self.embedding_features + self.text_features) if c in cols]
        out = X.drop(columns=drop) if drop else X
        if new_blocks:
            out = pd.concat([out, pd.DataFrame(new_blocks, index=X.index)], axis=1)
        return out

    def __getstate__(self):
        # The acquired HF model/tokenizer is unpicklable + large; drop it and re-acquire lazily on next transform.
        state = self.__dict__.copy()
        state.pop("_provider", None)
        return state

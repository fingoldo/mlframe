"""Embedding / free-text passthrough detection for MRMR feature selection.

MRMR's MI estimator discretises every candidate column into integer bins, which requires scalar (hashable, orderable) cell values. Two column kinds violate
that contract and must NOT enter the MI candidate set:

* EMBEDDING-VECTOR columns -- object-dtype cells holding ``list`` / ``np.ndarray`` / ``tuple`` of floats (a learned/precomputed embedding per row). Passing such a
  cell to the discretiser raises ``TypeError: unhashable type: 'numpy.ndarray'`` or silently mis-bins.
* FREE-TEXT columns -- object/string cells holding long natural-language strings. Treated as a categorical with ~N distinct levels they carry no usable MI signal
  and blow up the bin count; the downstream PyTorch-Lightning network's ``_encode_emb_text_fit`` boundary encoder is the correct consumer.

These columns are detected at fit time, EXCLUDED from the discretisation / MI / FE candidate set, and PASSED THROUGH to the transform output unchanged so the
learnable-embedding MLP / recurrent network (and the boundary encoder) receive them end-to-end. The detector is sampling-based (never materialises a 100+ GB
column) and is enabled by default -- the legacy "drop / crash on non-scalar column" behaviour was silently wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sampling cap: inspect at most this many cells per column to classify it. A handful of rows is enough to decide list-vs-scalar / long-text-vs-short; bounding the
# scan keeps detection O(ncols) regardless of frame height (100+ GB frames must never be fully materialised here).
_SAMPLE_ROWS = 50
# A string column whose sampled mean length exceeds this many characters is treated as FREE-TEXT (boundary-encoder territory) rather than a low-cardinality cat.
_TEXT_MEAN_LEN_THRESHOLD = 32


def _cell_is_vector(v) -> bool:
    """True when a single cell holds an embedding vector (list / tuple / ndarray of length >= 1)."""
    if isinstance(v, np.ndarray):
        return v.ndim >= 1
    if isinstance(v, (list, tuple)):
        return len(v) >= 1
    return False


def detect_passthrough_columns(
    X: pd.DataFrame,
    *,
    detect_embeddings: bool = True,
    detect_text: bool = True,
    text_mean_len_threshold: int = _TEXT_MEAN_LEN_THRESHOLD,
) -> tuple[list, list]:
    """Detect embedding-vector and free-text columns in ``X``.

    Returns ``(embedding_cols, text_cols)`` -- two lists of column labels. Only object/string-dtype columns are inspected (numeric / category / bool columns can
    never carry list cells and are cheap to bin, so they stay in the MI candidate set). Detection samples at most ``_SAMPLE_ROWS`` non-null cells per column.

    A column is an EMBEDDING column when its sampled non-null cells are predominantly list/tuple/ndarray. It is a FREE-TEXT column when its sampled cells are
    predominantly strings whose mean length exceeds ``text_mean_len_threshold``. Short-string object columns (typical categoricals) are left untouched.
    """
    embedding_cols: list = []
    text_cols: list = []
    if not isinstance(X, pd.DataFrame):
        return embedding_cols, text_cols
    if not (detect_embeddings or detect_text):
        return embedding_cols, text_cols

    n = len(X)
    if n == 0:
        return embedding_cols, text_cols
    step = max(1, n // _SAMPLE_ROWS)

    for col in X.columns:
        s = X[col]
        # Duplicate-labelled columns yield a DataFrame on ``X[col]``; skip (ambiguous, handled by the dedup pass upstream).
        if getattr(s, "ndim", 1) != 1:
            continue
        # Only object / string dtypes can hold vectors or long text. ``is_object_dtype`` covers ndarray-cell columns; string dtype covers pandas StringArray.
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        sample = s.iloc[::step]
        n_vec = 0
        n_str = 0
        n_seen = 0
        total_len = 0
        for v in sample.to_numpy():
            if v is None or (np.isscalar(v) and pd.isna(v)):
                continue
            n_seen += 1
            if detect_embeddings and _cell_is_vector(v):
                n_vec += 1
            elif isinstance(v, str):
                n_str += 1
                total_len += len(v)
        if n_seen == 0:
            continue

        if detect_embeddings and n_vec * 2 >= n_seen:  # majority of sampled cells are vectors
            embedding_cols.append(col)
            continue
        if detect_text and n_str * 2 >= n_seen and total_len / max(n_str, 1) > text_mean_len_threshold:
            text_cols.append(col)

    return embedding_cols, text_cols

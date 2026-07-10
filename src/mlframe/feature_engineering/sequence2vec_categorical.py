"""Word2Vec-style (skip-gram + negative sampling) embeddings for entity-sequence categorical IDs.

Treats each entity's chronological sequence of categorical IDs (e.g. per-user app_codes, per-session item
views) as a "sentence" and trains skip-gram-with-negative-sampling embeddings so IDs that co-occur in similar
LOCAL windows end up nearby in embedding space -- capturing sequential/behavioral structure that a global
bag-of-interactions SVD (:func:`mlframe.feature_engineering.latent_interaction_svd.latent_interaction_features`)
does not (that tool ignores within-entity ORDER entirely). Implemented directly in numpy rather than via
gensim: gensim's compiled Cython extensions do not currently build against this Python version (a real
environment incompatibility, not a shortcut) -- SGNS is a small, well-defined algorithm and reasonable to
implement directly rather than block on an upstream build fix.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, fastmath=True)
def _sgns_update_njit(target_emb: np.ndarray, context_emb: np.ndarray, center_idx: int, sample_indices: np.ndarray, labels: np.ndarray, lr: float) -> None:
    """One SGNS gradient step (1 positive + n_negative negatives) for a single center token, in place.

    Plain nested loops instead of numpy calls on tiny (embedding_dim, n_negative+1)-sized arrays -- the
    original per-pair implementation (``_sigmoid``, ``@``, fancy-indexed ``.sum(axis=0)``) paid numpy's
    per-call dispatch overhead on every one of the millions of (center, context) pairs a real training run
    touches; profiled at ~85us/update. This njit kernel measures ~1.5us/update (~58x), consistent with the
    project's numerical-kernel-ladder convention for a hot loop this small and this frequently called.
    """
    d = target_emb.shape[1]
    v = target_emb[center_idx].copy()
    grad_center = np.zeros(d)
    for k in range(sample_indices.shape[0]):
        j = sample_indices[k]
        dot = 0.0
        for x in range(d):
            dot += context_emb[j, x] * v[x]
        if dot > 30.0:
            dot = 30.0
        elif dot < -30.0:
            dot = -30.0
        score = 1.0 / (1.0 + np.exp(-dot))
        g = score - labels[k]
        for x in range(d):
            grad_center[x] += g * context_emb[j, x]
        for x in range(d):
            context_emb[j, x] -= lr * g * v[x]
    for x in range(d):
        target_emb[center_idx, x] -= lr * grad_center[x]


def train_sequence2vec(
    sequences: Sequence[Sequence[str]],
    embedding_dim: int = 32,
    window: int = 3,
    n_negative: int = 5,
    n_epochs: int = 5,
    learning_rate: float = 0.05,
    min_count: int = 1,
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
    """Train skip-gram-with-negative-sampling embeddings over token sequences.

    Parameters
    ----------
    sequences
        One sequence (list of token strings) per entity, in chronological order.
    embedding_dim
        Output embedding dimensionality.
    window
        Max distance between a target token and a context token within the same sequence.
    n_negative
        Negative samples drawn per positive (target, context) pair.
    n_epochs
        Full passes over all sequences.
    learning_rate
        SGD step size.
    min_count
        Tokens appearing fewer than this many times total are dropped from the vocabulary (never embedded;
        callers should treat them as missing/OOV).
    random_state
        Seed for negative sampling and weight initialization.

    Returns
    -------
    dict[str, np.ndarray]
        ``{token: (embedding_dim,) vector}`` for every token meeting ``min_count``.
    """
    rng = np.random.default_rng(random_state)

    token_counts: Dict[str, int] = {}
    for seq in sequences:
        for tok in seq:
            token_counts[tok] = token_counts.get(tok, 0) + 1
    vocab = sorted(tok for tok, c in token_counts.items() if c >= min_count)
    if not vocab:
        return {}
    tok_to_idx = {tok: i for i, tok in enumerate(vocab)}
    n_vocab = len(vocab)

    freqs = np.array([token_counts[tok] for tok in vocab], dtype=np.float64)
    neg_sample_probs = freqs**0.75
    neg_sample_probs /= neg_sample_probs.sum()

    target_emb = (rng.random((n_vocab, embedding_dim)) - 0.5) / embedding_dim
    context_emb = np.zeros((n_vocab, embedding_dim), dtype=np.float64)

    labels_buf = np.zeros(n_negative + 1, dtype=np.float64)
    labels_buf[0] = 1.0
    sample_indices_buf = np.empty(n_negative + 1, dtype=np.int64)

    for _ in range(n_epochs):
        for seq in sequences:
            idx_seq = [tok_to_idx[tok] for tok in seq if tok in tok_to_idx]
            n = len(idx_seq)
            for center_pos in range(n):
                center_idx = idx_seq[center_pos]
                w = rng.integers(1, window + 1)
                lo, hi = max(0, center_pos - w), min(n, center_pos + w + 1)
                for context_pos in range(lo, hi):
                    if context_pos == center_pos:
                        continue
                    context_idx = idx_seq[context_pos]

                    sample_indices_buf[0] = context_idx
                    sample_indices_buf[1:] = rng.choice(n_vocab, size=n_negative, p=neg_sample_probs)

                    _sgns_update_njit(target_emb, context_emb, center_idx, sample_indices_buf, labels_buf, learning_rate)

    return {tok: target_emb[i] for tok, i in tok_to_idx.items()}


def sequence2vec_entity_features(
    df: pd.DataFrame,
    entity_col: str,
    token_col: str,
    time_col: Optional[str] = None,
    embedding_dim: int = 32,
    window: int = 3,
    n_negative: int = 5,
    n_epochs: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Train sequence2vec embeddings from ``events_df`` and return per-entity mean/last embedding features.

    Parameters
    ----------
    df
        Event-level frame with one row per (entity, token) occurrence.
    entity_col
        Grouping key defining each "sentence" (one sequence per entity).
    token_col
        Categorical id column to embed.
    time_col
        Optional ordering column; rows are sorted by this within each entity before forming sequences
        (defaults to the input row order).
    embedding_dim, window, n_negative, n_epochs, random_state
        Passed through to :func:`train_sequence2vec`.

    Returns
    -------
    pd.DataFrame
        One row per entity: ``entity_col`` plus ``emb_mean_{0..d-1}`` (mean embedding across the entity's
        tokens) and ``emb_last_{0..d-1}`` (the entity's most recent token's embedding) columns.
    """
    ordered = df.sort_values([entity_col, time_col] if time_col else [entity_col], kind="mergesort")
    sequences: List[List[str]] = []
    entity_order: List[object] = []
    for entity, grp in ordered.groupby(entity_col, sort=False):
        sequences.append(grp[token_col].astype(str).tolist())
        entity_order.append(entity)

    embeddings = train_sequence2vec(
        sequences, embedding_dim=embedding_dim, window=window, n_negative=n_negative, n_epochs=n_epochs, random_state=random_state
    )

    rows = []
    for entity, seq in zip(entity_order, sequences):
        vecs = [embeddings[tok] for tok in seq if tok in embeddings]
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            last_vec = vecs[-1]
        else:
            mean_vec = np.zeros(embedding_dim)
            last_vec = np.zeros(embedding_dim)
        row = {entity_col: entity}
        row.update({f"emb_mean_{i}": mean_vec[i] for i in range(embedding_dim)})
        row.update({f"emb_last_{i}": last_vec[i] for i in range(embedding_dim)})
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["train_sequence2vec", "sequence2vec_entity_features"]

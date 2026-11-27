"""Cluster 24 regression: the whole SGNS training loop runs inside numba and still learns, deterministically (audit 2026-11-20, 24-F39)."""

import numpy as np
from numba import njit

from mlframe.feature_engineering.sequence2vec_categorical import _draw_negative, train_sequence2vec


@njit
def _draws(cum_probs, n, seed):
    """Draw ``n`` negatives from numba's generator after seeding it."""
    np.random.seed(seed)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = _draw_negative(cum_probs)
    return out


def test_negative_sampler_follows_the_smoothed_frequency_distribution():
    """The inverse-transform sampler reproduces freq**0.75 / sum, including a rare token, within sampling error (24-F39)."""
    freqs = np.array([1000.0, 300.0, 100.0, 10.0, 1.0])
    probs = freqs**0.75
    probs /= probs.sum()
    cum = np.cumsum(probs)
    cum[-1] = 1.0
    n = 400_000
    draws = _draws(cum, n, 123)
    assert draws.min() >= 0 and draws.max() < len(probs)
    freq_hat = np.bincount(draws, minlength=len(probs)) / n
    np.testing.assert_allclose(freq_hat, probs, atol=0.004)


def _two_topic_corpus(seed=0, n_seq=200, length=40):
    """Sequences that stay inside one of two disjoint token groups, so same-group tokens co-occur and cross-group tokens never do."""
    rng = np.random.default_rng(seed)
    a, b = [f"a{i}" for i in range(6)], [f"b{i}" for i in range(6)]
    return [list(rng.choice(a if rng.random() < 0.5 else b, size=length)) for _ in range(n_seq)]


def _cos(u, v):
    """Cosine similarity of two vectors."""
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def test_cooccurring_tokens_end_up_closer_than_tokens_that_never_co_occur():
    """Mean same-group cosine is clearly above mean cross-group cosine after training (24-F39)."""
    emb = train_sequence2vec(_two_topic_corpus(), embedding_dim=16, window=3, n_negative=5, n_epochs=8, learning_rate=0.05, random_state=0)
    same = [_cos(emb[f"a{i}"], emb[f"a{j}"]) for i in range(6) for j in range(i + 1, 6)] + [_cos(emb[f"b{i}"], emb[f"b{j}"]) for i in range(6) for j in range(i + 1, 6)]
    cross = [_cos(emb[f"a{i}"], emb[f"b{j}"]) for i in range(6) for j in range(6)]
    assert np.mean(same) > np.mean(cross) + 0.3


def test_same_seed_is_reproducible_and_a_different_seed_differs():
    """Two runs with one random_state give identical embeddings; another random_state gives different ones (24-F39)."""
    corpus = _two_topic_corpus(n_seq=40, length=20)
    kw = dict(embedding_dim=8, window=2, n_negative=3, n_epochs=2)
    first = train_sequence2vec(corpus, random_state=5, **kw)
    again = train_sequence2vec(corpus, random_state=5, **kw)
    other = train_sequence2vec(corpus, random_state=6, **kw)
    for tok in first:
        np.testing.assert_array_equal(first[tok], again[tok])
    assert any(not np.allclose(first[t], other[t]) for t in first)


def test_empty_and_single_token_sequences_are_harmless():
    """Sequences with nothing to pair (empty, one token, out-of-vocabulary only) do not break the compiled loop (24-F39)."""
    emb = train_sequence2vec([[], ["x"], ["x", "y", "x"], ["y"]], embedding_dim=4, window=2, n_negative=2, n_epochs=2, min_count=1, random_state=0)
    assert set(emb) == {"x", "y"} and all(np.isfinite(v).all() for v in emb.values())
    assert train_sequence2vec([["a"], ["b"]], min_count=5) == {}

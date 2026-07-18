"""Unit tests for ``CategoricalEmbedding`` (learnable entity embeddings for tabular MLPs).

Covers forward-pass shapes (cats embed, numerics pass through, ``out_features`` correct), unseen / overflow code clamping to the reserved row
without IndexError, and a pickle round-trip preserving the learned embedding weights.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural._categorical_embeddings import CategoricalEmbedding, default_embed_dim


def test_forward_shape_with_numerics():
    # 2 cats (card 4 and 3) + 5 numeric columns -> out = embed_dims_sum + 5.
    """Forward shape with numerics."""
    cards = [4, 3]
    emb = CategoricalEmbedding(cardinalities=cards, embed_dim=6)
    emb.set_num_numeric(5)
    assert emb.out_features == 6 * 2 + 5
    n = 10
    cat_codes = np.column_stack(
        [
            np.random.randint(0, 4, size=n),
            np.random.randint(0, 3, size=n),
        ]
    ).astype(np.float32)
    numerics = np.random.randn(n, 5).astype(np.float32)
    x = torch.tensor(np.column_stack([cat_codes, numerics]), dtype=torch.float32)
    out = emb(x)
    assert out.shape == (n, emb.out_features)
    assert torch.isfinite(out).all()
    # The trailing 5 numeric columns must be passed through UNCHANGED (the last embed-block boundary onward).
    passed = out[:, 12:]
    assert torch.allclose(passed, x[:, 2:], atol=0.0)


def test_forward_cats_only_no_numerics():
    """Forward cats only no numerics."""
    cards = [5]
    emb = CategoricalEmbedding(cardinalities=cards, embed_dim=8)
    emb.set_num_numeric(0)
    assert emb.out_features == 8
    n = 7
    x = torch.tensor(np.random.randint(0, 5, size=(n, 1)).astype(np.float32), dtype=torch.float32)
    out = emb(x)
    assert out.shape == (n, 8)


def test_heuristic_embed_dim_used_when_none():
    """Heuristic embed dim used when none."""
    cards = [10, 100]
    emb = CategoricalEmbedding(cardinalities=cards, embed_dim=None)
    assert emb.embed_dims == [default_embed_dim(10), default_embed_dim(100)]
    emb.set_num_numeric(2)
    assert emb.out_features == sum(emb.embed_dims) + 2


def test_unseen_and_overflow_code_clamps_to_reserved_row_no_indexerror():
    # card=4 means valid codes 0..3 and a reserved unknown row at index 4 (table has 5 rows). Codes 4 (reserved), 99 (overflow), -1
    # (negative) must all clamp into [0, card] without raising.
    """Unseen and overflow code clamps to reserved row no indexerror."""
    emb = CategoricalEmbedding(cardinalities=[4], embed_dim=3)
    emb.set_num_numeric(0)
    x = torch.tensor([[0.0], [3.0], [4.0], [99.0], [-1.0]], dtype=torch.float32)
    out = emb(x)  # must not raise IndexError
    assert out.shape == (5, 3)
    # The reserved row (index 4) is what 4 / 99 map to; -1 clamps to 0.
    reserved_vec = emb.embeddings[0](torch.tensor([4]))
    assert torch.allclose(out[2], reserved_vec[0])
    assert torch.allclose(out[3], reserved_vec[0])
    row0 = emb.embeddings[0](torch.tensor([0]))
    assert torch.allclose(out[4], row0[0])


def test_pickle_round_trip_preserves_weights():
    """Pickle round trip preserves weights."""
    emb = CategoricalEmbedding(cardinalities=[6, 4], embed_dim=5)
    emb.set_num_numeric(3)
    # Perturb the weights away from init so the round-trip check is meaningful.
    with torch.no_grad():
        for e in emb.embeddings:
            e.weight.add_(torch.randn_like(e.weight))
    n = 8
    cat = np.column_stack([np.random.randint(0, 6, n), np.random.randint(0, 4, n)]).astype(np.float32)
    num = np.random.randn(n, 3).astype(np.float32)
    x = torch.tensor(np.column_stack([cat, num]), dtype=torch.float32)
    emb.eval()
    out1 = emb(x)
    emb2 = pickle.loads(pickle.dumps(emb))  # nosec B301 -- round-trip of a locally-created, trusted object
    emb2.eval()
    out2 = emb2(x)
    assert torch.allclose(out1, out2, atol=0.0)
    for e1, e2 in zip(emb.embeddings, emb2.embeddings):
        assert torch.allclose(e1.weight, e2.weight, atol=0.0)


def test_invalid_args_raise():
    """Invalid args raise."""
    with pytest.raises(ValueError):
        CategoricalEmbedding(cardinalities=[])
    with pytest.raises(ValueError):
        CategoricalEmbedding(cardinalities=[0])
    with pytest.raises(ValueError):
        CategoricalEmbedding(cardinalities=[3], embed_dim=0)


def test_out_features_zero_numeric_before_set():
    # Before set_num_numeric, out_features counts only the cat block (num_numeric treated as 0).
    """Out features zero numeric before set."""
    emb = CategoricalEmbedding(cardinalities=[4, 4], embed_dim=2)
    assert emb.out_features == 4

"""keep_columns gives a Mode B batch exactly the columns of the Mode A frame, whatever the batch would have deduped."""
import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.row_attention import compute_row_attention


def _data():
    """A training set and a query batch."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    return X, (X[:, 0] > 0).astype(np.float32), X[:1]


def test_one_row_batch_keeps_the_mode_a_columns():
    """Without keep_columns a 1-row batch has zero variance in every column and the dedupe collapses it to a single column; with it the width matches Mode A."""
    X, y, q = _data()
    kw = dict(seed=1, n_heads=3, head_dim=4, k=5, gpu_stage4=False, dedupe_threshold=0.9)
    a = compute_row_attention(X, y, None, KFold(3, shuffle=True, random_state=0), **kw)
    bare = compute_row_attention(X, y, q, KFold(3), **kw)
    pinned = compute_row_attention(X, y, q, KFold(3), keep_columns=a.columns, **kw)
    assert bare.width != a.width
    assert pinned.columns == a.columns


def test_unknown_keep_column_raises():
    """A name the call does not produce is an error, not a silently dropped column."""
    X, y, q = _data()
    with pytest.raises(ValueError, match="keep_columns"):
        compute_row_attention(X, y, q, KFold(3), seed=1, n_heads=2, head_dim=4, k=5, gpu_stage4=False, keep_columns=["nope"])


def test_residual_attention_mode_b_matches_the_deduped_mode_a_columns():
    """Mode A applies dedupe_threshold, Mode B never did; keep_columns makes Mode B return the Mode A columns."""
    from mlframe.feature_engineering.transformer.residual_attention import compute_residual_attention

    X, y, q = _data()
    kw = dict(seed=1, n_heads=3, head_dim=4, k=5, task="binary")
    a = compute_residual_attention(X, y, None, KFold(3, shuffle=True, random_state=0), dedupe_threshold=0.5, **kw)
    b = compute_residual_attention(X, y, q, KFold(3), keep_columns=a.columns, **kw)
    assert b.columns == a.columns and b.height == 1

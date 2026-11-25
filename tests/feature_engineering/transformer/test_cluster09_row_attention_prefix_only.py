"""Cluster 09 regression: the column-prefix rebrand touches only the leading prefix (audit 2026-11-20, finding 09-F42)."""

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.row_attention import compute_row_attention


def test_column_prefix_rebrands_leading_prefix_of_every_column():
    """Every output column starts with the caller's prefix and the default-prefix names map one-to-one (09-F42)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    splitter = KFold(3, shuffle=True, random_state=0)
    kwargs = dict(seed=5, n_heads=1, head_dim=4, k=5, gpu_stage4=False, dedupe_threshold=None)
    default = compute_row_attention(X, y, None, splitter, **kwargs)
    rebranded = compute_row_attention(X, y, None, splitter, column_prefix="pfx", **kwargs)
    assert rebranded.columns == [c.replace("attn_", "pfx_", 1) for c in default.columns]
    assert all(c.startswith("pfx_") for c in rebranded.columns)

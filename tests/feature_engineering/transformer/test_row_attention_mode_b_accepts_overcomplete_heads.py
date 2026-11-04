"""Mode B (query batch) must accept ``allow_overcomplete=True`` exactly as Mode A does."""
import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.row_attention import compute_row_attention


def test_head_dim_larger_than_the_input_width_works_in_mode_b():
    """head_dim 8 > d 4: Mode A passes with allow_overcomplete=True, and Mode B used to raise because the flag was not forwarded to the key bank."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    kw = dict(seed=1, n_heads=2, head_dim=8, k=5, allow_overcomplete=True, gpu_stage4=False)
    a = compute_row_attention(X, y, None, KFold(3, shuffle=True, random_state=0), **kw)
    b = compute_row_attention(X, y, X[:10], KFold(3), **kw)
    assert a.width == b.width and b.height == 10

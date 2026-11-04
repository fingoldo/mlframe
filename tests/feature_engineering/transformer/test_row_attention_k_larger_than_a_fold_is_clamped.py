"""A neighbour count above the training-fold size is clamped per fold instead of crashing inside the ANN query."""
import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.row_attention import compute_row_attention


def test_k_equal_to_n_train_works_with_five_folds():
    """k = 200 on 200 rows: every fold's index holds 160 rows, and the query asked for 200 neighbours."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    df = compute_row_attention(X, y, None, KFold(5, shuffle=True, random_state=0), seed=1, n_heads=2, head_dim=4, k=200, gpu_stage4=False)
    assert df.height == 200 and np.isfinite(df.to_numpy()).all()

"""local_linear fits its ridge on the Euclidean nearest neighbours, not on the rows in the same direction from the centre."""
import numpy as np

from mlframe.feature_engineering.transformer.local_linear import compute_local_linear_attention


def test_local_slope_of_a_quadratic_is_the_slope_at_the_query_point():
    """y = x0^2 on N(0, I): at the query (0.2, 0) the local slope in x0 is 0.4. A cosine index treats (3, 0) and (0.05, 0) as neighbours and gave about 2.5."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 2)).astype(np.float32)
    y = (X[:, 0] ** 2).astype(np.float32)
    q = np.array([[0.2, 0.0]], dtype=np.float32)
    out = compute_local_linear_attention(X, y, q, None, seed=1, k=40, standardize=False).to_numpy()
    assert abs(float(out[0, 1]) - 0.4) < 0.6, out

"""The NCA projection is learned in the space it is applied to (audit 2026-11-20, finding 08-F1)."""
import numpy as np

from mlframe.feature_engineering.transformer._projection import build_nca_projection


def test_projection_separates_the_classes_in_the_space_it_is_applied_to():
    """Fitted on rows X and applied to the same rows, the first head separates the classes; a projection learned in a different
    standardised space did not."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    X = np.column_stack([y + 0.3 * rng.standard_normal(n), 40.0 * rng.standard_normal(n), 5.0 * rng.standard_normal(n)])
    # The caller standardises every column before the projection is built and applied (see _oof.py); mirror that here.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    projections = build_nca_projection(X=X, y=y, n_heads=1, head_dim=1, seed=0, dtype=np.float64)
    score = (X @ projections[0][:, 0]) if projections.ndim == 3 else (X @ projections[:, 0])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, score) > 0.85 or roc_auc_score(y, -score) > 0.85

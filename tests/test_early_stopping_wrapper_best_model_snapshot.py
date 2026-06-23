"""Regression test for EarlyStoppingWrapper best-model snapshotting.

``EarlyStoppingWrapper.fit`` iterates ``partial_fit`` and records the best
validation score. Pre-fix it stored ``self.best_model_ = self.base_model`` -- a
reference to the LIVE model that keeps mutating on later partial_fit calls. So
``best_model_`` ended up holding the FINAL (often degraded) weights, not the
best ones: ``best_score_`` and ``best_model_``'s actual val score diverged,
silently defeating early stopping. The fix deep-copies the model at the best
iteration so ``best_model_`` matches the recorded ``best_score_``.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _fit():
    rng = np.random.RandomState(1)
    X = rng.randn(120, 6)
    y = (X[:, 0] + 0.3 * rng.randn(120) > 0).astype(int)
    # Large eta0 + no tolerance so weights overshoot AFTER the best iter,
    # degrading val accuracy -- exposes the live-reference bug.
    base = SGDClassifier(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=5.0)
    # Seeded so the shuffled/stratified val fold the wrapper held out can be reproduced exactly here.
    m = EarlyStoppingWrapper(base, patience=50, max_iter=60, random_state=0)
    m.fit(X, y)
    # Reconstruct the SAME validation fold the wrapper used (deterministic given random_state),
    # rather than assuming a last-rows holdout.
    _, Xv, _, yv = m._split(X, y)
    return m, Xv, yv


def test_best_model_is_a_snapshot_not_a_live_reference():
    m, _, _ = _fit()
    # Pre-fix this was True (same object); the model kept mutating after best.
    assert m.best_model_ is not m.base_model


def test_best_model_val_score_matches_recorded_best_score():
    m, Xv, yv = _fit()
    realized = accuracy_score(yv, m.best_model_.predict(Xv))
    # Pre-fix best_model_ pointed at the degraded final model: realized (0.833)
    # was well below the recorded best_score_ (1.0). Post-fix they agree.
    assert realized >= m.best_score_ - 1e-9, (
        f"best_model_ val score {realized} should match best_score_ {m.best_score_}"
    )
    # And the live model genuinely degraded -- proves the snapshot mattered.
    degraded = accuracy_score(yv, m.base_model.predict(Xv))
    assert realized > degraded

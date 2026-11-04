"""The regression flip threshold of adversarial_flip is the 10th percentile of OUT-OF-FOLD absolute residuals."""
import numpy as np

from mlframe.feature_engineering.transformer import adversarial_flip as af


def test_regression_flip_threshold_is_not_the_in_sample_residual_decile(monkeypatch):
    """On a pure-noise target the in-sample decile (about 0.10 here) sits below the out-of-fold one (about 0.15)."""
    seen = []
    real = af._stacked_flip_update

    def spy(model, task, Xq, feat_std, combos, orig_class, pred_orig, thr, flip_dists):
        """Record the threshold handed to the real update."""
        seen.append(float(thr))
        return real(model, task, Xq, feat_std, combos, orig_class, pred_orig, thr, flip_dists)

    monkeypatch.setattr(af, "_stacked_flip_update", spy)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    y = rng.standard_normal(300).astype(np.float32)
    af.compute_adversarial_flip_features(X, y, X[:20], seed=1, task="regression")
    assert seen and seen[0] > 0.125, seen

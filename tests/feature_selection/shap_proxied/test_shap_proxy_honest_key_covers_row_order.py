"""The honest-loss cache key changes when interior rows are reordered.

The key used a head/tail/column-sum summary of the arrays. Two training frames that differ only in the order of interior rows then shared a
key, and the cache replayed a loss computed on different data.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _shap_proxy_loss as loss


class _Template:
    """A minimal estimator stand-in exposing the parameters the key reads."""

    def get_params(self, deep=False):
        """No hyper-parameters, so the key depends only on the data."""
        return {}


def test_honest_loss_key_differs_when_only_interior_rows_are_reordered():
    """Swapping two rows away from the head and tail of an integer frame changes the key."""
    n = 400
    X = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    y = np.arange(n, dtype=np.int64)
    X_perm = X.copy()
    y_perm = y.copy()
    X_perm[[100, 101]] = X[[101, 100]]
    y_perm[[100, 101]] = y[[101, 100]]
    kwargs = dict(model_template=_Template(), X_ev=X[:50], y_ev=y[:50], idx=[0, 1, 2], classification=False, metric="rmse", seed=0,
                  n_estimators_cap=10, template_id="t")
    key_a = loss._build_honest_loss_disk_key(X_tr=X, y_tr=y, **kwargs)
    key_b = loss._build_honest_loss_disk_key(X_tr=X_perm, y_tr=y_perm, **kwargs)
    assert key_a is not None and key_b is not None
    assert key_a != key_b

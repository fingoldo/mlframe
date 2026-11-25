"""Cluster 41 regression: predict kwargs are dropped only when the regressor's predict cannot accept them (audit 2026-11-20, finding 41-F43).

The scaled-target predict used to retry without kwargs on any TypeError, which would hide a genuine TypeError raised inside a regressor that does
accept them. The decision now comes from the predict signature.
"""

from mlframe.training.targets._ttr_eval_set_scaling import _predict_accepts


class _LinearLike:
    """Mimics sklearn linear models: predict takes only X."""

    def predict(self, X):
        """Return X unchanged."""
        return X


class _Accepts:
    """Regressor whose predict takes extra keyword parameters."""

    def predict(self, X, **kwargs):
        """Return X unchanged."""
        return X


def test_predict_without_kwargs_is_not_given_them():
    """A predict that takes only X is called without the params (41-F43)."""
    assert _predict_accepts(_LinearLike().predict, {"return_std": True}) is False


def test_predict_with_kwargs_receives_them():
    """A predict that accepts the params is called with them (41-F43)."""
    assert _predict_accepts(_Accepts().predict, {"return_std": True}) is True


def test_no_params_is_always_accepted():
    """With no params the call is unchanged (41-F43)."""
    assert _predict_accepts(_LinearLike().predict, {}) is True

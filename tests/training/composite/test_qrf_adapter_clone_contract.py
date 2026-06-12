"""sklearn-contract regression for ``_QuantileForestAdapter``.

The adapter previously delegated ``get_params``/``set_params`` to ``self.model``,
so it exposed the inner forest's params (``n_estimators`` ...) instead of its own
``model`` arg. ``sklearn.clone`` then reconstructed it as
``_QuantileForestAdapter(n_estimators=..., bootstrap=...)`` and raised ``TypeError``
because ``__init__`` only accepts ``model`` -- breaking any Pipeline / cross_val /
GridSearch that cloned a fitted ``CompositeQRFEstimator`` with a quantile-forest backend.
"""

from sklearn import clone
from sklearn.ensemble import RandomForestRegressor

from mlframe.training.composite.qrf import _QuantileForestAdapter


def test_quantile_forest_adapter_clone_roundtrips_on_model_param():
    adapter = _QuantileForestAdapter(model=RandomForestRegressor(n_estimators=7))

    assert list(adapter.get_params(deep=False)) == ["model"]

    cloned = clone(adapter)
    assert isinstance(cloned, _QuantileForestAdapter)
    assert cloned.model is not adapter.model
    assert cloned.model.n_estimators == 7

    adapter.set_params(**adapter.get_params(deep=False))
    assert adapter.get_params(deep=False)["model"] is adapter.model

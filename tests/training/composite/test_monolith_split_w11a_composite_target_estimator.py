"""Monolith-split sensor for ``mlframe.training.composite.estimator._estimator``.

Carve pattern: method rebinding via class-attribute assignment at parent bottom (mirror of RFECV.fit). Identity must be preserved so downstream isinstance / hasattr / sklearn introspection keeps working.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training.composite.estimator import _estimator
    return _estimator


@pytest.fixture(scope="module")
def siblings():
    from mlframe.training.composite.estimator import (
        _predict,
        _update,
        _utils,
    )
    return {
        "predict": _predict,
        "update": _update,
        "utils": _utils,
    }


def test_predict_method_identity(parent_module, siblings):
    cls = parent_module.CompositeTargetEstimator
    p = siblings["predict"]
    # ``_predict_unclipped`` / ``predict_pre_clip`` are class-attr bound to the carved sibling (identity preserved).
    assert cls._predict_unclipped is p._predict_unclipped
    assert cls.predict_pre_clip is p.predict_pre_clip


def test_predict_stubs_delegate_to_sibling(parent_module, siblings):
    """``predict`` / ``predict_quantile`` are in-body stubs (discoverable on the class) that delegate to the carved sibling.

    They are NOT class-attr bound, so ``cls.predict is sibling.predict`` is intentionally False; assert instead that they
    are real methods declared on the class and that calling them routes through the sibling implementation.
    """
    cls = parent_module.CompositeTargetEstimator
    p = siblings["predict"]
    for name in ("predict", "predict_quantile"):
        assert name in vars(cls), f"{name} must be declared on the class body, not inherited"
        assert callable(getattr(cls, name))

    import pandas as pd
    from sklearn.linear_model import LinearRegression

    calls = {"predict": 0}
    real_predict = p.predict

    def spy(self, X):
        calls["predict"] += 1
        return real_predict(self, X)

    rng = np.random.default_rng(0)
    n = 50
    base = rng.normal(10.0, 2.0, size=n)
    y = base + rng.normal(0.0, 0.5, size=n)
    X = pd.DataFrame({"b": base})
    est = cls(base_estimator=LinearRegression(), transform_name="diff", base_column="b")
    est.fit(X, y)

    monkey = p.predict
    p.predict = spy
    try:
        out = est.predict(X)
    finally:
        p.predict = monkey
    assert calls["predict"] == 1, "in-body predict stub must delegate to sibling _pred.predict"
    assert out.shape == (n,)


def test_update_method_identity(parent_module, siblings):
    cls = parent_module.CompositeTargetEstimator
    u = siblings["update"]
    assert cls.update is u.update
    assert cls.get_buffer_state is u.get_buffer_state


def test_utils_require_fitted_identity(parent_module, siblings):
    cls = parent_module.CompositeTargetEstimator
    util = siblings["utils"]
    assert cls._require_fitted is util._require_fitted
    assert cls.get_booster is util.get_booster


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 1000, f"facade is {n_lines} LOC, expected < 1000"


def test_isinstance_preserved(parent_module):
    from sklearn.linear_model import LinearRegression
    cls = parent_module.CompositeTargetEstimator
    inst = cls(base_estimator=LinearRegression(), transform_name="diff", base_column="b")
    assert isinstance(inst, cls)
    # sklearn BaseEstimator inheritance
    from sklearn.base import BaseEstimator
    assert isinstance(inst, BaseEstimator)


def test_smoke_fit_predict_round_trip(parent_module):
    """Exercise the carved predict + fit + properties on a tiny synthetic. Per CLAUDE.md AST-audit rule, sensor must actually call the moved body, not just import."""
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(10.0, 2.0, size=n)
    noise = rng.normal(0.0, 0.5, size=n)
    y = base + 0.5 + noise
    X = pd.DataFrame({"b": base, "x1": rng.normal(0.0, 1.0, size=n)})

    cls = parent_module.CompositeTargetEstimator
    est = cls(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="b",
    )
    est.fit(X, y)
    y_hat = est.predict(X)
    assert y_hat.shape == (n,)
    assert np.isfinite(y_hat).all()
    y_hat_pre = est.predict_pre_clip(X)
    assert y_hat_pre.shape == (n,)
    # Carved utils property: coef_ delegated to inner LinearRegression
    assert est.coef_ is not None
    assert est.n_features_in_ == 2


def test_update_not_enabled_raises(parent_module):
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 80
    base = rng.normal(10.0, 2.0, size=n)
    y = base + rng.normal(0.0, 0.5, size=n)
    X = pd.DataFrame({"b": base})

    cls = parent_module.CompositeTargetEstimator
    est = cls(base_estimator=LinearRegression(), transform_name="linear_residual", base_column="b")
    est.fit(X, y)
    with pytest.raises(RuntimeError, match="online_refit_enabled is False"):
        est.update(y[:5], base[:5])

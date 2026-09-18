"""Regression: composite-target and ensemble entries must not collide on chart / ensemble names.

Composite-target entries are wrapped in CompositeTargetEstimator before ensembling and finalization, and ensemble
entries carry ``model=None``. Naming by ``type(entry.model).__name__`` therefore gave every composite model the name
"CompositeTargetEstimator" and every ensemble "NoneType": ensembles of cb/xgb/lgb were saved as
"EnsARITHM-CompositeTargetEstimatorCompositeTargetEstimator...", and one ``*_split_comparison`` file per target was
overwritten by each model in turn.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from mlframe.training._format import short_model_tag, unwrap_target_wrapper


class CatBoostRegressor:  # stand-in inner model (only the class name matters to the tagger)
    pass


class CompositeTargetEstimator:  # stand-in carrying the real wrapper's inner-model attribute
    def __init__(self, inner):
        self.estimator_ = inner


def test_short_model_tag_sees_through_the_composite_wrapper():
    """A wrapped CatBoost tags as cb, so an ensemble of wrapped members reads [cb+xgb+lgb]."""
    assert short_model_tag(CompositeTargetEstimator(CatBoostRegressor())) == "cb"


def test_short_model_tag_sees_through_transformed_target_regressor():
    """sklearn's TransformedTargetRegressor is a target-only wrapper too."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression

    X = np.arange(20, dtype=float).reshape(-1, 1)
    ttr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1).fit(X, X.ravel())
    assert short_model_tag(ttr) == "LinearRegression"


def test_unwrap_leaves_other_models_alone():
    """Meta-estimators that are models in their own right keep their own name."""
    from sklearn.ensemble import BaggingRegressor

    bag = BaggingRegressor()
    assert unwrap_target_wrapper(bag) is bag


def test_split_comparison_uses_each_entry_chart_prefix(tmp_path, monkeypatch):
    """Each entry's panel lands at its own chart prefix; ensembles are named by prefix, not "NoneType"."""
    import mlframe.reporting.diagnostics_dispatch as dispatch
    from mlframe.training.core._phase_finalize import _render_split_comparison_panels

    seen = []

    def spy(*, entry, target_type, plot_outputs, base_path, metrics_dict, model_name):
        seen.append((base_path, model_name))
        return True

    monkeypatch.setattr(dispatch, "render_split_comparison_from_suite", spy)
    d = os.path.join(str(tmp_path), "charts", "t", "m", "regression", "y-logY")
    entries = [
        SimpleNamespace(model=CompositeTargetEstimator(CatBoostRegressor()), plot_file=os.path.join(d, "recency__CatBoostRegressor")),
        SimpleNamespace(model=CompositeTargetEstimator(CatBoostRegressor()), plot_file=os.path.join(d, "CatBoostRegressor")),
        SimpleNamespace(model=None, plot_file=os.path.join(d, "EnsARITHM-cbxgblgb")),
        SimpleNamespace(model=None, plot_file=os.path.join(d, "EnsMEDIAN-cbxgblgb")),
    ]
    ctx = SimpleNamespace(
        data_dir=str(tmp_path), save_charts=True, verbose=0, metadata={}, target_name="t", model_name="m",
        reporting_config=SimpleNamespace(split_comparison_charts=True, plot_outputs="matplotlib[png]"),
        models={"regression": {"y-logY": entries}},
    )
    _render_split_comparison_panels(ctx)
    bases = [b for b, _ in seen]
    assert len(bases) == 4 and len(set(bases)) == 4, bases
    assert bases == [e.plot_file for e in entries]
    names = [n for _, n in seen]
    assert "NoneType" not in names and "CompositeTargetEstimator" not in names
    assert names[2] == "EnsARITHM-cbxgblgb"

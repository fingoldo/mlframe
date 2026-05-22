"""MBH (Model-Based Heuristic) optimizer construction for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s pre-while setup. Builds the
``MBHOptimizer`` instance with adaptive surrogate-model dispatch:

* Default plotting mode is ``No`` (avoids blocking pytest / headless
  runs on Qt event loops); user opts in via ``optimizer_plotting=``.
* Surrogate model is auto-picked by evaluation budget:
  budget <= mbh_adaptive_threshold (default 30) -> ``ETR n_estimators=20``
  budget 31..100                                -> ``CBQ iterations=50``
  budget > 100                                  -> ``CBQ iterations=150``
* User overrides via ``optimizer_config={"model_name":..., "model_params": {...}}``.

When ``top_predictors_search_method != ModelBasedHeuristic``, returns
None (caller uses the non-MBH search path).

Re-imported at the parent's module bottom so historical
``from ._rfecv_fit import _build_mbh_optimizer`` keeps resolving
transparently.
"""
from __future__ import annotations

import logging

import numpy as np

from mlframe.models.optimization import (
    CandidateSamplingMethod,
    MBHOptimizer,
    OptimizationDirection,
    OptimizationProgressPlotting,
)

from ._enums import OptimumSearch

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def _build_mbh_optimizer(self, *, original_features, max_refits, top_predictors_search_method):
    """Return an ``MBHOptimizer`` instance, or None if MBH search isn't selected."""
    if top_predictors_search_method != OptimumSearch.ModelBasedHeuristic:
        return None

    # Default plotting mode is 'No': OnScoreImprovement calls plt.show() inside the optimizer on every score improvement, which
    # blocks pytest / headless runs indefinitely (Qt event loop). Users must opt in explicitly.
    _plotting_map = {
        "No": OptimizationProgressPlotting.No,
        "Final": OptimizationProgressPlotting.Final,
        "OnScoreImprovement": OptimizationProgressPlotting.OnScoreImprovement,
        "Regular": OptimizationProgressPlotting.Regular,
    }
    if self.optimizer_plotting is None:
        plotting_mode = OptimizationProgressPlotting.No
    else:
        plotting_mode = _plotting_map.get(
            self.optimizer_plotting, OptimizationProgressPlotting.No
        )

    # Adaptive MBH surrogate. CatBoost has a fixed ~500ms per-fit overhead (Python<->C++ marshalling, roughly independent of
    # n_estimators), which dominates wall-clock on tiny RFECV problems where the outer estimator fits in <10ms. Switch to a sklearn
    # ExtraTreesRegressor surrogate when the evaluation budget is small: ETR n_estimators=20 fits in ~20ms with equivalent quality
    # on the 1D score curve. Quantile-style uncertainty via per-tree-prediction std (Breiman 2001 OOB variance estimate).
    #
    # Decision tree:
    #   evaluation budget <= 30:  ETR n_estimators=20
    #   31..100:                  CatBoost iterations=50
    #   >100:                     CatBoost iterations=150
    #
    # Users override via ``optimizer_config={"model_name":..., "model_params": {...}}``.
    _search_space_size = (
        min(self.max_nfeatures, len(original_features)) + 1
        if self.max_nfeatures
        else len(original_features) + 1
    )
    _max_evals_budget = (
        min(max_refits, _search_space_size)
        if max_refits
        else _search_space_size
    )
    _user_cfg = dict(self.optimizer_config) if self.optimizer_config else {}
    _user_model_name = _user_cfg.pop("model_name", None)
    _user_model_params = dict(_user_cfg.pop("model_params", {}) or {})
    # Adaptive surrogate threshold is operator-tunable: 30 was the historical hardcoded crossover, but very-cheap outer estimators benefit from higher cutoffs (CB overhead still dominates at budgets in the 30..60 range) while heavy outer estimators may prefer the noisier ETR even lower.
    _mbh_adaptive_threshold = getattr(self, "mbh_adaptive_threshold", 30)
    if _user_model_name is None:
        if _max_evals_budget <= _mbh_adaptive_threshold:
            _auto_model_name = "ETR"
        else:
            _auto_model_name = "CBQ"
            if "iterations" not in _user_model_params:
                _user_model_params["iterations"] = 50 if _max_evals_budget <= 100 else 150
        _model_name = _auto_model_name
    else:
        _model_name = _user_model_name
        # Only fill iterations default when the user picked a CatBoost-family surrogate.
        if _model_name in ("CBQ", "CB") and "iterations" not in _user_model_params:
            _user_model_params["iterations"] = 50 if _max_evals_budget <= 100 else 150
    _mbh_kwargs = dict(
        search_space=(
            np.array(np.arange(min(self.max_nfeatures, len(original_features)) + 1).tolist() + [len(original_features)])
            if self.max_nfeatures
            else np.arange(len(original_features) + 1)
        ),
        direction=OptimizationDirection.Maximize,
        init_sampling_method=CandidateSamplingMethod.Equidistant,
        init_evaluate_ascending=False,
        init_evaluate_descending=True,
        plotting=plotting_mode,
        seeded_inputs=[min(2, len(original_features))],
        model_name=_model_name,
        model_params=_user_model_params,
    )
    # Apply the rest of optimizer_config last so user's explicit kwargs (plotting=..., direction=..., etc.) override our defaults.
    _mbh_kwargs.update(_user_cfg)
    return MBHOptimizer(**_mbh_kwargs)

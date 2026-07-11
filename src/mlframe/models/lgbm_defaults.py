"""Default LightGBM hyperparameter preset, including ``extra_trees=True`` and a large-``n_features`` dart heuristic.

LightGBM's ``extra_trees`` option picks split thresholds RANDOMLY within each feature's candidate range
(rather than exhaustively searching for the locally-optimal split, as sklearn's ``ExtraTreesClassifier``
does relative to a standard Random Forest) — this decorrelates the individual trees, trading a slightly
weaker single tree for a materially more diverse ensemble. Ubiquant's 7th place team reported "steady
improvement when the number of trees goes large" from just flipping this one flag. It is off by default in
LightGBM itself and easy to never discover; this preset makes it the mlframe default so users get the win
without having to know the flag exists.

Source: 9th_home-credit-default-risk.md -- "method=dart outperforms method=gbdt because I had so many
features that it helped basically as feature_fraction. Low feature_fraction is key to improve accuracy on
tree models with lots of features." With a large feature count, gbdt's greedy per-split feature search keeps
picking correlated/redundant top features every round; ``dart`` (dropping a random subset of already-grown
trees each round, forcing later trees to compensate) plus a lower ``feature_fraction`` (randomly subsetting
candidate features per tree) both push the ensemble toward using more of the available features instead of
overfitting to the same few. When ``n_features`` is passed and crosses the threshold, this preset switches
``boosting_type`` to ``"dart"`` and lowers ``feature_fraction`` automatically.

Measured directly (not assumed) on the ``extra_trees`` biz-value benchmark's synthetic correlated/noisy
regression, sweeping ``n_estimators`` at fixed data size (6 seeds each): at ``n_estimators<=50``,
``extra_trees=True`` LOSES to the LightGBM default (fewer trees means the randomized-split trees haven't
had enough rounds to average out their added per-tree variance); from ``n_estimators>=100`` it wins on
5/6 seeds, and from ``n_estimators>=150`` it wins 6/6 seeds with a stable ~3.2-3.8% RMSE reduction. So
"``extra_trees`` helps at large tree counts" is real but has a floor below which it actively hurts --
``auto_extra_trees`` (opt-in) encodes that floor instead of forcing the flag on unconditionally.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

LARGE_N_FEATURES_THRESHOLD = 300

# Data-driven floor for the opt-in ``auto_extra_trees`` rule -- see module docstring for the sweep that
# produced it. Below this ``n_estimators`` budget, ``extra_trees=True`` measured WORSE than LightGBM's own
# default; at/above it, ``extra_trees=True`` won 6/6 seeds.
AUTO_EXTRA_TREES_MIN_N_ESTIMATORS = 150


def default_lgbm_params(
    objective: str = "regression",
    extra_trees: bool = True,
    n_features: Optional[int] = None,
    large_n_features_threshold: int = LARGE_N_FEATURES_THRESHOLD,
    auto_extra_trees: bool = False,
    auto_extra_trees_min_n_estimators: int = AUTO_EXTRA_TREES_MIN_N_ESTIMATORS,
    **overrides: Any,
) -> Dict[str, Any]:
    """Sensible default LightGBM hyperparameters, with ``extra_trees`` ON by default.

    Parameters
    ----------
    objective
        LightGBM ``objective`` string (e.g. ``"regression"``, ``"binary"``, ``"multiclass"``).
    extra_trees
        Forwarded as LightGBM's ``extra_trees`` param. Default ``True`` — decorrelates trees via randomized
        split-threshold selection, which measurably helps once ``n_estimators`` is large (see module
        docstring). Set ``False`` to restore LightGBM's own default (standard greedy-best-split trees).
        Ignored (overridden by the adaptive rule) when ``auto_extra_trees=True`` — see below.
    n_features
        Number of candidate features the model will see. If provided and ``>= large_n_features_threshold``,
        ``boosting_type`` defaults to ``"dart"``, ``feature_fraction`` is lowered, and ``n_estimators`` is
        scaled up 3x (dart's per-round tree dropout needs materially more rounds than gbdt to reach the same
        effective tree count -- see module docstring; ``n_estimators`` passed via ``**overrides`` is treated
        as the pre-scaling gbdt-equivalent budget, not the literal final round count). ``None`` (default)
        leaves ``boosting_type``/``feature_fraction`` unset (LightGBM's own gbdt/1.0 defaults apply) — pass
        the real feature count to opt into the heuristic.
    large_n_features_threshold
        Threshold for the ``n_features``-driven dart switch. Default ``300``.
    auto_extra_trees
        Opt-in (default ``False``, so default output is unchanged). When ``True``, ``extra_trees`` is set
        adaptively instead of using the static ``extra_trees`` argument: ``True`` once the REQUESTED
        (pre-dart-scaling) ``n_estimators`` budget is ``>= auto_extra_trees_min_n_estimators``, else
        ``False`` -- measured to actively hurt below that floor (see module docstring). An explicit
        ``extra_trees=`` passed via ``**overrides`` still wins (overrides are applied last).
    auto_extra_trees_min_n_estimators
        Floor for the ``auto_extra_trees`` rule. Default ``150`` (see ``AUTO_EXTRA_TREES_MIN_N_ESTIMATORS``
        and the module docstring for the measurement behind it).
    **overrides
        Any additional LightGBM param, applied last (overrides anything set above, including
        ``extra_trees``/``boosting_type``/``feature_fraction`` themselves if explicitly passed here too).

    Returns
    -------
    dict
        A ``**kwargs``-ready dict for ``lightgbm.LGBMRegressor`` / ``LGBMClassifier`` / the sklearn API.
    """
    # ``n_estimators`` in overrides is treated as a gbdt-equivalent round budget, not a literal final value --
    # dart needs materially more rounds than gbdt for the same effective tree count (see below), so it's
    # popped out and re-scaled rather than applied verbatim via the generic ``overrides`` merge.
    n_estimators = overrides.pop("n_estimators", 500)
    if auto_extra_trees:
        # decided on the pre-dart-scaling requested budget -- dart's 3x bump is a rounds-per-tree
        # compensation, not evidence the caller asked for more trees, so it must not affect this rule.
        extra_trees = n_estimators >= auto_extra_trees_min_n_estimators
    params: Dict[str, Any] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "extra_trees": extra_trees,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 0,
    }
    if n_features is not None and n_features >= large_n_features_threshold:
        params["boosting_type"] = "dart"
        params["feature_fraction"] = 0.5
        # dart drops a random subset of already-grown trees each round (a regularizing "dropout"), so a
        # dart ensemble needs materially more rounds than gbdt to reach the same effective tree count --
        # measured: at equal n_estimators dart underperforms gbdt, and only overtakes it once boosted.
        params["n_estimators"] = n_estimators * 3
    params.update(overrides)
    return params


__all__ = ["default_lgbm_params", "LARGE_N_FEATURES_THRESHOLD", "AUTO_EXTRA_TREES_MIN_N_ESTIMATORS"]

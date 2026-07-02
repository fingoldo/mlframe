"""Distribution-driven composite-estimator recommendation (Workstream E3).

The target-distribution analyzer already detects pathologies (heavy-tail, multi-modal, skew, ...). This
maps that verdict to the composite estimator best suited to it, so the suite can recommend (and, in a
follow-up, auto-dispatch) the right specialised estimator instead of always using the default regressor.

Heavy-tail / skew -> TailCompositeEstimator (GPD tail). Multi-modal -> CompositeDistributionEstimator
(full predictive distribution / CRPS). Returns a recommendation dict or None when the default fits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional


def recommend_composite_estimator(pathologies: Sequence[str]) -> Optional[dict[str, Any]]:
    """Map analyzer pathology strings to a recommended composite estimator (or None for the default).

    ``pathologies`` are the prefixed strings from the target-distribution analyzer (e.g.
    ``"heavy_tail(excess_kurt=12.3)"``, ``"multi_modal_target(peaks=3, ...)"``). Matching is by prefix.
    Returns ``{"estimator": <class name>, "module": <import path>, "reason": <pathology>}`` for the FIRST
    matching pathology by priority (heavy-tail/skew before multi-modal), else None.
    """
    pats = [str(p) for p in (pathologies or [])]

    def _first(prefixes):
        for p in pats:
            if any(p.startswith(pre) for pre in prefixes):
                return p
        return None

    heavy = _first(("heavy_tail", "skewed_target"))
    if heavy is not None:
        return {
            "estimator": "TailCompositeEstimator",
            "module": "mlframe.training.composite.extremes",
            "reason": heavy,
        }
    multimodal = _first(("multi_modal_target", "multimodal"))
    if multimodal is not None:
        return {
            "estimator": "CompositeDistributionEstimator",
            "module": "mlframe.training.composite.distributional",
            "reason": multimodal,
        }
    return None


def instantiate_recommended_estimator(recommendation: Optional[dict[str, Any]], **kwargs: Any) -> Optional[Any]:
    """Construct the recommended composite estimator from a ``recommend_composite_estimator`` dict (or None).

    Imports ``recommendation["module"].recommendation["estimator"]`` and instantiates it with ``kwargs``.
    Returns ``None`` when ``recommendation`` is None so the caller keeps its default estimator.

    ``TailCompositeEstimator`` / ``CompositeDistributionEstimator`` wrap a ``CompositeTargetEstimator`` and REQUIRE a
    ``base_column`` / ``base_columns`` (``.fit(X, y)`` without one raises) plus a ``base_estimator`` -- pass both via
    ``kwargs``. ``maybe_inject_distribution_driven_estimator`` is the suite-level integration: it auto-picks the
    base column (max |corr| to y) and a light GBM base, appends the instance to ``mlframe_models``, and extends the
    suite strategy/sort maps so the per-target loop actually TRAINS it (the generic estimator-instance path added to
    ``configure_training_params`` + the entry/``id()`` keying in ``_phase_train_one_target_body`` make this work).
    """
    if not recommendation:
        return None
    import importlib

    mod = importlib.import_module(recommendation["module"])
    cls = getattr(mod, recommendation["estimator"])
    return cls(**kwargs)


def _pick_base_column(train_df: Any, y: Any) -> Optional[str]:
    """Pick the numeric feature with the largest |Pearson corr| to ``y`` (train rows) as the composite base.

    Returns the column name, or None when no usable numeric column exists. Works on a pandas or polars frame;
    only columns whose values are finite-variant numeric are considered. This is the documented E3 heuristic --
    the composite estimator needs a ``base_column`` and full composite discovery has not run at suite-injection time.

    Cost: a one-time O(n_features * n_rows) corr pass per suite call (not a hot loop); cProfile shows no actionable
    hotspot. On very large frames the gated opt-in caller pays this once; ``np.corrcoef`` per column is vectorised.
    """
    import numpy as np

    _cols_attr = getattr(train_df, "columns", None)
    cols = list(_cols_attr) if _cols_attr is not None else []
    if not cols:
        return None
    y_arr = np.asarray(getattr(y, "to_numpy", lambda: y)(), dtype=np.float64).reshape(-1)
    if y_arr.size == 0 or not np.isfinite(y_arr).any() or float(np.nanstd(y_arr)) == 0.0:
        return None

    def _col_values(name):
        col = train_df[name]
        arr = getattr(col, "to_numpy", lambda: col)()
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    best_name, best_abs = None, -1.0
    for name in cols:
        try:
            x = _col_values(name)
        except (TypeError, ValueError):
            continue
        if x.shape[0] != y_arr.shape[0]:
            continue
        mask = np.isfinite(x) & np.isfinite(y_arr)
        if int(mask.sum()) < 3:
            continue
        xs, ys = x[mask], y_arr[mask]
        if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
            continue
        r = float(np.corrcoef(xs, ys)[0, 1])
        if np.isfinite(r) and abs(r) > best_abs:
            best_abs, best_name = abs(r), name
    return best_name


def _default_base_estimator():
    """A light GBM base estimator for the distribution-driven composite (LGBM preferred, HGB fallback)."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, num_leaves=31, verbose=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        # random_state pinned: HistGB early_stopping="auto" draws a random train/val split at n>10000,
        # which would otherwise make the default composite base nondeterministic run-to-run.
        return HistGradientBoostingRegressor(max_iter=200, random_state=0)


def maybe_inject_distribution_driven_estimator(
    ctx: Any,
    metadata: dict,
    mlframe_models: list,
    target_by_type: Any,
    train_idx: Any,
    train_df: Any,
    behavior_config: Any,
) -> list:
    """E3: append the analyzer-recommended composite estimator to ``mlframe_models`` so it actually trains.

    Gated by ``behavior_config.distribution_driven_estimator`` (default OFF). Reads the analyzer verdict from
    ``metadata["target_distribution_report"]["pathologies"]``; if it recommends a TailComposite / CompositeDistribution
    estimator AND a regression target with a usable base column exists, builds the estimator (auto base_column +
    light GBM base), appends it to the returned model list, and extends ``ctx.strategy_by_model`` /
    ``ctx.sorted_mlframe_models`` to mirror ``setup_configuration`` (so the per-target loop trains it).

    Returns the (possibly-extended) ``mlframe_models`` list. A no-op (returns the input unchanged) when the flag is
    off, no recommendation fires, there is no regression target, or no base column can be picked.
    """
    if not getattr(behavior_config, "distribution_driven_estimator", False):
        return mlframe_models

    report = (metadata or {}).get("target_distribution_report") or {}
    rec = recommend_composite_estimator(report.get("pathologies") or [])
    if rec is None:
        return mlframe_models

    from .._configs_base import TargetTypes

    reg_targets = (target_by_type or {}).get(TargetTypes.REGRESSION) or {}
    if not reg_targets:
        return mlframe_models
    _first_name, _first_y = next(iter(reg_targets.items()))

    import numpy as np

    y_full = np.asarray(getattr(_first_y, "to_numpy", lambda: _first_y)()).reshape(-1)
    if train_idx is not None:
        try:
            y_train = y_full[np.asarray(train_idx)]
        except (IndexError, TypeError):
            y_train = y_full
    else:
        y_train = y_full

    base_column = _pick_base_column(train_df, y_train)
    if not base_column:
        return mlframe_models

    estimator = instantiate_recommended_estimator(
        rec, base_estimator=_default_base_estimator(), base_column=base_column
    )
    if estimator is None:
        return mlframe_models

    new_models = list(mlframe_models) + [estimator]

    # Mirror setup_configuration: extend the strategy map + tier-sort so the per-target loop trains the new entry.
    from ..strategies import get_strategy as _get_strategy
    from ..models import is_neural_model as _is_neural_model

    sbm = getattr(ctx, "strategy_by_model", None)
    if sbm is not None:
        sbm[id(estimator)] = _get_strategy(estimator)

        def _tier(m):
            strat = sbm.get(id(m))
            if strat is None:
                strat = _get_strategy(m)
                sbm[id(m)] = strat
            return tuple(-int(t) for t in strat.feature_tier())

        ctx.sorted_mlframe_models = sorted(new_models, key=lambda m: (_is_neural_model(m), _tier(m)))
    # The per-target setup (``_setup_per_target_mlframe_models``) reads ``ctx.mlframe_models`` to build
    # ``models_params`` via ``select_target`` -> ``configure_training_params``; keep it in sync so the
    # injected estimator gets a params entry and is not skipped as "not known".
    ctx.mlframe_models = new_models

    metadata.setdefault("distribution_driven_estimator", {})[rec["estimator"]] = {
        "reason": rec["reason"],
        "base_column": base_column,
    }
    return new_models

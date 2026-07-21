"""Post-fit diagnostics + training-curve rendering for classification/regression reports.

Carved out of ``_reporting.py`` to keep that facade under the 1k-line ceiling. The functions here render the
additive diagnostic panels (PDP/ICE, SHAP, slice-finder, decision curve, model card, learning curve) and the
per-model train-vs-val iteration curves. The facade re-exports every public name below so existing
``from ...reporting._reporting import _render_post_fit_diagnostics`` imports keep resolving.

Parent helpers (``_unwrap_booster``, ``_canonicalize_split_names``, ``model_name_for_title``,
``display_estimator_name``) are imported from the partially-loaded parent: the facade's bottom re-export triggers
this module's load AFTER those helpers are defined at the parent top, so the partial-module lookup succeeds.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from ._reporting import (
    _canonicalize_split_names,
    _unwrap_booster,
    display_estimator_name,
    model_name_for_title,
)

logger = logging.getLogger(__name__)


def _extract_training_history(model: Any) -> tuple[dict | None, int | None]:
    """Extract ``{metric: {split: [...]}}`` history + the early-stopping iteration from a fitted booster.

    Handles lgb (``evals_result_``: ``{split: {metric: [...]}}``), xgb (``evals_result()``: same shape), and
    catboost (``get_evals_result()``: ``{split: {metric: [...]}}``). Returns ``(history_by_metric, es_iteration)``
    transposed to the metric-major shape ``compose_training_curve_figure`` expects, or ``(None, None)`` when the
    model carries no usable iteration history (non-boosting estimators, or boosters fit without an eval set).
    """
    est = _unwrap_booster(model)
    raw = None
    try:
        if hasattr(est, "evals_result_") and getattr(est, "evals_result_"):
            raw = getattr(est, "evals_result_")
        elif hasattr(est, "get_evals_result"):
            r = est.get_evals_result()
            raw = r if r else None
        elif hasattr(est, "evals_result") and callable(getattr(est, "evals_result")):
            r = est.evals_result()
            raw = r if r else None
    except Exception:
        logger.debug("training-curve: evals_result extraction raised; skipping curves for this model.", exc_info=True)
        return None, None
    if not raw:
        return None, None

    # raw is split-major ``{split: {metric: [...]}}``; transpose to metric-major ``{metric: {split: [...]}}`` and
    # canonicalise the split names so ``normalize_history`` keeps both curves. lightgbm names eval sets ``valid_0``,
    # ``valid_1``, ... in eval_set order (NOT recognised by the canonical alias set); by mlframe convention the FIRST
    # eval set is train and the LAST is the holdout, so map the lowest ``valid_N`` -> train and the highest -> val.
    split_names = list(raw.keys())
    canon_split = _canonicalize_split_names(split_names)
    by_metric: dict = {}
    try:
        for split, metrics_map in raw.items():
            if not hasattr(metrics_map, "items"):
                continue
            split_key = canon_split.get(split, str(split))
            for metric, series in metrics_map.items():
                if series is None or not hasattr(series, "__len__") or len(series) == 0:
                    continue
                by_metric.setdefault(str(metric), {}).setdefault(split_key, list(series))
    except Exception:
        logger.debug("training-curve: evals_result had an unexpected shape; skipping.", exc_info=True)
        return None, None
    if not by_metric:
        return None, None

    es_iteration = None
    for attr in ("best_iteration_", "best_iteration"):
        if hasattr(est, attr):
            try:
                bi = int(getattr(est, attr))
                if bi >= 0:
                    es_iteration = bi
                break
            except (TypeError, ValueError):
                pass
    return by_metric, es_iteration


def _render_training_curves(
    model: Any,
    *,
    model_name: str,
    plot_file: str,
    plot_outputs: str | None,
    plot_dpi: int | None,
    metrics: dict | None,
    reporting_config: Any,
) -> None:
    """Render per-model train-vs-val iteration curves when charts are saved AND the model carries boosting history.

    Default-ON via ``ReportingConfig.training_curves`` (no-op when off, when charts are not being saved to disk, or
    when the model exposes no iteration history). Failures are logged + swallowed -- the curve panel is additive.
    """
    if not (plot_file and plot_outputs):
        return
    if not getattr(reporting_config, "training_curves", True):
        return
    history, es_iteration = _extract_training_history(model)
    if not history:
        return
    try:
        from mlframe.reporting.charts.training_curve import compose_training_curve_figure
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save

        spec = compose_training_curve_figure(history, es_iteration=es_iteration, suptitle=f"{model_name} training curves")
        if plot_dpi is not None:
            from dataclasses import replace
            spec = replace(spec, dpi=plot_dpi)
        base = plot_file + "_training_curve"
        render_and_save(spec, parse_plot_output_dsl(plot_outputs), base)
        if isinstance(metrics, dict):
            _charts = metrics.setdefault("charts", {"saved": [], "failed": []})
            _charts.setdefault("saved", []).append("training_curve")
            _charts.setdefault("paths", []).append(base)
            # INV-56: optionally retain the PURE-DATA FigureSpec (no live matplotlib/plotly handle, so it stays
            # pickle-safe) when the caller wants to re-render or post-tweak the panel later.
            if getattr(reporting_config, "keep_figure_handles", False):
                metrics.setdefault("figure_specs", {})["training_curve"] = spec
    except Exception:
        logger.exception("training-curve render failed; continuing.")
        if isinstance(metrics, dict):
            _charts = metrics.setdefault("charts", {"saved": [], "failed": []})
            _charts.setdefault("failed", []).append("training_curve")


def _binary_positive_score(probs) -> np.ndarray | None:
    """Pull the positive-class probability column from a proba matrix / 1-D score, else None."""
    if probs is None:
        return None
    arr = np.asarray(probs)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    return None


def _ranked_feature_names(metrics, model, columns) -> tuple[list[str] | None, list[float] | None]:
    """Best-effort (names, importances) for PDP ranking: prefer the report's FI dict, else the model's native source."""
    names = list(columns) if columns is not None and len(columns) > 0 else None
    if not names:
        return None, None
    fi_dict = metrics.get("feature_importances") if isinstance(metrics, dict) else None
    if isinstance(fi_dict, dict):
        return names, [float(fi_dict.get(c, 0.0)) for c in names]
    native = getattr(model, "feature_importances_", None)
    if native is not None:
        # ``feature_importances_`` can be an unsized object (0-d array / scalar) or a 2-D per-class matrix on some
        # estimators (e.g. CatBoost with embedding features); ``len(...)`` raises on the 0-d case. Normalise to 1-D
        # and only use it when its length matches the column set.
        native_arr = np.atleast_1d(np.asarray(native))
        if native_arr.ndim == 1 and native_arr.shape[0] == len(names):
            return names, [float(v) for v in native_arr]
    return names, None


def _build_learning_curve(model, df, targets, columns, target_type, lc_cfg, metrics, metadata_target_name="learning_curve"):
    """Opt-in learning curve: refit a fresh clone of ``model`` on log-spaced train-size prefixes; store + render.

    Returns the ``FigureSpec`` panel (or None when skipped). Uses ``sklearn.clone`` for the estimator factory and the
    task-appropriate sklearn scorer. K full refits by construction -- only invoked when ``lc_cfg.enabled``.
    """
    if lc_cfg is None or not getattr(lc_cfg, "enabled", False):
        return None
    if model is None or df is None or not columns:
        return None
    try:
        from sklearn.base import clone
        from sklearn.metrics import get_scorer

        from mlframe.training.diagnostics import compute_learning_curve, learning_curve_panel

        tt = (target_type or "").lower()
        if "regress" in tt:
            scorer_name, higher_is_better = "r2", True
        else:
            scorer_name, higher_is_better = "roc_auc", True
        scorer = get_scorer(scorer_name)
        X = df[list(columns)] if hasattr(df, "__getitem__") else df
        base = clone(model)
        result = compute_learning_curve(
            lambda: clone(base), X, np.asarray(targets).ravel(),
            sizes=lc_cfg.sizes, scorer=scorer, holdout=lc_cfg.holdout, n_jobs=lc_cfg.n_jobs,
            parallel_backend=lc_cfg.parallel_backend,
            warm_start=lc_cfg.warm_start, random_state=lc_cfg.random_state,
            time_budget_s=lc_cfg.time_budget_s, score_repeats=lc_cfg.score_repeats,
            scorer_name=scorer_name, higher_is_better=higher_is_better,
        )
        if isinstance(metrics, dict):
            from dataclasses import asdict, is_dataclass

            metrics.setdefault(metadata_target_name, {})
            metrics[metadata_target_name] = asdict(result) if is_dataclass(result) else result
        return learning_curve_panel(result)
    except Exception:
        logger.exception("learning_curve diagnostic failed; continuing.")
        return None


def _render_post_fit_diagnostics(
    *,
    targets,
    model,
    df,
    columns,
    preds,
    probs,
    target_type,
    plot_file,
    plot_outputs,
    metrics,
    reporting_config,
    model_name=None,
):
    """Fire the model/preds-based standalone diagnostics default-ON (PDP, slice-finder, decision-curve, SHAP, learning
    curve) and stitch the combined HTML index. Each is gated by a ``ReportingConfig`` knob and skips cheaply when its
    inputs are absent; failures are swallowed (additive panels never abort a run). RAM-safe via the orchestrators.
    """
    if not plot_file or not plot_outputs or reporting_config is None:
        return
    cfg = reporting_config
    tt = (target_type or "").lower()
    task = "regression" if "regress" in tt else "classification"
    _targets_arr = np.asarray(targets) if targets is not None else None
    # A genuinely 2-D multilabel target (n, n_labels) must be excluded from the single-target diagnostics
    # BEFORE ravel() -- ravel() flattens it to a corrupted length-(n*n_labels) array and forces ndim back to 1,
    # which silently defeated the previous downstream ``y_arr.ndim == 1`` guards (they always saw the post-ravel
    # shape, never the true one). That corrupted array then reached df-paired diagnostics (row count n) and
    # crashed/length-mismatched deep inside them instead of being skipped at the gate as intended.
    _multilabel = _targets_arr is not None and _targets_arr.ndim > 1 and _targets_arr.shape[1] > 1
    y_arr = _targets_arr.ravel() if _targets_arr is not None else None
    names, importances = _ranked_feature_names(metrics, model, columns)

    from mlframe.reporting.diagnostics_dispatch import (
        build_combined_html_report, render_category_discriminability_diagnostic, render_class_structure_diagnostic,
        render_decile_table_diagnostic, render_decision_curve_diagnostic,
        render_engineered_separability_diagnostic,
        render_interaction_strength_diagnostic, render_model_card_diagnostic, render_pdp_2d_diagnostic,
        render_pdp_ice_diagnostic, render_shap_diagnostic,
        render_shap_interactions_diagnostic, render_shap_per_instance_diagnostic,
        render_slice_finder_diagnostic,
    )

    # 1-D point prediction for the error-based slice finder (binary uses the positive-class probability).
    y_pred = np.asarray(preds).ravel() if preds is not None else None
    if tt == "binary_classification":
        _bs = _binary_positive_score(probs)
        if _bs is not None:
            y_pred = _bs

    # Split label for the model card: the per-split plot_file is suffixed ``_<split>`` (val / test / train / oof / ...).
    _split = os.path.basename(plot_file).rsplit("_", 1)[-1] if plot_file else "test"

    # Collapse short-circuit: when a regression model's predictions have degenerated to ~constant
    # (pred_std << target_std AND R^2 < 0 -- the group-shift inverse-collapse the sensor flags), the
    # EXPENSIVE diagnostics carry no signal to slice/explain and just burn minutes of slice_finder
    # combo-enumeration + Chromium/kaleido PDP/SHAP charts per collapsed (composite) target. Detect it
    # cheaply from the predictions already in hand and skip those panels; cheap tabular diagnostics run.
    _collapsed = False
    if task == "regression" and y_arr is not None and y_pred is not None and len(y_pred) == len(y_arr):
        try:
            _ya = np.asarray(y_arr, dtype=np.float64).ravel()
            _yp = np.asarray(y_pred, dtype=np.float64).ravel()
            _fin = np.isfinite(_ya) & np.isfinite(_yp)
            if int(_fin.sum()) > 2:
                _ss_res = float(np.sum((_ya[_fin] - _yp[_fin]) ** 2))
                _ss_tot = float(np.sum((_ya[_fin] - _ya[_fin].mean()) ** 2))
                # A model with R^2 < 0 is worse than predicting the mean -> NO learnable structure to
                # slice/explain. Trigger on R^2 < 0 regardless of prediction spread: the constant-collapse
                # (pred_std ~ 0) AND the group-OOD-shift collapse (HIGH pred_std but predictions drift far
                # off, e.g. addres/diff on an extrapolating per-well base, R^2=-333) are both pathological.
                _collapsed = (_ss_tot > 0) and (1.0 - _ss_res / _ss_tot < 0.0)
        except Exception:  # -- collapse gate is a perf heuristic; never abort reporting
            _collapsed = False
    if _collapsed and getattr(cfg, "skip_expensive_diagnostics_on_collapse", True):
        logger.info(
            "[diagnostics] %s [%s]: R2<0 (predictions worse than the mean -- collapse / OOD-shift) -- "
            "skipping expensive slice_finder / PDP / SHAP panels (no signal to explain); cheap tabular diagnostics still run.",
            type(model).__name__ if model is not None else "model", _split,
        )
    else:
        _collapsed = False  # gate disabled -> behave as before

    if getattr(cfg, "pdp_ice", True) and y_arr is not None and not _collapsed:
        render_pdp_ice_diagnostic(
            model=model, df=df, feature_names=names, feature_importances=importances,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            top_features=getattr(cfg, "pdp_top_features", 4), sample=getattr(cfg, "pdp_sample", 2000),
            grid=getattr(cfg, "pdp_grid", 20),
        )

    if getattr(cfg, "pdp_2d_charts", False) and y_arr is not None and not _collapsed:
        render_pdp_2d_diagnostic(
            model=model, df=df, feature_names=names, feature_importances=importances,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            sample=getattr(cfg, "pdp_sample", 2000), grid=getattr(cfg, "pdp_grid", 20),
        )

    if getattr(cfg, "interaction_strength_charts", False) and y_arr is not None and not _collapsed:
        render_interaction_strength_diagnostic(
            model=model, df=df, feature_names=names, feature_importances=importances,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            max_features=getattr(cfg, "interaction_strength_max_features", 8),
            sample=getattr(cfg, "pdp_sample", 2000), grid=getattr(cfg, "pdp_grid", 20),
            max_seconds=getattr(cfg, "interaction_strength_max_seconds", 20.0),
        )

    if getattr(cfg, "engineered_separability_charts", True) and df is not None and y_arr is not None and not _collapsed and not _multilabel:
        render_engineered_separability_diagnostic(
            df=df, y_true=y_arr, feature_names=names, feature_importances=importances,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            sample=getattr(cfg, "pdp_sample", 2000) * 2,
        )

    if getattr(cfg, "class_structure_charts", True) and df is not None and y_arr is not None and not _multilabel:
        render_class_structure_diagnostic(
            df=df, y_true=y_arr, feature_names=names,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            max_groups=getattr(cfg, "class_structure_max_groups", 30),
            n_time_bins=getattr(cfg, "class_structure_time_bins", 20),
        )

    if getattr(cfg, "category_discriminability_charts", True) and df is not None and tt == "binary_classification" and y_arr is not None and not _multilabel:
        render_category_discriminability_diagnostic(
            df=df, y_true=y_arr, feature_names=names,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            top_k=getattr(cfg, "category_discriminability_top_k", 15),
            max_columns=getattr(cfg, "category_discriminability_max_columns", 40),
        )

    if (
        getattr(cfg, "slice_finder", True) and df is not None
        and y_arr is not None and y_pred is not None and not _multilabel
        and len(y_pred) == len(y_arr) and not _collapsed
    ):
        render_slice_finder_diagnostic(
            df=df, y_true=y_arr, y_pred=y_pred, task=task, feature_names=names,
            plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
        )

    if getattr(cfg, "decision_curve", True) and tt == "binary_classification" and y_arr is not None:
        _bs = _binary_positive_score(probs)
        if _bs is not None and len(_bs) == len(y_arr):
            render_decision_curve_diagnostic(
                y_true=y_arr, y_score=_bs, plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            )

    if getattr(cfg, "decile_table", True) and tt == "binary_classification" and y_arr is not None:
        _bs = _binary_positive_score(probs)
        if _bs is not None and len(_bs) == len(y_arr):
            render_decile_table_diagnostic(
                y_true=y_arr, y_score=_bs, plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            )

    if getattr(cfg, "risk_coverage_charts", True) and y_arr is not None and not _multilabel:
        from mlframe.reporting import render_risk_coverage_diagnostic
        if tt == "binary_classification":
            _bs = _binary_positive_score(probs)
            if _bs is not None and len(_bs) == len(y_arr):
                render_risk_coverage_diagnostic(
                    y_true=y_arr, y_score=_bs, task="binary", plot_outputs=plot_outputs,
                    base_path=plot_file, metrics_dict=metrics, model_label=model_name_for_title(target_type),
                )
        elif tt == "multiclass_classification" and probs is not None:
            _proba = np.asarray(probs, dtype=np.float64)
            if _proba.ndim == 2 and _proba.shape[0] == len(y_arr):
                render_risk_coverage_diagnostic(
                    y_true=y_arr, y_score=_proba, task="multiclass", plot_outputs=plot_outputs,
                    base_path=plot_file, metrics_dict=metrics, model_label=model_name_for_title(target_type),
                )
        elif task == "regression" and y_pred is not None and len(y_pred) == len(y_arr):
            # Confidence proxy: negative distance from the prediction's own mean (central preds = more typical = more certain).
            conf = -np.abs(y_pred - float(np.mean(y_pred)))
            render_risk_coverage_diagnostic(
                y_true=y_arr, y_score=y_pred, task="regression", confidence=conf, plot_outputs=plot_outputs,
                base_path=plot_file, metrics_dict=metrics, model_label=model_name_for_title(target_type),
            )

    if getattr(cfg, "model_card", True) and y_arr is not None:
        _mc_task = "regression" if task == "regression" else ("binary" if tt == "binary_classification" else "classification")
        # Card title must carry the ESTIMATOR identity (e.g. "LGBMRegressor"), not the target_type --
        # ``model_name_for_title(target_type)`` returns "regression"/"classification", which rendered a
        # nameless "Model card -- regression (test)" suptitle. Prefer the estimator class name (with the
        # internal shim suffix stripped so "LGBMRegressorWithDatasetReuse" displays as "LGBMRegressor");
        # for model-less paths (dummy baselines pass model=None + a descriptive ``model_name`` such as
        # "DummyBaseline:mean") use that explicit name so the card is never blank. Fall back to the
        # target-type label only when neither is available.
        if model is not None:
            _card_name = display_estimator_name(type(model).__name__)
        elif model_name:
            _card_name = str(model_name).strip()
        else:
            _card_name = model_name_for_title(target_type)
        # Card is defined for binary + regression; multiclass/multilabel have no single positive-class score.
        if _mc_task == "regression" and y_pred is not None and len(y_pred) == len(y_arr):
            render_model_card_diagnostic(
                task="regression", y_true=y_arr, y_pred=y_pred, plot_outputs=plot_outputs,
                base_path=plot_file, metrics_dict=metrics, model_name=_card_name, split=_split,
            )
        elif _mc_task == "binary":
            _bs = _binary_positive_score(probs)
            if _bs is not None and len(_bs) == len(y_arr):
                render_model_card_diagnostic(
                    task="binary", y_true=y_arr, y_score=_bs, plot_outputs=plot_outputs,
                    base_path=plot_file, metrics_dict=metrics, model_name=_card_name, split=_split,
                )

    if getattr(cfg, "shap_panels", True) and model is not None and df is not None and not _collapsed:
        render_shap_diagnostic(
            model=model, df=df, feature_names=names, plot_outputs=plot_outputs, base_path=plot_file,
            metrics_dict=metrics, max_rows=getattr(cfg, "shap_max_rows", 20000),
            top_k=getattr(cfg, "shap_top_k", 6), allow_kernel=getattr(cfg, "shap_allow_kernel", False),
        )

    if getattr(cfg, "shap_interactions", False) and model is not None and df is not None and not _collapsed:
        render_shap_interactions_diagnostic(
            model=model, df=df, feature_names=names, plot_outputs=plot_outputs, base_path=plot_file,
            metrics_dict=metrics, max_rows=getattr(cfg, "shap_interaction_max_rows", 2000),
        )

    if getattr(cfg, "shap_per_instance", False) and model is not None and df is not None and y_arr is not None:
        # Per-instance needs a 1-D score: binary positive-class prob, else the regression prediction.
        _yscore = _binary_positive_score(probs) if tt == "binary_classification" else (y_pred if task == "regression" else None)
        if _yscore is not None and len(_yscore) == len(y_arr):
            render_shap_per_instance_diagnostic(
                model=model, df=df, y_true=y_arr, y_score=_yscore, feature_names=names,
                plot_outputs=plot_outputs, base_path=plot_file, metrics_dict=metrics,
            )

    lc_panel = _build_learning_curve(model, df, targets, columns, target_type, getattr(cfg, "learning_curve", None), metrics)
    if lc_panel is not None:
        try:
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save

            base = plot_file + "_learning_curve"
            render_and_save(lc_panel, parse_plot_output_dsl(plot_outputs), base)
            if isinstance(metrics, dict):
                _c = metrics.setdefault("charts", {"saved": [], "failed": []})
                _c.setdefault("saved", []).append("learning_curve")
                _c.setdefault("paths", []).append(base)
        except Exception:
            logger.exception("learning_curve render failed; continuing.")

    # Combined single-page HTML index stitching every chart artifact recorded for this (model, split).
    if getattr(cfg, "combined_html", True) and isinstance(metrics, dict):
        paths = metrics.get("charts", {}).get("paths", [])
        if paths:
            build_combined_html_report(
                base_path=plot_file, chart_paths=paths, plot_outputs=plot_outputs,
                title=f"{model_name_for_title(target_type)} report".strip(), metrics_dict=metrics,
            )

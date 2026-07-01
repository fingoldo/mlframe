"""Model performance reporting functions extracted from ``evaluation.py``.

``report_model_perf`` — top-level dispatch (regression vs classification).
``report_regression_model_perf`` — MAE, RMSE, MaxError, R2, scatter plots.
``report_probabilistic_model_perf`` — AUC, logloss, calibration, ICE, PR curves.

All three accept pre-computed ``preds`` / ``probs`` (from cached models)
or live-compute via ``model.predict`` when none are provided.
"""

from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment] -- plot branches gated; headless envs skip

from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
# Metrics: use mlframe's fast njit versions, not sklearn
from sklearn.preprocessing import LabelEncoder

from mlframe.metrics.core import compute_fairness_metrics, fast_calibration_report, fast_mean_absolute_error, fast_max_error, fast_r2_score, fast_regression_metrics_block, fast_roc_auc, fast_root_mean_squared_error
from pyutilz.pythonlib import get_human_readable_set_size

# .evaluation imports back from ._reporting; deferring breaks the cycle.
# (See line 477 / 580 for the two call-sites.)
from ..phases import phase

if TYPE_CHECKING:
    from ..configs import MultilabelDispatchConfig  # forward annotation only; importing at runtime is unnecessary

# Inline to avoid circular import (_reporting <- evaluation <- _reporting)
DEFAULT_PLOT_SAMPLE_SIZE = 5000
DEFAULT_REPORT_NDIGITS = 2
DEFAULT_CALIB_REPORT_NDIGITS = 2
DEFAULT_NBINS = 10
DEFAULT_FIGSIZE = (15, 5)
DEFAULT_RANDOM_SEED = 42

logger = logging.getLogger(__name__)


def _reporting_field_default(field_name: str):
    """Field default of ReportingConfig, used to detect a left-at-default value the operator never customized. Cached on first call; None if the config is unavailable."""
    cache = _reporting_field_default.__dict__
    if field_name not in cache:
        try:
            from ..configs import ReportingConfig
            cache[field_name] = ReportingConfig.model_fields[field_name].default
        except Exception:
            cache[field_name] = None
    return cache[field_name]


try:
    from IPython.display import display as _ipython_display
except ImportError:  # pragma: no cover
    _ipython_display = None


def _maybe_display(obj):
    """Display ``obj`` in Jupyter; no-op in scripts/CI.

    ``UnicodeEncodeError`` guard: outside Jupyter, IPython's ``display``
    falls back to printing ``repr(obj)`` to stdout. On a Windows console
    under the cp1251 / cp1252 code page, a frame carrying non-Latin-1 cell
    values (emoji, CJK, combining-mark cyrillic -- the ``weird_cat_content``
    fuzz axis) raises ``UnicodeEncodeError: 'charmap' codec can't encode``
    and crashes the whole report. The display is cosmetic, so swallow the
    encode error rather than abort the run. Surfaced by fuzz (4 combos with
    unicode-heavy categorical columns)."""
    if _ipython_display is not None:
        try:
            _ipython_display(obj)
        except UnicodeEncodeError:
            logger.debug("_maybe_display: skipped display() (console encoding cannot represent the object).")
        return
    # No IPython kernel (script / CI): the styled fairness / error-segmentation tables would vanish silently. Log a
    # plain text rendering instead so the only error-segmentation artifact survives in script logs (INV-52).
    text = _frame_to_text(obj)
    if text is not None:
        logger.info("%s", text)


def _frame_to_text(obj) -> Optional[str]:
    """Best-effort plain-text rendering of a DataFrame / pandas Styler for the no-IPython log fallback.

    Returns ``None`` for objects that are not frame-like (nothing to log)."""
    data = getattr(obj, "data", obj)  # a pandas Styler exposes the underlying frame at ``.data``
    to_string = getattr(data, "to_string", None)
    if not callable(to_string):
        return None
    try:
        return to_string()
    except Exception:
        return None


def _style_with_caption(df, caption: str):
    """Return ``df.style.set_caption(caption)`` when the pandas Styler is
    usable, else the bare ``df``.

    ``DataFrame.style`` requires the optional ``jinja2`` dependency; pandas
    raises ``AttributeError: The '.style' accessor requires jinja2`` on import
    failure. The caption is purely cosmetic Jupyter-HTML decoration, so a
    headless / CI environment without jinja2 must NOT crash the whole report
    over it -- fall back to the plain frame (which ``_maybe_display`` /
    ``display`` render as a normal repr). Surfaced by fuzz c0005 (mlp combo,
    fairness-report styling) on a box without jinja2 installed.
    """
    try:
        return df.style.set_caption(caption)
    except (AttributeError, ImportError):
        return df


def _labels_are_arange(classes: Sequence | None, probs: np.ndarray) -> bool:
    """True when class labels are exactly 0..K-1, so the batched-kernel per-class ICE
    (indicator ``targets == column_index``) is bit-identical to the report's per-class
    recompute (``targets == class_name``) and can be indexed by class_id. Non-0-indexed
    labels (e.g. [1,2,3] / strings) make the column index differ from the label -> unsafe."""
    k = probs.shape[1] if hasattr(probs, "shape") and probs.ndim == 2 else (len(classes) if classes is not None else 0)
    return bool(classes is not None and len(classes) == k and all(
        isinstance(c, (int, np.integer)) and int(c) == i for i, c in enumerate(classes)
    ))


def _canonical_multilabel_y(targets) -> np.ndarray:
    """Coerce a multilabel-shaped target to a clean ``(N, K) ndarray``.

    Accepts the three shapes the suite encounters in production:
    1. ``np.ndarray`` 2-D ``(N, K)`` — passed through.
    2. ``pd.DataFrame`` — ``.values``.
    3. ``np.ndarray`` 1-D ``object`` dtype where each cell is a
       length-K array-like (the polars ``pl.List(pl.Int8)`` /
       ``pl.List(pl.Float32)`` -> pandas object roundtrip). Stacked
       to 2-D via ``np.stack``.
    Otherwise returns ``np.asarray(targets)`` unchanged (1-D / scalar
    / unhandled — caller decides).

    For float-typed cells (``pl.List(pl.Float32)`` source), the
    output is cast to ``int64`` via ``>= 0.5`` threshold:
    downstream metrics expect ``{0, 1}`` indicators.

    Extracted from the inline canonicalization at evaluation.py so
    ``mlframe.training.baselines.dummy`` and other consumers share
    one path.
    """
    if isinstance(targets, pd.DataFrame):
        targets_arr = targets.values
    elif isinstance(targets, np.ndarray):
        targets_arr = targets
    else:
        targets_arr = np.asarray(targets)

    if targets_arr.dtype == object and targets_arr.ndim == 1 and targets_arr.shape[0] > 0:
        first = targets_arr[0]
        if hasattr(first, "shape") or (
            hasattr(first, "__len__") and not isinstance(first, (str, bytes))
        ):
            try:
                targets_arr = np.stack([np.asarray(c) for c in targets_arr], axis=0)
            except Exception:
                # Unstackable (jagged / mixed shape) — leave as-is so the
                # caller's exception path surfaces the underlying issue.
                return targets_arr

    # Coerce float-indicator multilabel to int via threshold.
    if targets_arr.ndim == 2 and targets_arr.dtype.kind == "f":
        targets_arr = (targets_arr >= 0.5).astype(np.int64)

    return targets_arr


def _unwrap_booster(model: Any) -> Any:
    """Peel common mlframe wrappers to reach the underlying lgb/xgb/catboost estimator that holds iteration history.

    Returns the first object in the wrapper chain exposing a recognised history accessor (or the original model).
    """
    seen = set()
    cur = model
    for _ in range(6):  # bounded: wrapper nesting is shallow (composite -> calibrated -> base)
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if any(hasattr(cur, a) for a in ("evals_result_", "evals_result", "get_evals_result")):
            return cur
        nxt = None
        for attr in ("base_estimator", "estimator", "best_estimator_", "model_", "_model", "regressor_", "classifier_"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                break
        if nxt is None:
            break
        cur = nxt
    return cur


def _canonicalize_split_names(split_names: list) -> dict:
    """Map booster eval-set names to ``compose_training_curve_figure``'s canonical ``train`` / ``val`` keys.

    Keys already recognised by the alias set (``train`` / ``valid`` / ``validation`` / ``learn`` / ...) pass through.
    lightgbm's positional ``valid_0`` / ``valid_1`` / ... default names are not in that set; by the mlframe convention
    that the FIRST eval set is the training fold and the LAST is the holdout, the lowest-index ``valid_N`` maps to
    ``train`` and the highest to ``val``. When an explicit train key (e.g. lgb's ``training``) is already present, a
    lone positional ``valid_N`` is the holdout and maps to ``val`` (not a second train). Leftovers keep their name.
    """
    out: dict = {}
    recognised_train = {"train", "training", "learn"}
    recognised_val = {"val", "valid", "validation", "test", "eval", "holdout"}
    has_train = False
    has_val = False
    positional = []  # (index, original_name) for the lgb ``valid_<N>`` / xgb ``validation_<N>`` families
    for name in split_names:
        low = str(name).strip().lower()
        if low in recognised_train:
            has_train = True
            continue
        if low in recognised_val:
            has_val = True
            continue
        for prefix in ("valid_", "validation_"):
            if low.startswith(prefix) and low[len(prefix):].isdigit():
                positional.append((int(low[len(prefix):]), name))
                break
    if positional:
        positional.sort()
        if has_train:
            # An explicit train split already exists; the (lone or last) positional is the holdout.
            if not has_val:
                out[positional[-1][1]] = "val"
        else:
            out[positional[0][1]] = "train"
            if len(positional) > 1:
                out[positional[-1][1]] = "val"
    return out


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
            warm_start=lc_cfg.warm_start, random_state=lc_cfg.random_state,
            time_budget_s=lc_cfg.time_budget_s, score_repeats=lc_cfg.score_repeats,
            scorer_name=scorer_name, higher_is_better=higher_is_better,
        )
        if isinstance(metrics, dict):
            from dataclasses import asdict, is_dataclass

            metrics.setdefault("learning_curve", {})
            metrics["learning_curve"] = asdict(result) if is_dataclass(result) else result
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
    y_arr = np.asarray(targets).ravel() if targets is not None else None
    names, importances = _ranked_feature_names(metrics, model, columns)

    from mlframe.reporting.diagnostics_dispatch import (
        build_combined_html_report, render_decile_table_diagnostic, render_decision_curve_diagnostic,
        render_model_card_diagnostic, render_pdp_2d_diagnostic, render_pdp_ice_diagnostic, render_shap_diagnostic,
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
        except Exception:  # noqa: BLE001 -- collapse gate is a perf heuristic; never abort reporting
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

    if (
        getattr(cfg, "slice_finder", True) and df is not None
        and y_arr is not None and y_pred is not None and y_arr.ndim == 1
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

    if getattr(cfg, "risk_coverage_charts", True) and y_arr is not None and y_arr.ndim == 1:
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


def model_name_for_title(target_type) -> str:
    """Short report-title tag from the target_type (combined-HTML page heading)."""
    return (str(target_type) if target_type else "Model")


# Internal shim-wrapper suffixes that mlframe appends to an estimator class to add dataset/eval-set plumbing (e.g.
# ``LGBMRegressorWithDatasetReuse``, ``...WithEvalSetScaling``). They are an implementation detail and must never leak
# into a public model-card / chart title -- strip them so the user sees the real estimator name (``LGBMRegressor``).
_SHIM_CLASS_SUFFIXES = ("WithDatasetReuse", "WithDMatrixReuse", "WithReuse", "WithEvalSetScaling")


def display_estimator_name(name: str) -> str:
    """Strip internal shim-wrapper suffixes from an estimator class name for public display.

    ``LGBMRegressorWithDatasetReuse`` -> ``LGBMRegressor``. Applied repeatedly so stacked shims collapse. A bare
    suffix (the whole name is a shim marker) is left untouched -- there is no real estimator name to recover.
    """
    out = str(name)
    changed = True
    while changed:
        changed = False
        for suffix in _SHIM_CLASS_SUFFIXES:
            if out.endswith(suffix) and len(out) > len(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def report_model_perf(
    targets: np.ndarray | pd.Series,
    columns: Sequence[str],
    model_name: str,
    model: ClassifierMixin | RegressorMixin | None,
    subgroups: dict[str, np.ndarray] | None = None,
    subset_index: np.ndarray | None = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Sequence | None = None,
    preds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    target_label_encoder: LabelEncoder | None = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    fi_kwargs: dict[str, Any] | None = None,
    plot_file: str = "",
    custom_ice_metric: Callable | None = None,
    custom_rice_metric: Callable | None = None,
    metrics: dict[str, Any] | None = None,
    group_ids: np.ndarray | None = None,
    n_features: int | None = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: tuple[str, ...] | None = None,
    multilabel_dispatch_config: MultilabelDispatchConfig | None = None,
    plot_outputs: str | None = None,
    plot_dpi: int | None = None,
    target_type: str | None = None,
    binary_panels: str | None = None,
    multiclass_panels: str | None = None,
    multilabel_panels: str | None = None,
    ltr_panels: str | None = None,
    quantile_panels: str | None = None,
    quantile_alphas: Sequence[float] | None = None,
    y_train_envelope_stats: Any = None,
    reporting_config: Any = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Generate a unified performance report for both classifiers and regressors.

    Automatically detects model type and routes to the appropriate reporting
    function (classification or regression).

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target values.
    columns : Sequence[str]
        Feature column names used for training.
    model_name : str
        Name of the model for display in reports.
    model : ClassifierMixin, RegressorMixin, or None
        Trained model. Can be None if preds/probs are provided.
    subgroups : dict, optional
        Dictionary mapping subgroup names to boolean arrays for fairness analysis.
    subset_index : np.ndarray, optional
        Indices to subset the data for fairness analysis.
    report_ndigits : int, default=4
        Number of decimal digits for metric reporting.
    figsize : tuple, default=(15, 5)
        Figure size for plots.
    report_title : str, default=""
        Title prefix for reports.
    use_weights : bool, default=True
        Whether to use weighted calibration metrics.
    calib_report_ndigits : int, default=2
        Decimal digits for calibration metrics.
    verbose : bool, default=False
        Enable verbose output.
    classes : Sequence, optional
        Class labels for classification.
    preds : np.ndarray, optional
        Pre-computed predictions.
    probs : np.ndarray, optional
        Pre-computed probabilities for classification.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    target_label_encoder : LabelEncoder, optional
        Encoder for target labels.
    nbins : int, default=10
        Number of bins for calibration analysis.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display performance charts.
    show_fi : bool, default=True
        Whether to show feature importances.
    fi_kwargs : dict, optional
        Additional kwargs for feature importance plotting.
    plot_file : str, default=""
        Base path for saving plots.
    custom_ice_metric : Callable, optional
        Custom integral calibration error metric.
    custom_rice_metric : Callable, optional
        Custom robust integral calibration error metric.
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).
    group_ids : np.ndarray, optional
        Group identifiers for grouped calibration analysis.

    Returns
    -------
    tuple
        (preds, probs) for classification, (preds, None) for regression.
    """
    if fi_kwargs is None:
        fi_kwargs = {}

    # Common parameters shared by both classification and regression
    common_params = dict(
        targets=targets,
        columns=columns,
        model_name=model_name,
        model=model,
        subgroups=subgroups,
        subset_index=subset_index,
        report_ndigits=report_ndigits,
        figsize=figsize,
        report_title=report_title,
        verbose=verbose,
        preds=preds,
        df=df,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        plot_file=plot_file,
        metrics=metrics,
        n_features=n_features,
    )

    # sklearn>=1.6 raises AttributeError when is_classifier(None) triggers
    # get_tags(None) (formerly just returned False). The just_evaluate=True
    # path passes model=None with pre-computed preds/probs -- infer task type
    # from whether probs were supplied (presence of probs => classification).
    if model is None:
        is_probabilistic = probs is not None
    else:
        is_probabilistic = is_classifier(model) or type(model).__name__ == "NGBClassifier"
    if is_probabilistic:
        # G7: calibration binning strategy + reliability-CI toggle from ReportingConfig
        # (default-ON binning="auto"; reliability_show_ci forwarded for when the wave-5 y_err
        # spec field lands so the Wilson band renders).
        _calib_binning = getattr(reporting_config, "calibration_binning", None)
        _reliability_show_ci = getattr(reporting_config, "reliability_show_ci", None)
        _fairness_calibration_charts = getattr(reporting_config, "fairness_calibration_charts", True)
        _calibration_by_feature_charts = getattr(reporting_config, "calibration_by_feature_charts", True)
        _calibration_heatmap_2d_charts = getattr(reporting_config, "calibration_heatmap_2d_charts", True)
        with phase(
            "report_probabilistic_model_perf",
            n_rows=(len(targets) if hasattr(targets, '__len__') else None),
        ):
            preds, probs = report_probabilistic_model_perf(
                **common_params,
                use_weights=use_weights,
                calib_report_ndigits=calib_report_ndigits,
                classes=classes,
                probs=probs,
                target_label_encoder=target_label_encoder,
                nbins=nbins,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                group_ids=group_ids,
                show_prob_histogram=show_prob_histogram,
                prob_histogram_yscale=prob_histogram_yscale,
                show_inline_population_labels=show_inline_population_labels,
                title_metrics_tokens=title_metrics_tokens,
                multilabel_dispatch_config=multilabel_dispatch_config,
                plot_dpi=plot_dpi,
                calibration_binning=_calib_binning,
                reliability_show_ci=_reliability_show_ci,
                plot_outputs=plot_outputs,
                fairness_calibration_charts=_fairness_calibration_charts,
                calibration_by_feature_charts=_calibration_by_feature_charts,
                calibration_heatmap_2d_charts=_calibration_heatmap_2d_charts,
            )
    else:
        with phase(
            "report_regression_model_perf",
            n_rows=(len(targets) if hasattr(targets, '__len__') else None),
        ):
            # Thread plot_outputs explicitly so the regression
            # report can pick the plotly DSL path. Probabilistic path
            # uses plot_outputs differently (multi_target_panels at L277);
            # regression's only consumer is the residual-diagnostics chart.
            #
            # Decompose ``y_train_envelope_stats`` (computed once per
            # (model, target) in the trainer) into the three legacy
            # kwargs so the prediction-envelope clip uses the TRAIN
            # bound (k=3 sigma) instead of the per-split eval-fallback
            # bound (k=10 sigma). Composite-target wrap pass attaches
            # its own y-scale stats; degenerate train targets pass
            # None and the reporter falls back to eval-derived bounds.
            _ytmin = _ytmax = _ytstd = None
            if y_train_envelope_stats is not None:
                _ytmin = getattr(y_train_envelope_stats, "y_min", None)
                _ytmax = getattr(y_train_envelope_stats, "y_max", None)
                _ytstd = getattr(y_train_envelope_stats, "y_std", None)
            _scatter_sample = getattr(reporting_config, "regression_scatter_sample_size", DEFAULT_PLOT_SAMPLE_SIZE)
            preds, probs = report_regression_model_perf(
                **common_params, plot_outputs=plot_outputs, plot_dpi=plot_dpi,
                y_train_min=_ytmin, y_train_max=_ytmax, y_train_std=_ytstd,
                plot_sample_size=_scatter_sample,
                reporting_config=reporting_config,
            )

    # Render multiclass / multilabel / LTR panel
    # grids when the caller has supplied per-target_type templates +
    # output DSL. No-op for binary classification / regression / when
    # templates are unset. Failures are logged + swallowed (panels are
    # additive; existing perf chart + FI still emit).
    if plot_file and plot_outputs and (
        binary_panels or multiclass_panels or multilabel_panels or ltr_panels or quantile_panels
    ):
        from mlframe.reporting.auto_dispatch import render_multi_target_panels
        # INV-37: stamp the [NF / M rows] shape annotation onto the panel-grid suptitle like the FI plot header, so a
        # multi-target panel grid is self-describing (how many features + rows produced it).
        _n_cols_hdr = n_features if n_features is not None else (len(columns) if columns is not None and len(columns) > 0 else 0)
        _nfeatures_hdr = f"{_n_cols_hdr:_}F/" if _n_cols_hdr > 0 else ""
        _n_rows_hdr = len(preds) if (preds is not None and hasattr(preds, "__len__")) else (len(targets) if hasattr(targets, "__len__") else 0)
        _shape_hdr = f" [{_nfeatures_hdr}{get_human_readable_set_size(_n_rows_hdr)} rows]" if _n_rows_hdr else ""
        # Data-aware binary panel emphasis is opt-in via ReportingConfig and applies only when the operator left binary_panels at its field default; a custom
        # template is never reordered/dropped. The dispatcher derives the base rate from the y_true it already holds (no extra full-n pass).
        _panel_emphasis = getattr(reporting_config, "panel_emphasis", "all")
        _emph_lo = getattr(reporting_config, "emphasis_imbalance_lo", 0.2)
        _emph_hi = getattr(reporting_config, "emphasis_imbalance_hi", 0.8)
        _binary_panels_is_default = (binary_panels == _reporting_field_default("binary_panels"))
        with phase("render_multi_target_panels"):
            _rendered_tag = render_multi_target_panels(
                targets=np.asarray(targets) if not isinstance(targets, np.ndarray) else targets,
                probs=probs, preds=preds,
                classes=classes, group_ids=group_ids,
                quantile_alphas=quantile_alphas,
                plot_outputs=plot_outputs,
                plot_dpi=plot_dpi,
                binary_panels=binary_panels,
                multiclass_panels=multiclass_panels,
                multilabel_panels=multilabel_panels,
                ltr_panels=ltr_panels,
                quantile_panels=quantile_panels,
                panel_emphasis=_panel_emphasis,
                binary_panels_is_default=_binary_panels_is_default,
                emphasis_imbalance_lo=_emph_lo,
                emphasis_imbalance_hi=_emph_hi,
                base_path=plot_file,
                suptitle=(report_title + " " + model_name).strip() + _shape_hdr,
                # Authoritative target_type gate — prevents regression
                # targets that happen to carry group_ids from FTE
                # (grouped-CV pattern) from incorrectly triggering the
                # LTR panels branch, which used to render NDCG/MRR
                # nonsense for regression + paid 10-30s on 5M rows.
                target_type=target_type,
            )
            # INV-48: account which panel grids rendered so a run can assert
            # chart presence. The no-crash contract holds -- a failed render is
            # logged + swallowed inside the dispatcher and recorded as "failed".
            # INV-56: also stamp the base path so the accounting points at the on-disk artifact, not just a tag.
            if isinstance(metrics, dict):
                _charts = metrics.setdefault("charts", {"saved": [], "failed": []})
                _which = (
                    "binary" if (target_type or "").lower() == "binary_classification"
                    else (target_type or "").lower()
                ) or "panels"
                if _rendered_tag:
                    _charts["saved"].append(f"{_rendered_tag}_panels")
                    _charts.setdefault("paths", []).append(f"{plot_file}_{_rendered_tag}_panels")
                else:
                    _charts["failed"].append(f"{_which}_panels")

    # Per-model train-vs-val iteration curves (INV-24): default-ON; no-op for non-boosting models, when charts are
    # not saved to disk, or when the model carries no eval history.
    if model is not None:
        with phase("render_training_curves"):
            _render_training_curves(
                model,
                model_name=(report_title + " " + model_name).strip(),
                plot_file=plot_file,
                plot_outputs=plot_outputs,
                plot_dpi=plot_dpi,
                metrics=metrics,
                reporting_config=reporting_config,
            )

    # Binary decile gains/lift/KS table -- surfaced in the metrics dict (not a
    # chart panel) so the operator gets the gains-table view alongside the curves.
    if (
        isinstance(metrics, dict)
        and (target_type or "").lower() == "binary_classification"
        and probs is not None
    ):
        try:
            from mlframe.reporting.charts.binary import binary_decile_table
            probs_arr = np.asarray(probs)
            if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
                _score = probs_arr[:, 1]
            elif probs_arr.ndim == 1:
                _score = probs_arr
            elif probs_arr.ndim == 2 and probs_arr.shape[1] == 1:
                _score = probs_arr.ravel()
            else:
                _score = None
            if _score is not None:
                _yt = np.asarray(targets).ravel()
                metrics["binary_decile_table"] = binary_decile_table(_yt, _score)
        except Exception:
            logger.exception("binary_decile_table computation failed; continuing.")

    if show_fi:
        n_cols = n_features if n_features is not None else (len(columns) if columns is not None and len(columns) > 0 else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        with phase(
            "plot_feature_importances",
            model=type(model).__name__,
            n_cols=len(columns) if columns is not None and len(columns) > 0 else 0,
        ):
            # Lazy import: _reporting <- evaluation <- _reporting cycle (see comment at top of file).
            from ..evaluation import plot_model_feature_importances
            # Thread df + targets through to power the permutation-FI
            # fallback for estimators without native ``feature_importances_``
            # / ``coef_`` (PyTorch-Lightning MLP, Keras nets, custom predict-
            # only wrappers). When the inner exposes a native source the
            # permutation path is skipped automatically.
            _fi_X = df[list(columns)] if (df is not None and columns is not None and len(columns) > 0) else None
            feature_importances = plot_model_feature_importances(
                model=model,
                columns=columns,
                model_name=(report_title + " " + model_name + f" [{nfeatures}{get_human_readable_set_size(len(preds))} rows]").strip(),
                plot_file=plot_file + "_fiplot.png" if plot_file else "",
                X=_fi_X,
                y=targets,
                **fi_kwargs,
            )
        if metrics is not None:
            metrics.update({"feature_importances": feature_importances})

    # Standalone post-fit diagnostics (PDP/ICE, slice-finder, decision-curve, SHAP, learning curve) + combined HTML.
    # Placed after the FI block so the importance ranking is available for PDP feature selection.
    with phase("render_post_fit_diagnostics"):
        _render_post_fit_diagnostics(
            targets=targets, model=model, df=df, columns=columns, preds=preds, probs=probs,
            target_type=target_type, plot_file=plot_file, plot_outputs=plot_outputs,
            metrics=metrics, reporting_config=reporting_config, model_name=model_name,
        )

    return preds, probs


# Wave 97 (2026-05-21): report_probabilistic_model_perf (~520 lines)
# moved to sibling file _reporting_probabilistic.py to drop this file
# below the 1k-line monolith threshold. Re-exported below so existing
# callers (`from .._reporting import report_probabilistic_model_perf`)
# keep working.
from ._reporting_probabilistic import report_probabilistic_model_perf  # noqa: F401, E402
# report_regression_model_perf (~650 lines) moved to sibling file
# _reporting_regression.py for the same 1k-line monolith threshold;
# re-exported below.
from ._reporting_regression import report_regression_model_perf  # noqa: F401, E402


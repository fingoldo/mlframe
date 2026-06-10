"""Auto-dispatch helper that picks a multi-target panel composer
based on input shapes and renders it via the active backend(s).

This is the glue between the per-(model, split) reporting hot path
in ``mlframe.training.evaluation.report_model_perf`` and the panel
composers in ``mlframe.reporting.charts.{multiclass,multilabel,ltr}``.

Dispatch rules (probabilistic targets only):
- ``targets.ndim == 2``                  -> multilabel (panels=multilabel_panels)
- ``probs.shape[1] >= 3 and targets.ndim == 1`` -> multiclass (panels=multiclass_panels)
- ``group_ids is not None`` (any shape)  -> LTR (panels=ltr_panels)
- 1-D targets + 1-class/2-column probs   -> binary curve panels (panels=binary_panels)
- regression                             -> skip (dedicated scatter/residual charts)

The dispatcher is opt-in per panel-template kwarg: if the relevant
``*_panels`` kwarg is None or empty, that branch is skipped.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def render_multi_target_panels(
    *,
    targets: np.ndarray,
    probs: Optional[np.ndarray] = None,
    preds: Optional[np.ndarray] = None,
    classes: Optional[Sequence[Any]] = None,
    group_ids: Optional[np.ndarray] = None,
    quantile_alphas: Optional[Sequence[float]] = None,
    plot_outputs: Optional[str] = None,
    binary_panels: Optional[str] = None,
    multiclass_panels: Optional[str] = None,
    multilabel_panels: Optional[str] = None,
    ltr_panels: Optional[str] = None,
    quantile_panels: Optional[str] = None,
    threshold: float = 0.5,
    cost_ratio: Optional[Any] = None,
    base_path: str = "",
    suptitle: str = "",
    max_cols: int = 2,
    target_type: Optional[str] = None,
    plot_dpi: Optional[int] = None,
) -> Optional[str]:
    """Pick the right composer for the input shapes and render.

    Returns the chosen target_type tag (``"multiclass"`` /
    ``"multilabel"`` / ``"ltr"`` / ``"quantile"``) or ``None`` if nothing
    was rendered (binary, regression, missing inputs, or all panel
    templates empty).

    No-op short-circuits (silent):
    - ``base_path`` empty -> nothing to write to.
    - ``plot_outputs`` empty -> no backend selected.
    - The matched branch's panel template is empty.

    Authoritative gate: when ``target_type`` is set (caller knows the
    target_type explicitly), only the matching branch fires. When
    ``target_type`` is None, falls back to shape-based heuristics for
    back-compat — but those heuristics misfire for regression-with-
    ``group_ids`` (a common pattern when ``FTE.group_field`` is set
    for grouped CV splits, NOT for ranking). Always pass ``target_type``
    when available.
    """
    if not base_path or not plot_outputs:
        return None

    targets_arr = np.asarray(targets) if targets is not None else None

    # Per-target_type gate (when caller provided target_type explicitly).
    # The shape-based heuristics below were ambiguous for regression
    # targets that happen to carry ``group_ids`` (FTE grouped-split
    # pattern) — the LTR branch's ``group_ids is not None AND scores.ndim
    # == 1`` condition fired incorrectly + paid 10-30s of NDCG/MRR
    # computation per split. Authoritative target_type fixes this:
    # regression / binary / quantile_regression / multilabel /
    # multiclass / learning_to_rank each gate exactly one branch.
    tt = (target_type or "").lower()
    if tt:
        # Regression has its own dedicated report charts (scatter / residual
        # panels); this dispatcher's panels would be redundant there.
        if tt == "regression":
            return None
        # Each remaining target_type maps to exactly one branch.
        # When the matching panel template is empty, return None
        # silently (operator opted out of that target_type's panels).
        if tt == "binary_classification" and not binary_panels:
            return None
        if tt == "learning_to_rank" and not ltr_panels:
            return None
        if tt == "quantile_regression" and not quantile_panels:
            return None
        if tt == "multilabel_classification" and not multilabel_panels:
            return None
        if tt == "multiclass_classification" and not multiclass_panels:
            return None

    # LTR: opt-in via group_ids + 1-D score (preds for rankers). When
    # ``target_type`` is provided, gate strictly on it; otherwise the
    # back-compat shape heuristic fires (note: misfires for
    # regression-with-group_ids — pass target_type to avoid).
    _ltr_allowed = (tt == "" or tt == "learning_to_rank")
    if _ltr_allowed and group_ids is not None and ltr_panels and targets_arr is not None:
        scores = preds if preds is not None else probs
        if scores is not None and np.ndim(scores) == 1:
            try:
                from mlframe.reporting.charts.ltr import compose_ltr_figure
                from mlframe.reporting.output import parse_plot_output_dsl
                from mlframe.reporting.renderers import render_and_save

                spec = compose_ltr_figure(
                    targets_arr, np.asarray(scores), np.asarray(group_ids),
                    panels_template=ltr_panels, suptitle=suptitle,
                    max_cols=max_cols,
                )
                if plot_dpi is not None:
                    import dataclasses as _dc
                    spec = _dc.replace(spec, dpi=plot_dpi)
                render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                                base_path + "_ltr_panels")
                return "ltr"
            except Exception:
                logger.exception("LTR panel rendering failed; continuing.")
                # Fall through -- still try multiclass/multilabel below.

    # Quantile regression: opt-in via quantile_alphas + 2-D preds. Like
    # LTR, this is order-sensitive vs the multilabel branch (multilabel
    # also wants 2-D preds), so check QR FIRST and fall through if the
    # caller didn't supply quantile_alphas.
    _quantile_allowed = (tt == "" or tt == "quantile_regression")
    if (
        _quantile_allowed
        and quantile_panels and quantile_alphas is not None
        and preds is not None and targets_arr is not None
    ):
        preds_arr_q = np.asarray(preds)
        if preds_arr_q.ndim == 2 and targets_arr.ndim == 1:
            try:
                from mlframe.reporting.charts.quantile import compose_quantile_figure
                from mlframe.reporting.output import parse_plot_output_dsl
                from mlframe.reporting.renderers import render_and_save

                spec = compose_quantile_figure(
                    targets_arr, preds_arr_q, quantile_alphas,
                    panels_template=quantile_panels, suptitle=suptitle,
                    max_cols=max_cols,
                )
                if plot_dpi is not None:
                    import dataclasses as _dc
                    spec = _dc.replace(spec, dpi=plot_dpi)
                render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                                base_path + "_quantile_panels")
                return "quantile"
            except Exception:
                logger.exception("Quantile panel rendering failed; continuing.")
                # Fall through.

    if probs is None or targets_arr is None:
        return None

    probs_arr = np.asarray(probs)

    # Multilabel: 2-D targets aligned with 2-D probs.
    _ml_allowed = (tt == "" or tt == "multilabel_classification")
    if _ml_allowed and targets_arr.ndim == 2 and probs_arr.ndim == 2 and multilabel_panels:
        if targets_arr.shape != probs_arr.shape:
            logger.warning(
                "render_multi_target_panels: multilabel targets %s != probs %s; "
                "skipping multilabel panels.",
                targets_arr.shape, probs_arr.shape,
            )
            return None
        try:
            from mlframe.reporting.charts.multilabel import compose_multilabel_figure
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save

            labels = list(classes) if classes is not None else \
                [f"label_{i}" for i in range(probs_arr.shape[1])]
            spec = compose_multilabel_figure(
                targets_arr, probs_arr, labels,
                panels_template=multilabel_panels, suptitle=suptitle,
                max_cols=max_cols,
            )
            if plot_dpi is not None:
                import dataclasses as _dc
                spec = _dc.replace(spec, dpi=plot_dpi)
            render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                            base_path + "_multilabel_panels")
            return "multilabel"
        except Exception:
            logger.exception("Multilabel panel rendering failed; continuing.")
            return None

    # Multiclass: 1-D targets, K>=3 classes in the proba matrix.
    _mc_allowed = (tt == "" or tt == "multiclass_classification")
    if (_mc_allowed and targets_arr.ndim == 1 and probs_arr.ndim == 2
            and probs_arr.shape[1] >= 3 and multiclass_panels):
        try:
            from mlframe.reporting.charts.multiclass import compose_multiclass_figure
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save

            classes_seq = list(classes) if classes is not None else \
                list(range(probs_arr.shape[1]))
            spec = compose_multiclass_figure(
                targets_arr, probs_arr, classes_seq,
                panels_template=multiclass_panels, suptitle=suptitle,
                max_cols=max_cols,
            )
            if plot_dpi is not None:
                import dataclasses as _dc
                spec = _dc.replace(spec, dpi=plot_dpi)
            render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                            base_path + "_multiclass_panels")
            return "multiclass"
        except Exception:
            logger.exception("Multiclass panel rendering failed; continuing.")
            return None

    # Binary classification: 1-D targets, 1-class-or-2-column probs. The score
    # is the positive-class column (probs[:, 1] for a 2-column proba matrix,
    # else the 1-D probs / preds). Regression is already excluded above by the
    # authoritative target_type gate; the shape heuristic here is the binary
    # back-compat path for callers that do not pass target_type.
    _bin_allowed = (tt == "" or tt == "binary_classification")
    if _bin_allowed and binary_panels and targets_arr is not None and targets_arr.ndim == 1:
        y_score = None
        if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
            y_score = probs_arr[:, 1]
        elif probs_arr.ndim == 1:
            y_score = probs_arr
        elif probs_arr.ndim == 2 and probs_arr.shape[1] == 1:
            y_score = probs_arr.ravel()
        if y_score is not None:
            try:
                from mlframe.reporting.charts.binary import compose_binary_figure
                from mlframe.reporting.output import parse_plot_output_dsl
                from mlframe.reporting.renderers import render_and_save

                spec = compose_binary_figure(
                    targets_arr, np.asarray(y_score),
                    panels_template=binary_panels, threshold=threshold,
                    cost_ratio=cost_ratio, suptitle=suptitle, max_cols=max_cols,
                )
                if plot_dpi is not None:
                    import dataclasses as _dc
                    spec = _dc.replace(spec, dpi=plot_dpi)
                render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                                base_path + "_binary_panels")
                return "binary"
            except Exception:
                logger.exception("Binary panel rendering failed; continuing.")
                return None

    # Regression -- existing reporting paths cover it.
    return None


__all__ = ["render_multi_target_panels"]

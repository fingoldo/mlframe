"""Auto-dispatch helper that picks a multi-target panel composer
based on input shapes and renders it via the active backend(s).

This is the glue between the per-(model, split) reporting hot path
in ``mlframe.training.evaluation.report_model_perf`` and the panel
composers in ``mlframe.reporting.charts.{multiclass,multilabel,ltr}``.

Dispatch rules (probabilistic targets only):
- ``targets.ndim == 2``                  -> multilabel (panels=multilabel_panels)
- ``probs.shape[1] >= 3 and targets.ndim == 1`` -> multiclass (panels=multiclass_panels)
- ``group_ids is not None`` (any shape)  -> LTR (panels=ltr_panels)
- otherwise                              -> binary (skip; existing
                                            calibration plot already
                                            covers this)

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
    plot_outputs: Optional[str] = None,
    multiclass_panels: Optional[str] = None,
    multilabel_panels: Optional[str] = None,
    ltr_panels: Optional[str] = None,
    base_path: str = "",
    suptitle: str = "",
    max_cols: int = 2,
) -> Optional[str]:
    """Pick the right composer for the input shapes and render.

    Returns the chosen target_type tag (``"multiclass"`` /
    ``"multilabel"`` / ``"ltr"``) or ``None`` if nothing was rendered
    (binary, regression, missing inputs, or all panel templates empty).

    No-op short-circuits (silent):
    - ``base_path`` empty -> nothing to write to.
    - ``plot_outputs`` empty -> no backend selected.
    - The matched branch's panel template is empty.
    """
    if not base_path or not plot_outputs:
        return None

    targets_arr = np.asarray(targets) if targets is not None else None

    # LTR: opt-in via group_ids + 1-D score (preds for rankers). When
    # the LTR guard rejects (no 1-D score available), we fall through
    # to multilabel/multiclass dispatch -- this lets a multiclass run
    # that happens to have group_ids in scope still emit multiclass
    # panels via the same call site.
    if group_ids is not None and ltr_panels and targets_arr is not None:
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
                render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                                base_path + "_ltr_panels")
                return "ltr"
            except Exception:
                logger.exception("LTR panel rendering failed; continuing.")
                # Fall through -- still try multiclass/multilabel below.

    if probs is None or targets_arr is None:
        return None

    probs_arr = np.asarray(probs)

    # Multilabel: 2-D targets aligned with 2-D probs.
    if targets_arr.ndim == 2 and probs_arr.ndim == 2 and multilabel_panels:
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
            render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                            base_path + "_multilabel_panels")
            return "multilabel"
        except Exception:
            logger.exception("Multilabel panel rendering failed; continuing.")
            return None

    # Multiclass: 1-D targets, K>=3 classes in the proba matrix.
    if (targets_arr.ndim == 1 and probs_arr.ndim == 2
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
            render_and_save(spec, parse_plot_output_dsl(plot_outputs),
                            base_path + "_multiclass_panels")
            return "multiclass"
        except Exception:
            logger.exception("Multiclass panel rendering failed; continuing.")
            return None

    # Binary or regression -- existing reporting paths cover them.
    return None


__all__ = ["render_multi_target_panels"]

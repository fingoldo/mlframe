"""Persist what composite post-processing changed, after it ran.

``finalize_suite`` saves the metadata and the ``_CT_ENSEMBLE__*`` entries, but composite post-processing runs after it and
creates or replaces the objects that ship: every composite-target entry's ``.model`` becomes a ``CompositeTargetEstimator``
(the per-model ``.dump`` written by ``process_model`` holds the bare T-scale inner), the cross-target ensemble and its MoE
gate are built, and the y-scale metrics, verdicts and value reports are stamped into metadata. This step re-saves each
changed per-model dump, persists the CT-ensemble entries and rewrites the metadata, so a suite loaded from disk serves the
same models and carries the same metadata as the in-memory result.
"""
from __future__ import annotations

import logging
from os.path import exists
from typing import Any

from mlframe.utils.log_throttle import log_throttle

from ._predict_composite_routing import is_composite_wrapper

logger = logging.getLogger(__name__)

# Attribute ``process_model`` stamps on each in-memory entry: the path of that entry's per-model ``.dump``.
DUMP_PATH_ATTR = "model_file_path"


# A model composite post-processing put on an entry after its dump was written is exactly a composite wrapper.
_is_post_processed_model = is_composite_wrapper


def resave_post_processed_entries(ctx: Any) -> int:
    """Re-save every per-model dump whose in-memory entry now holds a composite wrapper; returns the number re-saved.

    The dump on disk is loaded, its ``model`` swapped for the in-memory wrapper and saved back, so every other field keeps
    exactly what the original save wrote (lean stripping, columns) while the served model becomes the y-scale predictor.
    """
    from ..io import load_mlframe_model, save_mlframe_model

    n_saved = 0
    for by_name in (getattr(ctx, "models", None) or {}).values():
        if not isinstance(by_name, dict):
            continue
        for tname, entries in by_name.items():
            for entry in entries if isinstance(entries, list) else ():
                model = getattr(entry, "model", None)
                path = getattr(entry, DUMP_PATH_ATTR, None)
                if not path or not _is_post_processed_model(model) or not exists(path):
                    continue
                try:
                    saved: Any = load_mlframe_model(path)
                    if saved is None or not hasattr(saved, "model"):
                        raise ValueError(f"dump at {path} did not load as a model entry")
                    saved.model = model
                    # Same object as the wrapper's inner pipeline, so the pickle stores it once.
                    if getattr(entry, "pre_pipeline", None) is not None:
                        saved.pre_pipeline = entry.pre_pipeline
                    if not save_mlframe_model(saved, path, verbose=0, lean=True):
                        raise RuntimeError(f"save_mlframe_model returned False for {path}")
                    n_saved += 1
                except Exception as exc:
                    log_throttle(
                        logger, "persist_post_resave_failed", logging.WARNING,
                        "[composite persist] re-saving the y-scale wrapper for target '%s' to %s failed: %s. The dump still holds "
                        "the T-scale inner model; a suite loaded from disk will serve T-scale values for it.",
                        tname, path, exc,
                    )
    return n_saved


def persist_after_composite_post(ctx: Any) -> None:
    """Re-save wrapped composite dumps, persist CT-ensemble entries and rewrite metadata after composite post-processing."""
    from ._phase_finalize import _persist_ct_ensemble_entries
    from ._setup_helpers_metadata import _finalize_and_save_metadata

    n_resaved = resave_post_processed_entries(ctx)
    if n_resaved and getattr(ctx, "verbose", 0):
        logger.info("[composite persist] re-saved %d composite-target dump(s) with their y-scale wrapper.", n_resaved)
    _persist_ct_ensemble_entries(ctx)
    _finalize_and_save_metadata(ctx, verbose=0)

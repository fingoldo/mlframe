"""Raw-input column names of a fitted pre-pipeline, for validating a cached model that sits behind a feature selector.

A cached model's own ``feature_names_`` are the pipeline's OUTPUT columns (what the selector kept or engineered), while the frame the
suite hands ``process_model`` holds the pipeline's INPUT columns. Comparing the two always mismatches; the comparable quantity is the
input column list the pipeline itself was fitted on.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def pipeline_input_names(pre_pipeline: Any) -> Optional[List[str]]:
    """Columns the pipeline's FIRST step was fitted on, or ``None`` when that step records no input names (old dumps, name-less steps).

    Only the first step is consulted: a later step's ``feature_names_in_`` describes an intermediate frame, not the suite's input.
    """
    if pre_pipeline is None:
        return None
    steps = getattr(pre_pipeline, "steps", None)
    first = steps[0][1] if steps else pre_pipeline
    if first is None or first == "passthrough":
        return None
    try:
        raw = getattr(first, "feature_names_in_", None)
        return None if raw is None else [str(c) for c in raw]
    except Exception as exc:  # an attribute that raises is the same as one that is absent
        logger.debug("pipeline_input_names: %s", exc)
        return None

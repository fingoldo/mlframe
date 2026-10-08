"""Merge of FE-stage outputs under the step-input contract.

Every FE stage of one step reads the frame the step started with (the base features for the first step, the survivors of the previous step afterwards) and
never another stage's output. A stage returns that frame plus its new columns; this module folds the new columns of each stage into the accumulated frame in
the fixed stage order, so the result does not depend on the order the stages ran in.
"""

from __future__ import annotations

import logging

from .._fe_frame_ops import fe_append_columns, fe_extract_columns

logger = logging.getLogger(__name__)


def _fe_merge_new_columns(acc, stage_out, step_input):
    """``acc`` plus the columns ``stage_out`` added on top of ``step_input``; a name already present in ``acc`` keeps its first owner."""
    if stage_out is step_input or stage_out is acc:
        return acc
    base = set(map(str, step_input.columns))
    have = set(map(str, acc.columns))
    new = [c for c in stage_out.columns if str(c) not in base]
    dup = [c for c in new if str(c) in have]
    if dup:
        logger.warning("FE stage emitted column(s) %s already produced by an earlier stage of the same step; keeping the first", dup[:6])
        new = [c for c in new if str(c) not in have]
    if not new:
        return acc
    return fe_append_columns(acc, fe_extract_columns(stage_out, new))

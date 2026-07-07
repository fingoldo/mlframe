"""Core training functions for mlframe."""

from __future__ import annotations


import logging

logger = logging.getLogger(__name__)


# CODE-P1-8: single consolidated import for all per-phase entry points (was 8 separate ``from
# ._phase_X import Y`` lines). Call e.g. ``pr.apply_polars_categorical_fixes(...)``.


from ._misc_helpers import _bulk_setattr_to_ctx, _split_preds_probs, _prep_polars_df  # noqa: F401

# The prelude patch handles (apply_loky_cpu_count_override /
# apply_third_party_patches_once) live on ``_main_train_suite`` -- the module
# that actually holds the ``train_mlframe_models_suite`` body and whose globals
# the live prelude resolves against. This facade only re-exports the callable,
# so no module-level seam is kept here.


# Re-export predict / load entry points for back-compat.
from .predict import (  # noqa: E402,F401
    predict_mlframe_models_suite,
    predict_from_models,
    load_mlframe_suite,
)

# ----------------------------------------------------------------------
# Sibling-module re-export. The 1008-LOC ``train_mlframe_models_suite``
# body lives in ``_main_train_suite.py`` so this file stays below the
# 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._main_train_suite import train_mlframe_models_suite  # noqa: E402,F401

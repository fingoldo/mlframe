"""Composite target transforms, estimator wrapper, and discovery.

Building blocks for composite-target discovery. This module ships:

1. The transform registry (forward / inverse / fit / domain check).
2. ``CompositeTargetEstimator`` -- sklearn-compatible wrapper that
   hides the transform-and-invert loop from downstream callers.
3. ``CompositeTargetDiscovery`` -- auto-finds the best (base, transform)
   pairs by MI gain over the raw target, with strict train-only
   fitting and forbidden-base filtering.

Concept. A composite target is a transform ``T = f(y, base)`` such
that the model learns ``T`` from features ``X`` (typically excluding
the dominant feature ``base``), and a wrapper applies ``f^{-1}`` at
predict time to recover ``y`` in the original scale. The structural
example: ``y = TVT`` and ``base = TVT_prev``, where the autoregressive
lag is captured natively by the transform and the model is forced to
explain the remaining residual.

Public surface
--------------
- :class:`Transform` -- frozen dataclass, one entry per transform.
- :data:`_TRANSFORMS_REGISTRY` / :func:`get_transform` /
  :func:`list_transforms` -- registry lookup.
- :class:`CompositeTargetEstimator` -- sklearn-compatible wrapper that
  fits an inner regressor on ``T`` and inverts at predict.
- :exc:`DomainViolationError`, :exc:`UnknownTransformError`.

Design choices
--------------
- Transforms are looked up by **name** at fit/predict time, never
  stored as per-instance callables. This keeps :func:`sklearn.clone`
  semantics honest, makes pickle work with the standard library
  (no closure traps -> no PII leakage via captured DataFrames), and
  lets the wrapper survive process boundaries (joblib, Optuna).
- Transforms are **frozen**: ``forward``, ``inverse``, ``fit``,
  ``domain_check`` are pure module-level functions registered in
  :data:`_TRANSFORMS_REGISTRY` at import time. Adding a new transform =
  one dataclass entry + one parametrized test row.
- Fitted parameters (``alpha``, ``beta``, MAD floor, post-inverse
  y-clip bounds) are computed **only on training rows passed to
  ``fit``**. The wrapper never re-fits at predict time; downstream
  composite-target discovery is responsible for using the same
  ``train_idx`` discipline at the screening step.
- Numerical safety: MAD-soft-cap with floor (against degenerate
  ``T_train`` collapsing to a constant), post-inverse y-clip to the
  ``[Q001/10, Q999*10]`` bounds of ``y_train`` (against ``exp(...)``
  blow-up in ``logratio``), and ``np.isfinite`` guards on incoming
  ``base`` values at predict (against adversarial ``+inf`` injection).

Out of scope for this module
----------------------------
- Discovery (auto-find ``base`` and best transform): future PR.
- Cross-target ensembling: future PR.
- ``base_margin`` / classification residuals: regression only here.
"""
from __future__ import annotations

import contextlib
import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)










# ----------------------------------------------------------------------
# CompositeSpec moved to composite_spec.py (broke circular import with
# composite_discovery). Re-export below preserves callers doing
# ``from mlframe.training.composite import CompositeSpec``.
# ----------------------------------------------------------------------
from .composite_spec import CompositeSpec  # noqa: F401




# Phase 3 split: re-export everything from composite_transforms for
# full back-compat. Existing callers ``from mlframe.training.composite
# import Transform, _TRANSFORMS_REGISTRY, get_transform, ...`` keep
# working unchanged.
# ----------------------------------------------------------------------
from .composite_transforms import (  # noqa: E402,F401
    DomainViolationError,
    UnknownTransformError,
    Transform,
    TAG_CORE,
    TAG_EXTENDED,
    TAG_REGRESSION,
    _MAD_FLOOR_FRAC,
    _MAD_SOFT_CAP_K,
    _MULTI_BASE_COND_NUMBER_MAX,
    _GROUPED_MIN_GROUP_SIZE,
    _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
    _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS,
    _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N,
    _EWMA_RESIDUAL_DEFAULT_K,
    _FRAC_DIFF_DEFAULT_D,
    _FRAC_DIFF_DEFAULT_LAGS,
    _ROLLING_QUANTILE_DEFAULT_K,
    _TRANSFORMS_REGISTRY,
    TRANSFORM_NAME_SHORT,
    compose_target_name,
    get_transform,
    is_composite_target_name,
    list_transforms,
    # Shared helpers used by transforms (tests import some directly).
    _ewma_compute,
    _frac_diff_weights,
    _james_stein_shrinkage_factor,
    _monotonic_residual_g,
    _quantile_residual_assign_bins,
    _rolling_median,
    _row_alpha_beta,
    # 11 transform impls (private helpers; re-exported because tests
    # and a few internal call sites import them by name).
    _diff_forward, _diff_inverse, _diff_fit, _diff_domain,
    _ratio_forward, _ratio_inverse, _ratio_fit, _ratio_domain,
    _logratio_forward, _logratio_inverse, _logratio_fit, _logratio_domain,
    _linear_residual_forward, _linear_residual_inverse, _linear_residual_fit, _linear_residual_domain,
    _linear_residual_multi_forward, _linear_residual_multi_inverse, _linear_residual_multi_fit, _linear_residual_multi_domain,
    _linear_residual_grouped_forward, _linear_residual_grouped_inverse, _linear_residual_grouped_fit, _linear_residual_grouped_domain,
    _quantile_residual_forward, _quantile_residual_inverse, _quantile_residual_fit, _quantile_residual_domain,
    _monotonic_residual_forward, _monotonic_residual_inverse, _monotonic_residual_fit, _monotonic_residual_domain,
    _ewma_residual_forward, _ewma_residual_inverse, _ewma_residual_fit, _ewma_residual_domain,
    _rolling_quantile_ratio_forward, _rolling_quantile_ratio_inverse, _rolling_quantile_ratio_fit, _rolling_quantile_ratio_domain,
    _frac_diff_forward, _frac_diff_inverse, _frac_diff_fit, _frac_diff_domain,
)


# ----------------------------------------------------------------------
# Phase 4a split: re-export CompositeTargetEstimator + helpers from
# composite_estimator for full back-compat. Existing callers
# ``from mlframe.training.composite import CompositeTargetEstimator``
# keep working unchanged.
# ----------------------------------------------------------------------
from .composite_estimator import (  # noqa: E402,F401
    CompositeTargetEstimator,
    _Y_CLIP_LOW_FRAC,
    _Y_CLIP_HIGH_FRAC,
    _y_train_clip_bounds,
    _to_1d_numpy,
    _extract_base,
    _extract_groups,
    _extract_base_matrix,
)


# ----------------------------------------------------------------------
# Phase 4b split: re-export CompositeProvenance + report_to_markdown.
# ----------------------------------------------------------------------
from .composite_provenance import (  # noqa: E402,F401
    CompositeProvenance,
    _format_transform_formulas,
    report_to_markdown,
)


# ----------------------------------------------------------------------
# Phase 4c split: re-export ensemble + OOF + util symbols.
# ----------------------------------------------------------------------
from .composite_ensemble import (  # noqa: E402,F401
    CompositeCrossTargetEnsemble,
    compute_oof_holdout_predictions,
    derive_seeds,
    detect_gpu_in_use,
    env_signature,
)


# ----------------------------------------------------------------------
# Phase 4d split: re-export screening helpers.
# ----------------------------------------------------------------------
from .composite_screening import (  # noqa: E402,F401
    _extract_column_array,
    _is_numeric_column,
    _safe_corr,
    _safe_abs_corr_all,
    _residualise,
    _mi_pair_bin,
    _mi_to_target,
    _silence_tiny_model_output,
    _build_tiny_model,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_y_scale_multiseed,
    _tiny_cv_rmse_raw_y_multiseed,
    _per_bin_rmse,
    _tiny_cv_rmse_y_scale,
    _sample_indices,
)

# ----------------------------------------------------------------------
# Phase 1+2 splits: re-export independent + dependent helper modules.
# Restored after the Phase 4e extraction inadvertently dropped them.
# ----------------------------------------------------------------------
from .composite_auto_detect import (  # noqa: E402,F401
    detect_time_column_candidates,
    sort_df_by_time_column,
    detect_group_column_candidates,
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
)
from .composite_cache import (  # noqa: E402,F401
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from .composite_stacking import (  # noqa: E402,F401
    residual_correlation_matrix,
    max_off_diagonal_correlation,
    stacking_aware_gate,
)
from .composite_interaction_bases import (  # noqa: E402,F401
    generate_interaction_bases,
)
from .composite_streaming import (  # noqa: E402,F401
    streaming_alpha_check_and_refit,
    _STREAMING_DEFAULT_Z_THRESHOLD,
    _STREAMING_DEFAULT_MIN_BUFFER_N,
)
from .composite_bayesian import (  # noqa: E402,F401
    bayesian_alpha_fit,
)
from .composite_forward_stepwise import (  # noqa: E402,F401
    forward_stepwise_multi_base,
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
)
from .composite_feature_stacking import (  # noqa: E402,F401
    composite_predictions_as_feature,
    composite_oof_predictions,
)


# ----------------------------------------------------------------------
# Phase 4e split: re-export CompositeTargetDiscovery.
# ----------------------------------------------------------------------
from .composite_discovery import CompositeTargetDiscovery  # noqa: E402,F401

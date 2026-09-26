"""Per-fold transform refit: the fold-local params that keep a tiny-CV held-out fold honest.

A leaf module (numpy + stdlib only) so the tiny-CV screening path can import it without pulling in ``_eval``, which
imports ``screening``, which imports the tiny-CV modules back.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.training.composite.transforms.shared import call_transform

logger = logging.getLogger(__name__)


def refit_transform_on_fold(
    transform: Any,
    y_fold: np.ndarray,
    base_fold: np.ndarray,
    *,
    groups_fold: np.ndarray | None = None,
    min_valid_rows: int = 2,
) -> tuple[dict[str, Any], np.ndarray] | None:
    """Re-fit a transform's params on ONE CV fold's TRAIN rows only.

    This is the ``_eval``-side contract for **per-fold transform refit**, the
    cure for the in-fold leakage: in :func:`eval_one_transform`
    the transform is fit ONCE on every valid train row (line ~71) and those
    GLOBAL ``fitted_params`` (e.g. ``linear_residual``'s alpha/beta) are then
    reused for every inner tiny-CV fold downstream
    (``_screening_tiny._tiny_cv_rmse_y_scale``). Because the global fit saw the
    rows that later become each fold's HELD-OUT validation set, the recovered
    ``T = forward(y, base, params)`` on the held-out fold is partly explained by
    parameters that already peeked at those very rows -> the held-out RMSE the
    tiny-CV reports is optimistic (the alpha/beta absorbed val-fold structure).

    The honest path is to re-fit the transform params on each fold's TRAIN rows
    ONLY, then ``forward``/``inverse`` the held-out fold with those fold-local
    params. This helper performs exactly that single-fold refit, reusing the
    SAME fit + fitted-domain-refinement logic ``eval_one_transform`` runs on the
    global sample, so the per-fold params are produced identically to how the
    shipped spec's params are produced -- just on a row subset.

    Parameters
    ----------
    transform
        A registry ``Transform`` (reads ``.fit``, ``.domain_check``,
        ``.domain_check_fitted``).
    y_fold
        Raw (un-transformed) target for THIS fold's train rows, NOT the globally-computed ``T``.
    base_fold
        Raw (un-transformed) target / base columns for THIS fold's train rows.
        These are the exact arrays the caller must expose per fold -- the raw
        ``y`` and ``base``, NOT the globally-computed ``T``.
    groups_fold
        Group labels for the fold's train rows (grouped transforms only). Passed
        through to ``transform.fit`` only when the fit signature accepts it.
    min_valid_rows
        Minimum surviving (domain-valid) fold rows required to attempt a refit.
        Below this the fold is too small to re-estimate params reliably and the
        caller should keep the global params for this fold (we return ``None``).

    Returns
    -------
    ``(fold_params, valid_fold_mask)`` on success, where ``fold_params`` is the
    transform's fitted-params dict for THIS fold's train rows and
    ``valid_fold_mask`` is the boolean mask (aligned to ``y_fold``) of rows that
    survived the (fitted-)domain filter; or ``None`` when the fold is degenerate
    (too few valid rows, an empty mask, or a fit that flags
    ``is_degenerate`` / non-dict params) -- in which case the caller falls back
    to the global params so the fold still scores rather than dropping out.

    Notes
    -----
    * **No leakage by construction**: only ``y_fold`` / ``base_fold`` rows enter
      the fit, so a held-out fold scored with these params is honest.
    * **Bit-stable fallback**: returning ``None`` (not raising) lets the caller
      preserve today's global-fit numerics on degenerate folds, so enabling the
      per-fold path never crashes a previously-scoring spec.
    * The caller (``_screening_tiny._one_fold``) still owns the fold split and
      the ``forward``/``inverse`` calls; this helper only produces the params.
    """
    y_fold = np.asarray(y_fold)
    base_fold = np.asarray(base_fold)
    # Pre-fit domain filter (same gate eval_one_transform applies before fit).
    valid = np.asarray(transform.domain_check(y_fold, base_fold), dtype=bool)
    if valid.shape != y_fold.shape:
        # Defensive: a domain_check that returns a mis-shaped mask cannot be
        # trusted to subset rows; signal the caller to keep global params.
        return None
    if int(valid.sum()) < min_valid_rows:
        return None
    y_v = y_fold[valid]
    base_v = base_fold[valid]
    g_fit = None
    if groups_fold is not None:
        g_arr = np.asarray(groups_fold)
        if g_arr.shape[0] == y_fold.shape[0]:
            g_fit = g_arr[valid]
    try:
        # The gateway passes groups only to a fit that declares them, and raises for a grouped transform refit without its groups.
        fold_params = call_transform(transform, "fit", y_v, base_v, groups=g_fit)
    except Exception as _fit_err:  # -- degenerate fold, keep global
        logger.debug(
            "refit_transform_on_fold: per-fold fit failed (%s); caller should " "fall back to global params for this fold.",
            _fit_err,
        )
        return None
    if not isinstance(fold_params, dict):
        return None
    # A fold whose fit collapses to a near-identity / degenerate function is no
    # better than the global params (and downstream forward on it can NaN); let
    # the caller keep global params rather than score on a degenerate refit.
    if fold_params.get("is_degenerate"):
        return None
    # Fitted-params-aware domain refinement: drop rows that are
    # only out-of-domain once the fold's params exist, so the mask the caller
    # uses to subset the fold matches the params it was fit on.
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None:
        try:
            valid_fitted = np.asarray(
                _dcf(y_fold, base_fold, fold_params), dtype=bool,
            )
        except Exception as e:  # -- treat as no refinement
            logger.warning("domain_check for fold refinement failed: %s", e)
            valid_fitted = None
        if valid_fitted is not None and valid_fitted.shape == valid.shape:
            refined = valid & valid_fitted
            if int(refined.sum()) < min_valid_rows:
                return None
            valid = refined
    return fold_params, valid

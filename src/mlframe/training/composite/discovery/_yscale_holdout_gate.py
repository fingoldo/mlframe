"""y-scale group-aware holdout gate for ``CompositeTargetDiscovery``.

Why this exists
---------------
Discovery selects composite specs on a FORWARD statistic -- ``mi_gain`` (and an
i.i.d. honest-holdout re-score of the same quantity). Production, however, runs
the INVERSE pipeline: a model predicts the transformed target ``T`` from the
features, then reconstructs ``y = transform.inverse(T_hat, base, params)``. On a
group-aware split (whole groups/wells held out) a residual spec whose inverse
amplifies a *standardized* base by ~``std(y)`` blows up: any train/holdout base
distribution shift is multiplied through the inverse, ``y_hat`` slams the
prediction envelope, collapses to ~constant, and ``R^2`` goes sharply negative.
The forward-only screens never observe this -- which is exactly how a prod run
selected, then spent hours training, ten composite targets that every collapsed
to ``R^2 = -146`` on the group-aware test split.

What it does
------------
After the winner set is final, replicate the production predict-T -> invert-to-y
path with a TINY model on a GROUP-DISJOINT holdout carved from the training
groups, and DROP any spec whose reconstructed y-scale RMSE either collapses
(predictions degenerate to ~constant / go non-finite) or loses to the raw-y tiny
baseline by more than ``yscale_holdout_gate_tolerance``. The gate no-ops (keeps
every spec) when group ids are absent or there are too few groups to carve a
disjoint holdout -- so non-group runs and small synthetics are unaffected.

100GB-frame rule: only the narrow per-column gathers on the bounded gate sample
are materialised; no frame copy.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..transforms import UnknownTransformError, get_transform
from .screening import _extract_column_array
from ._screening_tiny import _build_tiny_model

logger = logging.getLogger(__name__)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _spec_base_columns(spec) -> list[str]:
    extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
    if not spec.base_column:
        return []
    return [spec.base_column, *extra]


def _base_arg(df, base_columns: Sequence[str], rows: np.ndarray) -> np.ndarray:
    """Materialise the base argument shape ``transform.forward/inverse`` expects."""
    if not base_columns:
        return np.zeros(rows.size, dtype=np.float64)
    if len(base_columns) == 1:
        return _extract_column_array(df, base_columns[0], rows=rows).astype(np.float64)
    return np.column_stack(
        [_extract_column_array(df, c, rows=rows).astype(np.float64) for c in base_columns]
    )


def _carve_group_disjoint(
    sample_idx: np.ndarray, groups_sample: np.ndarray, holdout_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``sample_idx`` into (fit_idx, eval_idx) with DISJOINT groups.

    Whole groups go to one side or the other -- mirroring the production group-aware split so the
    eval rows come from groups the tiny model never trained on (the regime where the inverse blows up).
    """
    uniq = np.unique(groups_sample)
    perm = rng.permutation(uniq.size)
    n_eval_groups = max(1, int(round(uniq.size * holdout_frac)))
    eval_groups = set(uniq[perm[:n_eval_groups]].tolist())
    eval_mask = np.array([g in eval_groups for g in groups_sample], dtype=bool)
    return sample_idx[~eval_mask], sample_idx[eval_mask]


def apply_yscale_holdout_gate(
    self,
    df: Any,
    target_col: str,
    kept_specs: list,
    usable_features: Sequence[str],
    screen_idx: np.ndarray,
    y_full: np.ndarray,
    val_df: Any = None,
    val_y: np.ndarray | None = None,
) -> list:
    """Drop specs whose predict-T -> invert-to-y pipeline collapses on UNSEEN groups.

    Evaluation set, in priority order:

    1. **The VAL split** (``val_df`` + ``val_y``) when supplied -- the production unseen-WELL
       distribution. Train groups and val groups are DISJOINT under the group-aware split, so fitting a
       tiny model on a train subsample and predicting on a val subsample reproduces the exact regime
       where a residual inverse extrapolates: the held-out wells' base values fall outside the train
       range, the tree's T_hat is clamped, ``y = T_hat + alpha*base`` blows past the envelope and
       collapses (R^2 << 0). A holdout carved from TRAIN groups CANNOT show this -- its base stays
       in-distribution -- which is why the train-group gate passed specs that then collapsed on test.
    2. **Group-disjoint carve from train** (fallback) when no val frame is available but group ids are.

    The discovery frame ``df`` is TRAIN-only, so the val split must be supplied as a separate frame
    (``val_df``, same feature columns) + its targets (``val_y``); the gate fits on ``df``/``y_full`` and
    evaluates the inversion on ``val_df``/``val_y``.

    Returns the surviving spec list (possibly empty -- the CORRECT outcome when every candidate
    collapses: better to train no composite than ten doomed ones). No-ops when the gate is disabled,
    there are no specs, or no usable unseen-group evaluation set can be formed.
    """
    cfg = self.config
    if not getattr(cfg, "yscale_holdout_gate_enabled", True) or not kept_specs:
        return kept_specs

    screen_idx = np.asarray(screen_idx)
    cap = int(getattr(cfg, "yscale_holdout_gate_sample_n", 30_000))
    rng = np.random.default_rng(int(getattr(cfg, "random_state", 0)))

    def _subsample(idx: np.ndarray, n_cap: int) -> np.ndarray:
        idx = np.asarray(idx)
        if n_cap <= 0 or idx.size <= n_cap:
            return idx
        return np.sort(rng.choice(idx, size=n_cap, replace=False))

    feats = list(usable_features)
    val_y = None if val_y is None else np.asarray(val_y)
    _eval_df = df  # frame the eval rows are gathered from; df (train) for the fallback path
    if val_df is not None and val_y is not None and val_y.size >= 50:
        # Preferred path: fit on TRAIN (seen wells), evaluate on the VAL frame (unseen wells);
        # group-disjoint by construction under the group-aware split.
        fit_idx = _subsample(screen_idx, cap)
        eval_idx = _subsample(np.arange(val_y.size), cap)
        _eval_df = val_df
        y_fit = y_full[fit_idx].astype(np.float64)
        y_eval = val_y[eval_idx].astype(np.float64)
        _gate_mode = "val-split"
    else:
        # Fallback: carve a group-disjoint holdout out of the training groups themselves.
        group_ids = getattr(self, "_group_ids_for_rerank", None)
        if group_ids is None:
            logger.info(
                "[CompositeTargetDiscovery.yscale_gate] no val frame and no group ids -- gate skipped "
                "(the inverse-collapse regime is unseen-group; nothing to validate against)."
            )
            return kept_specs
        group_ids = np.asarray(group_ids)
        sample_idx = _subsample(screen_idx, cap)
        try:
            groups_sample = group_ids[sample_idx]
        except (IndexError, TypeError):
            logger.warning("[CompositeTargetDiscovery.yscale_gate] group ids not row-aligned -- gate skipped.")
            return kept_specs
        min_groups = int(getattr(cfg, "yscale_holdout_gate_min_groups", 4))
        if np.unique(groups_sample).size < min_groups:
            logger.info(
                "[CompositeTargetDiscovery.yscale_gate] only %d group(s) in the gate sample (<%d) and no "
                "val frame -- cannot carve a disjoint holdout; gate skipped.",
                np.unique(groups_sample).size, min_groups,
            )
            return kept_specs
        fit_idx, eval_idx = _carve_group_disjoint(
            sample_idx, groups_sample, float(getattr(cfg, "yscale_holdout_gate_holdout_group_frac", 0.3)), rng
        )
        y_fit = y_full[fit_idx].astype(np.float64)
        y_eval = y_full[eval_idx].astype(np.float64)
        _gate_mode = "train-group-holdout"
    if fit_idx.size < 50 or eval_idx.size < 50:
        logger.info("[CompositeTargetDiscovery.yscale_gate] unseen-group eval set too small -- gate skipped.")
        return kept_specs

    x_fit = self._build_feature_matrix(df, feats, fit_idx)
    x_eval = self._build_feature_matrix(_eval_df, feats, eval_idx)
    y_eval_std = float(np.std(y_eval))

    n_estimators = int(getattr(cfg, "tiny_model_n_estimators", 60))
    num_leaves = int(getattr(cfg, "tiny_model_num_leaves", 15))
    learning_rate = float(getattr(cfg, "tiny_model_learning_rate", 0.1))
    rs = int(getattr(cfg, "random_state", 0))

    def _fit_predict(y_target_fit: np.ndarray, valid_fit: np.ndarray | None = None) -> np.ndarray:
        xf = x_fit if valid_fit is None else x_fit[valid_fit]
        yf = y_target_fit if valid_fit is None else y_target_fit[valid_fit]
        model = _build_tiny_model(
            "lgb", n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, random_state=rs,
        )
        model.fit(xf, yf)
        return np.asarray(model.predict(x_eval), dtype=np.float64)

    # Raw-y tiny baseline on the SAME group-disjoint split (apples-to-apples).
    try:
        raw_pred = _fit_predict(y_fit)
        raw_rmse = _rmse(y_eval, raw_pred)
    except Exception as exc:  # noqa: BLE001 -- baseline failure -> can't gate, keep all
        logger.warning("[CompositeTargetDiscovery.yscale_gate] raw-y baseline fit failed (%s); gate skipped.", exc)
        return kept_specs
    if not np.isfinite(raw_rmse) or raw_rmse <= 0:
        return kept_specs

    tol = float(getattr(cfg, "yscale_holdout_gate_tolerance", 1.10))
    threshold = raw_rmse * tol
    survivors: list = []
    rejected: list[tuple[str, str, float]] = []

    for spec in kept_specs:
        try:
            transform = get_transform(spec.transform_name)
        except UnknownTransformError:
            survivors.append(spec)  # cannot evaluate -> do not penalise
            continue
        params = dict(spec.fitted_params)
        base_cols = _spec_base_columns(spec)
        base_fit = _base_arg(df, base_cols, fit_idx)
        base_eval = _base_arg(_eval_df, base_cols, eval_idx)
        try:
            valid = np.asarray(transform.domain_check(y_fit, base_fit), dtype=bool)
            if valid.shape != y_fit.shape:
                valid = np.ones(y_fit.shape, dtype=bool)
        except Exception:  # noqa: BLE001
            valid = np.ones(y_fit.shape, dtype=bool)
        if int(valid.sum()) < 50:
            survivors.append(spec)
            continue
        base_fit_v = base_fit[valid] if base_fit.ndim == 1 else base_fit[valid, :]
        try:
            t_fit = np.asarray(transform.forward(y_fit[valid], base_fit_v, params), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[yscale_gate] forward failed for %s: %s", spec.name, exc)
            survivors.append(spec)
            continue
        # Fit the tiny model on the transformed target over the valid fit rows, predict on eval rows.
        try:
            model = _build_tiny_model(
                "lgb", n_estimators=n_estimators, num_leaves=num_leaves,
                learning_rate=learning_rate, random_state=rs,
            )
            model.fit(x_fit[valid], t_fit)
            t_hat = np.asarray(model.predict(x_eval), dtype=np.float64)
            y_hat = np.asarray(transform.inverse(t_hat, base_eval, params), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[yscale_gate] fit/inverse failed for %s: %s", spec.name, exc)
            survivors.append(spec)
            continue

        finite = np.isfinite(y_hat)
        n_finite = int(finite.sum())
        if n_finite < max(50, int(0.5 * y_hat.size)):
            rejected.append((spec.name, f"non-finite inverse ({n_finite}/{y_hat.size} finite)", float("inf")))
            continue
        pred_std = float(np.std(y_hat[finite]))
        if y_eval_std > 0 and pred_std < 1e-4 * y_eval_std:
            rejected.append((spec.name, f"collapsed (pred_std={pred_std:.3g} vs y_std={y_eval_std:.3g})", float("inf")))
            continue
        rmse_y = _rmse(y_eval[finite], y_hat[finite])
        if not np.isfinite(rmse_y) or rmse_y > threshold:
            rejected.append((spec.name, f"y-RMSE={rmse_y:.4g} > raw {raw_rmse:.4g} x {tol:.2f}", float(rmse_y)))
            continue
        object.__setattr__(spec, "yscale_holdout_rmse", float(rmse_y))
        object.__setattr__(spec, "yscale_holdout_raw_rmse", float(raw_rmse))
        survivors.append(spec)

    if rejected:
        logger.warning(
            "[CompositeTargetDiscovery.yscale_gate] dropped %d/%d spec(s) that collapse on the "
            "%s y-scale eval (raw-y baseline RMSE=%.4g, tol=%.2f): %s",
            len(rejected), len(kept_specs), _gate_mode, raw_rmse, tol,
            ", ".join(f"{n}({why})" for n, why, _ in rejected),
        )
    else:
        logger.info(
            "[CompositeTargetDiscovery.yscale_gate] all %d spec(s) passed the %s "
            "y-scale eval (raw-y baseline RMSE=%.4g).", len(kept_specs), _gate_mode, raw_rmse,
        )
    return survivors

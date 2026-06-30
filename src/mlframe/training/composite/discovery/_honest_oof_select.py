"""Honest group-OOF reconstruction RMSE as a load-bearing spec-RANK key.

Discovery ranks specs on the optimistic in-sample ``mi_gain`` and a group-INTERNAL CV-RMSE; the two honest constructs
(the y-scale gate, the holdout MI re-score) run AFTER selection and can only delete a promoted spec, never reorder by the
production objective. The structural defect is that honest reconstruction RMSE is a janitor, not a chooser.

This module promotes the production predict-T -> invert-to-y reconstruction RMSE -- measured on the never-touched,
group-DISJOINT honest holdout (``honest_holdout_idx_``, whole upper/tail wells carved at fit entry) -- into a ranking
key the tiny-rerank consumes. The holdout contains the out-of-range base tail where a base-additive inverse
``y = T_hat + alpha*base`` extrapolates and blows up; a group-INTERNAL CV fold (drawn from the screening pool, bounded
by the train base range) never samples that tail, which is why a fragile spec wins the internal CV (~9) yet collapses
on the disjoint holdout (~13.6).

Leak-free by construction: the transform params were fit on the screening pool (``screen_idx``) and the tiny model is
also fit on ``screen_idx``; reconstruction is evaluated on the disjoint ``holdout_idx``. No per-fold refit is needed.
A collapse / non-finite inverse -> ``+inf`` (sinks to the bottom of the rank, exactly as the gate would have dropped it);
a degenerate MEASUREMENT (too few valid rows, transform missing) -> ``None`` so the caller falls back to the existing
group-internal CV-RMSE rather than auto-killing the spec.

100GB-frame rule: only narrow per-column gathers on the bounded screen/holdout samples are materialised; no frame copy.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from pyutilz.parallel import cpu_count_physical

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


def honest_oof_reconstruction_rmse(
    self,
    df: Any,
    target_col: str,
    kept_specs: list,
    usable_features: Sequence[str],
    screen_idx: np.ndarray,
    holdout_idx: np.ndarray | None,
    y_full: np.ndarray,
) -> dict[str, float]:
    """Per-spec predict-T -> invert-to-y reconstruction RMSE on the group-disjoint honest holdout.

    For each spec: fit a tiny model on the screening rows (seen wells) on the spec's transformed target, predict on the
    holdout rows (unseen wells), invert to the y-scale, RMSE vs raw y. Returns ``{spec.name -> rmse}``. A genuine
    COLLAPSE (non-finite / near-constant inverse, RMSE non-finite) maps to ``+inf``; a degenerate MEASUREMENT (no
    holdout, too few valid rows, missing transform) leaves the spec OUT of the dict so the caller falls back to the
    group-internal CV-RMSE. Also records the raw-y honest-OOF baseline on ``self._honest_oof_raw_rmse`` for the gate.
    """
    out: dict[str, float] = {}
    if holdout_idx is None or not kept_specs:
        return out
    screen_idx = np.asarray(screen_idx)
    holdout_idx = np.asarray(holdout_idx)
    if screen_idx.size < 50 or holdout_idx.size < 50:
        return out

    cfg = self.config
    cap = int(getattr(cfg, "yscale_holdout_gate_sample_n", 30_000))
    rng = np.random.default_rng(int(getattr(cfg, "random_state", 0)))

    def _subsample(idx: np.ndarray, n_cap: int) -> np.ndarray:
        if n_cap <= 0 or idx.size <= n_cap:
            return idx
        return np.sort(rng.choice(idx, size=n_cap, replace=False))

    fit_idx = _subsample(screen_idx, cap)
    eval_idx = _subsample(holdout_idx, cap)
    feats = list(usable_features)
    x_fit = self._build_feature_matrix(df, feats, fit_idx)
    x_eval = self._build_feature_matrix(df, feats, eval_idx)
    y_fit = np.asarray(y_full)[fit_idx].astype(np.float64)
    y_eval = np.asarray(y_full)[eval_idx].astype(np.float64)
    y_eval_std = float(np.std(y_eval))

    n_estimators = int(getattr(cfg, "tiny_model_n_estimators", 60))
    num_leaves = int(getattr(cfg, "tiny_model_num_leaves", 15))
    learning_rate = float(getattr(cfg, "tiny_model_learning_rate", 0.1))
    rs = int(getattr(cfg, "random_state", 0))

    def _new_model():
        return _build_tiny_model(
            "lgb", n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, random_state=rs,
        )

    try:
        raw_model = _new_model()
        raw_model.fit(x_fit, y_fit)
        raw_rmse = _rmse(y_eval, np.asarray(raw_model.predict(x_eval), dtype=np.float64))
        self._honest_oof_raw_rmse = float(raw_rmse) if np.isfinite(raw_rmse) else float("nan")
    except Exception as exc:  # noqa: BLE001 -- baseline failure -> no ranking key produced
        logger.warning("[CompositeTargetDiscovery.honest_oof_select] raw-y baseline fit failed (%s); selector skipped.", exc)
        return out

    def _score_one(spec) -> tuple[str, float | None]:
        try:
            transform = get_transform(spec.transform_name)
        except UnknownTransformError:
            return spec.name, None
        params = dict(getattr(spec, "fitted_params", {}) or {})
        base_cols = _spec_base_columns(spec)
        base_fit = _base_arg(df, base_cols, fit_idx)
        base_eval = _base_arg(df, base_cols, eval_idx)
        try:
            valid = np.asarray(transform.domain_check(y_fit, base_fit), dtype=bool)
            if valid.shape != y_fit.shape:
                valid = np.ones(y_fit.shape, dtype=bool)
        except Exception:  # noqa: BLE001
            valid = np.ones(y_fit.shape, dtype=bool)
        if int(valid.sum()) < 50:
            return spec.name, None
        base_fit_v = base_fit[valid] if base_fit.ndim == 1 else base_fit[valid, :]
        try:
            t_fit = np.asarray(transform.forward(y_fit[valid], base_fit_v, params), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 -- cannot transform -> fall back
            logger.debug("[honest_oof_select] forward failed for %s: %s", spec.name, exc)
            return spec.name, None
        try:
            model = _new_model()
            model.fit(x_fit[valid], t_fit)
            t_hat = np.asarray(model.predict(x_eval), dtype=np.float64)
            y_hat = np.asarray(transform.inverse(t_hat, base_eval, params), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 -- fit/inverse blew up -> fall back
            logger.debug("[honest_oof_select] fit/inverse failed for %s: %s", spec.name, exc)
            return spec.name, None
        finite = np.isfinite(y_hat)
        if int(finite.sum()) < max(50, int(0.5 * y_hat.size)):
            return spec.name, float("inf")  # non-finite inverse -> genuine collapse
        pred_std = float(np.std(y_hat[finite]))
        if y_eval_std > 0 and pred_std < 1e-4 * y_eval_std:
            return spec.name, float("inf")  # collapsed to ~constant -> genuine collapse
        rmse_y = _rmse(y_eval[finite], y_hat[finite])
        if not np.isfinite(rmse_y):
            return spec.name, float("inf")
        return spec.name, float(rmse_y)

    # Per-spec scores are independent (each fits its own tiny model, reads shared read-only arrays). LightGBM releases
    # the GIL, so thread across physical cores.
    n_jobs = min(len(kept_specs), cpu_count_physical())
    if n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(_score_one)(s) for s in kept_specs
        )
    else:
        results = [_score_one(s) for s in kept_specs]

    for name, score in results:
        if score is not None:
            out[name] = score
    return out

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

from mlframe.training.composite.estimator.shared import smeared_prediction
from ._spec_shared import spec_base_columns, rmse

import logging
import threading
from typing import Any, Sequence

import numpy as np
from pyutilz.parallel import cpu_count_physical

from ..transforms import UnknownTransformError, get_transform
from ._causal_lag import causal_lag_predict_rmse, detect_causal_lag_column
from .screening import _extract_column_array, base_arg as _base_arg
from ._screening_tiny import _build_tiny_model
from ._yscale_scoring import median_filled_with_std
from mlframe.training.composite.transforms.shared import call_transform

logger = logging.getLogger(__name__)


def prediction_key(valid: np.ndarray) -> int:
    """A fingerprint of the fit-row mask a prediction was made under; two measurements match only on equal masks."""
    return hash(np.packbits(np.asarray(valid, dtype=bool)).tobytes())


def cached_honest_prediction(self, fit_idx: np.ndarray, eval_idx: np.ndarray, spec_name: str | None = None, valid: np.ndarray | None = None):
    """Honest-OOF's holdout prediction for ``spec_name`` (or the raw baseline when ``None``), if it was made on these rows.

    The honest RMSE gate fits the same tiny model on the same screen rows and predicts the same holdout rows whenever
    both samples fit under their caps, which is when neither draws at random: its numbers were measured identical to
    honest-OOF's to full precision. A prediction is returned only when the fit rows, the eval rows and the spec's fit
    mask all match, so a gate whose sample or domain refinement differs always refits.
    """
    cache = getattr(self, "_honest_oof_predictions", None)
    if not cache or not np.array_equal(cache["fit_idx"], fit_idx) or not np.array_equal(cache["eval_idx"], eval_idx):
        return None
    if spec_name is None:
        return cache["raw"]
    hit = cache["specs"].get(spec_name)
    if hit is None or valid is None or hit[0] != prediction_key(valid):
        return None
    return hit[1]


def _raw_baseline_prediction(new_model, x_fit, y_fit, x_eval):
    """The raw-y tiny model's holdout prediction, or ``None`` (logged) when it cannot be fit, which disables the selector."""
    try:
        raw_model = new_model()
        raw_model.fit(x_fit, y_fit)
        return np.asarray(raw_model.predict(x_eval), dtype=np.float64)
    except Exception as exc:  # -- baseline failure -> no ranking key produced
        logger.warning("[CompositeTargetDiscovery.honest_oof_select] raw-y baseline fit failed (%s); selector skipped.", exc)
        return None


def _honest_lag_floor(df, target_col, eval_idx, y_eval) -> float:
    """The AR failsafe (``y_hat = y_prev``) RMSE on the honest holdout rows, or ``nan`` when there is no usable lag column."""
    lag_col = detect_causal_lag_column(df, target_col)
    if lag_col is None:
        return float("nan")
    try:
        lag_eval = _extract_column_array(df, lag_col, rows=eval_idx).astype(np.float64)
        return float(causal_lag_predict_rmse(lag_eval, y_eval))
    except Exception as exc:  # -- lag probe failure -> no lag floor (raw floor still applies)
        logger.debug("[honest_oof_select] lag floor probe failed for %s: %s", lag_col, exc)
        return float("nan")


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
    group-internal CV-RMSE. Also records the raw-y honest-OOF baseline on ``self._honest_oof_raw_rmse`` and the AR
    failsafe (lag_predict) baseline on ``self._honest_oof_lag_rmse`` so the gate can floor specs against ``min(raw, lag)``.
    """
    from .._row_roles import note_rows

    note_rows("honest_holdout", "select", "honest_oof_rerank", holdout_idx)
    out: dict[str, float] = {}
    # Reset the per-target honest-OOF floor references so a stale value from a prior target cannot leak into this gate.
    self._honest_oof_raw_rmse = float("nan")
    self._honest_oof_lag_rmse = float("nan")
    self._honest_oof_predictions = None
    if holdout_idx is None or not kept_specs:
        return out
    screen_idx = np.asarray(screen_idx)
    holdout_idx = np.asarray(holdout_idx)
    if screen_idx.size < 50 or holdout_idx.size < 50:
        return out

    cfg = self.config
    cap = int(getattr(cfg, "yscale_holdout_gate_sample_n", 30_000))
    rng = np.random.default_rng(int(getattr(cfg, "random_state", 42)))

    def _subsample(idx: np.ndarray, n_cap: int) -> np.ndarray:
        """Cap ``idx`` to ``n_cap`` rows via a sorted random draw (sorted so downstream gathers stay cache-friendly), leaving it unchanged when already within budget."""
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
    rs = int(getattr(cfg, "random_state", 42))
    # Per-spec scoring below fits one tiny LightGBM model per outer joblib thread; LightGBM's own OpenMP thread
    # pool defaults to all physical cores per fit, so outer_threads x inner_OpenMP_threads oversubscribes when
    # more than one spec is scored concurrently. Cap LightGBM to a single thread whenever the outer parallel
    # scoring loop below will actually use more than one worker (mirrors the same n_jobs=1 cap applied to the
    # composite_screening CV-fold loop in _screening_tiny.py).
    _outer_n_jobs = min(len(kept_specs), cpu_count_physical())
    _inner_n_jobs = 1 if _outer_n_jobs > 1 else -1

    def _new_model(inner_n_jobs: int = -1):
        """Build a fresh tiny LightGBM regressor with the configured (small) capacity, so each spec/baseline gets an unfitted model rather than reusing fitted state."""
        return _build_tiny_model(
            "lgb", n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, random_state=rs, inner_n_jobs=inner_n_jobs,
        )

    raw_pred = _raw_baseline_prediction(_new_model, x_fit, y_fit, x_eval)
    if raw_pred is None:
        return out
    raw_rmse = rmse(y_eval, raw_pred)
    self._honest_oof_raw_rmse = float(raw_rmse) if np.isfinite(raw_rmse) else float("nan")

    # AR failsafe (lag_predict) floor on the SAME group-disjoint holdout rows. On a strong-AR sequential target the
    # deployed model is often just ``y_hat = y_prev``; a composite spec whose honest reconstruction cannot beat that
    # failsafe is worthless even if it beats the raw-y model. Measure the lag baseline here so the gate can floor every
    # spec against ``min(raw, lag)`` rather than raw alone (prod incident: 13.30 ensemble vs 11.58 lag floor).
    self._honest_oof_lag_rmse = _honest_lag_floor(df, target_col, eval_idx, y_eval)

    # Keep what was measured so the honest RMSE gate, which fits the same models on the same rows when both samples are
    # under their caps, can reuse a prediction instead of refitting it (see ``cached_honest_prediction``).
    _spec_preds: dict = {}
    _pred_lock = threading.Lock()
    self._honest_oof_predictions = {"fit_idx": fit_idx, "eval_idx": eval_idx, "raw": raw_pred, "specs": _spec_preds}

    def _score_one(spec) -> tuple[str, float | None]:
        """Fit-and-invert one spec's transform on the screen/holdout split, returning ``(spec.name, rmse)``; ``rmse=None`` signals a degenerate measurement (unknown transform, no valid rows) that should NOT auto-kill the spec, distinct from ``+inf`` which signals a genuine reconstruction collapse."""
        try:
            transform = get_transform(spec.transform_name)
        except UnknownTransformError:
            return spec.name, None
        params = dict(getattr(spec, "fitted_params", {}) or {})
        base_cols = spec_base_columns(spec)
        base_fit = _base_arg(df, base_cols, fit_idx)
        base_eval = _base_arg(df, base_cols, eval_idx)
        try:
            valid = np.asarray(transform.domain_check(y_fit, base_fit), dtype=bool)
            if valid.shape != y_fit.shape:
                valid = np.ones(y_fit.shape, dtype=bool)
        except Exception as exc:  # -- domain_check itself crashed; fails OPEN (no restriction), so surface it loudly
            logger.warning("[honest_oof_select] domain_check raised for %s; treating all rows as in-domain: %s", spec.name, exc)
            valid = np.ones(y_fit.shape, dtype=bool)
        if int(valid.sum()) < 50:
            return spec.name, None
        base_fit_v = base_fit[valid] if base_fit.ndim == 1 else base_fit[valid, :]
        try:
            t_fit = np.asarray(call_transform(transform, "forward", y_fit[valid], base_fit_v, params), dtype=np.float64)
        except Exception as exc:  # -- cannot transform -> fall back
            logger.debug("[honest_oof_select] forward failed for %s: %s", spec.name, exc)
            return spec.name, None
        try:
            model = _new_model(inner_n_jobs=_inner_n_jobs)
            model.fit(x_fit[valid], t_fit)
            t_hat = np.asarray(model.predict(x_eval), dtype=np.float64)
            # Smearing for curved unary inverses: score the conditional mean of y, as the trained composite predicts.
            y_hat = smeared_prediction(spec.transform_name, model, x_fit[valid], t_fit, t_hat, lambda t: call_transform(transform, "inverse", t, base_eval, params))
            with _pred_lock:
                _spec_preds[spec.name] = (prediction_key(valid), y_hat)
        except Exception as exc:  # -- fit/inverse blew up -> fall back
            logger.debug("[honest_oof_select] fit/inverse failed for %s: %s", spec.name, exc)
            return spec.name, None
        finite = np.isfinite(y_hat)
        if int(finite.sum()) < max(50, int(0.5 * y_hat.size)):
            return spec.name, float("inf")  # non-finite inverse -> genuine collapse
        y_hat, pred_std = median_filled_with_std(y_hat, y_fit)  # score what predict() ships, on every eval row
        if y_eval_std > 0 and pred_std < 1e-4 * y_eval_std:
            return spec.name, float("inf")  # collapsed to ~constant -> genuine collapse
        rmse_y = rmse(y_eval, y_hat)
        if not np.isfinite(rmse_y):
            return spec.name, float("inf")
        return spec.name, float(rmse_y)

    # Per-spec scores are independent (each fits its own tiny model, reads shared read-only arrays). LightGBM releases
    # the GIL, so thread across physical cores. Each fit is capped to inner_n_jobs=1 above (via ``_inner_n_jobs``)
    # whenever this loop actually runs multi-worker, so LightGBM's own OpenMP pool doesn't oversubscribe on top
    # of the outer joblib threads.
    if _outer_n_jobs > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=_outer_n_jobs, backend="threading", prefer="threads")(delayed(_score_one)(s) for s in kept_specs)
    else:
        results = [_score_one(s) for s in kept_specs]

    out.update({name: score for name, score in results if score is not None})
    return out

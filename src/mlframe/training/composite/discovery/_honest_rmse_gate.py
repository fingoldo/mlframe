"""Honest-holdout OOS predictive-error (RMSE) gate for ``CompositeTargetDiscovery``.

Why this exists
---------------
The honest holdout (``_honest_holdout.py``) re-scores the FINAL specs with the same
``MI(T, X) - MI(y, X)`` statistic the screen used. MI is monotone-invariant and
bias-inflated: a transform can RAISE MI while WORSENING y-scale OOS RMSE -- the
canonical case being a ratio dividing by a small noisy base, which amplifies noise
into the reconstructed y while the forward MI still climbs. The ``screening="mi"``
config additionally has NO out-of-sample predictive gate anywhere end-to-end (the
tiny-rerank / raw-baseline gates only run for ``tiny_model`` / ``hybrid``).

What it does
------------
On the SAME never-touched honest holdout, replicate the actual prediction objective
for every final spec: fit the tiny screening model on (a capped sample of) the
screening rows for the spec's ``T`` (already-fitted params, no refit), predict on
the holdout, invert to y-scale, and compare the holdout y-RMSE against the same
tiny model trained on raw ``y`` over the identical rows. A spec whose y-scale
holdout RMSE loses to raw beyond ``honest_rmse_gate_tolerance`` (or whose inverse
degenerates to non-finite / near-constant output) is DROPPED. Survivors get
``honest_holdout_rmse`` / ``honest_holdout_raw_rmse`` / ``honest_holdout_rmse_gain``
stamped alongside the MI-based ``honest_holdout_gain``.

Relation to the y-scale group holdout gate (``_yscale_holdout_gate.py``): that gate
targets the unseen-GROUP inverse-collapse regime and no-ops without group ids or a
val frame; this gate is the i.i.d. holdout analogue that always runs when the
honest holdout exists, so plain non-grouped runs (including ``screening="mi"``) get
an OOS predictive floor too. Default ON; opt out via ``honest_rmse_gate_enabled``.

100GB-frame rule: only capped per-column gathers on the fit/eval row samples are
materialised; never a frame copy.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..transforms import UnknownTransformError, get_transform
from .screening import _extract_column_array
from ._rejection_ledger import RejectStage, ledger_append
from ._screening_tiny import _build_tiny_model

logger = logging.getLogger(__name__)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain RMSE in float64 (upcast so wide-range targets don't lose precision)."""
    d = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _spec_base_columns(spec: Any) -> list[str]:
    """Ordered base-column list the spec's forward/inverse needs (primary + multi-base extras); empty for unary."""
    extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
    if not spec.base_column:
        return []
    return [spec.base_column, *extra]


def _base_arg(df: Any, base_columns: Sequence[str], rows: np.ndarray) -> np.ndarray:
    """Materialise the ``base`` argument shape ``transform.forward/inverse`` expects on the given rows."""
    if not base_columns:
        return np.zeros(np.asarray(rows).size, dtype=np.float64)
    if len(base_columns) == 1:
        return _extract_column_array(df, base_columns[0], rows=rows).astype(np.float64)
    return np.column_stack([_extract_column_array(df, c, rows=rows).astype(np.float64) for c in base_columns])


def apply_honest_rmse_gate(
    self: Any,
    df: Any,
    target_col: str,
    kept_specs: list,
    usable_features: Sequence[str],
    screen_idx: np.ndarray,
    holdout_idx: np.ndarray | None,
    y_full: np.ndarray,
) -> list:
    """Drop specs whose predict-T -> invert-to-y holdout RMSE loses to a raw-y tiny baseline.

    Returns the surviving spec list (possibly empty -- the correct outcome when every
    candidate worsens the y-scale prediction). No-ops (keeps every spec) when the gate is
    disabled, there are no specs, the honest holdout is absent/too small, or the raw-y
    baseline itself cannot be fit (nothing sound to gate against).
    """
    cfg = self.config
    if not getattr(cfg, "honest_rmse_gate_enabled", True) or not kept_specs:
        return kept_specs
    if holdout_idx is None or np.asarray(holdout_idx).size < 50:
        logger.info(
            "[CompositeTargetDiscovery.honest_rmse_gate] no usable honest holdout " "(honest_holdout_frac disabled or too small) -- OOS RMSE gate skipped."
        )
        return kept_specs

    screen_idx = np.asarray(screen_idx)
    holdout_idx = np.asarray(holdout_idx)
    cap = int(getattr(cfg, "honest_rmse_gate_sample_n", 20_000))
    rng = np.random.default_rng(int(getattr(cfg, "random_state", 0)))

    def _subsample(idx: np.ndarray) -> np.ndarray:
        """Cap ``idx`` to the gate budget via a sorted seeded draw; unchanged when already within budget."""
        if cap <= 0 or idx.size <= cap:
            return idx
        return np.sort(rng.choice(idx, size=cap, replace=False))

    fit_idx = _subsample(screen_idx)
    eval_idx = _subsample(holdout_idx)
    if fit_idx.size < 50 or eval_idx.size < 50:
        return kept_specs

    feats = list(usable_features)
    x_fit = self._build_feature_matrix(df, feats, fit_idx)
    x_eval = self._build_feature_matrix(df, feats, eval_idx)
    y_fit = np.asarray(y_full)[fit_idx].astype(np.float64)
    y_eval = np.asarray(y_full)[eval_idx].astype(np.float64)
    y_eval_std = float(np.std(y_eval[np.isfinite(y_eval)])) if y_eval.size else 0.0

    n_estimators = int(getattr(cfg, "tiny_model_n_estimators", 60))
    num_leaves = int(getattr(cfg, "tiny_model_num_leaves", 15))
    learning_rate = float(getattr(cfg, "tiny_model_learning_rate", 0.1))
    rs = int(getattr(cfg, "random_state", 0))

    def _fit_predict(target_fit: np.ndarray, row_mask: np.ndarray | None = None) -> np.ndarray:
        """Fit a fresh tiny model on ``target_fit`` (optionally masked to the transform's valid fit rows) and predict on the shared holdout matrix."""
        xf = x_fit if row_mask is None else x_fit[row_mask]
        tf = target_fit if row_mask is None else target_fit[row_mask]
        model = _build_tiny_model(
            "lgb", n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, random_state=rs,
        )
        model.fit(xf, tf)
        return np.asarray(model.predict(x_eval), dtype=np.float64)

    try:
        raw_rmse = _rmse(y_eval, _fit_predict(y_fit))
    except Exception as exc:  # -- no baseline, no sound gate
        logger.warning("[CompositeTargetDiscovery.honest_rmse_gate] raw-y baseline fit failed (%s); gate skipped.", exc)
        return kept_specs
    if not np.isfinite(raw_rmse) or raw_rmse <= 0:
        return kept_specs

    tol = float(getattr(cfg, "honest_rmse_gate_tolerance", 1.05))
    threshold = raw_rmse * tol
    survivors: list = []
    rejected: list[tuple[str, str]] = []

    for spec in kept_specs:
        try:
            transform = get_transform(spec.transform_name)
        except UnknownTransformError:
            survivors.append(spec)  # cannot evaluate -> never penalise
            continue
        params = dict(spec.fitted_params)
        base_cols = _spec_base_columns(spec)
        base_fit = _base_arg(df, base_cols, fit_idx)
        base_eval = _base_arg(df, base_cols, eval_idx)
        # Same two-stage domain gate the screen applies, so T is fit on the spec's real domain.
        try:
            valid = np.asarray(transform.domain_check(y_fit, base_fit), dtype=bool)
            if valid.shape != y_fit.shape:
                valid = np.ones(y_fit.shape, dtype=bool)
        except Exception:
            valid = np.ones(y_fit.shape, dtype=bool)
        _dcf = getattr(transform, "domain_check_fitted", None)
        if _dcf is not None:
            try:
                vf = np.asarray(_dcf(y_fit, base_fit, params), dtype=bool)
                if vf.shape == valid.shape:
                    valid = valid & vf
            except Exception as e:  # -- treat as no refinement
                logger.debug("honest_rmse_gate domain_check_fitted failed for %s: %s", spec.name, e)
        if int(valid.sum()) < 50:
            survivors.append(spec)
            continue
        base_fit_v = base_fit[valid] if base_fit.ndim == 1 else base_fit[valid, :]
        try:
            t_fit = np.asarray(transform.forward(y_fit[valid], base_fit_v, params), dtype=np.float64)
            t_hat = _fit_predict_masked(_fit_predict, t_fit, valid)
            y_hat = np.asarray(transform.inverse(t_hat, base_eval, params), dtype=np.float64)
        except Exception as exc:  # -- a spec the tiny pipeline cannot evaluate keeps its MI verdict
            logger.debug("honest_rmse_gate fit/inverse failed for %s: %s", spec.name, exc)
            survivors.append(spec)
            continue

        finite = np.isfinite(y_hat)
        n_finite = int(finite.sum())
        _led_kw = dict(base_column=getattr(spec, "base_column", ""), transform_name=getattr(spec, "transform_name", ""))
        if n_finite < max(50, int(0.5 * y_hat.size)):
            _r = f"non-finite inverse on holdout ({n_finite}/{y_hat.size} finite)"
            rejected.append((spec.name, _r))
            ledger_append(self, spec_name=spec.name, stage=RejectStage.HONEST_RMSE, reason=_r,
                          numbers={"n_finite": n_finite, "n_total": int(y_hat.size)}, **_led_kw)
            continue
        pred_std = float(np.std(y_hat[finite]))
        if y_eval_std > 0 and pred_std < 1e-4 * y_eval_std:
            _r = f"collapsed inverse (pred_std={pred_std:.3g} vs y_std={y_eval_std:.3g})"
            rejected.append((spec.name, _r))
            ledger_append(self, spec_name=spec.name, stage=RejectStage.HONEST_RMSE, reason=_r,
                          numbers={"pred_std": pred_std, "y_eval_std": y_eval_std}, **_led_kw)
            continue
        rmse_y = _rmse(y_eval[finite], y_hat[finite])
        if not np.isfinite(rmse_y) or rmse_y > threshold:
            _r = f"honest y-RMSE={rmse_y:.4g} > raw {raw_rmse:.4g} x {tol:.2f}"
            rejected.append((spec.name, _r))
            ledger_append(self, spec_name=spec.name, stage=RejectStage.HONEST_RMSE, reason=_r,
                          numbers={"rmse_y": float(rmse_y), "raw_rmse": float(raw_rmse), "tol": float(tol)}, **_led_kw)
            continue
        object.__setattr__(spec, "honest_holdout_rmse", float(rmse_y))
        object.__setattr__(spec, "honest_holdout_raw_rmse", float(raw_rmse))
        object.__setattr__(spec, "honest_holdout_rmse_gain", float(raw_rmse - rmse_y))
        survivors.append(spec)

    if rejected:
        logger.warning(
            "[CompositeTargetDiscovery.honest_rmse_gate] dropped %d/%d spec(s) whose y-scale honest-holdout "
            "RMSE loses to the raw-y tiny baseline (raw RMSE=%.4g, tol=%.2f): %s",
            len(rejected), len(kept_specs), raw_rmse, tol,
            ", ".join(f"{n}({why})" for n, why in rejected),
        )
    else:
        logger.info(
            "[CompositeTargetDiscovery.honest_rmse_gate] all %d spec(s) passed the honest-holdout "
            "y-scale RMSE gate (raw-y baseline RMSE=%.4g).", len(kept_specs), raw_rmse,
        )
    return survivors


def _fit_predict_masked(fit_predict: Any, t_fit_valid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Adapter: ``t_fit_valid`` is already gathered to the valid rows, while ``fit_predict`` masks internally.

    Rebuilds a full-length target where invalid rows are placeholders and passes the mask so the tiny model
    trains on exactly the valid rows with their transformed targets.
    """
    if bool(valid.all()):
        return np.asarray(fit_predict(t_fit_valid), dtype=np.float64)
    full = np.zeros(valid.shape[0], dtype=np.float64)
    full[valid] = t_fit_valid
    return np.asarray(fit_predict(full, valid), dtype=np.float64)

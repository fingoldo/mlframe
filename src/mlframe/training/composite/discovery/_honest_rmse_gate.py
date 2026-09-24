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

_DUPLICATE_OF_RAW_RMSE_FRAC = 1e-3
"""How close a spec's honest y-RMSE must be to raw's before it is even considered a possible copy of raw."""

_DUPLICATE_OF_RAW_MIN_CORR = 0.9999
"""And how strongly its reconstruction must track the raw model's prediction for it to be one."""

from ._spec_shared import spec_base_columns, rmse
from ._honest_oof_select import cached_honest_prediction

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..estimator._smearing import N_SMEAR_QUANTILES, SMEARED_TRANSFORMS, smeared_inverse
from ..transforms import UnknownTransformError, get_transform
from .screening import _extract_column_array
from ._yscale_scoring import median_filled_with_std
from ._rejection_ledger import RejectStage, ledger_append
from .._row_roles import note_rows
from ._rejection_ledger import gate_error_reject as _gate_error_reject
from ._rejection_ledger import spec_inverse
from mlframe.training.composite.transforms._call_gateway import call_transform

logger = logging.getLogger(__name__)


_GAIN_SIGNIFICANCE_Z: float = 2.0
"""Standard errors a gain must clear to count as measured rather than as noise. Used for the gate's own reporting and
exported on each spec so the ship/no-ship floor downstream can be noise-aware instead of a fixed constant."""


def _paired_rmse_gain_se(y_true: np.ndarray, y_hat_raw: np.ndarray, y_hat_spec: np.ndarray, raw_rmse: float, spec_rmse: float) -> float:
    """Standard error of ``RMSE_raw - RMSE_spec`` from the PAIRED per-row squared errors; NaN when undefined.

    Both predictions cover the same holdout rows from the same model class, so the difference of squared errors is
    paired and its mean is ``MSE_raw - MSE_spec``. The delta method turns that into the RMSE scale::

        RMSE_raw - RMSE_spec = (MSE_raw - MSE_spec) / (RMSE_raw + RMSE_spec)
        se(dRMSE)            ~ se(mean of per-row squared-error differences) / (RMSE_raw + RMSE_spec)

    O(n) on arrays the gate already holds, with no extra model fits. The raw prediction used to be discarded on the
    line that computed its RMSE; pairing against it is the cheapest available evidence for whether a gain is real.
    """
    denom = raw_rmse + spec_rmse
    if y_true.size < 30 or denom <= 0:
        return float("nan")
    d = (y_true - y_hat_raw) ** 2 - (y_true - y_hat_spec) ** 2
    d = d[np.isfinite(d)]
    if d.size < 30:
        return float("nan")
    return float(np.std(d, ddof=1) / np.sqrt(d.size) / denom)


def _log_gate_summary(survivors: list, n_candidates: int, raw_rmse: float, tol: float) -> None:
    """Log how the gate's survivors actually did against raw, split into beat-raw / beat-by-z-SE / merely-within-tol.

    "Passed" means "not more than tol worse than raw", which includes being worse. Reporting only the count reads
    as an endorsement: a production run logged "all 10 spec(s) passed" for specs that later scored y-scale R2 of
    -0.024 and -0.026.
    """
    _better = [s for s in survivors if (getattr(s, "honest_holdout_rmse_gain", None) or 0.0) > 0]
    _sig = [
        s
        for s in _better
        if (getattr(s, "honest_holdout_rmse_gain_se", None) or 0.0) > 0 and s.honest_holdout_rmse_gain >= _GAIN_SIGNIFICANCE_Z * s.honest_holdout_rmse_gain_se
    ]
    logger.info(
        "[CompositeTargetDiscovery.honest_rmse_gate] %d/%d spec(s) survived the honest-holdout y-scale RMSE gate "
        "(raw-y baseline RMSE=%.4g, tol=%.2f): %d beat raw, of which %d by at least %.1f paired standard errors; "
        "the remaining %d are within tolerance but do NOT beat raw.",
        len(survivors), n_candidates, raw_rmse, tol,
        len(_better), len(_sig), _GAIN_SIGNIFICANCE_Z, len(survivors) - len(_better),
    )


def _base_arg(df: Any, base_columns: Sequence[str], rows: np.ndarray) -> np.ndarray:
    """Materialise the ``base`` argument shape ``transform.forward/inverse`` expects on the given rows."""
    if not base_columns:
        return np.zeros(np.asarray(rows).size, dtype=np.float64)
    if len(base_columns) == 1:
        return _extract_column_array(df, base_columns[0], rows=rows).astype(np.float64)
    return np.column_stack([_extract_column_array(df, c, rows=rows).astype(np.float64) for c in base_columns])


def _spec_fit_mask(transform, y_fit, base_fit, params, spec_name: str) -> np.ndarray:
    """The rows a spec's T is fit on: the transform's domain, refined by its fitted-domain check (the screen's two stages)."""
    try:
        valid = np.asarray(transform.domain_check(y_fit, base_fit), dtype=bool)
        if valid.shape != y_fit.shape:
            valid = np.ones(y_fit.shape, dtype=bool)
    except Exception as e:
        logger.warning("domain_check failed, treating all rows as valid: %s", e)
        valid = np.ones(y_fit.shape, dtype=bool)
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None:
        try:
            vf = np.asarray(_dcf(y_fit, base_fit, params), dtype=bool)
            if vf.shape == valid.shape:
                valid = valid & vf
        except Exception as e:  # -- treat as no refinement
            logger.debug("honest_rmse_gate domain_check_fitted failed for %s: %s", spec_name, e)
    return valid


def _correlation_if_duplicate_of_raw(y_hat: np.ndarray, raw_pred: np.ndarray, rmse_y: float, raw_rmse: float) -> float | None:
    """The spec-vs-raw prediction correlation when the reconstruction duplicates the raw model, else ``None``.

    A spec whose reconstruction IS the raw model's prediction ships a second trained model for nothing: no lift and no
    ensemble diversity, since the two prediction vectors are the same. The 5% tolerance keeps specs that trade a little
    accuracy for a different view of the data, not copies of raw. The canonical case is a unary transform whose T is y
    shifted by a constant on data with none of the structure it models.
    """
    if raw_pred.size <= 2 or abs(rmse_y - raw_rmse) > _DUPLICATE_OF_RAW_RMSE_FRAC * raw_rmse:
        return None
    if float(np.std(y_hat)) <= 0 or float(np.std(raw_pred)) <= 0:
        return None
    corr = float(np.corrcoef(y_hat, raw_pred)[0, 1])
    return corr if corr >= _DUPLICATE_OF_RAW_MIN_CORR else None


def _digest(a: Any) -> bytes:
    """Content digest of an array (or None) for the gate memo keys."""
    import hashlib

    if a is None:
        return b"none"
    arr = np.ascontiguousarray(np.asarray(a))
    return hashlib.blake2b(f"{arr.dtype.str}{arr.shape}".encode() + arr.view(np.uint8).data, digest_size=16).digest()


def _gate_matrix(self: Any, df: Any, feats: list, rows: np.ndarray) -> np.ndarray:
    """The feature matrix for ``rows``, through the honest-stage memo when one is open.

    The gate runs twice per fit (the selection half, then the record-only report half) over the same fit rows and the two
    halves of one holdout: with the memo the fit rows are gathered once and the whole holdout once, and each half is a
    slice of it, instead of four gathers.
    """
    memo = getattr(self, "_honest_gate_memo", None)
    if memo is None or memo["key"] != (id(df), tuple(feats)):
        return cast(np.ndarray, self._build_feature_matrix(df, feats, rows))
    rows = np.asarray(rows)
    k = _digest(rows)
    if k in memo["mats"]:
        return cast(np.ndarray, memo["mats"][k])
    hold = memo.get("holdout")
    if hold is not None and np.isin(rows, hold).all():
        if memo.get("holdout_x") is None:
            memo["holdout_x"] = self._build_feature_matrix(df, feats, hold)
        pos = np.searchsorted(hold, rows)
        return cast(np.ndarray, memo["holdout_x"][pos])
    memo["mats"][k] = self._build_feature_matrix(df, feats, rows)
    return cast(np.ndarray, memo["mats"][k])


def _holdout_fit_context(self: Any, df: Any, usable_features: Sequence[str], screen_idx: np.ndarray, holdout_idx: np.ndarray,
                         y_full: np.ndarray) -> tuple | None:
    """The rows, matrices and tiny-model fitter one gate pass scores with, or None when a side is below 50 rows.

    Returns ``(fit_idx, eval_idx, y_fit, y_eval, y_eval_std, fit_predict, last_residual_q)``: ``fit_predict(target, mask)``
    fits a fresh tiny model on the (masked) fit rows and predicts the eval rows, and ``last_residual_q["q"]`` holds the
    residual quantiles of its last fit for the smearing correction.
    """
    cfg = self.config
    screen_idx, holdout_idx = np.asarray(screen_idx), np.asarray(holdout_idx)
    cap = int(getattr(cfg, "honest_rmse_gate_sample_n", 20_000))
    rng = np.random.default_rng(int(getattr(cfg, "random_state", 42)))

    def _subsample(idx: np.ndarray) -> np.ndarray:
        """Cap ``idx`` to the gate budget via a sorted seeded draw; unchanged when already within budget."""
        if cap <= 0 or idx.size <= cap:
            return idx
        return np.sort(rng.choice(idx, size=cap, replace=False))

    fit_idx = _subsample(screen_idx)
    eval_idx = _subsample(holdout_idx)
    if fit_idx.size < 50 or eval_idx.size < 50:
        return None

    feats = list(usable_features)
    x_fit = _gate_matrix(self, df, feats, fit_idx)
    x_eval = _gate_matrix(self, df, feats, eval_idx)
    y_fit = np.asarray(y_full)[fit_idx].astype(np.float64)
    y_eval = np.asarray(y_full)[eval_idx].astype(np.float64)
    y_eval_std = float(np.std(y_eval[np.isfinite(y_eval)])) if y_eval.size else 0.0

    n_estimators = int(getattr(cfg, "tiny_model_n_estimators", 60))
    num_leaves = int(getattr(cfg, "tiny_model_num_leaves", 15))
    learning_rate = float(getattr(cfg, "tiny_model_learning_rate", 0.1))
    rs = int(getattr(cfg, "random_state", 42))

    # Residual quantiles of the last tiny-model fit on its own fit rows, for the smearing correction (``_smearing``).
    last_residual_q: dict = {"q": None}

    def _fit_predict(target_fit: np.ndarray, row_mask: np.ndarray | None = None) -> np.ndarray:
        """Fit a fresh tiny model on ``target_fit`` (optionally masked to the transform's valid fit rows) and predict on the shared holdout matrix."""
        xf = x_fit if row_mask is None else x_fit[row_mask]
        tf = target_fit if row_mask is None else target_fit[row_mask]
        # The report pass fits exactly what the selection pass fitted (same fit rows, target and mask): reuse the model.
        memo = getattr(self, "_honest_gate_memo", None)
        key = (_digest(fit_idx), _digest(target_fit), _digest(row_mask))
        if memo is not None and key in memo["fits"]:
            model, last_residual_q["q"] = memo["fits"][key]
            return np.asarray(model.predict(x_eval), dtype=np.float64)
        # The native fit on a dataset binned once per (fit matrix, rows) and shared by every spec the gate scores with
        # the label swapped; same booster as the sklearn wrapper's fit, which re-binned the fit rows per spec.
        from ._lgb_shared_fold import fit_on_rows, lgb_params

        rows = np.arange(x_fit.shape[0]) if row_mask is None else np.flatnonzero(row_mask)
        model = fit_on_rows(np.asarray(x_fit), rows, np.asarray(tf, dtype=np.float64), n_estimators=n_estimators,
                            params=lgb_params(num_leaves=num_leaves, learning_rate=learning_rate, random_state=rs, deterministic=False,
                                              num_threads=-1))
        _res = np.asarray(tf, dtype=np.float64) - np.asarray(model.predict(xf), dtype=np.float64)
        _res = _res[np.isfinite(_res)]
        last_residual_q["q"] = np.quantile(_res, (np.arange(N_SMEAR_QUANTILES) + 0.5) / N_SMEAR_QUANTILES) if _res.size >= 4 * N_SMEAR_QUANTILES else None
        if memo is not None:
            memo["fits"][key] = (model, last_residual_q["q"])
        return np.asarray(model.predict(x_eval), dtype=np.float64)

    return fit_idx, eval_idx, y_fit, y_eval, y_eval_std, _fit_predict, last_residual_q

def apply_honest_rmse_gate(
    self: Any,
    df: Any,
    target_col: str,
    kept_specs: list,
    usable_features: Sequence[str],
    screen_idx: np.ndarray,
    holdout_idx: np.ndarray | None,
    y_full: np.ndarray,
    record_only: bool = False,
) -> list:
    """Drop specs whose predict-T -> invert-to-y holdout RMSE loses to a raw-y tiny baseline.

    Returns the surviving spec list (possibly empty -- the correct outcome when every
    candidate worsens the y-scale prediction). No-ops (keeps every spec) when the gate is
    disabled, there are no specs, the honest holdout is absent/too small, or the raw-y
    baseline itself cannot be fit (nothing sound to gate against).

    ``record_only`` re-scores the given specs on rows no decision reads and drops nothing: the numbers the gate decided
    on are conditioned on having passed it, so exporting them as the honest gain carried the winner's curse into the
    cross-target budget. In this mode every spec is kept, the ``honest_holdout_rmse*`` fields are stamped from these
    rows (``None`` where they cannot score the spec), and the gate's own gain moves to ``selection_holdout_rmse_gain``.
    """
    note_rows("honest_holdout", "report" if record_only else "select", "honest_rmse_gate", holdout_idx)
    cfg = self.config
    if not getattr(cfg, "honest_rmse_gate_enabled", True) or not kept_specs:
        return kept_specs
    if holdout_idx is None or np.asarray(holdout_idx).size < 50:
        logger.info("[CompositeTargetDiscovery.honest_rmse_gate] no usable honest holdout (honest_holdout_frac disabled or too small) -- OOS RMSE gate skipped.")
        return kept_specs

    ctx = _holdout_fit_context(self, df, usable_features, screen_idx, holdout_idx, y_full)
    if ctx is None:
        return kept_specs
    fit_idx, eval_idx, y_fit, y_eval, y_eval_std, _fit_predict, _last_residual_q = ctx

    try:
        # Keep the raw PREDICTION VECTOR, not just its RMSE: specs' squared errors pair with it (see _paired_rmse_gain_se).
        _raw_pred = cached_honest_prediction(self, fit_idx, eval_idx)
        if _raw_pred is None:
            _raw_pred = _fit_predict(y_fit)
        _raw_pred = np.asarray(_raw_pred, dtype=np.float64).reshape(-1)
        raw_rmse = rmse(y_eval, _raw_pred)
    except Exception as exc:  # -- no baseline, no sound gate
        logger.warning("[CompositeTargetDiscovery.honest_rmse_gate] raw-y baseline fit failed (%s); gate skipped.", exc)
        return kept_specs
    if not np.isfinite(raw_rmse) or raw_rmse <= 0:
        return kept_specs

    tol = float(getattr(cfg, "honest_rmse_gate_tolerance", 1.05))
    threshold = raw_rmse * tol
    const_rmse = rmse(y_eval, np.full(y_eval.shape, float(np.mean(y_fit))))  # the null a spec must beat (the constant, not the raw tiny model)
    survivors: list = []
    rejected: list[tuple[str, str]] = []

    def _reject(spec: Any, reason: str, numbers: dict | None = None, error: bool = False) -> None:
        """Drop ``spec`` with ``reason``; in ``record_only`` mode keep it and record that these rows could not score it."""
        if record_only:
            _move_rmse_stamps_to_selection(spec)
            survivors.append(spec)
        elif error:
            _gate_error_reject(self, spec, rejected, RejectStage.HONEST_RMSE, reason, with_score=False)
        else:
            rejected.append((spec.name, reason))
            ledger_append(self, spec_name=spec.name, stage=RejectStage.HONEST_RMSE, reason=reason, numbers=numbers or {},
                          base_column=getattr(spec, "base_column", ""), transform_name=getattr(spec, "transform_name", ""))

    for spec in kept_specs:
        try:
            transform = get_transform(spec.transform_name)
        except UnknownTransformError:  # not in the registry, so no predict can invert it either
            _reject(spec, "transform is not registered", error=True)
            continue
        params = dict(spec.fitted_params)
        base_cols = spec_base_columns(spec)
        base_fit = _base_arg(df, base_cols, fit_idx)
        base_eval = _base_arg(df, base_cols, eval_idx)
        valid = _spec_fit_mask(transform, y_fit, base_fit, params, spec.name)
        if int(valid.sum()) < 50:
            if record_only:  # too few valid rows to score on these rows either: no honest number, not the gate's one
                _move_rmse_stamps_to_selection(spec)
            survivors.append(spec)
            continue
        base_fit_v = base_fit[valid] if base_fit.ndim == 1 else base_fit[valid, :]
        try:
            y_hat = cached_honest_prediction(self, fit_idx, eval_idx, spec.name, valid)
            if y_hat is None:
                t_fit = np.asarray(call_transform(transform, "forward", y_fit[valid], base_fit_v, params), dtype=np.float64)
                t_hat = _fit_predict_masked(_fit_predict, t_fit, valid)
                # Score the spec the way the trained composite will predict: with smearing for the curved unary inverses,
                # so a log/cbrt target is judged on the conditional mean of y, not on the (lower) geometric mean.
                _q = _last_residual_q["q"] if spec.transform_name in SMEARED_TRANSFORMS else None
                y_hat = smeared_inverse(spec_inverse(transform, base_eval, params), t_hat, _q)
        except Exception as exc:
            # The same forward/inverse raises at predict time on rows like these; keeping the spec disabled the gate for it.
            _reject(spec, f"fit/inverse raised {type(exc).__name__}: {exc}", error=True)
            continue

        n_finite = int(np.isfinite(y_hat).sum())
        if n_finite < max(50, int(0.5 * y_hat.size)):
            _reject(spec, f"non-finite inverse on holdout ({n_finite}/{y_hat.size} finite)", {"n_finite": n_finite, "n_total": int(y_hat.size)})
            continue
        y_hat, pred_std = median_filled_with_std(y_hat, y_fit)  # score what predict() ships, on every eval row
        if y_eval_std > 0 and pred_std < 1e-4 * y_eval_std:
            _reject(spec, f"collapsed inverse (pred_std={pred_std:.3g} vs y_std={y_eval_std:.3g})", {"pred_std": pred_std, "y_eval_std": y_eval_std})
            continue
        rmse_y = rmse(y_eval, y_hat)
        if record_only:  # every spec here already passed the gate; these rows only measure it, whatever the result
            _move_rmse_stamps_to_selection(spec)
        elif not np.isfinite(rmse_y) or rmse_y > threshold:
            _reject(spec, f"honest y-RMSE={rmse_y:.4g} > raw {raw_rmse:.4g} x {tol:.2f}",
                    {"rmse_y": float(rmse_y), "raw_rmse": float(raw_rmse), "tol": float(tol)})
            continue
        elif np.isfinite(const_rmse) and rmse_y >= const_rmse:
            # The null is the constant, not the raw tiny model: on a signal-free target the raw model overfits noise and
            # loses to the constant, so a composite "beat raw" by 2-3 standard errors while predicting nothing.
            _reject(spec, f"honest y-RMSE={rmse_y:.4g} is no better than the constant train mean ({const_rmse:.4g})",
                    {"rmse_y": float(rmse_y), "constant_rmse": float(const_rmse)})
            continue
        else:
            # A reconstruction that duplicates the raw model ships a second model for nothing (see the helper).
            _corr = _correlation_if_duplicate_of_raw(y_hat, np.asarray(_raw_pred, dtype=np.float64), rmse_y, raw_rmse)
            if _corr is not None:
                _reject(spec, f"reconstruction duplicates the raw model (corr={_corr:.6f}, y-RMSE={rmse_y:.6g} vs raw {raw_rmse:.6g})",
                        {"rmse_y": float(rmse_y), "raw_rmse": float(raw_rmse), "corr_with_raw": _corr})
                continue
        object.__setattr__(spec, "honest_holdout_rmse", float(rmse_y))
        object.__setattr__(spec, "honest_holdout_raw_rmse", float(raw_rmse))
        object.__setattr__(spec, "honest_holdout_rmse_gain", float(raw_rmse - rmse_y))
        object.__setattr__(spec, "honest_holdout_rmse_gain_se", _paired_rmse_gain_se(y_eval, _raw_pred, y_hat, raw_rmse, float(rmse_y)))
        survivors.append(spec)

    if record_only:
        return survivors
    if rejected:
        logger.warning("[CompositeTargetDiscovery.honest_rmse_gate] dropped %d/%d spec(s) whose y-scale honest-holdout RMSE loses to the raw-y "
                       "tiny baseline or the constant (raw RMSE=%.4g, tol=%.2f): %s", len(rejected), len(kept_specs), raw_rmse, tol,
                       ", ".join(f"{n}({why})" for n, why in rejected))
    _log_gate_summary(survivors, len(kept_specs), raw_rmse, tol)
    return survivors


def _move_rmse_stamps_to_selection(spec: Any) -> None:
    """Keep the gate's own gain as ``selection_holdout_rmse_gain`` and clear the honest fields for the report pass to fill.

    Called once per spec in ``record_only`` mode, before that pass stamps the report-half numbers (or leaves them ``None``
    when those rows cannot score the spec): whatever it writes, the number the gate decided on is no longer exported as
    the honest one.
    """
    if getattr(spec, "selection_holdout_rmse_gain", None) is None:
        object.__setattr__(spec, "selection_holdout_rmse_gain", getattr(spec, "honest_holdout_rmse_gain", None))
    for field in ("honest_holdout_rmse", "honest_holdout_raw_rmse", "honest_holdout_rmse_gain", "honest_holdout_rmse_gain_se"):
        object.__setattr__(spec, field, None)


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

"""Measured achievable-ceiling pre-screen for composite-target discovery.

CompositeTargetDiscovery decides whether to build residual-target composites. On a strong-AR sequential target (well log;
lag-1 autocorr ~1) composites cannot beat the ``lag_predict`` failsafe (``y_hat = y_prev``), so running the full MI /
Wilcoxon / tiny-rerank discovery is minutes of wasted compute that ship specs which all lose to the lag. The legacy skip
(``_ar_skip._extreme_ar_discovery_skip``) fired off a crude per-group ``lag1_autocorr >= 0.99`` heuristic gated on the
single ``extreme_ar_group_aware_skip`` flag -- so flipping that one flag OFF disabled the ENTIRE skip and let discovery
churn (the prod footgun).

This module replaces the heuristic with a MEASUREMENT on a bounded subsample (cap ~30k rows). It measures three RMSEs on
a group-disjoint holdout (or a random holdout when no group key is available):

  (a) ``raw_rmse``            -- a tiny model fit on the seen rows predicting raw y, evaluated on the holdout;
  (b) ``lag_rmse``           -- the AR failsafe ``y_hat = y_prev`` on the holdout (reuses ``detect_causal_lag_column`` +
                                ``causal_lag_predict_rmse``); NaN when no causal-lag column exists;
  (c) ``best_composite_rmse`` -- an OPTIMISTIC achievable-composite ceiling: for the best level-carrying base (the causal
                                lag plus the top features by ``|corr(base, y)|``) fit ``alpha,beta`` by OLS of y on the
                                base on the seen rows, residualise ``T = y - (alpha*base + beta)``, fit a tiny model on
                                the seen rows predicting T, invert ``y_hat = T_hat + alpha*base + beta`` on the holdout,
                                and take the BEST (min RMSE) across candidate bases. Base selection is the only "oracle"
                                (we peek at which base is best); the fit itself is honest group-disjoint, so a residual of
                                pure noise (the strong-AR case) cannot be over-fit into a spurious win.

If the optimistic composite cannot beat ``min(raw, lag)`` by a configurable relative margin, the verdict is SKIP:
even the best-case composite loses to the model / failsafe already deployed, so discovery is futile. This MEASURED signal
is authoritative; the crude per-group lag-1 autocorr is retained only as a cheap observability field (``lag1_ar``) in the
verdict, never as the decision.

Footgun guard (the whole point): the measured precheck reads its OWN flag ``composite_achievable_ceiling_precheck``
(``getattr(config, ..., True)``) and is ORTHOGONAL to ``extreme_ar_group_aware_skip``. Setting the legacy flag to False
disables ONLY the legacy autocorr-heuristic skip in ``_ar_skip`` -- it does NOT disable this measured ceiling precheck nor
the ``lag_predict`` failsafe.

100GB-frame rule: only narrow per-column gathers on the bounded (<=30k) subsample are materialised; no frame copy.

cProfile (n=30k, 6 numeric features, 40 groups; ``python -m mlframe.training.core._achievable_ceiling``): wall is
dominated by the LightGBM tiny-model fits (raw baseline + one per candidate base = ~5 fits), each ~40-90ms warm; the
column gathers + OLS + RMSE are <2ms combined. The candidate-base ``|corr|`` scan over W features is O(W) narrow gathers
(~0.2ms/col). No actionable speedup below the tiny-model fits themselves: they are already the smallest useful model
(60 trees / 15 leaves), the bases are capped (``_MAX_BASE_CANDIDATES``), and the fits run on the bounded subsample -- the
whole precheck is sub-second and negligible next to the multi-minute discovery it decides to skip. The per-base composite
fits are independent and could be threaded, but at ~5 fits the joblib spawn overhead outweighs the gain, so serial stays.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..composite.discovery._causal_lag import causal_lag_predict_rmse, detect_causal_lag_column
from ..composite.discovery._screening_tiny import _build_tiny_model
from ..composite.discovery.screening import _extract_column_array, _is_numeric_column
from ._ar_skip import _recompute_lag1_ar_per_group

logger = logging.getLogger(__name__)

_METHOD = "measured_achievable_ceiling"

# The optimistic composite must beat ``min(raw, lag)`` by at least this RELATIVE RMSE margin to justify discovery.
_DEFAULT_CEILING_MARGIN = 0.02
# SKIP only when the floor (raw model / AR failsafe) is ALREADY a strong predictor: floor_rmse <= this fraction of
# std(y). On a weak / near-random target the floor is close to std(y) and the optimistic-ceiling estimate is unreliable
# (everything is noise), so we PROCEED and let discovery decide rather than skipping on a low-confidence measurement.
# 0.5 => the failsafe must explain >=75% of variance (R^2 >= 1 - 0.5^2) before a "composites are futile" skip fires.
_DEFAULT_STRONG_FLOOR_FRAC = 0.5
# Bounded subsample size -- MI / RMSE saturate far below millions of rows, so measuring on <=30k is representative.
_DEFAULT_CEILING_SAMPLE_N = 30_000
# Fraction of groups (or rows) held out to evaluate the honest disjoint reconstruction.
_DEFAULT_HOLDOUT_FRAC = 0.3
# Below this many usable rows the measurement is too noisy to trust -> never SKIP (proceed, let discovery decide).
_MIN_ROWS_FOR_MEASUREMENT = 200
_MIN_HOLDOUT_ROWS = 50
# Optimistic base set: the causal lag + the top level-carrying features by |corr(base, y)|.
_MAX_BASE_CANDIDATES = 4


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE, or ``nan`` if the result is non-finite (e.g. empty input)."""
    d = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(d * d)))
    return rmse if np.isfinite(rmse) else float("nan")


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """|Pearson corr| over finite-on-both rows; 0.0 on a degenerate (constant / too-few-rows) input."""
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 20:
        return 0.0
    av = a[m] - a[m].mean()
    bv = b[m] - b[m].mean()
    da = float(np.dot(av, av))
    db = float(np.dot(bv, bv))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    c = float(np.dot(av, bv)) / float(np.sqrt(da * db))
    return abs(c) if np.isfinite(c) else 0.0


def _numeric_feature_matrix(df: Any, feature_cols: Sequence[str], target_col: str, rows: np.ndarray):
    """Stack the numeric feature columns into a ``(len(rows), used)`` float32 matrix via narrow gathers.

    Non-numeric columns and the target are skipped; returns ``(X, used_cols)`` (X is ``None`` when nothing numeric
    remains). Only the requested rows are materialised -- no whole-frame conversion.
    """
    used: list[str] = []
    cols: list[np.ndarray] = []
    for c in feature_cols:
        if c == target_col:
            continue
        if not _is_numeric_column(df, c):
            continue
        try:
            arr = _extract_column_array(df, c, rows=rows).astype(np.float64, copy=False)
        except Exception as e:
            logger.debug("swallowed exception in _achievable_ceiling.py: %s", e)
            continue
        if arr.shape[0] != rows.shape[0]:
            continue
        used.append(c)
        cols.append(arr)
    if not cols:
        return None, used
    return np.column_stack(cols), used


def _ols_alpha_beta(base: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope+intercept of ``y ~ base`` over finite-on-both rows; ``(0, mean(y))`` on a degenerate base."""
    m = np.isfinite(base) & np.isfinite(y)
    if int(m.sum()) < 20:
        ym = float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0
        return 0.0, ym
    b = base[m]
    yy = y[m]
    bc = b - b.mean()
    denom = float(np.dot(bc, bc))
    if denom <= 0.0:
        return 0.0, float(yy.mean())
    alpha = float(np.dot(bc, yy - yy.mean()) / denom)
    beta = float(yy.mean() - alpha * b.mean())
    return alpha, beta


def _group_disjoint_split(group_ids: np.ndarray, holdout_frac: float, rng: np.random.Generator, min_groups: int):
    """Hold out whole groups (leave-groups-out). Returns ``(fit_idx, hold_idx)`` or ``None`` when too few groups."""
    uniq = np.unique(group_ids)
    if uniq.size < max(2, min_groups):
        return None
    n_hold = round(uniq.size * holdout_frac)
    n_hold = max(1, min(n_hold, uniq.size - 1))
    hold_groups = rng.permutation(uniq)[:n_hold]
    hold_mask = np.isin(group_ids, hold_groups)
    fit_idx = np.nonzero(~hold_mask)[0]
    hold_idx = np.nonzero(hold_mask)[0]
    if fit_idx.size < _MIN_HOLDOUT_ROWS or hold_idx.size < _MIN_HOLDOUT_ROWS:
        return None
    return fit_idx, hold_idx


def _random_split(n: int, holdout_frac: float, rng: np.random.Generator):
    """Random fit/holdout index split (sorted output), clamping the holdout size to ``[1, n-1]``."""
    n_hold = round(n * holdout_frac)
    n_hold = max(1, min(n_hold, n - 1))
    perm = rng.permutation(n)
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])


def _pick_base_candidates(df: Any, feature_cols: Sequence[str], target_col: str, lag_col: Optional[str],
                          y_sub: np.ndarray, rows_sub: np.ndarray, max_bases: int) -> list[str]:
    """Optimistic level-carrying base set: the causal lag (first) + top numeric features by ``|corr(feature, y)|``."""
    cands: list[str] = []
    if lag_col:
        cands.append(lag_col)
    scored: list[tuple[float, str]] = []
    for c in feature_cols:
        if c == target_col or c == lag_col:
            continue
        if not _is_numeric_column(df, c):
            continue
        try:
            col = _extract_column_array(df, c, rows=rows_sub).astype(np.float64, copy=False)
        except Exception as e:
            logger.debug("swallowed exception in _achievable_ceiling.py: %s", e)
            continue
        if col.shape != y_sub.shape:
            continue
        cc = _abs_corr(col, y_sub)
        if cc > 0.0:
            scored.append((cc, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    for _, c in scored:
        if len(cands) >= max_bases:
            break
        cands.append(c)
    return cands


def _composite_rmse_for_base(
    *, base_fit: np.ndarray, base_hold: np.ndarray, y_fit: np.ndarray, y_hold: np.ndarray,
    x_fit: np.ndarray, x_hold: np.ndarray, model_factory, y_hold_std: float,
) -> float:
    """Honest disjoint reconstruction RMSE for a single linear-residual base. ``inf`` on collapse / non-finite inverse."""
    alpha, beta = _ols_alpha_beta(base_fit, y_fit)
    t_fit = y_fit - (alpha * base_fit + beta)
    # T is NaN wherever base / y is; the tree tolerates NaN in x (Ridge fallback imputes), so gate only on a finite target.
    fit_ok = np.isfinite(t_fit)
    if int(fit_ok.sum()) < _MIN_HOLDOUT_ROWS:
        return float("inf")
    try:
        model = model_factory()
        model.fit(x_fit[fit_ok], t_fit[fit_ok])
        t_hat = np.asarray(model.predict(x_hold), dtype=np.float64)
    except Exception as exc:
        logger.debug("[achievable_ceiling] composite fit failed: %s", exc)
        return float("inf")
    y_hat = t_hat + alpha * base_hold + beta
    finite = np.isfinite(y_hat) & np.isfinite(y_hold)
    if int(finite.sum()) < max(_MIN_HOLDOUT_ROWS, int(0.5 * y_hat.size)):
        return float("inf")
    pred_std = float(np.std(y_hat[finite]))
    if y_hold_std > 0 and pred_std < 1e-4 * y_hold_std:
        return float("inf")  # collapsed to ~constant
    return _rmse(y_hold[finite], y_hat[finite])


def _verdict(*, raw_rmse, lag_rmse, best_composite_rmse, best_base, floor_rmse, headroom, decision, reason, margin, n_fit, n_hold, lag1_ar) -> dict:
    """Assemble the achievable-ceiling diagnostic's final verdict dict from its computed RMSEs/decision/metadata."""
    return {
        "raw_rmse": float(raw_rmse),
        "lag_rmse": float(lag_rmse),
        "best_composite_rmse": float(best_composite_rmse),
        "headroom_vs_min": float(headroom),
        "decision": decision,
        "reason": reason,
        "method": _METHOD,
        "floor_rmse": float(floor_rmse),
        "best_base": best_base,
        "margin": float(margin),
        "n_fit": int(n_fit),
        "n_holdout": int(n_hold),
        "lag1_ar": (float(lag1_ar) if (lag1_ar is not None and np.isfinite(lag1_ar)) else None),
    }


def measure_achievable_ceiling(
    *,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    y_train: np.ndarray,
    group_ids_train: Optional[np.ndarray] = None,
    lag_values: Optional[np.ndarray] = None,
    cap: int = _DEFAULT_CEILING_SAMPLE_N,
    holdout_frac: float = _DEFAULT_HOLDOUT_FRAC,
    margin: float = _DEFAULT_CEILING_MARGIN,
    random_state: int = 0,
    n_estimators: int = 60,
    num_leaves: int = 15,
    learning_rate: float = 0.1,
    max_base_candidates: int = _MAX_BASE_CANDIDATES,
    min_groups: int = 4,
    strong_floor_frac: float = _DEFAULT_STRONG_FLOOR_FRAC,
) -> dict:
    """Measure raw / lag / optimistic-composite RMSE on a bounded group-disjoint holdout and return the verdict dict.

    All arrays are train-aligned to ``df`` rows. ``group_ids_train`` (optional) enables a leave-groups-out holdout; when
    absent a random holdout is used. ``lag_values`` (optional, train-aligned) overrides column detection for the AR
    failsafe. The verdict schema is the single source of truth: ``{raw_rmse, lag_rmse, best_composite_rmse,
    headroom_vs_min, decision('proceed'|'skip'), reason, method, ...}``. The decision is conservative toward PROCEED:
    SKIP fires ONLY on a confident measurement that the optimistic composite cannot beat ``min(raw, lag)`` by ``margin``.
    """
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    rng = np.random.default_rng(int(random_state))

    def _proceed(reason: str, **kw) -> dict:
        """Build a conservative default 'proceed' verdict (nan-filled measurements) for early-exit paths, overridable via ``kw``."""
        base = dict(raw_rmse=float("nan"), lag_rmse=float("nan"), best_composite_rmse=float("nan"),
                    best_base=None, floor_rmse=float("nan"), headroom=float("nan"), n_fit=0, n_hold=0,
                    lag1_ar=None)
        base.update(kw)
        return _verdict(decision="proceed", reason=reason, margin=margin, **base)

    finite_y = np.isfinite(y)
    if int(finite_y.sum()) < _MIN_ROWS_FOR_MEASUREMENT:
        return _proceed(f"insufficient finite target rows (n_finite={int(finite_y.sum())} < {_MIN_ROWS_FOR_MEASUREMENT})")

    # Bounded subsample of the finite-y rows (keeps the whole thing sub-second at prod scale).
    idx_all = np.nonzero(finite_y)[0]
    if cap > 0 and idx_all.size > cap:
        idx_all = np.sort(rng.choice(idx_all, size=cap, replace=False))
    y_sub = y[idx_all]
    grp_sub = None
    if group_ids_train is not None:
        g = np.asarray(group_ids_train).reshape(-1)
        if g.shape[0] == n:
            grp_sub = g[idx_all]

    # Cheap observability signal only (NOT the decision): per-group lag-1 autocorr of the target.
    lag1_ar = None
    if grp_sub is not None:
        lag1_ar = _recompute_lag1_ar_per_group(y_sub, grp_sub, np.arange(y_sub.shape[0]))

    # Group-disjoint holdout when a usable group key exists; random holdout otherwise.
    split = None
    if grp_sub is not None:
        split = _group_disjoint_split(grp_sub, holdout_frac, rng, min_groups)
    if split is None:
        split = _random_split(y_sub.shape[0], holdout_frac, rng)
    fit_pos, hold_pos = split
    if fit_pos.size < _MIN_HOLDOUT_ROWS or hold_pos.size < _MIN_HOLDOUT_ROWS:
        return _proceed(f"holdout too small (fit={fit_pos.size}, hold={hold_pos.size})", lag1_ar=lag1_ar)

    rows_fit = idx_all[fit_pos]
    rows_hold = idx_all[hold_pos]
    y_fit = y_sub[fit_pos]
    y_hold = y_sub[hold_pos]
    y_hold_std = float(np.std(y_hold))

    x_fit, _used_cols = _numeric_feature_matrix(df, feature_cols, target_col, rows_fit)
    x_hold, _ = _numeric_feature_matrix(df, feature_cols, target_col, rows_hold)
    if x_fit is None or x_hold is None or x_fit.shape[1] == 0:
        return _proceed("no numeric feature columns to fit a raw baseline", lag1_ar=lag1_ar, n_fit=fit_pos.size, n_hold=hold_pos.size)

    def _model_factory():
        """Build the tiny probe model: LightGBM if available, else a plain linear fallback."""
        try:
            return _build_tiny_model("lgb", n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, random_state=int(random_state))
        except Exception:
            return _build_tiny_model("linear", n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, random_state=int(random_state))

    # (a) raw-y tiny-model baseline.
    try:
        raw_model = _model_factory()
        raw_model.fit(x_fit, y_fit)
        raw_rmse = _rmse(y_hold, np.asarray(raw_model.predict(x_hold), dtype=np.float64))
    except Exception as exc:
        logger.debug("[achievable_ceiling] raw baseline fit failed: %s", exc)
        return _proceed(f"raw baseline fit failed ({exc})", lag1_ar=lag1_ar, n_fit=fit_pos.size, n_hold=hold_pos.size)
    if not np.isfinite(raw_rmse):
        return _proceed("raw baseline RMSE non-finite", lag1_ar=lag1_ar, n_fit=fit_pos.size, n_hold=hold_pos.size)

    # (b) AR failsafe (lag_predict) RMSE on the SAME holdout rows.
    lag_rmse = float("nan")
    lag_col = detect_causal_lag_column(df, target_col)
    if lag_values is not None:
        lv = np.asarray(lag_values, dtype=np.float64).reshape(-1)
        if lv.shape[0] == n:
            lag_rmse = causal_lag_predict_rmse(lv[rows_hold], y_hold)
    elif lag_col is not None:
        try:
            lag_hold = _extract_column_array(df, lag_col, rows=rows_hold).astype(np.float64, copy=False)
            lag_rmse = causal_lag_predict_rmse(lag_hold, y_hold)
        except Exception as exc:
            logger.debug("[achievable_ceiling] lag probe failed for %s: %s", lag_col, exc)

    # (c) OPTIMISTIC achievable-composite ceiling: best (min-RMSE) linear-residual reconstruction over candidate bases.
    base_cands = _pick_base_candidates(df, feature_cols, target_col, lag_col, y_sub, idx_all, max_base_candidates)
    best_composite_rmse = float("inf")
    best_base: Optional[str] = None
    for bcol in base_cands:
        try:
            base_fit = _extract_column_array(df, bcol, rows=rows_fit).astype(np.float64, copy=False)
            base_hold = _extract_column_array(df, bcol, rows=rows_hold).astype(np.float64, copy=False)
        except Exception as e:
            logger.debug("swallowed exception in _achievable_ceiling.py: %s", e)
            continue
        if base_fit.shape != y_fit.shape or base_hold.shape != y_hold.shape:
            continue
        comp = _composite_rmse_for_base(
            base_fit=base_fit, base_hold=base_hold, y_fit=y_fit, y_hold=y_hold,
            x_fit=x_fit, x_hold=x_hold, model_factory=_model_factory, y_hold_std=y_hold_std,
        )
        if np.isfinite(comp) and comp < best_composite_rmse:
            best_composite_rmse = comp
            best_base = bcol

    # Floor = the honest min over the raw model and the AR failsafe.
    floor_candidates = [v for v in (raw_rmse, lag_rmse) if np.isfinite(v)]
    floor_rmse = float(min(floor_candidates)) if floor_candidates else float("nan")

    common = dict(raw_rmse=raw_rmse, lag_rmse=lag_rmse, best_composite_rmse=best_composite_rmse,
                  best_base=best_base, floor_rmse=floor_rmse, margin=margin,
                  n_fit=fit_pos.size, n_hold=hold_pos.size, lag1_ar=lag1_ar)

    if not np.isfinite(floor_rmse):
        return _verdict(headroom=float("nan"), decision="proceed", reason="no measurable floor (raw + lag both non-finite)", **common)
    if not np.isfinite(best_composite_rmse):
        return _verdict(
            headroom=float("nan"), decision="proceed", reason="optimistic composite ceiling unmeasurable (all candidate bases collapsed / absent)", **common
        )

    headroom = (floor_rmse - best_composite_rmse) / floor_rmse if floor_rmse > 0 else float("nan")
    if np.isfinite(headroom) and headroom >= margin:
        reason = (f"optimistic composite RMSE {best_composite_rmse:.4g} beats min(raw={raw_rmse:.4g}, "
                  f"lag={lag_rmse:.4g}) floor {floor_rmse:.4g} by {headroom:.1%} >= margin {margin:.1%}")
        return _verdict(headroom=headroom, decision="proceed", reason=reason, **common)

    # The optimistic composite cannot beat the floor. Only SKIP when that floor is ALREADY strong (the deployed model /
    # AR failsafe explains most variance) -- on a weak / near-random target the ceiling estimate is unreliable, so proceed.
    strong_floor = y_hold_std > 0 and floor_rmse <= strong_floor_frac * y_hold_std
    if not strong_floor:
        reason = (f"optimistic composite cannot beat the floor (headroom {headroom:.1%}) but the floor {floor_rmse:.4g} "
                  f"is weak vs std(y)={y_hold_std:.4g} (> {strong_floor_frac:.0%}); low-confidence, discovery proceeds")
        return _verdict(headroom=headroom, decision="proceed", reason=reason, **common)
    reason = (f"optimistic composite RMSE {best_composite_rmse:.4g} cannot beat min(raw={raw_rmse:.4g}, "
              f"lag={lag_rmse:.4g}) floor {floor_rmse:.4g} by margin {margin:.1%} "
              f"(headroom {headroom:.1%}); the failsafe already explains the target, lag_predict deployed instead")
    return _verdict(headroom=headroom, decision="skip", reason=reason, **common)


def run_achievable_ceiling_precheck(
    *,
    config: Any,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    y_train: np.ndarray,
    group_ids_train: Optional[np.ndarray] = None,
    lag_values: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Config-reading wrapper. Returns the verdict dict, or ``None`` when the measured precheck is disabled.

    Reads ITS OWN flag ``composite_achievable_ceiling_precheck`` (default True) -- ORTHOGONAL to the legacy
    ``extreme_ar_group_aware_skip``. Logs the verdict once at INFO (the single source of truth for did-we-skip-and-why).
    """
    if not bool(getattr(config, "composite_achievable_ceiling_precheck", True)):
        return None
    margin = float(getattr(config, "composite_achievable_ceiling_margin", _DEFAULT_CEILING_MARGIN))
    cap = int(getattr(config, "composite_achievable_ceiling_sample_n", _DEFAULT_CEILING_SAMPLE_N))
    holdout_frac = float(getattr(config, "composite_achievable_ceiling_holdout_frac", _DEFAULT_HOLDOUT_FRAC))
    n_estimators = int(getattr(config, "tiny_model_n_estimators", 60))
    num_leaves = int(getattr(config, "tiny_model_num_leaves", 15))
    learning_rate = float(getattr(config, "tiny_model_learning_rate", 0.1))
    _rs = getattr(config, "random_state", 0)
    random_state = int(_rs if _rs is not None else 0)
    min_groups = int(getattr(config, "yscale_holdout_gate_min_groups", 4))
    strong_floor_frac = float(getattr(config, "composite_achievable_ceiling_strong_floor_frac", _DEFAULT_STRONG_FLOOR_FRAC))

    verdict = measure_achievable_ceiling(
        df=df, target_col=target_col, feature_cols=feature_cols, y_train=y_train,
        group_ids_train=group_ids_train, lag_values=lag_values, cap=cap, holdout_frac=holdout_frac,
        margin=margin, random_state=random_state, n_estimators=n_estimators, num_leaves=num_leaves,
        learning_rate=learning_rate, min_groups=min_groups, strong_floor_frac=strong_floor_frac,
    )
    logger.info(
        "[CompositeTargetDiscovery] achievable-ceiling precheck target=%r decision=%s: raw=%.4g lag=%.4g "
        "best_composite=%.4g headroom=%.4g base=%s lag1_ar=%s reason=%s",
        target_col, verdict["decision"], verdict["raw_rmse"], verdict["lag_rmse"],
        verdict["best_composite_rmse"], verdict["headroom_vs_min"], verdict["best_base"],
        verdict["lag1_ar"], verdict["reason"],
    )
    return verdict


def _profile_main() -> None:  # pragma: no cover -- manual cProfile harness (see module docstring)
    """Manual cProfile harness: run ``measure_achievable_ceiling`` on a synthetic grouped dataset and print the top hotspots."""
    import cProfile
    import pstats

    rng = np.random.default_rng(0)
    n_groups, per = 40, 800
    levels = rng.uniform(0.0, 50.0, n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    y_prev = np.empty(groups.size)
    y = np.empty(groups.size)
    k = 0
    for g in range(n_groups):
        prev = float(levels[g])
        for _ in range(per):
            y_prev[k] = prev
            cur = levels[g] + 0.95 * (prev - levels[g]) + rng.normal(0.0, 1.0)
            y[k] = cur
            prev = cur
            k += 1
    import pandas as pd
    df = pd.DataFrame({
        "x1": rng.normal(size=groups.size), "x2": rng.normal(size=groups.size),
        "x3": rng.normal(size=groups.size), "x4": rng.normal(size=groups.size),
        "y_prev": y_prev,
    })
    feature_cols = ["x1", "x2", "x3", "x4", "y_prev"]
    # Warm the JIT / import path once, then profile.
    measure_achievable_ceiling(df=df, target_col="y", feature_cols=feature_cols, y_train=y, group_ids_train=groups)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        measure_achievable_ceiling(df=df, target_col="y", feature_cols=feature_cols, y_train=y, group_ids_train=groups)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":  # pragma: no cover
    _profile_main()

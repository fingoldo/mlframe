"""Structured model card for a fitted ``CompositeTargetEstimator``.

Governance / UX surface: :func:`composite_model_card` produces a single dict
(plus a markdown rendering) that documents everything a reviewer, a deploying
engineer, or a downstream stakeholder needs to understand and safely operate a
fitted composite:

* identity -- transform, base column(s), inner estimator type;
* provenance -- the forward/inverse formula + the stakeholder description, reused
  verbatim from :mod:`provenance` (no algebra re-derived here);
* fitted params -- the reproducible inversion coefficients;
* training summary -- n_train + valid-domain fraction (recorded at fit time);
* evaluation -- RMSE / MAE on a supplied ``(X, y)`` plus conformal interval
  coverage when the estimator was calibrated;
* base-vs-residual attribution -- how much of the prediction the base column
  carries vs the inner model (via :mod:`attribution`);
* leakage check -- :func:`detect_base_target_leakage` on the base vs ``y`` when
  ``X`` carries the base column, surfacing a base that is a trivial re-encoding
  of the current target;
* deployment-readiness checklist -- conformal calibrated? online-refit enabled?
  a drift-monitor sketch (the recorded fit-time alpha/beta + the live runtime
  counters) present?

This module is pure orchestration over existing helpers. It does ONE
``predict`` (via the attribution / evaluation paths reusing the same point
prediction where possible), never copies the frame, and pulls only the narrow
base column for the leakage check. A no-data call (``X=None``) still renders
identity / provenance / params / readiness, so the card is useful even before a
holdout set is available.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from .attribution import attribution_summary
from .provenance import CompositeProvenance, _format_transform_formulas
from .spec import CompositeSpec
from .transforms import get_transform

logger = logging.getLogger(__name__)


def _is_fitted(estimator: Any) -> bool:
    """A fitted composite carries an inner ``estimator_`` and ``fitted_params_``."""
    return hasattr(estimator, "estimator_") and hasattr(estimator, "fitted_params_")


def _inner_type_name(estimator: Any) -> str:
    """Class name of the fitted inner model, falling back to the unfitted template."""
    inner = getattr(estimator, "estimator_", None)
    if inner is None:
        inner = getattr(estimator, "base_estimator", None)
    return type(inner).__name__ if inner is not None else "?"


def _resolve_base_columns(estimator: Any) -> tuple[str, ...]:
    """Resolve the base column tuple (multi-base first, then single, else ())."""
    try:
        return tuple(estimator._resolve_base_columns())
    except Exception:  # pragma: no cover - defensive; estimator may be a bare mock
        cols = getattr(estimator, "base_columns", None)
        if cols:
            return tuple(cols)
        bc = getattr(estimator, "base_column", None)
        return (bc,) if bc else ()


def _synth_spec(estimator: Any, *, target_col: str) -> CompositeSpec:
    """Build a minimal :class:`CompositeSpec` from the fitted estimator's own
    attributes so provenance formulas can be rendered without the original
    discovery spec (the estimator does not retain it). MI numbers are unknown
    post-fit and recorded as NaN -- the card never claims a discovery MI it
    cannot prove."""
    base_cols = _resolve_base_columns(estimator)
    base_column = base_cols[0] if base_cols else (getattr(estimator, "base_column", "") or "")
    extra = base_cols[1:] if len(base_cols) > 1 else ()
    params = dict(getattr(estimator, "fitted_params_", {}) or {})
    n_valid = int(params.get("n_train_valid", 0) or 0)
    n_invalid = int(params.get("n_train_invalid", 0) or 0)
    n_total = n_valid + n_invalid
    valid_frac = (n_valid / n_total) if n_total > 0 else float("nan")
    return CompositeSpec(
        name=f"{target_col}-{estimator.transform_name}-{base_column}",
        target_col=target_col,
        transform_name=estimator.transform_name,
        base_column=base_column,
        fitted_params=params,
        mi_gain=float("nan"),
        mi_y=float("nan"),
        mi_t=float("nan"),
        valid_domain_frac=valid_frac,
        n_train_rows=n_valid,
        extra_base_columns=tuple(extra),
    )


def _identity(estimator: Any, base_cols: tuple[str, ...]) -> dict[str, Any]:
    """Identity block: transform / base column(s) / inner type / fitted flag."""
    return {
        "transform_name": estimator.transform_name,
        "base_columns": list(base_cols),
        "base_column": base_cols[0] if base_cols else None,
        "is_multi_base": len(base_cols) > 1,
        "inner_estimator_type": _inner_type_name(estimator),
        "is_fitted": _is_fitted(estimator),
    }


def _training_summary(params: dict[str, Any]) -> dict[str, Any]:
    """n_train + valid-domain fraction from the fit-recorded counters."""
    n_valid = int(params.get("n_train_valid", 0) or 0)
    n_invalid = int(params.get("n_train_invalid", 0) or 0)
    n_total = n_valid + n_invalid
    return {
        "n_train": n_valid,
        "n_train_invalid": n_invalid,
        "valid_domain_frac": (n_valid / n_total) if n_total > 0 else None,
    }


def _evaluation(estimator: Any, X: Any, y: Any) -> dict[str, Any]:
    """RMSE / MAE on (X, y) plus conformal interval coverage when calibrated.

    Uses a single ``predict`` for the point metrics; the interval coverage path
    reuses the calibrated radius via ``predict_interval`` only when the
    estimator was calibrated (no calibration -> coverage omitted, never faked).
    """
    y_true = np.asarray(y, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(estimator.predict(X), dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_hat)
    out: dict[str, Any] = {"n_eval": int(finite.sum())}
    if finite.any():
        err = y_hat[finite] - y_true[finite]
        out["rmse"] = float(np.sqrt(np.mean(err * err)))
        out["mae"] = float(np.mean(np.abs(err)))
    else:
        out["rmse"] = None
        out["mae"] = None

    q = getattr(estimator, "_conformal_q_", None) or {}
    if q and hasattr(estimator, "predict_interval"):
        levels: dict[str, Any] = {}
        for alpha in sorted(q.keys()):
            try:
                lo, hi = estimator.predict_interval(X, alpha=alpha)
            except Exception:  # pragma: no cover - level uncalibrated mid-iter
                continue
            lo = np.asarray(lo, dtype=np.float64).reshape(-1)
            hi = np.asarray(hi, dtype=np.float64).reshape(-1)
            m = finite & np.isfinite(lo) & np.isfinite(hi)
            if not m.any():
                continue
            covered = (y_true[m] >= lo[m]) & (y_true[m] <= hi[m])
            levels[str(alpha)] = {
                "target_coverage": 1.0 - float(alpha),
                "empirical_coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(hi[m] - lo[m])),
                "n": int(m.sum()),
            }
        if levels:
            out["interval_coverage"] = levels
    return out


def _attribution(estimator: Any, X: Any) -> Optional[dict[str, Any]]:
    """Base-vs-residual share via :func:`attribution_summary`.

    Defined only for base-consuming transforms; a base-free unary y-transform
    has no base term to attribute against, so the section is recorded as a
    skip-with-reason rather than raising out of the card."""
    transform = get_transform(estimator.transform_name)
    if not transform.requires_base:
        return {"available": False, "reason": "base-free unary transform"}
    try:
        summary = attribution_summary(estimator, X)
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        return {"available": False, "reason": f"attribution failed: {exc}"}
    summary["available"] = True
    return summary


def _leakage_check(estimator: Any, X: Any, y: Any, base_cols: tuple[str, ...]) -> Optional[dict[str, Any]]:
    """Run :func:`detect_base_target_leakage` on the primary base vs ``y``.

    Only runs when ``X`` is present, carries the (single) base column, and a
    base-consuming transform is in play. The narrow base column is pulled via
    the estimator's own ``_extract_base_for_transform`` (one ndarray, no frame
    copy). Multi-base composites probe the primary base only."""
    if X is None or y is None or not base_cols:
        return None
    transform = get_transform(estimator.transform_name)
    if not transform.requires_base:
        return {"available": False, "reason": "base-free unary transform"}
    from .discovery._leakage import detect_base_target_leakage  # local: avoid cycle

    try:
        base_arr = estimator._extract_base_for_transform(X, base_cols)
    except Exception as exc:  # pragma: no cover - X may lack the base column
        return {"available": False, "reason": f"base column unavailable in X: {exc}"}
    base_arr = np.asarray(base_arr, dtype=np.float64)
    if base_arr.ndim > 1:
        base_arr = base_arr[:, 0]
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    res = detect_base_target_leakage(y_arr, base_arr.reshape(-1))
    res["available"] = True
    res["probed_base_column"] = base_cols[0]
    return res


def _readiness(estimator: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Deployment-readiness checklist.

    * ``conformal_calibrated`` -- at least one alpha level has a stored radius;
    * ``online_refit_enabled`` -- the streaming alpha-refit flag is on;
    * ``drift_monitor_sketch`` -- the fitted alpha/beta baseline plus the live
      runtime counters that a drift monitor would track (present once fit ran).
    """
    q = getattr(estimator, "_conformal_q_", None) or {}
    runtime = getattr(estimator, "runtime_stats_", None)
    drift_sketch = None
    if "alpha" in params or "beta" in params or runtime is not None:
        drift_sketch = {
            "baseline_alpha": params.get("alpha"),
            "baseline_beta": params.get("beta"),
            "runtime_counters": dict(runtime) if runtime else {},
        }
    return {
        "conformal_calibrated": bool(q),
        "calibrated_levels": sorted(q.keys()) if q else [],
        "n_calibration": int(getattr(estimator, "_conformal_n_cal_", 0) or 0),
        "online_refit_enabled": bool(getattr(estimator, "online_refit_enabled", False)),
        "drift_monitor_sketch": drift_sketch,
        "drift_monitor_present": drift_sketch is not None,
    }


def _fmt_num(value: Any) -> str:
    """Compact render of an optional scalar; ``-`` for missing / non-finite."""
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "-"
    return f"{f:.4g}"


def _render_markdown(card: dict[str, Any]) -> str:
    """Render the card dict into a stakeholder-ready markdown document."""
    ident = card["identity"]
    prov = card["provenance"]
    lines: list[str] = []
    lines.append(f"# Composite model card: `{prov['name']}`")
    lines.append("")
    lines.append(
        f"_Not fitted._" if not ident["is_fitted"] else
        f"Fitted composite over inner `{ident['inner_estimator_type']}`."
    )
    lines.append("")

    lines.append("## Identity")
    lines.append("")
    lines.append(f"- transform: `{ident['transform_name']}`")
    lines.append(f"- base column(s): {', '.join(f'`{c}`' for c in ident['base_columns']) or '(none)'}")
    lines.append(f"- inner estimator: `{ident['inner_estimator_type']}`")
    lines.append(f"- multi-base: {ident['is_multi_base']}")
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append(f"_{prov['stakeholder_description']}_")
    lines.append("")
    lines.append(f"- forward: `{prov['forward_formula_human']}`")
    lines.append(f"- inverse: `{prov['inverse_formula_human']}`")
    lines.append("")

    params = card["fitted_params"]
    lines.append("## Fitted parameters")
    lines.append("")
    if params:
        lines.append("| param | value |")
        lines.append("|-------|-------|")
        for k in sorted(params.keys()):
            lines.append(f"| `{k}` | {_fmt_num(params[k])} |")
    else:
        lines.append("_(none recorded -- estimator not fitted)_")
    lines.append("")

    train = card["training"]
    lines.append("## Training summary")
    lines.append("")
    vf = train.get("valid_domain_frac")
    lines.append(f"- n_train: {train.get('n_train', 0)}")
    lines.append(f"- valid-domain fraction: {('%.1f%%' % (100 * vf)) if vf is not None else '-'}")
    lines.append("")

    ev = card.get("evaluation")
    if ev is not None:
        lines.append("## Evaluation")
        lines.append("")
        lines.append(f"- n_eval: {ev.get('n_eval', 0)}")
        lines.append(f"- RMSE: {_fmt_num(ev.get('rmse'))}")
        lines.append(f"- MAE: {_fmt_num(ev.get('mae'))}")
        cov = ev.get("interval_coverage")
        if cov:
            lines.append("")
            lines.append("| alpha | target_cov | empirical_cov | mean_width | n |")
            lines.append("|-------|-----------|---------------|-----------|---|")
            for a, c in cov.items():
                lines.append(
                    f"| {a} | {_fmt_num(c['target_coverage'])} | "
                    f"{_fmt_num(c['empirical_coverage'])} | "
                    f"{_fmt_num(c['mean_width'])} | {c['n']} |"
                )
        lines.append("")

    attr = card.get("attribution")
    if attr is not None:
        lines.append("## Base-vs-residual attribution")
        lines.append("")
        if attr.get("available"):
            lines.append(f"- base share: {_fmt_num(attr.get('base_share'))}")
            lines.append(f"- residual share: {_fmt_num(attr.get('residual_share'))}")
            lines.append(f"- mode: {attr.get('mode')} (n={attr.get('n_rows')})")
        else:
            lines.append(f"_not available: {attr.get('reason')}_")
        lines.append("")

    leak = card.get("leakage")
    if leak is not None:
        lines.append("## Leakage check")
        lines.append("")
        if leak.get("available"):
            verdict = "LEAKY" if leak.get("is_leaky") else "clean"
            lines.append(f"- base `{leak.get('probed_base_column')}` vs y: **{verdict}**")
            lines.append(f"- score: {_fmt_num(leak.get('score'))}")
            lines.append(f"- reason: {leak.get('reason')}")
        else:
            lines.append(f"_not run: {leak.get('reason')}_")
        lines.append("")

    rd = card["readiness"]
    lines.append("## Deployment-readiness checklist")
    lines.append("")
    lines.append(f"- [{'x' if rd['conformal_calibrated'] else ' '}] conformal calibrated "
                 f"(levels: {rd['calibrated_levels'] or 'none'}, n_cal={rd['n_calibration']})")
    lines.append(f"- [{'x' if rd['online_refit_enabled'] else ' '}] online-refit enabled")
    lines.append(f"- [{'x' if rd['drift_monitor_present'] else ' '}] drift-monitor sketch present")
    lines.append("")
    return "\n".join(lines)


def composite_model_card(
    estimator: Any,
    X: Any = None,
    y: Any = None,
    *,
    target_col: str = "y",
) -> dict[str, Any]:
    """Build a structured model card for a fitted ``CompositeTargetEstimator``.

    Parameters
    ----------
    estimator
        A FITTED :class:`CompositeTargetEstimator`. An unfitted estimator still
        yields identity / provenance / params / readiness (the data-dependent
        sections are simply omitted).
    X, y
        Optional held-out feature frame + target. When both are given the card
        adds evaluation metrics (RMSE / MAE, plus conformal interval coverage
        when calibrated), the base-vs-residual attribution share, and the
        base-vs-target leakage check. Never copied; only the narrow base column
        is pulled for the leakage probe.
    target_col
        Name used in the rendered formulas / spec name (the fitted estimator
        does not retain the discovery target name). Defaults to ``"y"``.

    Returns
    -------
    dict with keys ``identity``, ``provenance``, ``fitted_params``,
    ``training``, ``readiness`` always present; ``evaluation``, ``attribution``,
    ``leakage`` present when ``X`` (and ``y`` for evaluation/leakage) is given;
    and ``markdown`` -- the full rendered document.
    """
    base_cols = _resolve_base_columns(estimator)
    params = dict(getattr(estimator, "fitted_params_", {}) or {})

    spec = _synth_spec(estimator, target_col=target_col)
    if _is_fitted(estimator):
        prov = CompositeProvenance.from_spec(spec, random_state=None).to_dict()
    else:
        # Unfitted: render formula text directly (no MI / training numbers).
        fwd, inv, desc = _format_transform_formulas(
            transform_name=estimator.transform_name,
            base_column=spec.base_column,
            target_col=target_col,
            fitted_params=params,
        )
        prov = {
            "name": spec.name,
            "forward_formula_human": fwd,
            "inverse_formula_human": inv,
            "stakeholder_description": desc,
        }

    card: dict[str, Any] = {
        "identity": _identity(estimator, base_cols),
        "provenance": prov,
        "fitted_params": params,
        "training": _training_summary(params),
        "readiness": _readiness(estimator, params),
    }

    if X is not None:
        card["attribution"] = _attribution(estimator, X)
    if X is not None and y is not None:
        card["evaluation"] = _evaluation(estimator, X, y)
        card["leakage"] = _leakage_check(estimator, X, y, base_cols)

    card["markdown"] = _render_markdown(card)
    return card

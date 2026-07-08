"""Unified explainability report for a fitted ``CompositeTargetEstimator``.

This is PURE ORCHESTRATION over helpers that already exist -- it assembles a
single human-readable document from:

* the provenance formula + stakeholder description
  (``provenance._format_transform_formulas``),
* the fitted parameters (alpha / beta / the clip envelope) read off
  ``estimator.fitted_params_``,
* the base-vs-residual attribution share (``attribution.attribution_summary``,
  only when an ``X`` is supplied),
* the conformal / CQR calibration state (the per-alpha radius dicts the
  estimator stores under ``_conformal_q_`` / ``_cqr_q_``),
* a compact diagnostics summary (prediction range over the supplied ``X`` plus
  the predict-time fallback / domain-violation rate from ``runtime_stats_``),
* interval coverage when both ``X`` and ``y`` are given and a level is
  calibrated.

No recompute beyond a SINGLE ``predict(X)`` (reused for range, coverage, and
fallback rate); no frame copy -- ``X`` is passed straight through to the
estimator's own predict / attribution path, which only pulls the narrow base
column. With ``X=None`` the static sections (provenance, fitted params,
conformal state) still render so the report is useful for a cold-loaded model.

Output is Markdown by default; ``fmt="html"`` wraps the same content in a
minimal HTML document (sections become ``<h2>`` headers, tables become real
``<table>``s) for embedding in a notebook / dashboard.
"""
from __future__ import annotations

import html as _html
from typing import Any, Optional

import numpy as np

from .provenance import _format_transform_formulas
from ._value_report import build_composite_value_report, render_composite_value_report

# Headline fitted params shown first, in this fixed order, when present. Mirrors
# the notebook repr's _HEADLINE_PARAM_KEYS so the report and the repr agree.
_HEADLINE_PARAM_KEYS: tuple[str, ...] = (
    "alpha", "beta", "alphas",
    "y_clip_low", "y_clip_high",
    "t_clip_low", "t_clip_high",
    "y_train_median",
)


def _fmt_num(v: Any) -> str:
    """Render a scalar param compactly; arrays as a shape tag, else ``str``."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f:
            return "nan"
        if f in (float("inf"), float("-inf")):
            return str(f)
        return f"{f:.6g}"
    if hasattr(v, "shape") and getattr(v, "shape", None):
        return f"array{tuple(v.shape)}"
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."


def _resolve_base_str(estimator: Any) -> str:
    """Resolved base column(s) as a display string (never raises)."""
    try:
        cols = estimator._resolve_base_columns()
    except Exception:
        cols = ()
    return ", ".join(map(str, cols)) if cols else "(none -- unary y-transform)"


def _is_fitted(estimator: Any) -> bool:
    return getattr(estimator, "fitted_params_", None) is not None and hasattr(estimator, "estimator_")


def _gather(estimator: Any, X: Any, y: Any) -> dict[str, Any]:
    """Collect every report fact in ONE pass (a single ``predict(X)``).

    Returns a plain dict the renderers consume. Each optional fact is present
    only when its inputs are; failures of an optional probe are swallowed (the
    section degrades gracefully rather than the whole report raising).
    """
    transform_name = getattr(estimator, "transform_name", "?")
    target_col = getattr(estimator, "target_col", None) or "y"
    base_str = _resolve_base_str(estimator)
    params = dict(getattr(estimator, "fitted_params_", None) or {})

    # Primary base column for the formula (first resolved column).
    try:
        base_cols = estimator._resolve_base_columns()
        primary_base = base_cols[0] if base_cols else "base"
    except Exception:
        primary_base = "base"

    forward, inverse, description = _format_transform_formulas(
        transform_name=transform_name,
        base_column=primary_base,
        target_col=target_col,
        fitted_params=params,
    )

    facts: dict[str, Any] = {
        "fitted": _is_fitted(estimator),
        "transform_name": transform_name,
        "target_col": target_col,
        "base_str": base_str,
        "inner_name": type(getattr(estimator, "base_estimator", None)).__name__,
        "params": params,
        "forward": forward,
        "inverse": inverse,
        "description": description,
        "n_train": params.get("n_train_valid"),
        "n_train_invalid": params.get("n_train_invalid", 0),
        "conformal_alphas": sorted((getattr(estimator, "_conformal_q_", None) or {}).keys()),
        "cqr_alphas": sorted((getattr(estimator, "_cqr_q_", None) or {}).keys()),
        "attribution": None,
        "pred_range": None,
        "fallback": None,
        "coverage": None,
    }

    if X is None:
        return facts

    # Single predict over X; reused for range + coverage + fallback snapshot.
    rs_before = dict(getattr(estimator, "runtime_stats_", None) or {})
    try:
        y_hat = np.asarray(estimator.predict(X), dtype=np.float64).reshape(-1)
    except Exception:
        y_hat = None

    if y_hat is not None and y_hat.size:
        finite = y_hat[np.isfinite(y_hat)]
        if finite.size:
            facts["pred_range"] = {
                "n": int(y_hat.size),
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "nonfinite": int(y_hat.size - finite.size),
            }

    # Fallback rate from the delta in runtime_stats_ caused by this predict.
    rs_after = dict(getattr(estimator, "runtime_stats_", None) or {})
    rows = rs_after.get("predict_rows_total", 0) - rs_before.get("predict_rows_total", 0)
    if rows > 0:
        viol = rs_after.get("domain_violation_rows", 0) - rs_before.get("domain_violation_rows", 0)
        facts["fallback"] = {
            "rows": int(rows),
            "domain_violation_rows": int(viol),
            "fallback_rate": float(viol) / float(rows),
        }

    # Base-vs-residual attribution (defined only for base-consuming transforms).
    try:
        from .attribution import attribution_summary
        facts["attribution"] = attribution_summary(estimator, X)
    except Exception as exc:
        facts["attribution_error"] = str(exc)

    # Interval coverage: needs y AND a calibrated level.
    if y is not None:
        facts["coverage"] = _interval_coverage(estimator, X, y, facts)

    return facts


def _interval_coverage(estimator: Any, X: Any, y: Any, facts: dict[str, Any]) -> Optional[list[dict]]:
    """Empirical coverage + mean width for each calibrated conformal / CQR level."""
    y_true = np.asarray(y, dtype=np.float64).reshape(-1)
    out: list[dict[str, Any]] = []
    for kind, alphas, fn_name in (
        ("conformal", facts["conformal_alphas"], "predict_interval"),
        ("cqr", facts["cqr_alphas"], "predict_interval_cqr"),
    ):
        fn = getattr(estimator, fn_name, None)
        if fn is None:
            continue
        for a in alphas:
            try:
                lo, hi = fn(X, alpha=a)
                lo = np.asarray(lo, dtype=np.float64).reshape(-1)
                hi = np.asarray(hi, dtype=np.float64).reshape(-1)
                if lo.shape[0] != y_true.shape[0]:
                    continue
                covered = (y_true >= lo) & (y_true <= hi)
                out.append({
                    "kind": kind,
                    "alpha": float(a),
                    "target_coverage": 1.0 - float(a),
                    "empirical_coverage": float(np.mean(covered)),
                    "mean_width": float(np.mean(hi - lo)),
                    "n": int(y_true.shape[0]),
                })
            except Exception:  # nosec B112 - best-effort path
                continue
    return out or None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(facts: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# Composite explainability report: `{facts['transform_name']}`")
    L.append("")
    fitted_tag = "fitted" if facts["fitted"] else "NOT fitted"
    L.append(f"_Status_: **{fitted_tag}**. Target `{facts['target_col']}`, " f"base column(s) `{facts['base_str']}`, inner `{facts['inner_name']}`.")
    L.append("")

    # Provenance / formula
    L.append("## Provenance")
    L.append("")
    L.append(facts["description"] or "(no description)")
    L.append("")
    L.append(f"- **Forward**: `{facts['forward']}`")
    L.append(f"- **Inverse**: `{facts['inverse']}`")
    if facts["n_train"] is not None:
        drop = f" ({int(facts['n_train_invalid'])} dropped)" if facts["n_train_invalid"] else ""
        L.append(f"- **n_train**: {int(facts['n_train'])}{drop}")
    L.append("")

    # Fitted params
    L.append("## Fitted parameters")
    L.append("")
    params = facts["params"]
    if params:
        L.append("| param | value |")
        L.append("|-------|-------|")
        shown = set()
        for k in _HEADLINE_PARAM_KEYS:
            if k in params:
                L.append(f"| `{k}` | {_fmt_num(params[k])} |")
                shown.add(k)
        for k in params:
            if k not in shown:
                L.append(f"| `{k}` | {_fmt_num(params[k])} |")
    else:
        L.append("_No fitted parameters (estimator not fitted)._")
    L.append("")

    # Attribution
    L.append("## Base-vs-residual attribution")
    L.append("")
    attr = facts["attribution"]
    if attr is not None:
        bs = attr.get("base_share")
        rs = attr.get("residual_share")
        L.append(f"Over {attr.get('n_rows', 0)} rows ({attr.get('mode')} combine): " f"**base share = {_fmt_num(bs)}**, residual share = {_fmt_num(rs)}.")
    elif "attribution_error" in facts:
        L.append(f"_Attribution unavailable: {facts['attribution_error']}_")
    else:
        L.append("_No data supplied (pass X to compute the base/residual share)._")
    L.append("")

    # Conformal / CQR
    L.append("## Conformal calibration")
    L.append("")
    ca = facts["conformal_alphas"]
    qa = facts["cqr_alphas"]
    L.append("- **Split-conformal**: " + ("calibrated @ alpha=" + ", ".join(f"{a:g}" for a in ca) if ca else "not calibrated"))
    L.append("- **CQR**: " + ("calibrated @ alpha=" + ", ".join(f"{a:g}" for a in qa) if qa else "not calibrated"))
    L.append("")
    cov = facts["coverage"]
    if cov:
        L.append("### Interval coverage on supplied data")
        L.append("")
        L.append("| kind | alpha | target_cov | empirical_cov | mean_width | n |")
        L.append("|------|-------|-----------|---------------|-----------|---|")
        for c in cov:
            L.append(f"| {c['kind']} | {c['alpha']:g} | {c['target_coverage']:.3f} | "
                     f"{c['empirical_coverage']:.3f} | {_fmt_num(c['mean_width'])} | {c['n']} |")
        L.append("")

    # Diagnostics
    L.append("## Diagnostics")
    L.append("")
    pr = facts["pred_range"]
    if pr is not None:
        L.append(f"- **Prediction range** ({pr['n']} rows): "
                 f"min={_fmt_num(pr['min'])}, mean={_fmt_num(pr['mean'])}, "
                 f"max={_fmt_num(pr['max'])}"
                 + (f", {pr['nonfinite']} non-finite" if pr["nonfinite"] else ""))
    fb = facts["fallback"]
    if fb is not None:
        L.append(f"- **Fallback / domain-violation rate**: " f"{fb['domain_violation_rows']}/{fb['rows']} rows " f"= {fb['fallback_rate']:.2%}")
    if pr is None and fb is None:
        L.append("_No data supplied (pass X for prediction range + fallback rate)._")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML rendering (same content, minimal document)
# ---------------------------------------------------------------------------

def _md_table_to_html(lines: list[str], i: int) -> tuple[str, int]:
    """Consume a contiguous Markdown table starting at ``lines[i]`` (header,
    separator, rows) and return its ``<table>`` HTML + the next line index."""
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    j = i + 2  # skip header + separator row
    body: list[str] = []
    while j < len(lines) and lines[j].lstrip().startswith("|"):
        cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
        body.append("<tr>" + "".join(f"<td style='padding:2px 8px;border:1px solid #d0d7de;'>" f"{_html.escape(c)}</td>" for c in cells) + "</tr>")
        j += 1
    head = "<tr>" + "".join(f"<th style='padding:2px 8px;border:1px solid #d0d7de;text-align:left;'>" f"{_html.escape(c)}</th>" for c in header) + "</tr>"
    table = "<table style='border-collapse:collapse;margin:6px 0;'>" + head + "".join(body) + "</table>"
    return table, j


def _render_html(facts: dict[str, Any]) -> str:
    """Reuse the Markdown structure and translate it line-by-line to HTML so the
    two formats can never drift in content (single source of facts)."""
    md = _render_markdown(facts)
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            out.append(f"<h3>{_html.escape(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{_html.escape(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{_html.escape(ln[2:])}</h1>")
        elif ln.lstrip().startswith("|"):
            table, i = _md_table_to_html(lines, i)
            out.append(table)
            continue
        elif ln.strip() == "":
            pass
        else:
            out.append(f"<p>{_html.escape(ln)}</p>")
        i += 1
    return "<div style='font-family:-apple-system,Segoe UI,sans-serif;" "font-size:13px;max-width:900px;'>" + "".join(out) + "</div>"


def composite_report(
    estimator: Any,
    X: Any = None,
    y: Any = None,
    fmt: str = "markdown",
) -> str:
    """Assemble ONE explainability report for a fitted composite estimator.

    Parameters
    ----------
    estimator
        A :class:`CompositeTargetEstimator`. Fitted is strongly preferred; an
        unfitted instance still renders the static config sections.
    X
        Optional predict-time feature frame (pandas / polars / ndarray). When
        given, the report adds the base-vs-residual attribution share, the
        prediction range, and the predict-time fallback rate -- via a SINGLE
        ``predict(X)`` (no frame copy; only the narrow base column is pulled).
    y
        Optional true target aligned with ``X``. When given alongside a
        calibrated conformal / CQR level, the report adds an empirical
        interval-coverage table.
    fmt
        ``"markdown"`` (default) or ``"html"``.

    Returns
    -------
    str
        The rendered report. Sections: status header, provenance (formula +
        stakeholder description), fitted parameters, base-vs-residual
        attribution, conformal/CQR calibration (+ coverage when ``y`` given),
        and diagnostics (prediction range + fallback rate).
    """
    fmt_l = (fmt or "markdown").lower()
    if fmt_l not in ("markdown", "md", "html"):
        raise ValueError(f"composite_report: fmt must be 'markdown' or 'html', got {fmt!r}")
    facts = _gather(estimator, X, y)
    if fmt_l == "html":
        return _render_html(facts)
    return _render_markdown(facts)


def composite_value_report(
    y_true: Any,
    y_pred_raw: Any,
    y_pred_composite: Any,
    group_ids: Any,
    y_pred_lag: Any = None,
    *,
    config: Any = None,
    **kwargs: Any,
) -> Optional[dict]:
    """Build the composite-vs-raw(-vs-lag) per-group VALUE report + attach a rendered text block.

    Answers "did the composite earn its keep, and where?" -- the per-group breakdown, the row-weighted
    net lift, and the count of groups where the composite is worse than the lag failsafe. See
    :func:`mlframe.training.composite._value_report.build_composite_value_report` for the full contract.

    A normal composite run gates emission on the config toggle: when ``config`` is passed and
    ``emit_composite_value_report`` is falsy, this returns ``None`` (the report is suppressed). The
    returned dict carries the rendered ASCII text under ``report["markdown"]`` so a caller can log or
    persist both the structured data and the human-readable block from one call.
    """
    if config is not None and not getattr(config, "emit_composite_value_report", True):
        return None
    report = build_composite_value_report(y_true, y_pred_raw, y_pred_composite, group_ids, y_pred_lag=y_pred_lag, **kwargs)
    report["markdown"] = render_composite_value_report(report)
    return report

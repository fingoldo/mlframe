"""Rich Jupyter HTML repr for ``CompositeTargetEstimator``.

Carved into a sibling so ``_estimator.py`` stays under the 1k-line monolith
threshold; bound onto the class at ``_estimator``'s module bottom (identity
preserved, so isinstance / sklearn introspection are unchanged).

The repr is a compact styled table of the transform name, the resolved base
column(s), the headline fitted parameters (alpha / beta / the clip envelope),
the valid-train row count, and whether split-conformal / CQR intervals have
been calibrated -- exactly the state an operator wants at a glance in a
notebook without printing the whole ``fitted_params_`` dict.
"""
from __future__ import annotations

import html
from typing import Any

# Headline fitted parameters shown first (in this order) when present; every
# other fitted_params_ key is summarised in a count so the table stays compact.
_HEADLINE_PARAM_KEYS: tuple[str, ...] = (
    "alpha",
    "beta",
    "y_clip_low",
    "y_clip_high",
    "t_clip_low",
    "t_clip_high",
    "y_train_median",
)


def _fmt_value(v: Any) -> str:
    """Format a fitted-param value for the table: floats to 4 sig-figs,
    arrays to a shape tag, everything else via ``str``. HTML-escaped."""
    if isinstance(v, bool):
        return html.escape(str(v))
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        if v in (float("inf"), float("-inf")):
            return html.escape(str(v))
        return html.escape(f"{v:.4g}")
    if hasattr(v, "shape") and getattr(v, "shape", None):  # ndarray-like
        return html.escape(f"array{tuple(v.shape)}")
    s = str(v)
    if len(s) > 60:
        s = s[:57] + "..."
    return html.escape(s)


def _row(label: str, value: str) -> str:
    """One styled ``<tr>`` (escaped label, pre-formatted/escaped value)."""
    return (
        "<tr>"
        f"<td style='text-align:left;padding:2px 10px 2px 0;"
        f"font-weight:600;color:#444;'>{html.escape(label)}</td>"
        f"<td style='text-align:left;padding:2px 0;"
        f"font-family:monospace;'>{value}</td>"
        "</tr>"
    )


def _repr_html_(self: Any) -> str:
    """Return a compact styled HTML table summarising the estimator.

    Safe to call on an UNFITTED instance (shows config only, marks the
    estimator not-fitted); never raises (a notebook repr must not blow up).
    """
    try:
        return _build_repr_html(self)
    except Exception as exc:  # pragma: no cover - a repr must never raise
        return "<div style='font-family:monospace;color:#a00;'>" f"CompositeTargetEstimator (repr failed: {html.escape(str(exc))})" "</div>"


def _build_repr_html(self: Any) -> str:
    """Assemble the actual HTML body for the notebook repr (config, resolved base column(s), fitted-state summary); split out so the public `_repr_html_` wrapper can catch any failure here and degrade to an error div instead of raising."""
    transform_name = getattr(self, "transform_name", "?")
    # Resolved base column(s): prefer the canonical resolver so the multi-base
    # path and the single-column legacy alias both render correctly.
    try:
        base_cols = self._resolve_base_columns()
    except Exception:
        base_cols = ()
    if base_cols:
        base_str = ", ".join(map(str, base_cols))
    else:
        base_str = "(none -- unary y-transform)"

    fitted = getattr(self, "fitted_params_", None)
    is_fitted = fitted is not None and hasattr(self, "estimator_")

    inner = getattr(self, "base_estimator", None)
    inner_name = type(inner).__name__ if inner is not None else "(unset)"

    rows: list[str] = []
    rows.append(_row("transform", _fmt_value(transform_name)))
    rows.append(_row("base column(s)", html.escape(base_str)))
    rows.append(_row("inner estimator", html.escape(inner_name)))
    rows.append(_row("fitted", "yes" if is_fitted else "no"))

    if is_fitted and fitted is not None:
        n_valid = fitted.get("n_train_valid")
        if n_valid is not None:
            n_invalid = fitted.get("n_train_invalid", 0)
            rows.append(_row("n_train_valid", _fmt_value(n_valid) + (f" ({_fmt_value(n_invalid)} dropped)" if n_invalid else "")))
        # Headline fitted params (alpha / beta / clip envelope) in fixed order.
        shown = 0
        for key in _HEADLINE_PARAM_KEYS:
            if key in fitted:
                rows.append(_row(key, _fmt_value(fitted[key])))
                shown += 1
        # Summarise the remaining fitted-param keys without dumping them all.
        other = [k for k in fitted if k not in _HEADLINE_PARAM_KEYS]
        if other:
            rows.append(_row("other fitted params", html.escape(f"{len(other)}: " + ", ".join(map(str, other[:6])) + ("..." if len(other) > 6 else ""))))
        if getattr(self, "recurrence_continuation", False):
            rows.append(_row("recurrence_continuation", "True"))

    # Conformal / CQR calibration state -- the dicts are keyed per alpha-level.
    conf_q = getattr(self, "_conformal_q_", None) or {}
    cqr_q = getattr(self, "_cqr_q_", None) or {}
    conf_str = "calibrated @ alpha=" + ", ".join(f"{a:g}" for a in sorted(conf_q)) if conf_q else "not calibrated"
    cqr_str = "calibrated @ alpha=" + ", ".join(f"{a:g}" for a in sorted(cqr_q)) if cqr_q else "not calibrated"
    rows.append(_row("conformal interval", html.escape(conf_str)))
    rows.append(_row("CQR interval", html.escape(cqr_str)))

    title_colour = "#1a7f37" if is_fitted else "#9a6700"
    return (
        "<div style='border:1px solid #d0d7de;border-radius:6px;"
        "padding:8px 12px;display:inline-block;"
        "font-family:-apple-system,Segoe UI,sans-serif;font-size:13px;'>"
        f"<div style='font-weight:700;color:{title_colour};"
        "margin-bottom:6px;'>CompositeTargetEstimator</div>"
        "<table style='border-collapse:collapse;'>" + "".join(rows) + "</table></div>"
    )

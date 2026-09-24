"""Risk-coverage (selective prediction) diagnostic render entry.

Kept in its own module because ``diagnostics_dispatch`` is already at the module-size limit; reuses that module's
save/record helpers so the per-(model, split) wiring stays consistent with the other chart diagnostics.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


def render_risk_coverage_diagnostic(
    *,
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    task: Literal["binary", "multiclass", "regression"] = "binary",
    confidence: Optional[np.ndarray] = None,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    model_label: str = "model",
    confidence_source: str = "model",
) -> bool:
    """Risk-coverage curve: accuracy/error as you abstain on the least-confident cases vs random-rejection.

    Binary / multiclass need only y_true + score/proba; regression needs an explicit ``confidence`` array. Records the
    AURC + accuracy@80% selective gain into ``metrics_dict``. Default-ON; a no-op when scores are absent or n==0.

    ``confidence_source`` names where the confidence came from. Anything but ``"model"`` is a PROXY, and its metrics are
    stored as ``risk_coverage_proxy_aurc`` / ``risk_coverage_proxy_selective_gain`` with the source recorded beside
    them: the regression path ranks by distance from the prediction mean, which shows a "selective gain" for nearly any
    model because central predictions tend to have smaller errors, and under the generic keys that read as evidence the
    model has usable uncertainty.
    """
    from mlframe.reporting.diagnostics_dispatch import _record, _record_path, _save_spec

    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or y_score is None:
        return False
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64)
    m = min(len(yt), ys.shape[0])
    if m == 0:
        return False
    try:
        from mlframe.reporting.charts.risk_coverage import build_risk_coverage_spec

        conf = None if confidence is None else np.asarray(confidence, dtype=np.float64).ravel()[:m]
        res = build_risk_coverage_spec(yt[:m], ys[:m], task=task, confidence=conf, model_label=model_label)
        ok = _save_spec(res.figure, plot_outputs, base_path + "_risk_coverage")
        _record(charts, "risk_coverage", ok)
        if ok:
            _record_path(charts, base_path + "_risk_coverage")
        if isinstance(metrics_dict, dict):
            prefix = "risk_coverage_" if confidence_source == "model" else "risk_coverage_proxy_"
            metrics_dict[f"{prefix}aurc"] = float(res.aurc)
            metrics_dict[f"{prefix}selective_gain"] = float(res.selective_gain)
            metrics_dict["risk_coverage_confidence_source"] = confidence_source
        return ok
    except Exception:
        logger.exception("risk_coverage diagnostic failed; continuing.")
        _record(charts, "risk_coverage", False)
        return False


__all__ = ["render_risk_coverage_diagnostic"]

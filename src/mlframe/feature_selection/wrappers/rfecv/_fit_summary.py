"""End-of-fit INFO summary for ``RFECV.fit``: how many and which columns were kept, why the search stopped, and why the kept size differs from the best-scoring one."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..._selection_log import format_name_list, is_quiet
from ._one_se_band import band_half_width, fold_counts, split_rule_band

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def build_rfecv_fit_summary(self: Any, *, stop_reason: Optional[str], n_iters: int, elapsed_s: float, ndigits: int = 4) -> str:
    """Compose the summary line from the fitted slots (``cv_results_``, ``resolved_n_features_rule_``, ``_selected_cols_cache``)."""
    selected = list(getattr(self, "_selected_cols_cache", None) or [])
    n_in = int(getattr(self, "n_features_in_", 0))
    rule = getattr(self, "resolved_n_features_rule_", None)
    parts = [
        (
            f"RFECV: selected {len(selected):_} of {n_in:_} features after {n_iters:_} iteration(s) in {elapsed_s / 60:_.1f} min "
            f"(stopped: {'every candidate subset size was evaluated' if stop_reason is None else stop_reason}; n_features_selection_rule={rule})."
        )
    ]

    cv_results = getattr(self, "cv_results_", None) or {}
    nf = np.asarray(cv_results.get("nfeatures", []), dtype=float)
    mean = np.asarray(cv_results.get("cv_mean_perf", []), dtype=float)
    std = np.asarray(cv_results.get("cv_std_perf", []), dtype=float)
    usable = (nf > 0) & np.isfinite(mean) if nf.size and nf.shape == mean.shape == std.shape else np.zeros(0, dtype=bool)
    if usable.any():
        # Same scalar the progress bar reports as "best was NF with score S".
        final = mean * float(getattr(self, "mean_perf_weight", 1.0)) - np.nan_to_num(std) * float(getattr(self, "std_perf_weight", 0.0))
        idx = np.flatnonzero(usable)
        best = int(idx[np.argmax(final[idx])])
        parts.append(f"Best-scoring subset: {int(nf[best]):_} features (score {final[best]:.{ndigits}f}).")
        n_kept = int(getattr(self, "n_features_", len(selected)))
        if n_kept != int(nf[best]) and isinstance(rule, str) and rule.startswith("one_se") and not getattr(self, "feature_cost", 0.0):
            top = int(idx[np.argmax(mean[idx])])
            base_rule, band_kind = split_rule_band(rule)
            which = "largest" if base_rule.startswith("one_se_max") else "smallest"
            top_std = std[top] if np.isfinite(std[top]) else 0.0
            if band_kind == "foldstd":
                half, band_txt, arith = top_std, "one across-fold std", f"{mean[top]:.{ndigits}f} - {top_std:.{ndigits}f}"
            else:
                k = int(fold_counts(self, [int(nf[top])])[0])
                half = float(band_half_width(np.array([top_std]), np.array([k]))[0])
                band_txt, arith = "one standard error (fold std / sqrt(k))", f"{mean[top]:.{ndigits}f} - {top_std:.{ndigits}f}/sqrt({k})"
            parts.append(
                f"Rule {rule} keeps the {which} evaluated size whose CV mean is >= best mean minus {band_txt} "
                f"({arith} = {mean[top] - half:.{ndigits}f}); "
                f"pass n_features_selection_rule='argmax' to keep the best-scoring subset instead."
            )
    parts.append(f"Selected: [{format_name_list(selected)}]")
    return " ".join(parts)


def log_rfecv_fit_summary(self: Any, *, stop_reason: Optional[str], n_iters: int, elapsed_s: float, ndigits: int = 4) -> None:
    """Emit the summary at INFO: a single line per fit, and the only place the kept size, stop reason and column names meet.

    Silent when RFECV is fitted inside a wrapper's ``quiet_nested`` scope (e.g. ``cascade_select``), whose own line is the one to read.
    """
    if is_quiet():
        return
    try:
        logger.info(build_rfecv_fit_summary(self, stop_reason=stop_reason, n_iters=n_iters, elapsed_s=elapsed_s, ndigits=ndigits))
    except Exception as exc:  # a reporting failure must never fail a multi-hour fit
        logger.debug("RFECV fit summary failed: %s", exc)

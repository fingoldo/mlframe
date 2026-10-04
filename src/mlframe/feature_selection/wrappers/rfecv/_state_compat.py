"""Pickle compatibility for ``RFECV``: attributes added after a release are backfilled when an older pickle is loaded."""

from __future__ import annotations

from mlframe.feature_selection._legacy_state import backfill_legacy_state

# Value an estimator fitted by an older release gets for each fitted attribute it lacked.
_SETSTATE_LEGACY_DEFAULTS: dict = {
    "_auto_tune_applied_": False,
    "_fit_sample_weight_": None,
    "_n_samples_fit_": None,
    "_wide_data_fi_applied_": False,
    "auto_tune_decision_": None,
    "consensus_ranking_": None,
    "cv_": None,
    "cv_results_": {},
    "estimators_": [],
    "eval_trace_": None,
    "feature_importances_": None,
    "futility_verdict_": None,
    "multioutput_skipped_": None,
    "multioutput_strategy_": None,
    "multioutput_supports_": None,
    "n_features_": None,
    "provenance_": None,
    "ranking_": None,
    "resolved_n_features_rule_": None,
    "scoring_": None,
    "stability_avg_selected_per_bootstrap_": None,
    "stability_pfer_bound_": None,
    "stability_selection_freq_": None,
}

# Core fitted state: an old pickle without it must stay unfitted, so no default is invented.
CORE_FITTED_ATTRS = frozenset({"support_", "selected_features_", "feature_names_in_", "n_features_in_"})


class RFECVStateCompatMixin:
    """Provides ``__setstate__`` for ``RFECV``."""

    def __setstate__(self, state: dict) -> None:
        """Restore state and backfill fitted attributes missing from a pickle written by an older release."""
        super().__setstate__(backfill_legacy_state(state, _SETSTATE_LEGACY_DEFAULTS, "support_"))  # type: ignore[misc]

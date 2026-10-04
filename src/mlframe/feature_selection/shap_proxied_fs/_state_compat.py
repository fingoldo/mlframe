"""Pickle compatibility for ``ShapProxiedFS``: attributes added after a release are backfilled when an older pickle is loaded."""

from __future__ import annotations

from mlframe.feature_selection._legacy_state import backfill_legacy_state

# Value an estimator fitted by an older release gets for each fitted attribute it lacked.
_SETSTATE_LEGACY_DEFAULTS: dict = {
    "shap_proxy_report_": None,
}

# Core fitted state: an old pickle without it must stay unfitted, so no default is invented.
CORE_FITTED_ATTRS = frozenset({"support_", "selected_features_", "feature_names_in_", "n_features_in_", "classes_"})


class ShapProxiedStateCompatMixin:
    """Provides ``__setstate__`` for ``ShapProxiedFS``."""

    def __setstate__(self, state: dict) -> None:
        """Restore state and backfill fitted attributes missing from a pickle written by an older release."""
        super().__setstate__(backfill_legacy_state(state, _SETSTATE_LEGACY_DEFAULTS, "support_"))  # type: ignore[misc]

"""sklearn protocol plumbing for ``BorutaShap``: lean pickling, old-pickle migration and ``get_feature_names_out``."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.exceptions import NotFittedError

from mlframe.feature_selection._legacy_state import backfill_legacy_state

# Fitted/working attributes, stored under ``<name>_``. Estimators pickled by earlier releases keyed them by the bare name; ``__setstate__`` migrates those.
PRE_RENAME_ATTRS = (
    "X", "y", "starting_X", "X_shadow", "X_boruta", "X_boruta_train", "X_boruta_test", "y_train", "y_test", "X_categorical",
    "X_feature_import", "Shadow_feature_import", "preds", "shap_values", "features_to_remove", "columns", "all_columns", "ncols", "order",
    "hits", "accepted_columns", "rejected_columns", "history_shadow", "history_hits", "history_x", "accepted", "rejected", "tentative",
)

# Training-data copies and per-trial scratch: valid right after ``fit`` only, never serialised with the estimator.
TRANSIENT_ATTRS = (
    "X_", "y_", "starting_X_", "X_shadow_", "X_boruta_", "X_boruta_train_", "X_boruta_test_", "y_train_", "y_test_", "X_categorical_",
    "X_feature_import_", "Shadow_feature_import_", "preds_", "shap_values_",
)

# Defaults an estimator pickled before an attribute existed gets on load; core selection state (support_, selected_features_) is deliberately absent.
_SETSTATE_LEGACY_DEFAULTS: dict = {
    "auto_dispatch_diagnostics_": None,
    "n_trials_run_": None,
    "stability_accept_counts_": None,
    "_auto_forced_test_split_": False,
}

# Fitted state without which a loaded estimator must behave as unfitted rather than be given a made-up default.
CORE_FITTED_ATTRS = frozenset({"selected_features_", "support_", "feature_names_in_", "n_features_in_", "model_"})

# Attributes assigned during fit that need no backfill: fit-internal scratch re-derived at the start of each fit, plus the renamed
# working/result state, which ``__setstate__`` migrates from its bare pre-rename keys.
SCRATCH_FITTED_ATTRS = frozenset(
    {"_current_trial_", "_premerge_active_", "_premerge_original_cols_", "_resolved_importance_measure_", "_train_or_test_"}
    | {name + "_" for name in PRE_RENAME_ATTRS}
)


class BorutaShapProtocolMixin:
    """Pickling and feature-name protocol for ``BorutaShap``; sits before ``BaseEstimator`` in the MRO."""

    def __getstate__(self) -> dict:
        """Pickle state without the training-data copies (X, y, shadow frames, SHAP values) that ``fit`` leaves on the instance."""
        state = dict(super().__getstate__())  # type: ignore[misc]
        for name in TRANSIENT_ATTRS:
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state, migrating pre-rename bare keys and backfilling attributes older releases did not have."""
        state = dict(state)
        for name in PRE_RENAME_ATTRS:
            if name in state and name + "_" not in state:
                state[name + "_"] = state.pop(name)
        super().__setstate__(backfill_legacy_state(state, _SETSTATE_LEGACY_DEFAULTS, "selected_features_"))  # type: ignore[misc]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the selected features, in the order ``transform`` emits them; ``input_features`` must match the fitted width when given."""
        if not hasattr(self, "selected_features_"):
            raise NotFittedError("BorutaShap is not fitted; call fit() first.")
        if input_features is not None:
            n_in = getattr(self, "n_features_in_", None)
            if n_in is not None and len(list(input_features)) != int(n_in):
                raise ValueError(f"input_features has {len(list(input_features))} elements, expected {n_in} (n_features_in_).")
        return np.asarray(list(self.selected_features_), dtype=object)

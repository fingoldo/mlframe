"""sklearn protocol plumbing for ``BorutaShap``: legacy attribute aliases, lean pickling, old-pickle migration and ``get_feature_names_out``."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.exceptions import NotFittedError

from mlframe.feature_selection._legacy_state import backfill_legacy_state

# Fitted/working attributes now stored under a trailing-underscore name. The bare name stays as a read/write alias for existing callers.
LEGACY_ALIASED_ATTRS = (
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
# working/result state, which ``__setstate__`` migrates from its bare legacy keys.
SCRATCH_FITTED_ATTRS = frozenset(
    {"_current_trial_", "_premerge_active_", "_premerge_original_cols_", "_resolved_importance_measure_", "_train_or_test_"}
    | {name + "_" for name in LEGACY_ALIASED_ATTRS}
)


class _LegacyAlias:
    """Data descriptor mapping the historical bare attribute name onto its trailing-underscore storage (``<name>_``)."""

    def __init__(self, name: str) -> None:
        self._storage = name + "_"
        self.__doc__ = f"Alias of ``{self._storage}``, kept for callers written against the pre-sklearn-convention names."

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        """Return the stored value, or the descriptor itself on class access."""
        if obj is None:
            return self
        try:
            return obj.__dict__[self._storage]
        except KeyError:
            raise AttributeError(f"{type(obj).__name__!r} object has no attribute {self._storage[:-1]!r}") from None

    def __set__(self, obj: Any, value: Any) -> None:
        """Store ``value`` under the trailing-underscore name."""
        obj.__dict__[self._storage] = value

    def __delete__(self, obj: Any) -> None:
        """Delete the stored value."""
        try:
            del obj.__dict__[self._storage]
        except KeyError:
            raise AttributeError(self._storage[:-1]) from None


class BorutaShapProtocolMixin:
    """Aliases, pickling and feature-name protocol for ``BorutaShap``; sits before ``BaseEstimator`` in the MRO."""

    def __getstate__(self) -> dict:
        """Pickle state without the training-data copies (X, y, shadow frames, SHAP values) that ``fit`` leaves on the instance."""
        state = dict(super().__getstate__())  # type: ignore[misc]
        for name in TRANSIENT_ATTRS:
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state, migrating pre-rename bare keys and backfilling attributes older releases did not have."""
        state = dict(state)
        for name in LEGACY_ALIASED_ATTRS:
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


def install_legacy_aliases(cls: type) -> None:
    """Attach a ``_LegacyAlias`` descriptor to ``cls`` for every name in ``LEGACY_ALIASED_ATTRS``."""
    for name in LEGACY_ALIASED_ATTRS:
        setattr(cls, name, _LegacyAlias(name))

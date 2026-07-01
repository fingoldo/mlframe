"""Return-shape contract + string-target encoding helpers for the suite facade.

Lifted from ``_main_train_suite.py`` so the facade stays within its LOC budget.
Re-exported from the parent so existing imports keep resolving.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from ..configs import TargetTypes


class SuiteResult(NamedTuple):
    """Suite return value: ``(models_dict, metadata_dict)``.

    Fully back-compatible with the historical bare 2-tuple: it unpacks as
    ``models, metadata = result`` and indexes as ``result[0]`` / ``result[1]``,
    and ``isinstance(result, tuple)`` is True (NamedTuple subclasses ``tuple``).
    ``.models`` / ``.metadata`` return the SAME dict objects, so nested-dict
    access like ``metadata["baseline_diagnostics"]["regression"]["y"]`` keeps
    working unchanged.

    ``models`` maps per-target/per-model keys to fitted estimators; ``metadata``
    carries the run's reports, splits, and provenance.
    """

    models: Dict[str, Any]
    metadata: Dict[str, Any]

    def get_model(self, name: str, task: Optional[str] = None) -> Any:
        """Return a fitted model by key.

        Looks up ``name`` directly in ``models`` first. When ``task`` is given
        (or on a miss), also tries the ``"{task}_{name}"`` / ``"{name}_{task}"``
        composite keys the suite emits. Returns None when nothing matches.
        """
        if task is None and name in self.models:
            return self.models[name]
        candidates = [name]
        if task is not None:
            candidates = [f"{task}_{name}", f"{name}_{task}", name]
        for key in candidates:
            if key in self.models:
                return self.models[key]
        return None

    def best_model(self, task: Optional[str] = None) -> Any:
        """Return a single representative fitted model.

        When ``task`` is provided, restrict to keys containing that task tag.
        Returns the first matching model (insertion order), or None when empty.
        This is a convenience accessor, not a metric-ranked selection.
        """
        for key, model in self.models.items():
            if task is None or task in key:
                return model
        return None


def _assert_suite_return_shape(result: Any, *, source: str) -> SuiteResult:
    """Verify an LTR-dispatch passthrough matches the suite's ``(models, metadata)`` contract.

    Coerces a bare 2-tuple into a :class:`SuiteResult` so the whole suite returns
    the richer type uniformly (LTR/dispatch passthroughs may still yield a plain tuple).
    """
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], dict)):
        raise TypeError(
            f"{source} returned {type(result).__name__} not matching the suite "
            f"(models_dict, metadata_dict) 2-tuple contract."
        )
    if isinstance(result, SuiteResult):
        return result
    return SuiteResult(result[0], result[1])


def _encode_string_multiclass_target(
    target_type: "TargetTypes",
    target_name: str,
    target_values: Any,
    metadata: Dict[str, Any],
) -> Any:
    """Label-encode a string/object 1-D multiclass target to integer codes.

    String-labelled multiclass targets ("a"/"b"/"c") otherwise reach numeric-only guards
    (np.isfinite in the regression-refit collapse detector) and the binary positive-rate
    summary in select_target, which mis-reports them as a single class. Factorizing to
    sorted-unique integer codes makes the string path behave identically to the int path.
    The bijection over the label space leaks no y-statistic into X, so encoding the full
    target is leakage-safe; the sorted classes_ are stamped into metadata for predict-time
    inverse mapping by callers that don't carry classes_ on the estimator.
    """
    if target_type != TargetTypes.MULTICLASS_CLASSIFICATION:
        return target_values
    if isinstance(target_values, (pd.Series, pl.Series)):
        arr = target_values.to_numpy()
    else:
        arr = np.asarray(target_values)
    if arr.ndim != 1 or arr.dtype.kind not in ("O", "U", "S"):
        return target_values
    classes_ = np.unique(arr)
    codes = np.searchsorted(classes_, arr).astype(np.int64)
    metadata.setdefault("target_label_classes", {})[target_name] = list(classes_)
    return codes

"""Return-shape contract + string-target encoding helpers for the suite facade.

Lifted from ``_main_train_suite.py`` so the facade stays within its LOC budget.
Re-exported from the parent so existing imports keep resolving.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl

from ..configs import TargetTypes


# Suite return contract: ``(models_dict, metadata_dict)``. ``models_dict`` maps
# per-target/per-model keys to fitted estimators; ``metadata_dict`` carries the
# run's reports, splits, and provenance. The LTR-dispatch passthrough must
# return this same 2-tuple shape (asserted by ``_assert_suite_return_shape``).
SuiteResult = Tuple[Dict[str, Any], Dict[str, Any]]


def _assert_suite_return_shape(result: Any, *, source: str) -> SuiteResult:
    """Verify an LTR-dispatch passthrough matches the suite's ``(models, metadata)`` contract."""
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], dict)):
        raise TypeError(
            f"{source} returned {type(result).__name__} not matching the suite "
            f"(models_dict, metadata_dict) 2-tuple contract."
        )
    return result


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

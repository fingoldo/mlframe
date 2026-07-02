"""Default-extractor construction for ``train_mlframe_models_suite``.

Carved out of ``_main_train_suite.py`` to keep the orchestration facade under
its LOC budget. Re-exported from the parent so historical import sites resolve
transparently.
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl


def _infer_target_is_classification(values: Any) -> bool:
    """Infer whether ``target_name``'s column is a classification target.

    Reuses the project heuristic (``slicing._slice_helpers._is_classification_target``:
    integer/bool dtype + small cardinality). Float continuous columns fall through to
    regression. Object/string columns are treated as classification (label targets).
    """
    from ..slicing._slice_helpers import _is_classification_target

    if isinstance(values, (pd.Series, pl.Series)):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    if arr.dtype.kind in ("O", "U", "S"):
        return True
    return _is_classification_target(arr)


def _build_default_extractor(df: Union[pl.DataFrame, pd.DataFrame, str], target_name: str):
    """Construct a ``SimpleFeaturesAndTargetsExtractor`` from ``target_name`` alone.

    Infers the task type from the target column's dtype/cardinality: a continuous
    float target becomes a regression target, a low-cardinality int/categorical/string
    target becomes a classification target. Used only when the caller omits an explicit
    extractor (the ergonomic ``train_mlframe_models_suite(df, target_name="y")`` path).
    """
    from ..extractors import SimpleFeaturesAndTargetsExtractor

    # Read just the target column so the dtype can be inspected without loading the
    # full frame when a parquet path was supplied.
    if isinstance(df, str):
        target_values = pl.read_parquet(df, columns=[target_name])[target_name]
    else:
        if target_name not in df.columns:
            # No extractor supplied AND the target column is absent, so no default can be
            # inferred. Surface as a ValueError naming ``features_and_targets_extractor`` -- the
            # caller must either pass one explicitly or a ``target_name`` that exists in ``df``.
            raise ValueError(
                f"target_name '{target_name}' not found in df columns and no "
                f"features_and_targets_extractor was supplied; pass an explicit "
                f"features_and_targets_extractor or a target_name present in df. "
                f"Available (first 10): {list(df.columns)[:10]}"
            )
        target_values = df[target_name]

    if _infer_target_is_classification(target_values):
        return SimpleFeaturesAndTargetsExtractor(classification_targets=[target_name])
    return SimpleFeaturesAndTargetsExtractor(regression_targets=[target_name])

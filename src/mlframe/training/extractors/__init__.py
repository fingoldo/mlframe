"""
Feature and target extraction utilities for mlframe.

Provides classes for preparing DataFrames before ML training, including:
- Target extraction and transformation
- Sample weight computation
- Feature engineering hooks
"""

from __future__ import annotations


import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.pythonlib import get_parent_func_args, store_params_in_object

from ._extractors_dtype_helpers import (  # -- re-export
    _safe_int_cast_numpy,
    _smallest_safe_int_dtype,
    get_dataframe_info,
    get_sample_weights_by_recency,
    intize_targets,
)
from ._extractors_showcase import showcase_features_and_targets  # -- re-export
from ..utils import get_pandas_view_of_polars_df, log_ram_usage

logger = logging.getLogger(__name__)


from ._extractors_base import (  # re-exported from the facade
    FeaturesAndTargetsExtractor,
    FeaturesAndTargetsExtractorProtocol,
)

from ._extractors_simple import SimpleFeaturesAndTargetsExtractor

__all__ = [
    # Helper functions
    "get_dataframe_info",
    "intize_targets",
    "get_sample_weights_by_recency",
    "showcase_features_and_targets",
    # Classes
    "FeaturesAndTargetsExtractor",
    "SimpleFeaturesAndTargetsExtractor",
    "FeaturesAndTargetsExtractorProtocol",
]

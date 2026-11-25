"""
Feature and target extraction utilities for mlframe.

Provides classes for preparing DataFrames before ML training, including:
- Target extraction and transformation
- Sample weight computation
- Feature engineering hooks
"""

from __future__ import annotations

import logging

from ._extractors_dtype_helpers import (  # -- re-export
    get_dataframe_info,
    get_sample_weights_by_recency,
    intize_targets,
)
from ._extractors_showcase import showcase_features_and_targets  # -- re-export

logger = logging.getLogger(__name__)


from ._extractors_base import (  # re-exported from the facade
    FeaturesAndTargetsExtractor,
    FeaturesAndTargetsExtractorProtocol,
)

from ._extractors_simple import SimpleFeaturesAndTargetsExtractor

__all__ = [
    "get_dataframe_info",
    "intize_targets",
    "get_sample_weights_by_recency",
    "showcase_features_and_targets",
    "FeaturesAndTargetsExtractor",
    "SimpleFeaturesAndTargetsExtractor",
    "FeaturesAndTargetsExtractorProtocol",
]

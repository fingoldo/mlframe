"""Carved out of ``mlframe.training.core.predict``.

Bound back into the parent's namespace via ``from ._predict_<name> import X``
at the parent's module bottom so historical
``from mlframe.training.core.predict import predict_from_models``
resolves transparently.
"""
from __future__ import annotations

import glob
import logging
import os
import pickle as _pickle
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join

from scipy import stats
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify

from ..configs import TargetTypes
from ..extractors import FeaturesAndTargetsExtractor
from ..io import load_mlframe_model
from ..pipeline import prepare_df_for_catboost
from ..cb import _predict_with_fallback
from ..utils import drop_columns_from_dataframe, get_pandas_view_of_polars_df
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _drop_cols_df,
    _setup_model_directories,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)

logger = logging.getLogger("mlframe.training.core.predict")


# ----------------------------------------------------------------------
# Sub-sibling re-exports. The two entry-points each live in their own
# file so this file stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._predict_main_from_models import predict_from_models  # noqa: E402,F401
from ._predict_main_suite import predict_mlframe_models_suite  # noqa: E402,F401

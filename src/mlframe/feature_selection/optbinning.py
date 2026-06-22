
from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

"""Categorical feature engineering for ML. Optimized & rich set of aggregates for 1d vectors."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import pandas as pd

import category_encoders as ce
from optbinning import BinningProcess
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def get_binningprocess_featureselectors(
    features: pd.DataFrame,
    memory: str = None,
    n_jobs: int = -1,
    iv_kwargs: dict = None,
) -> tuple:
    """Returns binningprocess pipelines for a specific featureset. Inserts categorical encoders where needed."""
    if iv_kwargs is None:
        iv_kwargs = {"min": 0.02, "strategy": "highest"}
    all_cols = features.columns.tolist()

    bp_withcats_fs = Pipeline(
        [
            ("enc", ce.CatBoostEncoder()),
            (
                "BP",
                BinningProcess(
                    variable_names=all_cols,
                    selection_criteria={
                        "iv": {"top": len(all_cols), **iv_kwargs},
                    },
                    n_jobs=n_jobs,
                ),
            ),
            ("identity", FunctionTransformer()),
        ],
        memory=memory,
    )
    bp_withcats_nofs = Pipeline(
        [("enc", ce.CatBoostEncoder()), ("BP", BinningProcess(variable_names=all_cols, n_jobs=n_jobs)), ("identity", FunctionTransformer())], memory=memory
    )

    nocat_cols = features.columns.tolist()
    # ``.head()`` is safe for category detection because dtype is a column-level attribute (the first 5 rows carry the same dtype as the full column). CONTRACT: column
    # names must be unique -- ``list.remove`` drops only the first match, so a duplicated name would leave a stale category column in ``nocat_cols`` (or raise if absent).
    for col in features.head().select_dtypes("category").columns:
        nocat_cols.remove(col)

    bp_nocats_fs = Pipeline(
        [
            (
                "BP",
                BinningProcess(
                    variable_names=nocat_cols,
                    selection_criteria={
                        "iv": {"top": len(nocat_cols), **iv_kwargs},
                    },
                    n_jobs=n_jobs,
                ),
            ),
            ("identity", FunctionTransformer()),
        ],
        memory=memory,
    )
    bp_nocats_nofs = Pipeline([("BP", BinningProcess(variable_names=nocat_cols, n_jobs=n_jobs)), ("identity", FunctionTransformer())], memory=memory)

    return bp_withcats_fs, bp_withcats_nofs, bp_nocats_fs, bp_nocats_nofs

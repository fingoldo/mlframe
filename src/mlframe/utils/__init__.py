"""Miscellaneous helpers.

Submodules:
    eda          - exploratory-data-analysis re-exports.
    experiments  - experiment-tracking helpers.
    text         - text preprocessing utilities.
    misc         - small generic helpers (get_pipeline_last_element, ...).
"""

from __future__ import annotations


from mlframe.utils.eda import *
from mlframe.utils.experiments import *
from mlframe.utils.text import *
from mlframe.utils.misc import *

# 2026-06-01: promote the param-oracle public surface so cross-package
# consumers (``feature_selection.filters._meta_fe_recommender``,
# the recommender CLI, etc.) can import the symbols from
# ``mlframe.utils`` instead of reaching into ``mlframe.utils._param_oracle``
# directly -- the cross-package underscore-import meta-linter flagged the
# bare ``_param_oracle`` reach as a private-API leak.
from mlframe.utils._param_oracle import (
    ParamOracle,
    bucketize_fingerprint,
    default_fingerprint,
    loads_json,
    stable_json,
)

# Curate the star-import surface explicitly (mirrors mlframe.metrics.__init__'s pattern).
__all__ = sorted(name for name in globals() if not name.startswith("_"))

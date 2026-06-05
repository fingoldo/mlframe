"""Miscellaneous helpers.

Submodules:
    eda          - exploratory-data-analysis re-exports.
    experiments  - experiment-tracking helpers.
    text         - text preprocessing utilities.
    misc         - small generic helpers (get_pipeline_last_element, ...).
"""

from __future__ import annotations


from mlframe.utils.eda import *  # noqa: F401,F403
from mlframe.utils.experiments import *  # noqa: F401,F403
from mlframe.utils.text import *  # noqa: F401,F403
from mlframe.utils.misc import *  # noqa: F401,F403

# 2026-06-01: promote the param-oracle public surface so cross-package
# consumers (``feature_selection.filters._meta_fe_recommender``,
# the recommender CLI, etc.) can import the symbols from
# ``mlframe.utils`` instead of reaching into ``mlframe.utils._param_oracle``
# directly -- the cross-package underscore-import meta-linter flagged the
# bare ``_param_oracle`` reach as a private-API leak.
from mlframe.utils._param_oracle import (  # noqa: F401
    ParamOracle,
    bucketize_fingerprint,
    default_fingerprint,
    loads_json,
    stable_json,
)

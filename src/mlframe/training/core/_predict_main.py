"""Carved out of ``mlframe.training.core.predict``.

Bound back into the parent's namespace via ``from ._predict_<name> import X``
at the parent's module bottom so historical
``from mlframe.training.core.predict import predict_from_models``
resolves transparently.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("mlframe.training.core.predict")


# ----------------------------------------------------------------------
# Sub-sibling re-exports. The two entry-points each live in their own
# file so this file stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._predict_main_from_models import predict_from_models  # noqa: E402,F401
from ._predict_main_suite import predict_mlframe_models_suite  # noqa: E402,F401

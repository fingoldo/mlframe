"""The frame a composite wrapper hands its inner model, in a flavour that model can read.

The wrapper passes its inner model whatever frame it was given, on purpose: converting a large polars frame to pandas is
a full copy, and most inner models read polars directly. Some cannot. LightGBM 4.6 under scikit-learn 1.8 raises on any
polars frame (scikit-learn tries to set ``feature_names_in_``, which LightGBM exposes as a read-only property), so a
composite target with a LightGBM inner failed to fit at all on the carrier the rest of the suite treats as first-class.

Whether a library reads polars is asked of the installed build (``accepts_polars``, a cached child-process probe) rather
than assumed, and a conversion only happens when the answer is no - that is, when the alternative is an exception. It is
logged, once per library, with the reason, so a copy of the frame is never silent.
"""

from __future__ import annotations

import logging
from typing import Any

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

__all__ = ["frame_for_inner"]

# Libraries whose polars support depends on the installed build; ``accepts_polars`` knows how to probe these two.
_PROBED_LIBRARIES = ("lightgbm", "catboost")


def _probed_library(estimator: Any) -> str | None:
    """The probed library ``estimator`` belongs to, looking through its class hierarchy for a wrapped subclass."""
    for cls in type(estimator).__mro__:
        root = (getattr(cls, "__module__", "") or "").split(".", 1)[0]
        if root in _PROBED_LIBRARIES:
            return root
    return None


def frame_for_inner(estimator: Any, X: Any) -> Any:
    """``X`` as-is, or its pandas copy when ``X`` is polars and ``estimator``'s library cannot read polars.

    Parameters
    ----------
    estimator
        The inner model about to fit on or predict from ``X``.
    X
        The frame the wrapper would hand it.

    Returns
    -------
    Any
        ``X`` itself for a non-polars frame, a library that reads polars, or a library this module does not probe;
        otherwise ``X.to_pandas()``.
    """
    try:
        import polars as pl
    except ImportError:
        return X
    if not isinstance(X, pl.DataFrame):
        return X
    library = _probed_library(estimator)
    if library is None:
        return X
    from mlframe.training._polars_native_support import accepts_polars

    if accepts_polars(library):
        return X
    log_throttle(
        logger, f"composite_inner_polars_to_pandas_{library}", logging.INFO,
        "[CompositeTargetEstimator] the inner %s cannot read a polars frame on this installation (probed), so its %d x %d "
        "input is converted to pandas; pass a polars-capable inner or a pandas frame to avoid the copy.",
        type(estimator).__name__, X.height, X.width,
    )
    return X.to_pandas()

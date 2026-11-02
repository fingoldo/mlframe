"""Name under which an estimator's ``fit`` accepts a per-row sample weight, looking through wrappers."""

from __future__ import annotations

from typing import Any, Optional

from ._data_helpers import get_function_param_names


def _accepts_weight(estimator: Any) -> bool:
    """True when ``estimator.fit`` declares ``sample_weight``."""
    return "sample_weight" in get_function_param_names(estimator.fit)


def oof_weight_fit_param(estimator: Any) -> Optional[str]:
    """The ``fit`` keyword that carries the sample weight, or None when the estimator cannot take one.

    A target-transform wrapper declares only ``**fit_params`` and forwards them, and a Pipeline routes ``<step>__sample_weight`` to its final step,
    so both are resolved through the inner estimator instead of being treated as unweighted.
    """
    if _accepts_weight(estimator):
        return "sample_weight"
    inner = getattr(estimator, "regressor", None) or getattr(estimator, "classifier", None)
    if inner is not None and _accepts_weight(inner):
        return "sample_weight"
    steps = getattr(estimator, "steps", None)
    if steps:
        name, last = steps[-1]
        if last is not None and last != "passthrough" and oof_weight_fit_param(last) is not None:
            return f"{name}__sample_weight"
    return None

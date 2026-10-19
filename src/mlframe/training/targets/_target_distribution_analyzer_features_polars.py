"""Polars-family dispatch used by ``_normalise_X``: convert a polars input to a pandas frame, or return None."""

from __future__ import annotations

# --- imports (managed) ---
import pandas as pd
from mlframe.training.targets._target_distribution_analyzer_features import logger
# --- end imports ---


def _polars_to_pandas_frame(X, feature_names):
    """Return the pandas frame for a polars DataFrame, LazyFrame or Series input, or None for anything else."""
    _module = type(X).__module__ if X is not None else ""
    _is_polars = isinstance(_module, str) and _module.startswith("polars")
    df = None
    if _is_polars:
        _typename = type(X).__name__
        # iter291: use mlframe's optimised pl->pd helper
        # for DataFrame conversion. The helper hits the pyarrow split-
        # blocks fastpath (no consolidation copy of every numeric
        # buffer) and the dict->int32 cast on Categorical columns, so
        # the bridge is ~5x faster than the naive ``.to_pandas()`` on
        # typical 200k x 100+ frames. c0140 iter291 profile attributed
        # 29.3s to a single _normalise_X call via raw to_pandas; the
        # helper drops it to ~6s. Lazy import to avoid utils <-> *
        # cycle.
        from mlframe.training.utils import get_pandas_view_of_polars_df as _pl2pd

        if _typename == "LazyFrame":
            # Materialise to DataFrame then convert. LazyFrame has neither
            # to_pandas nor columns directly, so caller must accept the
            # collect() cost; a LazyFrame walked through analyze_feature_
            # distribution is going to be materialised anyway downstream.
            try:
                df = _pl2pd(X.collect())
            except Exception as e:
                logger.debug("_pl2pd(X.collect()) failed: %s", e)
                df = None
        elif _typename == "Series":
            # 1-D series -> single-column frame. Use the series name if
            # caller didn't supply feature_names. ``get_pandas_view_of_
            # polars_df`` raises on Series; fall through to the legacy
            # ``.to_pandas()`` path which returns a pd.Series directly.
            _name = feature_names[0] if feature_names else (getattr(X, "name", None) if getattr(X, "name", None) not in (None, "") else "f0")
            try:
                df = pd.DataFrame({_name: X.to_pandas()})
            except Exception as e:
                logger.debug("building a DataFrame from a polars Series failed: %s", e)
                df = None
        else:
            try:
                df = _pl2pd(X)
            except Exception as e:
                # Helper may raise on unusual polars subclasses; fall
                # back to the naive call so the suite still progresses.
                logger.debug("_pl2pd(X) failed, falling back to X.to_pandas() if available: %s", e)
                _to_pandas = getattr(X, "to_pandas", None)
                if callable(_to_pandas):
                    df = _to_pandas()
    return df

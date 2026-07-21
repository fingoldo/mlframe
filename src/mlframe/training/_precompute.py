"""Train-set feature stats + precompute bundle for suite fast-path.

Wave 95 (2026-05-21): split out from `helpers.py` to keep that file
below the 1k-line threshold. Behaviour preserved bit-for-bit; every
moved symbol is re-exported from `helpers` so existing
``from mlframe.training.helpers import precompute_all`` (and the other
moved names) imports continue to work.

What lives here:
  - ``get_trainset_features_stats`` (pandas backend)
  - ``get_trainset_features_stats_polars`` (polars backend)
  - ``TrainMlframeSuitePrecomputed`` (dataclass bundle)
  - ``precompute_composite_target_specs`` (NotImplementedError stub)
  - ``precompute_dummy_baselines`` (NotImplementedError stub)
  - ``precompute_trainset_features_stats`` (backend dispatcher)
  - ``precompute_all`` (one-shot helper that fills the stats slot)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import polars as pl
import polars.selectors as cs

from pyutilz.system import tqdmu

from .utils import get_numeric_columns, get_categorical_columns

logger = logging.getLogger(__name__)


def get_trainset_features_stats(train_df: pd.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables.

    Numeric ranges are computed via a single ``df[num_cols].agg(['min','max'])``
    call rather than a per-column Python loop. On 1M rows x 60 numeric cols the
    vectorised path measures ~9x faster (single C-level reduction over a
    contiguous block versus N separate column reductions with attribute
    look-up overhead per iteration).
    """
    res = {}
    num_cols = get_numeric_columns(train_df)
    if num_cols:
        if len(num_cols) == train_df.shape[1]:
            res["min"] = train_df.min(axis=0)
            res["max"] = train_df.max(axis=0)
        else:
            # Vectorised aggregation: pandas reduces all numeric columns in a
            # single pass instead of issuing one .min() / .max() call per col.
            # Slicing once via train_df[num_cols] avoids the "Categorical is
            # not ordered for operation min" error that surfaced when the
            # whole frame was reduced (and that the previous per-col loop
            # worked around row-by-row).
            agg_df = train_df[num_cols].agg(["min", "max"])
            res["min"] = agg_df.loc["min"]
            res["max"] = agg_df.loc["max"]

    cat_cols = get_categorical_columns(train_df, include_string=False)
    if cat_cols:
        cat_vals = {}
        # Fast path for pd.CategoricalDtype columns: ``.cat.categories`` is the dtype-declared domain
        # and returns in O(1) without a data scan. Measured ~1880x faster than ``.unique()`` on a
        # 1M-row x 30-col all-Categorical frame (207ms -> 0.1ms). Object/string columns still need a
        # data scan via ``.unique()`` since their domain is not pinned at dtype level.
        for col in tqdmu(cat_cols, desc="cat vars stats", leave=False):
            _dt = train_df[col].dtype
            if isinstance(_dt, pd.CategoricalDtype):
                unique_vals = _dt.categories.to_numpy()
            else:
                unique_vals = train_df[col].unique()
            if not max_ncats_to_track or (len(unique_vals) <= max_ncats_to_track):
                cat_vals[col] = unique_vals
        res["cat_vals"] = cat_vals
    return res


def get_trainset_features_stats_polars(train_df: pl.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables using Polars.

    Uses lazy mode and selectors for parallel computation.

    Args:
        train_df: Polars DataFrame
        max_ncats_to_track: Max unique values to track for categorical columns

    Returns:
        dict with "min", "max" (as pd.Series) and "cat_vals" (dict of arrays)
    """

    res = {}
    lf = train_df.lazy()

    # Compute numeric min/max and categorical n_unique in a single parallel select
    stats = lf.select(
        # Numeric: min and max
        cs.numeric().min().name.suffix("__min"),
        cs.numeric().max().name.suffix("__max"),
        # Categorical: n_unique to filter before getting unique values
        cs.by_dtype(pl.String, pl.Categorical).n_unique().name.suffix("__n_unique"),
    ).collect()

    # Extract numeric stats
    if len(stats.columns) > 0:
        mins = {}
        maxs = {}
        for col in stats.columns:
            if col.endswith("__min"):
                orig_col = col[:-5]
                mins[orig_col] = stats[col][0]
            elif col.endswith("__max"):
                orig_col = col[:-5]
                maxs[orig_col] = stats[col][0]

        if mins:
            res["min"] = pd.Series(mins)
        if maxs:
            res["max"] = pd.Series(maxs)

    # Extract categorical columns that are under the threshold
    cat_cols_to_fetch = []
    for col in stats.columns:
        if col.endswith("__n_unique"):
            orig_col = col[:-10]
            n_unique = stats[col][0]
            if not max_ncats_to_track or n_unique <= max_ncats_to_track:
                cat_cols_to_fetch.append(orig_col)

    # Get unique values for qualifying categorical columns. Batched into
    # ONE collect() via implode() so per-column unique-vectors arrive as
    # rows of one frame; saves ``len(cat_cols_to_fetch) - 1`` LazyFrame
    # materializations (each costing 5-10ms on a typical mid-size frame).
    # On a 100k×15-cat-cols frame this dropped 14 collects -> 1 collect
    # without changing semantics (implode collapses unique values into
    # a single list-typed cell per column; we then unpack via [0]).
    if cat_cols_to_fetch:
        unique_lists = lf.select([pl.col(c).unique().implode().alias(c) for c in cat_cols_to_fetch]).collect()
        cat_vals = {col: unique_lists[col][0].to_numpy() for col in cat_cols_to_fetch}
        res["cat_vals"] = cat_vals

    return res


# -----------------------------------------------------------------------------
# Precomputed suite bundle (opt-in fast path for repeated-suite-on-same-train benchmarking)
# -----------------------------------------------------------------------------


@dataclass
class TrainMlframeSuitePrecomputed:
    """Bundle of pre-computed train-set artifacts that ``train_mlframe_models_suite`` would otherwise compute inline.

    Populated via the ``precompute_*`` helpers in this module. Pass to the suite as
    ``precomputed=TrainMlframeSuitePrecomputed(...)`` to skip the matching in-suite compute steps;
    each field is independently opt-in (None = compute inline as today).

    ``train_df_fingerprint`` is reserved for a future cross-process disk-cache layer so a bundle
    persisted from one run can be safely re-attached only when the train frame hasn't changed.
    """
    trainset_features_stats: Optional[dict] = None
    dummy_baselines: Optional[dict] = None
    composite_target_specs: Optional[dict] = None
    train_df_fingerprint: Optional[str] = None  # for cross-process disk-cache reuse later


def precompute_composite_target_specs(
    train_df=None,
    target_by_type: Optional[dict] = None,
    config: Optional[Any] = None,
) -> dict:
    """NOT IMPLEMENTED -- always raises.

    A faithful precompute would have to mirror ``run_composite_target_discovery``: composite_cache
    wiring, library version signatures, DiscoveryCache fingerprints. That surface is large and lives
    behind locked files; until the helper can reuse the same cache key path as the suite and stay
    byte-equal across runs, returning an empty dict here would silently disable discovery on the
    suite side -- worse than recomputing.

    Callers who already have a prior run's ``metadata["composite_target_specs"]`` saved to disk can
    still feed the suite directly via ``TrainMlframeSuitePrecomputed(composite_target_specs=...)``;
    the bundle's skip-when-supplied gate is content-truthy, not just non-None, so an empty dict will
    NOT disable the in-suite compute (see ``train_mlframe_models_suite`` for the gate).

    Raises:
        NotImplementedError: always. Use ``TrainMlframeSuitePrecomputed(composite_target_specs=<dict from prior run>)`` instead.
    """
    raise NotImplementedError(
        "precompute_composite_target_specs is not implemented. Load metadata['composite_target_specs'] from a prior run "
        "and pass it directly via TrainMlframeSuitePrecomputed(composite_target_specs=...).",
    )


def precompute_dummy_baselines(
    train_df,
    target_by_type: dict,
    config: Optional[Any] = None,
) -> dict:
    """NOT IMPLEMENTED -- always raises.

    The in-suite dummy-baseline compute lives in ``core/_phase_dummy_baselines.py`` and needs the
    post-split train/val/test frames plus per-target slices, which the caller does NOT have access to
    before the suite has run the split phase. A faithful precompute helper would have to either
    (a) replicate the suite's split logic here (duplication risk) or (b) accept the already-split
    frames + per-target targets as arguments (large signature). Both are deferred.

    Callers who already have a prior run's ``metadata["dummy_baselines"]`` saved to disk can feed
    the suite directly via ``TrainMlframeSuitePrecomputed(dummy_baselines=...)``; the bundle's
    skip-when-supplied gate is content-truthy, so an empty dict will NOT silently disable the
    per-target in-suite compute.

    Raises:
        NotImplementedError: always. Use ``TrainMlframeSuitePrecomputed(dummy_baselines=<dict from prior run>)`` instead.
    """
    raise NotImplementedError(
        "precompute_dummy_baselines is not implemented. Load metadata['dummy_baselines'] from a prior run and pass it "
        "directly via TrainMlframeSuitePrecomputed(dummy_baselines=...).",
    )


def precompute_trainset_features_stats(train_df, max_ncats_to_track: int = 1000) -> dict:
    """Compute the trainset_features_stats dict the suite would compute inline.

    Dispatches to the polars or pandas backend based on the input type so the output dict is
    byte-equal (same key order, same value shapes) to what ``train_mlframe_models_suite`` produces
    on the same frame. Use the returned dict as ``TrainMlframeSuitePrecomputed.trainset_features_stats``
    to skip the in-suite recompute on repeat runs.

    Args:
        train_df: Pandas or Polars DataFrame -- the same frame that will later be passed to the suite
            (post-split, post-pipeline-fit form). For pre-split callers, slice the train rows yourself
            first; the suite's stats step runs AFTER train/val/test split.
        max_ncats_to_track: forwarded to the underlying stats function.

    Returns:
        dict with at least ``min``, ``max`` (pd.Series) and ``cat_vals`` (dict[str, np.ndarray]) keys.
    """
    if isinstance(train_df, pl.DataFrame):
        return get_trainset_features_stats_polars(train_df, max_ncats_to_track=max_ncats_to_track)
    return get_trainset_features_stats(train_df, max_ncats_to_track=max_ncats_to_track)


def precompute_all(
    train_df,
    target_by_type: Optional[dict] = None,
    *,
    fs_config: Optional[Any] = None,
    dummy_baselines_config: Optional[Any] = None,
    composite_config: Optional[Any] = None,
) -> TrainMlframeSuitePrecomputed:
    """Fill the ``trainset_features_stats`` precompute slot; leave the rest at ``None``.

    Despite the name, this is NOT a one-shot helper for every slot: only ``trainset_features_stats``
    has a real precompute path. ``precompute_dummy_baselines`` and ``precompute_composite_target_specs``
    raise ``NotImplementedError`` -- the dummy helper needs post-split frames that aren't reachable
    pre-suite, and the composite helper would have to mirror the full discovery cache surface
    (deferred). The bundle slots themselves still work: callers who have a prior run's metadata
    saved to disk can build the bundle by hand:

        from mlframe.training.helpers import precompute_all, TrainMlframeSuitePrecomputed
        bundle = precompute_all(train_df, target_by_type)
        bundle.dummy_baselines = prior_run_metadata["dummy_baselines"]
        bundle.composite_target_specs = prior_run_metadata["composite_target_specs"]

    Args:
        train_df: Pandas or Polars train frame.
        target_by_type: per-target mapping (forwarded to dummy stub).
        fs_config: feature-stats kwargs container (currently only ``max_ncats_to_track`` is honored
            if present as an attribute; pass None for defaults).
        dummy_baselines_config: forwarded to the dummy stub.
        composite_config: forwarded to the composite stub.

    Returns:
        A populated ``TrainMlframeSuitePrecomputed`` bundle.
    """
    _max_ncats = 1000
    if fs_config is not None:
        _maybe = getattr(fs_config, "max_ncats_to_track", None)
        if isinstance(_maybe, int) and _maybe > 0:
            _max_ncats = _maybe
    stats = precompute_trainset_features_stats(train_df, max_ncats_to_track=_max_ncats)

    # precompute_dummy_baselines / precompute_composite_target_specs always raise
    # NotImplementedError (see their own docstrings) and are never called here; leave both
    # bundle slots at None so the suite's "if precomputed.X is not None" gate keeps recomputing
    # inline rather than silently skipping with no data.
    return TrainMlframeSuitePrecomputed(
        trainset_features_stats=stats,
        dummy_baselines=None,
        composite_target_specs=None,
        train_df_fingerprint=None,
    )

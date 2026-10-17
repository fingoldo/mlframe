"""Helpers carved out of ``_phase_pandas_conversion_and_cat_prep``: defer-conversion decision, size estimation and conversion logging."""
from __future__ import annotations

import logging
from functools import partial
from timeit import default_timer as timer
from typing import Any, Callable

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# CACHE-P1-7: cat-heavy frames inflate roughly 1.4-1.6x when going
# polars->pandas (string columns gain per-row object overhead, category
# columns lose dictionary compression). Apply a 1.5x safety factor when
# the pre-conversion size is the pandas-sizing input; downstream GPU /
# RAM heuristics over-allocate slightly rather than starving allocators.
_CAT_SIZE_SAFETY_FACTOR = 1.5


def _cat_heavy_size(df: Any, raw_bytes: float, cat_features: list[str] | None) -> float:
    """Inflate a polars frame's raw byte size by up to ``_CAT_SIZE_SAFETY_FACTOR`` proportional to its categorical-column fraction, approximating the pandas-conversion size increase; non-polars frames pass through unchanged."""
    try:
        if not isinstance(df, pl.DataFrame):
            return raw_bytes
        cat_cols = [c for c in (cat_features or []) if c in df.columns]
        n_cols = max(len(df.columns), 1)
        cat_frac = float(len(cat_cols)) / n_cols
        # Linear interp between 1.0x (no cat cols) and 1.5x (>=50% cat).
        scale = 1.0 + (_CAT_SIZE_SAFETY_FACTOR - 1.0) * min(cat_frac / 0.5, 1.0)
        return float(raw_bytes) * float(scale)
    except Exception as exc:
        logger.debug("_cat_heavy_size: categorical-fraction size estimate failed, using raw bytes: %s", exc)
        return raw_bytes


def _make_cat_heavy_size(cat_features: list[str] | None) -> Callable[[Any, float], float]:
    """Bind ``cat_features`` into ``_cat_heavy_size`` so it can be passed as a two-argument sizing callback."""
    return partial(_cat_heavy_size, cat_features=cat_features)


def _resolve_defer_pandas_conv(
    *,
    was_polars_input: bool,
    all_models_polars_native: bool,
    polars_pipeline_applied: bool,
    pipeline_stages_requested: bool,
    recurrent_models: list[str],
    has_rfecv: bool,
) -> tuple[bool, bool]:
    """Return ``(defer_pandas_conv, pipeline_state_lives_in_pandas)``."""
    # The earlier disjunction ``all_models_polars_native OR _has_non_native_mlframe_strategy`` was always True
    # given ``was_polars_input`` (the second disjunct collapses to ``was_polars_input AND NOT first``), so
    # the whole conjunction reduces to the three real gates: had a polars input, no recurrent model, no RFECV.
    # ``polars_pipeline_applied`` captures whether a polars-aware pipeline actually fitted on the polars frame;
    # when False the downstream pipeline state lives only in pandas representation, so the lazy-pandas fastpath
    # cannot keep frames as polars without losing that state.
    #
    # But "not applied" and "not requested" are different things, and conflating them cost a production run a
    # multi-GB conversion for nothing: a caller asking for no encoder, no scaler and no imputer has NO pipeline
    # state in either representation, so there is nothing for pandas to carry and the polars frames can stay.
    # That configuration is the normal one for a CatBoost-only run, which needs none of those transforms.
    #
    # Deferring needs BOTH halves though. ``polars_pipeline_applied`` used to imply "some model reads polars"
    # on its own, so relaxing the first half alone started deferring for an XGB-only run -- whose models cannot
    # read a polars frame, so each fit converted its own copy, and the cross-clone DMatrix cache (keyed on the
    # frame signature) missed on every weight schema. Ask about the models explicitly instead of inferring it.
    _pipeline_state_lives_in_pandas = pipeline_stages_requested and not polars_pipeline_applied
    _needs_pandas_frames = bool(_pipeline_state_lives_in_pandas or not all_models_polars_native)
    defer_pandas_conv = was_polars_input and not _needs_pandas_frames and not recurrent_models and not has_rfecv
    return defer_pandas_conv, _pipeline_state_lives_in_pandas


def _log_deferred_conversion(
    all_models_polars_native: bool,
    mlframe_models: list[str],
    strategies_for: Callable[[list[str]], list],
) -> None:
    """Log why the polars->pandas conversion was deferred (verbose only)."""
    if all_models_polars_native:
        logger.info("  Skipped pandas conversion -- all models are Polars-native")
    else:
        _strats = strategies_for(mlframe_models or [])
        non_native = [m for m, s in zip(mlframe_models or [], _strats) if not s.supports_polars]
        logger.info(
            "  Deferred pandas conversion -- Polars-native models run on the fastpath; " "non-native %s will convert lazily at their strategy branch.",
            non_native,
        )


def _log_conversion_reasons(
    *,
    was_polars_input: bool,
    pipeline_state_lives_in_pandas: bool,
    recurrent_models: list[str],
    has_rfecv: bool,
    rfecv_models: list[str],
    all_models_polars_native: bool,
    mlframe_models: list[str],
    strategies_for: Callable[[list[str]], list],
) -> None:
    """Log why the polars->pandas conversion is needed (verbose only)."""
    # These must mirror the ``defer_pandas_conv`` gate above TERM FOR TERM. They previously did
    # not: the gate gives up deferral when ``polars_pipeline_applied`` is False, but that term was
    # missing from the list, while ``all_models_polars_native`` -- which the gate never consults --
    # was in it. So the one genuinely common case (a caller passing
    # PreprocessingBackendConfig(prefer_polarsds=False), which leaves the polars-ds pipeline
    # unapplied) matched no listed reason and printed "unknown", leaving a multi-GB conversion with
    # no stated cause.
    reasons = []
    if not was_polars_input:
        reasons.append("input is not a Polars DataFrame")
    if pipeline_state_lives_in_pandas:
        reasons.append(
            "transform stages were requested (encoder / scaler / imputer) but the polars-ds pipeline did "
            "not apply them, so the fitted state exists only in pandas representation"
        )
    if recurrent_models:
        reasons.append(f"recurrent_models={recurrent_models}")
    if has_rfecv:
        reasons.append(f"rfecv_models={rfecv_models}")
    if not all_models_polars_native:
        _strats = strategies_for(mlframe_models or [])
        non_native = [m for m, s in zip(mlframe_models or [], _strats) if not s.supports_polars]
        if non_native:
            reasons.append(f"non-Polars-native models requested: {non_native}")
    logger.info(
        "  polars->pandas conversion needed because: %s",
        "; ".join(reasons) if len(reasons) > 0 else "unknown",
    )


def _cache_pandas_frame_sizes(
    train_df_pd: Any,
    val_df_pd: Any,
    train_df_size_bytes_cached: float | None,
    val_df_size_bytes_cached: float | None,
    verbose: bool,
) -> tuple[float | None, float | None]:
    """Fill in missing train/val byte sizes with a shallow ``memory_usage`` scan; keep sizes already estimated from polars."""
    # Sizing for downstream GPU / RAM heuristics. Two-tier strategy:
    # 1. If ``was_polars_input`` already populated ``train_df_size_bytes_cached`` via
    #    ``estimated_size()`` + cat-heavy inflation, KEEP that value.
    #    On a 4M-row x 25-col object-heavy frame ``memory_usage(deep=True)`` scans every cell
    #    and takes ~17.6s (); the polars estimate plus the 1.5x cat-heavy
    #    factor is accurate enough for GPU-sizing heuristics that already over-allocate.
    # 2. Otherwise fall back to ``memory_usage(deep=False)`` -- shallow scan returns in <1ms
    #    by reading buffer-block sizes per column. The object-dtype undercount that
    #    ``deep=True`` was guarding against (5GB vs 0.72GB on the test fixture) is acceptable
    #    given how much wall-time the deep scan costs; downstream consumers can request the
    #    exact value via a follow-up ``deep=True`` call if needed (none currently do).
    _t0_memsize = timer()
    try:
        if isinstance(train_df_pd, pd.DataFrame) and train_df_size_bytes_cached is None:
            train_df_size_bytes_cached = float(train_df_pd.memory_usage(deep=False, index=False).sum())
        if val_df_pd is not None and isinstance(val_df_pd, pd.DataFrame) and val_df_size_bytes_cached is None:
            val_df_size_bytes_cached = float(val_df_pd.memory_usage(deep=False, index=False).sum())
    except Exception as e:
        logger.debug("swallowed exception in _phase_helpers.py: %s", e)
        pass
    if verbose:
        _memsize_elapsed = timer() - _t0_memsize
        if _memsize_elapsed > 1.0:
            logger.info(
                "  trainset_features_stats memory_usage: %.1fs " "(train=%.1fMB, val=%.1fMB) -- runs once per suite",
                _memsize_elapsed,
                (train_df_size_bytes_cached or 0) / 1e6,
                (val_df_size_bytes_cached or 0) / 1e6,
            )
    return train_df_size_bytes_cached, val_df_size_bytes_cached

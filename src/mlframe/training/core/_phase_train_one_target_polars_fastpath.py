"""Polars-fastpath cat-prep + lazy-pandas-conversion branch, carved out of
``_train_one_target`` (in ``_phase_train_one_target_body``).

Runs once per (pre_pipeline, mlframe_model) iteration BEFORE the inner weight
loop. The polars branch warms ``prepared_frames_cache`` (a feature-side cache
that survives across targets); the pandas branch performs lazy conversion
via the ctx-scoped pandas-view cache.

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _prepare_strategy_inputs`` keeps
resolving transparently.
"""
from __future__ import annotations

import logging
import os

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from ..utils import filter_existing, get_pandas_view_of_polars_df
from ._misc_helpers import _build_tier_dfs, _prep_polars_df

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _pandas_view_cache_bytes(cache) -> int:
    """Sum of approximate byte usage across cached pandas views.

    Uses ``memory_usage(deep=False)`` for the dominant block-manager footprint without paying the
    deep-walk cost on every eviction probe; per CLAUDE.md the goal is OOM avoidance, so a slight
    under-estimate is preferable to a deep walk on a 100 GB frame at eviction time.
    """
    total = 0
    for _v in cache.values():
        try:
            total += int(_v.memory_usage(deep=False).sum())
        except Exception:
            pass
    return total


def resolve_pandas_view_cache_budget_bytes() -> float:
    """Byte budget for the polars->pandas view cache, configurable by TYPE x SIZE.

    Env:
      ``MLFRAME_PANDAS_VIEW_CACHE_TYPE`` = ``FREE_RAM_SHARE`` (default) | ``TOTAL_RAM_SHARE`` | ``ABSOLUTE_MB``
      ``MLFRAME_PANDAS_VIEW_CACHE_SIZE`` = ``0.2`` (default) -- a fraction in [0, 1] for the ``*_SHARE``
        types, or a megabyte count for ``ABSOLUTE_MB``.

    Default 0.2 of FREE RAM: on a 60 GB-free host that is a 12 GB budget -- enough to reuse a ~10 GB
    transformed-feature view across the per-target loop (so 10 composite targets pay ONE polars->pandas
    conversion, not ten); on a memory-tight host it scales DOWN automatically, preserving the OOM-safety
    property the old fixed 2 GB cap provided. Deprecated alias ``MLFRAME_PANDAS_VIEW_CACHE_MAX_MB`` is
    still honoured as ``ABSOLUTE_MB`` when the new vars are unset. Falls back to 2 GB absolute when
    psutil is unavailable or the env is malformed.
    """
    _DEFAULT_ABS_BYTES = 2048.0 * (1024**2)
    _legacy_mb = os.environ.get("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB")
    ctype = os.environ.get("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "").strip().upper()
    size_raw = os.environ.get("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "").strip()
    # Deprecated-alias path: old MAX_MB set and the new vars not -> treat as ABSOLUTE_MB.
    if _legacy_mb is not None and not ctype and not size_raw:
        try:
            return max(0.0, float(_legacy_mb)) * (1024**2) or _DEFAULT_ABS_BYTES
        except ValueError:
            return _DEFAULT_ABS_BYTES
    if not ctype:
        ctype = "FREE_RAM_SHARE"
    try:
        size = float(size_raw) if size_raw else (0.2 if ctype != "ABSOLUTE_MB" else 2048.0)
    except ValueError:
        size = 0.2 if ctype != "ABSOLUTE_MB" else 2048.0
    if size <= 0:
        return _DEFAULT_ABS_BYTES
    if ctype == "ABSOLUTE_MB":
        return size * (1024**2)
    try:
        import psutil
        vm = psutil.virtual_memory()
        base = vm.available if ctype == "FREE_RAM_SHARE" else vm.total
        budget = float(size) * float(base)
        return budget if budget > 0 else _DEFAULT_ABS_BYTES
    except Exception:
        return _DEFAULT_ABS_BYTES


def _prepare_strategy_inputs(
    *,
    polars_fastpath_active: bool,
    mlframe_model_name: str,
    strategy,
    cat_features,
    text_features,
    embedding_features,
    train_df_polars,
    val_df_polars,
    test_df_polars,
    prepared_frames_cache: dict,
    tier_dfs_cache: dict,
    tier_enum_map_cache: dict,
    common_params: dict,
    pre_pipeline_name,
    ctx,
    verbose: bool,
) -> dict:
    """Build the per-(pre_pipeline, model) feature-side cache or perform lazy
    pandas conversion, then return a dict of the bindings the parent expects
    to use further down the iteration.

    Polars-fastpath branch sets:
      ``polars_fastpath_active=True``, ``prepared_train`` / ``prepared_val``
      / ``prepared_test``, ``xgb_category_map``, ``cat_features``.

    Pandas branch sets:
      ``polars_fastpath_active=False``, ``tier_pandas`` dict (and mutates
      ``common_params`` in place so the caller's reference picks up the
      pandas views).

    Caller binds via dict-unpack so missing keys raise KeyError on misuse.
    """
    if polars_fastpath_active:
        if verbose:
            logger.info(
                "  Polars fastpath active for %s (strategy=%s)",
                mlframe_model_name, type(strategy).__name__,
            )
        # MUST use the post-promotion `cat_features` (post-auto-detect reassignment), NOT the stale
        # `cat_features_polars` snapshot from before auto-detect ran - the latter would still list
        # text-promoted columns and trip CB's polars-categorical fastpath on String dtypes.
        _cat_features = list(cat_features or [])

        # Cross-target reuse: cache key is (feature_tier, supports_polars=True, strategy_class,
        # cb_text_pass) where cb_text_pass tracks whether the CB-only Categorical->String text-
        # column cast must be applied (CB requires it; other CB-tier polars-native models don't).
        # All target-independent so the prepared frames carry from target 1 to target N.
        _prep_key = (
            strategy.feature_tier(),
            True,
            type(strategy).__name__,
            bool(text_features and mlframe_model_name == "cb"),
        )
        _cached_prep = prepared_frames_cache.get(_prep_key)
        if _cached_prep is not None:
            prepared_train = _cached_prep["prepared_train"]
            prepared_val = _cached_prep["prepared_val"]
            prepared_test = _cached_prep["prepared_test"]
            _xgb_category_map = _cached_prep["xgb_category_map"]
            if verbose:
                logger.info(
                    "  feature-side cache hit for %s (strategy=%s, pp=%s): reusing prepared polars frames across targets",
                    mlframe_model_name, type(strategy).__name__, pre_pipeline_name or "<ordinary>",
                )
        else:
            tier_base = {
                "train_df": train_df_polars,
                "val_df": val_df_polars,
                "test_df": test_df_polars,
            }
            tier_polars = _build_tier_dfs(
                tier_base, strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
            )

            # Enum map: leak-free, train+val union only; cached by (tier, strategy class).
            _enum_cache_key = (strategy.feature_tier(), type(strategy).__name__)
            if _enum_cache_key in tier_enum_map_cache:
                _xgb_category_map = tier_enum_map_cache[_enum_cache_key]
            elif hasattr(strategy, "build_polars_enum_map"):
                try:
                    _xgb_category_map = strategy.build_polars_enum_map(
                        tier_polars["train_df"],
                        tier_polars.get("val_df"),
                        _cat_features,
                    )
                except Exception as _emb_exc:
                    logger.warning(
                        "build_polars_enum_map failed for %s; " "falling back to per-DF Enum cast: %s",
                        type(strategy).__name__,
                        _emb_exc,
                    )
                    _xgb_category_map = None
                tier_enum_map_cache[_enum_cache_key] = _xgb_category_map
            else:
                _xgb_category_map = None
                tier_enum_map_cache[_enum_cache_key] = None

            prepared_train = _prep_polars_df(tier_polars["train_df"], strategy, _cat_features, _xgb_category_map)
            prepared_val = _prep_polars_df(tier_polars.get("val_df"), strategy, _cat_features, _xgb_category_map)
            prepared_test = _prep_polars_df(tier_polars.get("test_df"), strategy, _cat_features, _xgb_category_map)

            # CatBoost's polars text-features path requires plain String with no nulls; cast Categorical/Enum
            # text columns and fill_null. The dtype mismatch happens whenever auto-detect promotes a
            # column from cat_features to text_features without changing its backing dtype.
            if text_features and mlframe_model_name == "cb":
                text_cols_present = filter_existing(prepared_train, text_features)
                if text_cols_present:
                    # Determine which of the text columns need a dtype cast.
                    needs_cast = [c for c in text_cols_present if prepared_train.schema[c] == pl.Categorical or isinstance(prepared_train.schema[c], pl.Enum)]
                    prep_exprs = []
                    for c in text_cols_present:
                        expr = pl.col(c)
                        if c in needs_cast:
                            expr = expr.cast(pl.String)
                        prep_exprs.append(expr.fill_null(""))
                    prepared_train = prepared_train.with_columns(prep_exprs)
                    if prepared_val is not None:
                        prepared_val = prepared_val.with_columns(prep_exprs)
                    if prepared_test is not None:
                        prepared_test = prepared_test.with_columns(prep_exprs)
                    if needs_cast and verbose:
                        logger.info(
                            "  Cast %d text feature(s) from Polars Categorical to String " "for CatBoost: %s",
                            len(needs_cast),
                            needs_cast,
                        )

            # Null-in-Categorical fill is applied upstream once on train_df_polars/val/test (search:
            # `_polars_fill_null_in_categorical`, marker "__MISSING__"); no per-model fill needed.

            # Store REFERENCES only (no clones / no copies): a 100GB train_df_polars is shared
            # with ctx.train_df_polars; the prepared variant is a polars LazyFrame-evaluation
            # result that's already eager but immutable in our path. Carrying across targets
            # costs ~one pointer per slot - never duplicates feature data.
            prepared_frames_cache[_prep_key] = {
                "prepared_train": prepared_train,
                "prepared_val": prepared_val,
                "prepared_test": prepared_test,
                "xgb_category_map": _xgb_category_map,
            }
        return {
            "polars_fastpath_active": True,
            "prepared_train": prepared_train,
            "prepared_val": prepared_val,
            "prepared_test": prepared_test,
            "xgb_category_map": _xgb_category_map,
            "cat_features": _cat_features,
            "tier_pandas": None,
        }

    # Lazy pandas conversion for non-Polars-native strategies. The upfront
    # _convert_dfs_to_pandas is skipped when all blockers are non-native;
    # per-strategy conversion happens here, which preserves RAM when CB/XGB
    # can run natively on polars. Two trigger cases get distinct log messages:
    # (a) strategy genuinely non-Polars-native; (b) strategy IS native but
    # polars originals were released earlier in the run.
    # Cache the polars->pandas view by id() of the source frame on ctx so
    # two non-Polars-native strategies sharing the same source polars frame
    # pay one conversion total, not one per strategy.
    _logged_lazy_conv = False
    _view_cache = ctx._pandas_view_cache
    for df_key in ("train_df", "val_df", "test_df"):
        df_ = common_params.get(df_key)
        if isinstance(df_, pl.DataFrame):
            if not _logged_lazy_conv and verbose:
                if strategy.supports_polars:
                    _reason = "Polars originals released " "(common_params still carries " "polars frames; converting to " "pandas for inner predict path)"
                else:
                    _reason = f"non-Polars-native strategy " f"{type(strategy).__name__}"
                logger.info(
                    "  Lazy pandas conversion for %s -- %s",
                    mlframe_model_name, _reason,
                )
                _logged_lazy_conv = True
            _src_id = id(df_)
            _pd_view = _view_cache.get(_src_id)
            # Pandas-view cache stats: count one HIT per reuse (id() match) and one MISS
            # per fresh conversion. Stamped on ctx so finalize_suite can read without
            # touching the cache backend.
            _cs_pv = ctx._cache_stats.setdefault("pandas_view_cache", {"hits": 0, "misses": 0})
            if _pd_view is None:
                _cs_pv["misses"] += 1
                _pd_view = get_pandas_view_of_polars_df(df_)
                _view_cache[_src_id] = _pd_view
                # Bound the cache to 4 entries (train/val/test + one slack) AND a byte budget so a
                # single oversized pandas-view does not pin GB-scale blockmgr buffers across targets.
                # Per CLAUDE.md the byte gate is the safety property; the count cap is a fallback.
                # Budget is RAM-relative by default (MLFRAME_PANDAS_VIEW_CACHE_TYPE=FREE_RAM_SHARE,
                # SIZE=0.2 -> 20% of free RAM) so a ~10 GB view IS reused across the per-target loop on
                # a high-RAM host but scales down on a tight one; see resolve_pandas_view_cache_budget_bytes.
                _max_count = 4
                _max_bytes = resolve_pandas_view_cache_budget_bytes()
                while len(_view_cache) > _max_count or _pandas_view_cache_bytes(_view_cache) > _max_bytes:
                    if not _view_cache:
                        break
                    _view_cache.popitem(last=False)
            else:
                _cs_pv["hits"] += 1
                try:
                    _view_cache.move_to_end(_src_id)
                except (AttributeError, KeyError):
                    pass
            common_params[df_key] = _pd_view

    # Defense-in-depth: after lazy conversion, every common_params DF must be non-polars.
    # Surfacing here (rather than at trainer.fit time) makes the cross-iteration leakage cause
    # visible with full strategy/common_params context.
    for df_key in ("train_df", "val_df", "test_df"):
        df_ = common_params.get(df_key)
        if isinstance(df_, pl.DataFrame):
            raise RuntimeError(
                f"Lazy pandas conversion produced incomplete "
                f"state for non-Polars-native strategy "
                f"{type(strategy).__name__} ({mlframe_model_name}): "
                f"common_params[{df_key!r}] is still pl.DataFrame "
                f"(shape={df_.shape}, id={id(df_)}). The lazy-"
                f"conversion hook iterated over train/val/test but "
                f"this key escaped. Likely cause: a ``common_params`` "
                f"override between lazy-conversion and here, or "
                f"pipeline_cache cross-stream leakage (see core.py "
                f"kind-suffix in cache_key)."
            )

    tier_pandas = _build_tier_dfs(
        {
            "train_df": common_params.get("train_df"),
            "val_df": common_params.get("val_df"),
            "test_df": common_params.get("test_df"),
        },
        strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
    )
    return {
        "polars_fastpath_active": False,
        "prepared_train": None,
        "prepared_val": None,
        "prepared_test": None,
        "xgb_category_map": None,
        "cat_features": cat_features,
        "tier_pandas": tier_pandas,
    }

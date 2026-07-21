"""Suite-scoped feature-side and dataset-reuse cache helpers.

Carved out of ``_phase_train_one_target`` to keep the parent under the LOC
budget. Centralises:

- The ``_DATASET_REUSE_CACHE_ATTRS`` list of XGB/LightGBM cache attribute
  names forwarded across ``sklearn.clone()`` to enable
  ``set_label``/``set_weight`` reuse of the binned dataset.
- Bookkeeping helpers for ``ctx.artifacts``: feature-side cache, dataset
  reuse cache, polars-tier invalidation, FH cache token purge, capture/
  restore of the dataset cache across per-target rebuilds.
- ``_release_ctx_polars_frames`` which drops the strong refs on
  ctx.{train,val,test}_df_polars and verifies the RSS reclaim.
"""
from __future__ import annotations

import logging

from .._ram_helpers import estimate_df_size_mb, get_process_rss_mb, maybe_clean_ram_and_gpu

logger = logging.getLogger(__name__)


# XGB DMatrix / LGB Dataset reuse cache attribute names: forwarded across sklearn.clone() in both
# directions (template -> clone before fit; clone -> template after fit) so the weight-schema loop
# reuses the heavy binned dataset via set_label / set_weight instead of rebuilding.
#
# "_cached_val_datasets" (plural) is lgb_shim's multi-slot val-Dataset cache -- a dict keyed by
# val content signature, not a single pointer (fixed 2026-07-21: the prior single
# "_cached_val_dataset" slot only ever held the LAST eval-set entry across a multi-eval-set fit. The generic getattr/setattr
# transfer below works unchanged for a dict value; only the skip_none/falsy guard below needed
# updating so an empty (not just None) source doesn't null out a populated destination.
_DATASET_REUSE_CACHE_ATTRS = (
    "_cached_train_dmatrix",
    "_cached_train_key",
    "_cached_val_dmatrix",
    "_cached_val_key",
    "_cached_train_dataset",
    "_cached_val_datasets",
)


def _forward_dataset_reuse_cache(src, dst, attrs=_DATASET_REUSE_CACHE_ATTRS, *, skip_none: bool = False):
    """Copy each present attr from ``src`` onto ``dst``.

    Both the template -> clone forward and the clone -> template back transfer used to
    inline the same loop with slightly different ``if _val is not None`` guard. Centralised here so
    additions to ``_DATASET_REUSE_CACHE_ATTRS`` flow to both call sites automatically.

    ``skip_none=True`` matches the back-transfer's behaviour: only carry POPULATED caches up to the
    template (a falsy check, so this covers both ``None`` pointer-attrs and an EMPTY
    ``_cached_val_datasets`` dict uniformly), otherwise a clone that did not populate the cache
    would null out (or empty out) the template's prior value and defeat the reuse.
    """
    for _attr in attrs:
        if not hasattr(src, _attr):
            continue
        _val = getattr(src, _attr)
        if skip_none and not _val:
            continue
        try:
            setattr(dst, _attr, _val)
        except Exception as _attr_err:
            logger.debug("Could not transfer %s from %r to %r: %s", _attr, type(src).__name__, type(dst).__name__, _attr_err)


# Heuristic: if reclaim is under this share of the dropped-frame footprint, something is still pinning the buffers.
_POLARS_RELEASE_MIN_RECLAIM_FRACTION = 0.05


_FEATURE_SIDE_CACHE_KEY = "feature_side_cache"
_DATASET_REUSE_CACHE_KEY = "dataset_reuse_cache"


def _ensure_ctx_artifacts(ctx) -> dict:
    """Return ctx.artifacts as a dict, materialising it if the dataclass default left it as None.

    ``ctx.artifacts`` is declared ``dict = field(default_factory=dict)`` in _training_context.py
    so normal construction produces an empty dict, BUT older test fixtures and direct field
    assignments can land ``None`` on the slot. Calling .setdefault() then AttributeErrors before
    the helper has a chance to install its key.
    """
    artifacts: dict = ctx.artifacts
    if artifacts is None:
        artifacts = {}
        ctx.artifacts = artifacts
    return artifacts


def _ensure_feature_side_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped feature-side cache off ctx.artifacts."""
    result: dict = _ensure_ctx_artifacts(ctx).setdefault(_FEATURE_SIDE_CACHE_KEY, {})
    return result


def _ensure_dataset_reuse_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped dataset-reuse cache off ctx.artifacts.

    Keyed by ``(mlframe_model_name, pp_name)`` (DSET-REUSE-NO-PP-KEY): the prior bare-name key
    let two pre-pipelines (e.g. MRMR vs ordinary) on the same target + model collide on the same
    cache slot, replaying the prior PP's binned dataset onto the next PP's fresh template.
    ``capture`` / ``restore`` build the same tuple key so the round-trip stays consistent. Entries
    are dicts of ``_DATASET_REUSE_CACHE_ATTRS`` -> value captured from the prior target's fitted
    model template before _maybe_clear_shim_cache nuked it.
    """
    result: dict = _ensure_ctx_artifacts(ctx).setdefault(_DATASET_REUSE_CACHE_KEY, {})
    return result


def _dataset_reuse_cache_key(mlframe_model_name: str, pp_name: str | None) -> tuple:
    """Build the (model_name, pp_name) cache key for the dataset-reuse cache.

    Centralised so capture-side and restore-side never disagree on key shape.
    """
    return (mlframe_model_name, pp_name or "")


def _invalidate_polars_feature_side_cache(ctx) -> None:
    """Drop every polars-tier entry from ctx.artifacts['feature_side_cache'].

    Called from ``_release_ctx_polars_frames`` (the only place where ctx polars frames go to
    None) so the next target's loop doesn't read back stale pointers into freed frames. Pandas-
    tier entries (``supports_polars=False``) are preserved - they live in their own keys and
    point at frames that are NOT being released here.
    """
    cache = (ctx.artifacts or {}).get(_FEATURE_SIDE_CACHE_KEY)
    if not cache:
        return
    # Cache shape: cache[pp_name] -> {"tier_dfs": {sub_key -> dict}, "prepared_frames":
    # {sub_key -> dict}, "tier_enum_map": {sub_key -> map}}. Sub-keys are tuples and we
    # drop only the polars-tier ones; the "tier_enum_map" group is polars-only by
    # construction so it can be cleared whole.
    for _pp_name, _pp_payload in list(cache.items()):
        if not isinstance(_pp_payload, dict):
            continue
        for _group in ("tier_dfs", "prepared_frames"):
            _group_map = _pp_payload.get(_group)
            if not isinstance(_group_map, dict):
                continue
            # tier_dfs sub-key is (tier_tuple, kind) where kind is "pl" / "pd"; prepared_frames
            # sub-key is (tier_tuple, supports_polars, strategy_class, cb_text_pass). Polars
            # marker: kind=="pl" OR supports_polars==True (positional element 1).
            _polars_sub_keys = []
            for _sub_key in list(_group_map.keys()):
                if not isinstance(_sub_key, tuple) or len(_sub_key) < 2:
                    continue
                _kind = _sub_key[1]
                # Membership, not `is True` identity -- a numpy bool or any other truthy-but-not-`True`-identical
                # supports_polars value must still match so this polars-tier entry gets invalidated.
                if _kind in ("pl", True):
                    _polars_sub_keys.append(_sub_key)
            for _k in _polars_sub_keys:
                _group_map.pop(_k, None)
        # tier_enum_map is polars-only by construction (the per-target loop only writes to
        # it on polars_fastpath_active); a polars frame release means all entries are stale.
        _enum_map = _pp_payload.get("tier_enum_map")
        if isinstance(_enum_map, dict):
            _enum_map.clear()


def _purge_fh_cache_by_df_tokens(ctx, df_tokens) -> None:
    """Scrub FH ``FeatureCache._mem`` entries whose ``df_token`` matches a just-released frame id.

    The FH cache is keyed by ``InMemoryKey(session_id, df_token=id(train_df), ...)``. While the
    strong ref is alive, ``id()`` is stable; once we drop it in ``_release_ctx_polars_frames``,
    Python may recycle the same integer for a freshly allocated frame. The session_id is rotated
    per-suite by ``reset_session`` so cross-suite reuse is already safe; this scrub handles the
    mid-suite tier-transition case where one suite call releases polars frames and a subsequent
    target-loop iteration re-builds them.

    FeatureCache instances live wherever the suite stashed them. v1 stores under
    ``ctx.artifacts["feature_handling_fitted"]`` (the FeatureHandlingResult holds no cache ref);
    later phases may park the cache itself under ``ctx.artifacts["feature_handling_cache"]`` (single
    instance for the whole suite). Honour either shape; tolerate absence.
    """
    if not df_tokens:
        return
    artifacts = getattr(ctx, "artifacts", None)
    if not isinstance(artifacts, dict):
        return
    candidates = []
    _cache = artifacts.get("feature_handling_cache")
    if _cache is not None:
        candidates.append(_cache)
    _fitted = artifacts.get("feature_handling_fitted")
    if isinstance(_fitted, dict):
        for _v in _fitted.values():
            _c = getattr(_v, "cache", None)
            if _c is not None and _c not in candidates:
                candidates.append(_c)
    for _cache in candidates:
        _purge_fn = getattr(_cache, "purge_by_df_token", None)
        if not callable(_purge_fn):
            continue
        for _tok in df_tokens:
            try:
                _purge_fn(_tok)
            except Exception as _purge_err:  # pragma: no cover -- defensive  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                logger.debug("FH cache purge_by_df_token(%s) raised %r; continuing", _tok, _purge_err)


def _capture_dataset_reuse_cache(
    ctx,
    mlframe_model_name: str,
    model_template,
    pp_name: str | None = None,
) -> None:
    """Snapshot ``_DATASET_REUSE_CACHE_ATTRS`` off ``model_template`` into ctx.artifacts.

    Runs BEFORE ``_maybe_clear_shim_cache`` so the next target gets the live binned dataset
    (XGB DMatrix / LGB Dataset) rather than the post-clear None. Skips entries whose value is
    falsy (None, or an empty ``_cached_val_datasets`` dict) - those entries would defeat the next
    target's cache-hit check.
    """
    if model_template is None:
        return
    captured = {}
    for _attr in _DATASET_REUSE_CACHE_ATTRS:
        if not hasattr(model_template, _attr):
            continue
        _val = getattr(model_template, _attr)
        if not _val:
            continue
        captured[_attr] = _val
    if captured:
        _ensure_dataset_reuse_cache(ctx)[_dataset_reuse_cache_key(mlframe_model_name, pp_name)] = captured


def _restore_dataset_reuse_cache(
    ctx,
    mlframe_model_name: str,
    model_template,
    pp_name: str | None = None,
) -> None:
    """Re-attach ``_DATASET_REUSE_CACHE_ATTRS`` from ctx.artifacts onto ``model_template``.

    The per-target rebuild of ``models_params`` produces a fresh estimator without the cache
    attributes; this restore wires the previous target's binned dataset back on so the next
    forward-transfer-into-clone() carries it forward, and the shim's signature_of(X) check
    detects the same X (ctx-pinned across targets) and triggers the set_label / set_weight
    swap instead of a fresh build. No-op when there is no prior capture, or when target 1
    has not run yet for this model.
    """
    if model_template is None:
        return
    _key = _dataset_reuse_cache_key(mlframe_model_name, pp_name)
    captured = (ctx.artifacts or {}).get(_DATASET_REUSE_CACHE_KEY, {}).get(_key)
    if not captured:
        return
    for _attr, _val in captured.items():
        try:
            setattr(model_template, _attr, _val)
        except Exception as _attr_err:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            logger.debug(
                "Could not restore %s on %s template: %s",
                _attr, mlframe_model_name, _attr_err,
            )


def _release_ctx_polars_frames(
    ctx,
    baseline_rss_mb: float,
    df_size_mb: float,
    *,
    verbose: bool,
    reason: str,
) -> float:
    """Drop ctx.{train,val,test}_df_polars strong refs, then trigger maybe_clean_ram_and_gpu and verify reclaim.

    The naked ``del train_df_polars`` at each call site only released the local alias inside
    ``_train_one_target``; the ctx attributes (assigned from ctx.*_df_polars) kept the
    real strong reference alive, so ``maybe_clean_ram_and_gpu`` had nothing to reclaim and the log line
    claiming a release was misleading. Centralised here so both call sites stay in sync and the post-release
    sanity check (RSS drop vs estimated frame footprint) flags any future regression where a new strong
    ref to the same frames is introduced upstream without being scrubbed here.
    """
    expected_mb = 0.0
    released_df_tokens: list = []
    for _attr in ("train_df_polars", "val_df_polars", "test_df_polars"):
        _frame = getattr(ctx, _attr, None)
        if _frame is None:
            continue
        try:
            _sz = estimate_df_size_mb(_frame)
        except Exception:
            _sz = 0.0
        if _sz and _sz != float("inf"):
            expected_mb += float(_sz)
        # Capture id() BEFORE clearing the ctx slot -- once we drop the strong ref, Python is free
        # to recycle the integer for a freshly-allocated object and we'd no longer be able to scrub
        # the matching FH cache entries.
        released_df_tokens.append(id(_frame))
    rss_before_mb = get_process_rss_mb()
    ctx.train_df_polars = None
    ctx.val_df_polars = None
    ctx.test_df_polars = None
    # Drop polars-tier entries from the suite-scoped feature-side cache so they don't pin the
    # frames we just released. Pandas-tier entries are preserved - they point at separate
    # frames not touched by this release.
    _invalidate_polars_feature_side_cache(ctx)
    # ``ctx._pandas_view_cache`` keys by ``id(polars_df)``. The frames we just released may have
    # their ids recycled by a freshly allocated polars frame the next time the suite enters
    # tier_pandas conversion -- the cache would silently serve the prior pandas view. Pop every
    # entry keyed by a just-released id so the next conversion always misses cleanly.
    _view_cache = getattr(ctx, "_pandas_view_cache", None)
    if isinstance(_view_cache, dict):
        for _tok in released_df_tokens:
            _view_cache.pop(_tok, None)
    # Same hygiene for the recurrent numpy-coercion cache (POLARS-PANDAS-CHURN). Keys are
    # ``(split, id(frame))`` so we pop every (_, tok) pair where the second element matches a
    # just-released id.
    _rec_cache = getattr(ctx, "_recurrent_numpy_cache", None)
    if isinstance(_rec_cache, dict) and released_df_tokens:
        _released_set = set(released_df_tokens)
        for _k in [k for k in _rec_cache.keys() if isinstance(k, tuple) and len(k) == 2 and k[1] in _released_set]:
            _rec_cache.pop(_k, None)
    # Scrub any FH FeatureCache in-memory entries keyed by the released df ids. Without this, a
    # future tier transition that re-allocates a polars frame at the same memory address would
    # silently hit a cached entry whose state belonged to the dropped frame.
    _purge_fh_cache_by_df_tokens(ctx, released_df_tokens)
    # Drop XGB DMatrix / LightGBM Dataset / CatBoost Pool wrappers from the suite-scoped reuse cache.
    # These wrappers hold binned column tensors that mirror the underlying polars buffers; without
    # this scrub the polars frames "release" succeeds only in name (frames lose their strong ref but
    # the buffers stay alive behind the dataset cache). Observed in prod 2026-05-29: 9.4 GB expected
    # reclaim showed as 0.0 MB delta because every column pointer was still pinned via dataset_reuse_cache.
    # Also clear the per-target cached pre_pipeline transforms (heavy fitted preprocessing) that pin
    # column-major arrays of the released frames.
    artifacts = getattr(ctx, "artifacts", None)
    if isinstance(artifacts, dict):
        _reuse_cache = artifacts.get(_DATASET_REUSE_CACHE_KEY)
        if isinstance(_reuse_cache, dict) and _reuse_cache:
            _reuse_cache.clear()
    # The single-entry get_pandas_view_of_polars_df memo retains an Arrow-backed pandas view that shares (pins) the
    # released frame's buffers zero-copy; without dropping it the gigabytes stay resident and the reclaim shows ~0 MB.
    try:
        from mlframe.training.utils import clear_pandas_view_cache
        clear_pandas_view_cache()
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    new_baseline = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason=reason)
    # Only emit the lingering-refs warning when the expected reclaim is large enough for the
    # actual delta to be measurable above RSS-measurement noise. Windows / Linux RSS reporting
    # rounds to page-granularity (~4 KiB) and the resident-set is also affected by OS-managed
    # caching outside our control. For small frames (<10 MB expected reclaim) a delta of 0
    # is well within noise and the warning is just chatter that fires on every fuzz / unit test
    # suite call. Keep the warning loud for the real-production case (gigabyte-scale frames
    # where a missed release is a real leak).
    _POLARS_RELEASE_MIN_EXPECTED_MB = 10.0
    if expected_mb >= _POLARS_RELEASE_MIN_EXPECTED_MB:
        rss_after_mb = get_process_rss_mb()
        delta_mb = rss_before_mb - rss_after_mb
        if delta_mb < _POLARS_RELEASE_MIN_RECLAIM_FRACTION * expected_mb:
            logger.warning(
                "ctx polars frames released but RSS dropped only %.1f MB; expected at least %.1f MB - check for lingering refs",
                delta_mb,
                expected_mb,
            )
    return new_baseline

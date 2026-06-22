"""Suite-once unsupervised pre-screen, carved out of ``_train_one_target``
(in ``_phase_train_one_target_body``).

Drops variance==0 and nulls>99% columns from the TRAIN split only, then
reapplies the drops to val / test mirrors. Train-only fit by contract so
no held-out distribution leaks into the drop decision. Latched on ctx
(``_pre_screen_done`` / ``_pre_screen_dropped_cols``) so multi-target
suites pay the cost once.

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _maybe_run_unsupervised_pre_screen``
keeps resolving transparently.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _maybe_run_unsupervised_pre_screen(ctx, targets):
    """Run the suite-once unsupervised pre-screen if configured and not yet done.

    Mirrors the prior in-line block byte-for-byte: same protected-column
    union (every (target_type, target_name) pair across the suite +
    cat/text/embedding features + group/timestamp columns), same lazy
    import of ``mlframe.feature_selection.pre_screen``, same per-frame
    apply with loud per-frame failure logging, same outer except that
    latches ``_pre_screen_done=True`` so a transient failure doesn't
    re-fire the screen on every target.
    """
    _fs_cfg = ctx.feature_selection_config
    if not (
        _fs_cfg is not None
        and getattr(_fs_cfg, "pre_screen_unsupervised", False)
        and not ctx._pre_screen_done
    ):
        return
    try:
        # Canonical home is ``mlframe.feature_selection.pre_screen`` (not under ``.filters``).
        # The shorter path avoids triggering ``filters/__init__.py``'s ``from ._legacy import *``,
        # which cascades into ``_numba_utils`` and pays ~0.8s of @njit decorator init on
        # cold-start. Saves that wall on every suite call that doesn't also use MRMR.
        from mlframe.feature_selection.pre_screen import compute_unsupervised_drops, apply_drops
        _protected: set[str] = set()
        # Pre-screen runs ONCE per suite (gated by ctx._pre_screen_done). The protected set
        # must therefore cover EVERY target across EVERY (target_type, target_name) pair in
        # the suite, not just the first target's siblings. Pre-fix used the local ``targets``
        # arg which is target_type-scoped, so a multi-target-type suite (regression + binary)
        # would protect only the type that won the iteration order and drop its sibling's
        # target column if it satisfied the variance/null thresholds. Same hazard for
        # group/timestamp columns referenced via ctx.
        for _tt_targets in (ctx.target_by_type or {}).values():
            if isinstance(_tt_targets, dict):
                _protected.update(str(k) for k in _tt_targets.keys())
        if isinstance(targets, dict):
            _protected.update(str(k) for k in targets.keys())
        # Cat features were meant to be added here but the original code had
        # ``if ctx.cat_features: pass`` -- a dead no-op that silently dropped categorical
        # columns from the suite if they happened to be near-constant or near-all-null.
        # Same for text / embedding features and group / timestamp columns: those carry
        # semantic meaning the model relies on, so they must never be pre-screened out.
        if ctx.cat_features:
            _protected.update(str(c) for c in ctx.cat_features)
        if getattr(ctx, "text_features", None):
            _protected.update(str(c) for c in ctx.text_features)
        if getattr(ctx, "embedding_features", None):
            _protected.update(str(c) for c in ctx.embedding_features)
        for _attr in ("group_id_col", "ts_field"):
            _val = getattr(ctx, _attr, None)
            if isinstance(_val, str) and _val:
                _protected.add(_val)
        # Defensive double-source: also pull group/ts column names from upstream extractor + split_config so a group-aware split with a missing ctx attribute still protects the columns. Without this fallback, variance/null pre-screen can drop the group_id column itself (high-cardinality string IDs often look like "near-all-unique strings") and break GroupShuffleSplit downstream.
        _extractor = getattr(ctx, "extractor", None) or getattr(ctx, "features_and_targets_extractor", None)
        if _extractor is not None:
            for _attr in ("group_field", "timestamps_column", "ts_column", "timestamp_field"):
                _val = getattr(_extractor, _attr, None)
                if isinstance(_val, str) and _val:
                    _protected.add(_val)
        _split_cfg = getattr(ctx, "split_config", None)
        if _split_cfg is not None:
            for _attr in ("group_field", "timestamps_column", "ts_column"):
                _val = getattr(_split_cfg, _attr, None)
                if isinstance(_val, str) and _val:
                    _protected.add(_val)
        _split_cfg_use_groups = bool(getattr(_split_cfg, "use_groups", False)) if _split_cfg is not None else False
        if _split_cfg_use_groups and not _protected:
            # SKIP the pre-screen entirely: with a group-aware split and an empty protected set we cannot
            # know which column is the group/ts key, so variance/null pruning could silently drop it (group
            # IDs frequently look like near-all-unique strings) and break GroupShuffleSplit downstream. A
            # WARN-and-proceed left that hazard live; skipping is the safe default.
            logger.warning(
                "[pre-screen] split_config.use_groups=True but protected_columns set is empty; skipping the "
                "unsupervised pre-screen to avoid pruning the (unidentified) group/ts column. Set "
                "ctx.group_id_col / extractor.group_field / split_config.group_field to re-enable it.",
            )
            ctx._pre_screen_done = True
            ctx._pre_screen_dropped_cols = []
            return
        # Do NOT use ``ctx.train_df_polars or ctx.train_df_pd``: bool(pl.DataFrame) raises a TypeError
        # (ambiguous truthiness) which the outer except swallows, silently skipping the pre-screen on
        # polars-only inputs and latching ``_pre_screen_done``. Explicit ``is not None`` avoids that.
        _train_for_screen = ctx.filtered_train_df if ctx.filtered_train_df is not None else (
            ctx.train_df_polars if ctx.train_df_polars is not None else ctx.train_df_pd
        )
        _drops = compute_unsupervised_drops(
            _train_for_screen,
            variance_threshold=getattr(_fs_cfg, "pre_screen_variance_threshold", 0.0),
            null_fraction_threshold=getattr(_fs_cfg, "pre_screen_null_fraction_threshold", 0.99),
            protected_columns=_protected,
        )
        ctx._pre_screen_dropped_cols = list(_drops)
        ctx._pre_screen_done = True
        try:
            from mlframe.training.provenance import record_provenance as _record_provenance
            _record_provenance(
                getattr(ctx, "metadata", None),
                "pre_screen",
                source="train_only",
                n_rows=int(_train_for_screen.shape[0]) if hasattr(_train_for_screen, "shape") else None,
                extra={"n_dropped": len(_drops)},
            )
        except Exception:
            pass
        if _drops:
            # Atomic across all train/val/test mirrors: compute every dropped frame into a staging dict
            # FIRST, and only reassign ctx attributes once ALL succeed. A per-mirror failure used to leave
            # some frames with the dropped columns and others without -> schema drift that surfaces later as
            # an opaque "feature missing" training error. Now any single failure raises (caught by the outer
            # except, which then SKIPS the whole pre-screen) so no frame is partially dropped.
            _frame_attrs = (
                "filtered_train_df", "filtered_val_df",
                "train_df_pd", "val_df_pd", "test_df_pd",
                "train_df_polars", "val_df_polars", "test_df_polars",
            )
            _staged: dict[str, object] = {}
            for _frame_attr in _frame_attrs:
                _f = getattr(ctx, _frame_attr, None)
                if _f is not None:
                    _staged[_frame_attr] = apply_drops(_f, _drops)
            for _frame_attr, _new_f in _staged.items():
                setattr(ctx, _frame_attr, _new_f)
            if ctx.verbose:
                logger.info(
                    "[pre-screen] dropped %d column(s) suite-wide (variance=%s, null_fraction>%s): %s",
                    len(_drops),
                    getattr(_fs_cfg, "pre_screen_variance_threshold", 0.0),
                    getattr(_fs_cfg, "pre_screen_null_fraction_threshold", 0.99),
                    _drops[:20] + (["..."] if len(_drops) > 20 else []),
                )
    except Exception as _e:
        # Pre-screen is a perf optimization; never block training on its failure.
        ctx._pre_screen_done = True
        ctx._pre_screen_dropped_cols = []
        logger.warning("[pre-screen] skipped due to error: %s", _e)

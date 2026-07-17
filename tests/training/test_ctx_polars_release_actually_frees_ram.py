"""Regression: ctx.{train,val,test}_df_polars must be cleared on release so RSS can actually drop.

Pre-fix the helper only ``del``'d the function-local aliases inside ``_train_one_target``; the ctx
attributes assigned at function entry kept the same DataFrames alive, so ``maybe_clean_ram_and_gpu``
had nothing to reclaim while the log line claimed a release.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import polars as pl

from mlframe.training.core._phase_train_one_target import _release_ctx_polars_frames


def _make_ctx_with_frames():
    """Minimal ctx-like object carrying three non-trivial polars frames on the expected attrs.

    Each frame is constructed independently (no .clone() / .copy()) so the test does not
    paper over a hypothetical regression where the helper might rely on shared buffers.
    """

    def _build():
        """Builds a ~1M-row polars frame large enough to clear the RSS-release warning-audit floor."""
        # 1M rows gives ~16MB per frame, comfortably above the 10MB
        # ``_POLARS_RELEASE_MIN_EXPECTED_MB`` floor that gates the warning
        # path (the helper skips the audit-warning altogether for tiny frames
        # where RSS-measurement noise would dominate the delta).
        return pl.DataFrame(
            {
                "a": list(range(1_000_000)),
                "b": [float(x) for x in range(1_000_000)],
            }
        )

    # ``artifacts`` is required by the polars-cache invalidation hook that
    # ``_release_ctx_polars_frames`` calls (added in the multi-target hoist).
    return SimpleNamespace(
        train_df_polars=_build(),
        val_df_polars=_build(),
        test_df_polars=_build(),
        artifacts={},
    )


def test_release_clears_ctx_attributes_to_none():
    """(a) ctx.*_df_polars is non-None before release and None after."""
    ctx = _make_ctx_with_frames()
    assert ctx.train_df_polars is not None
    assert ctx.val_df_polars is not None
    assert ctx.test_df_polars is not None
    # No warning path is exercised here; RSS readings stay flat so the audit math degrades to "no expected drop"
    # only if expected_mb were 0 -- but the frames are real so expected_mb > 0; to avoid a stray warning during
    # this attribute-clear test we make the mocked RSS drop generously.
    with (
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.get_process_rss_mb",
            side_effect=[1_000.0, 0.0],  # huge drop -> well above 5% threshold
        ),
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.maybe_clean_ram_and_gpu",
            return_value=0.0,
        ),
    ):
        _release_ctx_polars_frames(ctx, baseline_rss_mb=1_000.0, df_size_mb=0.0, verbose=False, reason="test")
    assert ctx.train_df_polars is None
    assert ctx.val_df_polars is None
    assert ctx.test_df_polars is None


def test_release_no_warning_when_rss_drops_enough(caplog):
    """Scenario A: mocked RSS drops by >= expected footprint -> no warning."""
    ctx = _make_ctx_with_frames()
    # Capture the actual expected footprint the helper will compute (sum of estimated_size("mb") across 3 frames).
    expected_mb = sum(float(getattr(ctx, a).estimated_size("mb")) for a in ("train_df_polars", "val_df_polars", "test_df_polars"))
    # Pre = 100 MB above post -> 100 MB drop. Make expected tiny relative to this drop.
    pre_rss = expected_mb + 50.0
    post_rss = 0.0
    with (
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.get_process_rss_mb",
            side_effect=[pre_rss, post_rss],
        ),
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.maybe_clean_ram_and_gpu",
            return_value=0.0,
        ),
        caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target_dataset_cache"),
    ):
        _release_ctx_polars_frames(ctx, baseline_rss_mb=pre_rss, df_size_mb=0.0, verbose=False, reason="test-A")
    matching = [r for r in caplog.records if "RSS dropped only" in r.getMessage()]
    assert matching == [], f"unexpected warning emitted: {[r.getMessage() for r in matching]}"


def test_release_warns_when_rss_drop_below_five_percent(caplog):
    """Scenario B: mocked RSS drop < 5% of expected -> warning with the documented format string."""
    ctx = _make_ctx_with_frames()
    expected_mb = sum(float(getattr(ctx, a).estimated_size("mb")) for a in ("train_df_polars", "val_df_polars", "test_df_polars"))
    assert expected_mb > 0.0
    # Drop only 1% of expected -> well below the 5% threshold.
    tiny_drop = 0.01 * expected_mb
    pre_rss = 10_000.0
    post_rss = pre_rss - tiny_drop
    with (
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.get_process_rss_mb",
            side_effect=[pre_rss, post_rss],
        ),
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.maybe_clean_ram_and_gpu",
            return_value=0.0,
        ),
        caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target_dataset_cache"),
    ):
        _release_ctx_polars_frames(ctx, baseline_rss_mb=pre_rss, df_size_mb=0.0, verbose=False, reason="test-B")
    matching = [r for r in caplog.records if "RSS dropped only" in r.getMessage()]
    assert len(matching) == 1, f"expected exactly one warning, got: {[r.getMessage() for r in caplog.records]}"
    msg = matching[0].getMessage()
    assert "RSS dropped only" in msg
    assert "expected at least" in msg
    assert "check for lingering refs" in msg


def test_release_skips_audit_when_all_frames_are_none(caplog):
    """When all frames are already None at entry, expected_mb == 0 -> no warning even if RSS doesn't drop."""
    ctx = SimpleNamespace(train_df_polars=None, val_df_polars=None, test_df_polars=None, artifacts={})
    with (
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.get_process_rss_mb",
            side_effect=[5_000.0, 5_000.0],
        ),
        mock.patch(
            "mlframe.training.core._phase_train_one_target_dataset_cache.maybe_clean_ram_and_gpu",
            return_value=0.0,
        ),
        caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target_dataset_cache"),
    ):
        _release_ctx_polars_frames(ctx, baseline_rss_mb=5_000.0, df_size_mb=0.0, verbose=False, reason="test-empty")
    matching = [r for r in caplog.records if "RSS dropped only" in r.getMessage()]
    assert matching == []

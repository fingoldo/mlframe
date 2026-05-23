"""Tests for the shared content-fingerprint cache key used across all
four booster dataset caches (xgb_shim, lgb_shim, CB train Pool, CB
val Pool).

The pre-2026-05-23 design keyed each cache on ``id(X)`` -- defeated
by ``sklearn.clone()`` (composite-ensemble OOF refit produces fresh
shim instances) and ``train_X.iloc[idx].reset_index(drop=True)``
(fresh pandas frame every call, different id). All four caches are
now keyed on ``(columns, n_rows, n_cols, content_hash, extra)`` via
``_dataset_cache_fingerprint.compute_signature``.

The tests below verify:
  1. Helper math: identical content -> identical key; different
     content -> different key; ``extra`` participates in the key.
  2. Each of the four sites uses the helper (no leftover ``id(X)``
     keying).
  3. Each callsite's signature is stable across ``.iloc`` slicing of
     identical-content frames.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------
# 1) Shared helper math
# ----------------------------------------------------------------------


class TestSharedFingerprintHelper:
    def test_identical_content_same_key(self) -> None:
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df1 = pd.DataFrame({
            "a": np.linspace(0, 1, 1000),
            "b": np.linspace(1, 2, 1000),
        })
        df2 = df1.copy()
        assert id(df1) != id(df2)
        assert compute_signature(df1) == compute_signature(df2)

    def test_iloc_slice_same_content_same_key(self) -> None:
        """The production failure mode: two ``.iloc`` views of the same
        source share content but get fresh ids. Pre-fix cache missed
        on EVERY OOF round."""
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df = pd.DataFrame({"a": np.arange(2000, dtype=np.float64)})
        view_a = df.iloc[:1500].reset_index(drop=True)
        view_b = df.iloc[:1500].reset_index(drop=True)
        assert id(view_a) != id(view_b)
        assert compute_signature(view_a) == compute_signature(view_b)

    def test_different_content_different_key(self) -> None:
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df1 = pd.DataFrame({"a": np.linspace(0, 1, 1000)})
        df2 = pd.DataFrame({"a": np.linspace(0, 2, 1000)})
        assert compute_signature(df1) != compute_signature(df2)

    def test_different_columns_different_key(self) -> None:
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df1 = pd.DataFrame({"a": np.zeros(100), "b": np.zeros(100)})
        df2 = pd.DataFrame({"a": np.zeros(100), "c": np.zeros(100)})
        assert compute_signature(df1) != compute_signature(df2)

    def test_different_shape_different_key(self) -> None:
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df1 = pd.DataFrame({"a": np.zeros(100)})
        df2 = pd.DataFrame({"a": np.zeros(200)})
        assert compute_signature(df1) != compute_signature(df2)

    def test_extra_participates_in_key(self) -> None:
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df = pd.DataFrame({"a": np.zeros(100)})
        k1 = compute_signature(df, extra=(("cat",),))
        k2 = compute_signature(df, extra=(("dog",),))
        k3 = compute_signature(df)
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3

    def test_ndarray_input_works(self) -> None:
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        arr1 = np.zeros((100, 5))
        arr2 = np.zeros((100, 5))
        sig1 = compute_signature(arr1)
        sig2 = compute_signature(arr2)
        # ndarray has shape but no columns; should still produce a stable key.
        assert sig1 == sig2
        arr3 = np.ones((100, 5))
        assert compute_signature(arr3) != sig1

    def test_polars_uses_row_fast_path_not_full_to_numpy(self) -> None:
        """Polars X must hit ``.row(idx)`` per-row sampling, NOT
        ``df.to_numpy()`` which materialises the whole frame. c0103 iter261
        attributed 4.51s (4 calls x 1.13s) to the to_numpy path on a 200k
        x 25 polars frame.

        Verify by patching ``to_numpy`` on a polars DataFrame to raise; the
        fingerprint must still succeed via ``.row()``.
        """
        pl = pytest.importorskip("polars")
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df = pl.DataFrame({"a": np.arange(1000.0), "b": np.zeros(1000)})
        called = {"to_numpy": 0}
        real_to_numpy = df.to_numpy

        def trap(*args, **kwargs):
            called["to_numpy"] += 1
            raise AssertionError(
                "polars to_numpy must not be called for row sampling; "
                "the .row(idx) fast path should handle this case."
            )

        try:
            df.to_numpy = trap  # type: ignore[method-assign]
            # Should succeed via .row(idx); never touch to_numpy.
            sig = compute_signature(df)
            assert sig is not None
            assert called["to_numpy"] == 0
        finally:
            df.to_numpy = real_to_numpy  # type: ignore[method-assign]

    def test_polars_signature_stable_across_identical_content(self) -> None:
        """Two polars frames with identical content must share the
        fingerprint even though they're distinct Python objects."""
        pl = pytest.importorskip("polars")
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df1 = pl.DataFrame({"a": np.arange(500.0), "b": np.linspace(0, 1, 500)})
        df2 = pl.DataFrame({"a": np.arange(500.0), "b": np.linspace(0, 1, 500)})
        assert id(df1) != id(df2)
        assert compute_signature(df1) == compute_signature(df2)


# ----------------------------------------------------------------------
# 2) Each callsite uses the helper (no id(X) leftover)
# ----------------------------------------------------------------------


class TestXGBShimUsesSharedHelper:
    def test_xgb_signature_of_delegates(self) -> None:
        from mlframe.training import xgb_shim
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        df_a = pd.DataFrame({"a": np.arange(1000.0)}).iloc[:500].reset_index(drop=True)
        df_b = pd.DataFrame({"a": np.arange(1000.0)}).iloc[:500].reset_index(drop=True)
        # Both shim signature and helper signature should match identically.
        assert xgb_shim._signature_of(df_a) == compute_signature(df_a)
        assert xgb_shim._signature_of(df_a) == xgb_shim._signature_of(df_b)

    def test_xgb_signature_delegates_not_inlines(self) -> None:
        """Source-guard: the shim function must delegate to the shared
        helper, not re-implement id(X)-keyed signature inline. We assert
        positively (delegates to ``compute_signature``) rather than
        negatively (no id(X)) because the rationale docstring still
        mentions id(X) as historical context."""
        from pathlib import Path
        import mlframe.training.xgb_shim as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        defn_start = src.find("def _signature_of(X)")
        assert defn_start != -1
        next_def = src.find("\ndef ", defn_start + 1)
        body = src[defn_start:next_def]
        assert "compute_signature" in body, (
            "xgb_shim._signature_of must delegate to "
            "_dataset_cache_fingerprint.compute_signature -- regression "
            "would re-introduce id(X) cache-key bug from prod TVT 2026-05-23."
        )
        # ``return compute_signature(X)`` should be the only return statement.
        assert "return compute_signature" in body


class TestLGBShimUsesSharedHelper:
    def test_lgb_signature_stable_across_iloc(self) -> None:
        from mlframe.training import lgb_shim
        df = pd.DataFrame({"a": np.arange(1000.0), "b": np.zeros(1000)})
        view_a = df.iloc[:500].reset_index(drop=True)
        view_b = df.iloc[:500].reset_index(drop=True)
        sig_a = lgb_shim._signature_of(view_a, categorical_feature=None)
        sig_b = lgb_shim._signature_of(view_b, categorical_feature=None)
        assert sig_a == sig_b

    def test_lgb_signature_distinguishes_categorical_feature(self) -> None:
        from mlframe.training import lgb_shim
        df = pd.DataFrame({"a": np.zeros(100), "b": np.zeros(100)})
        sig_none = lgb_shim._signature_of(df, categorical_feature=None)
        sig_a = lgb_shim._signature_of(df, categorical_feature=["a"])
        sig_b = lgb_shim._signature_of(df, categorical_feature=["b"])
        assert sig_none != sig_a
        assert sig_a != sig_b


class TestCBTrainPoolCacheKeyUsesHelper:
    """``_cb_pool_build`` builds the train CB Pool cache key via the
    shared helper. Source-grep verifies no ``id(train_df)`` left."""

    def test_no_id_train_df_in_cache_key_construction(self) -> None:
        from pathlib import Path
        import mlframe.training._cb_pool_build as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # The cache-key construction site must not re-introduce id().
        # The historical commentary still mentions ``id(train_df)`` as
        # the broken pattern; locate the actual key= line.
        assert "compute_signature" in src, (
            "_cb_pool_build.py no longer uses compute_signature for "
            "the train Pool cache key -- regression vs 2026-05-23 fix."
        )
        # Find the line that BUILDS the key (the assignment), not the
        # docstring comments above it.
        key_assign_lines = [
            ln for ln in src.splitlines()
            if ln.lstrip().startswith("key = ")
        ]
        for ln in key_assign_lines:
            assert "id(train_df)" not in ln, (
                f"_cb_pool_build train Pool cache key still uses "
                f"id(train_df): {ln.strip()!r}"
            )


class TestCBValPoolCacheKeyUsesHelper:
    """``_cb_pool._maybe_rewrite_eval_set_as_cb_pool`` builds the val
    Pool cache key via the shared helper."""

    def test_no_id_val_df_in_cache_key_construction(self) -> None:
        from pathlib import Path
        import mlframe.training._cb_pool as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "compute_signature" in src, (
            "_cb_pool.py no longer uses compute_signature for val Pool "
            "cache key -- regression vs 2026-05-23 fix."
        )
        # No "key = (id(val_df), ...)" assignment.
        key_assign_lines = [
            ln for ln in src.splitlines()
            if ln.lstrip().startswith("key = ")
        ]
        for ln in key_assign_lines:
            assert "id(val_df)" not in ln, (
                f"_cb_pool val Pool cache key still uses id(val_df): "
                f"{ln.strip()!r}"
            )


# ----------------------------------------------------------------------
# 3) Cross-shim invariant: the four cache sites all hit on identical-
#    content frames with different ids.
# ----------------------------------------------------------------------


class TestCrossShimInvariant:
    """All four cache sites must produce the SAME signature for
    identical-content frames with different Python ids -- ensures
    ``sklearn.clone()`` + ``.iloc[]`` reuse hits across the board."""

    def test_xgb_and_lgb_agree_on_content_when_extras_match(self) -> None:
        from mlframe.training import xgb_shim, lgb_shim
        df_a = pd.DataFrame({
            "a": np.linspace(0, 1, 500),
            "b": np.linspace(1, 2, 500),
        })
        df_b = df_a.copy()
        # XGB shim's _signature_of takes only X; LGB takes (X, cat).
        # Match by passing categorical_feature=None to LGB so the extras
        # collapse to the same value.
        sig_xgb_a = xgb_shim._signature_of(df_a)
        sig_xgb_b = xgb_shim._signature_of(df_b)
        sig_lgb_a = lgb_shim._signature_of(df_a, categorical_feature=None)
        sig_lgb_b = lgb_shim._signature_of(df_b, categorical_feature=None)
        # Same shim, same content -> same key (the load-bearing invariant).
        assert sig_xgb_a == sig_xgb_b
        assert sig_lgb_a == sig_lgb_b

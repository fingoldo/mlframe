"""Round 8 sensors for the four deferred findings from round 7 probe.

1. **Atomic joblib dump** (`training/io.py::atomic_write_bytes` used in
   `core.py::_finalize_and_save_metadata` + `io.py::save_mlframe_model`):
   pre-fix, ``with open(path, "wb") as f: joblib.dump(obj, f)``
   race-corrupted when two train runs wrote to the same metadata path.
   Now: write to ``<target>.xyz.tmp`` in the same directory, then
   ``os.replace()`` atomically.

2. **Polars bridge nested types** (`training/utils.py::get_pandas_view_of_polars_df`):
   columns with ``pl.List[pl.Float32]`` (embedding_features) silently
   became ``object`` dtype with Python list elements, breaking
   downstream CatBoost embedding fastpath with opaque "expected
   numeric". Now: WARN naming the columns so the operator traces back
   here instead of CatBoost internals.

3. **NaN per-fold importances** (`feature_selection/wrappers.py::get_feature_importances`):
   a degenerate CV fold (single-class target, zero-variance features)
   makes the model's ``feature_importances_`` contain NaN. Before:
   silently folded into the per-feature aggregate ranking. Now: WARN
   with the count, model type, and likely cause.

4. **Schema drift in fit_and_transform_pipeline** (`training/pipeline.py::_warn_on_schema_drift`):
   val/test with missing columns, extra columns, or dtype mismatches
   hit the pipeline and raised opaquely. Now: WARN with the diff before
   transform, so the operator knows exactly what's different.
"""
from __future__ import annotations

import logging
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest


# =============================================================================
# Fix #1 — atomic_write_bytes
# =============================================================================


class TestAtomicWriteBytes:

    def test_writes_target_atomically(self, tmp_path):
        from mlframe.training.io import atomic_write_bytes
        target = str(tmp_path / "out.bin")
        atomic_write_bytes(target, lambda f: f.write(b"hello world"))
        assert os.path.exists(target)
        assert open(target, "rb").read() == b"hello world"

    def test_overwrites_existing_file(self, tmp_path):
        """The key property: concurrent writers should see ALL-of-A or
        ALL-of-B, never a mix. The overwrite path must succeed on
        both POSIX and Windows — os.replace(), not os.rename()."""
        from mlframe.training.io import atomic_write_bytes
        target = str(tmp_path / "existing.bin")
        open(target, "wb").write(b"OLD CONTENT" * 100)
        atomic_write_bytes(target, lambda f: f.write(b"NEW"))
        assert open(target, "rb").read() == b"NEW"

    def test_failed_write_leaves_no_tmp_file(self, tmp_path):
        """If the writer_fn raises mid-write, the tmp file must be
        cleaned up — pre-fix no-op (``with open``) would leak the
        partial file under its original name, which is worse. Now:
        tmp file deleted, target path untouched if it existed."""
        from mlframe.training.io import atomic_write_bytes
        target = str(tmp_path / "wont_exist.bin")

        def _bad_writer(f):
            f.write(b"partial data")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            atomic_write_bytes(target, _bad_writer)

        # Target path was never created.
        assert not os.path.exists(target)
        # No tmp files leaked alongside.
        leaked = [p for p in os.listdir(tmp_path) if p.endswith(".tmp") or ".tmp" in p]
        assert not leaked, f"tmp files leaked: {leaked}"

    def test_joblib_dump_through_atomic_write(self, tmp_path):
        """End-to-end: actual joblib.dump path used in core.py.
        Metadata saved and loaded must round-trip exactly."""
        from mlframe.training.io import atomic_write_bytes
        target = str(tmp_path / "metadata.joblib")
        metadata = {
            "columns": ["a", "b", "c"],
            "cat_features": ["a"],
            "text_features": [],
            "embedding_features": [],
            "nested": {"k": [1, 2, 3]},
        }
        atomic_write_bytes(target, lambda f: joblib.dump(metadata, f))

        loaded = joblib.load(target)
        assert loaded == metadata

    def test_partial_existing_target_preserved_on_write_fail(self, tmp_path):
        """Specific atomicity guarantee: if write fails, the pre-existing
        target file must remain intact (NOT be truncated like
        ``open(path, 'wb')`` would)."""
        from mlframe.training.io import atomic_write_bytes
        target = str(tmp_path / "preserved.bin")
        open(target, "wb").write(b"IMPORTANT EXISTING DATA")

        def _bad_writer(f):
            raise RuntimeError("fail before write")

        with pytest.raises(RuntimeError):
            atomic_write_bytes(target, _bad_writer)

        # Original file untouched.
        assert open(target, "rb").read() == b"IMPORTANT EXISTING DATA"


# =============================================================================
# Fix #2 — polars bridge nested-types WARN
# =============================================================================


class TestPolarsBridgeNestedTypesWarn:

    def test_list_float_column_triggers_warning(self, caplog):
        """pl.List[pl.Float32] embedding column must produce a WARN
        naming it — downstream CatBoost fastpath rejects object dtype
        with list elements."""
        from mlframe.training.utils import get_pandas_view_of_polars_df
        df = pl.DataFrame({
            "num": np.arange(5, dtype=np.float32),
            "emb": [[1.0, 2.0, 3.0]] * 5,
        }).with_columns(pl.col("emb").cast(pl.List(pl.Float32)))
        with caplog.at_level(logging.WARNING, logger="mlframe.training.utils"):
            get_pandas_view_of_polars_df(df)
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("emb" in m and "List" in m for m in warns), warns

    def test_no_warning_on_flat_schema(self, caplog):
        """Clean numeric+string+categorical must be silent — the warn
        runs on every bridge call, false positives would drown logs."""
        from mlframe.training.utils import get_pandas_view_of_polars_df
        df = pl.DataFrame({
            "x": np.arange(10, dtype=np.float32),
            "y": ["a", "b"] * 5,
            "c": pl.Series("c", ["p", "q"] * 5).cast(pl.Categorical),
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.utils"):
            get_pandas_view_of_polars_df(df)
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns, f"Unexpected WARN on clean schema: {[r.message for r in warns]}"


# =============================================================================
# Fix #3 — get_feature_importances NaN WARN
# =============================================================================


class TestGetFeatureImportancesNaNWarn:

    def test_nan_importance_emits_warning(self, caplog):
        from mlframe.feature_selection.wrappers import get_feature_importances

        class _DegenModel:
            """Stand-in for a model that fitted on a degenerate CV fold
            and produced NaN importances (CatBoost / LightGBM both do
            this on single-class targets)."""
            feature_importances_ = np.array([0.3, float("nan"), 0.1, float("nan")])

        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers"):
            get_feature_importances(
                model=_DegenModel(),
                current_features=[0, 1, 2, 3],
                importance_getter="feature_importances_",
            )
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("NaN" in m and "2 / 4" in m for m in warns), warns

    def test_all_finite_no_warning(self, caplog):
        from mlframe.feature_selection.wrappers import get_feature_importances

        class _CleanModel:
            feature_importances_ = np.array([0.3, 0.2, 0.5])

        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers"):
            get_feature_importances(
                model=_CleanModel(),
                current_features=[0, 1, 2],
                importance_getter="feature_importances_",
            )
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns


# =============================================================================
# Fix #4 — pipeline schema drift WARN
# =============================================================================


class TestPipelineSchemaDriftWarn:

    def _train_schema(self):
        return {
            "a": pl.Int32,
            "b": pl.Float64,
            "c": pl.Utf8,
        }

    def test_missing_column_warns(self, caplog):
        from mlframe.training.pipeline import _warn_on_schema_drift
        other = pl.DataFrame({"a": [1], "b": [1.0]})  # 'c' missing
        with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
            _warn_on_schema_drift(self._train_schema(), other, "val")
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("missing" in m and "'c'" in m for m in msgs), msgs

    def test_extra_column_warns(self, caplog):
        from mlframe.training.pipeline import _warn_on_schema_drift
        other = pl.DataFrame({
            "a": [1], "b": [1.0], "c": ["x"],
            "surprise": [0.1],  # extra
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
            _warn_on_schema_drift(self._train_schema(), other, "test")
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("extra" in m and "surprise" in m for m in msgs), msgs

    def test_dtype_mismatch_warns(self, caplog):
        from mlframe.training.pipeline import _warn_on_schema_drift
        # 'a' was Int32 at fit, now Int64
        other = pl.DataFrame({
            "a": pl.Series("a", [1], dtype=pl.Int64),
            "b": [1.0],
            "c": ["x"],
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
            _warn_on_schema_drift(self._train_schema(), other, "val")
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("dtype" in m and "'a'" in m for m in msgs), msgs

    def test_identical_schema_silent(self, caplog):
        """False-positive sensor: clean val must not warn — schema
        validation runs on every fit+transform cycle; false positives
        would spam logs."""
        from mlframe.training.pipeline import _warn_on_schema_drift
        other = pl.DataFrame({
            "a": pl.Series("a", [1], dtype=pl.Int32),
            "b": pl.Series("b", [1.0], dtype=pl.Float64),
            "c": pl.Series("c", ["x"], dtype=pl.Utf8),
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
            _warn_on_schema_drift(self._train_schema(), other, "val")
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns, f"Identical schema must not warn: {[r.message for r in warns]}"

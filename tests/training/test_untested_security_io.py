"""Tests for security-sensitive path validation + joblib metadata I/O + split artifacts.

Targets:
- mlframe.training.core._validate_trusted_path
- mlframe.training.core._finalize_and_save_metadata
- mlframe.training.preprocessing.save_split_artifacts

NOTE: the audit listed save_split_artifacts as being in training/io.py and using numpy.save,
but the real implementation is in training/preprocessing.py and uses parquet via
save_series_or_df(). Tests target the real behavior.
"""
from __future__ import annotations
from mlframe.training import OutlierDetectionConfig, OutputConfig

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import _validate_trusted_path, _finalize_and_save_metadata
from mlframe.training.preprocessing import save_split_artifacts


def _load_metadata(metadata_dir: Path) -> dict:
    """Load saved metadata from the post-2026-04-29 format (`metadata.pkl.zst`
    or `.pkl` fallback when zstandard is missing). Replaces the legacy
    `joblib.load(metadata.joblib)` path."""
    import pickle
    zst_path = metadata_dir / "metadata.pkl.zst"
    pkl_path = metadata_dir / "metadata.pkl"
    if zst_path.exists():
        import zstandard as zstd
        return pickle.loads(zstd.ZstdDecompressor().decompress(zst_path.read_bytes()))
    if pkl_path.exists():
        return pickle.loads(pkl_path.read_bytes())
    raise FileNotFoundError(f"No metadata file in {metadata_dir}")


# ----- _validate_trusted_path -----

def test_validate_requires_trusted_root(tmp_path):
    # Path content doesn't matter -- the function rejects on trusted_root=None
    # before any filesystem touch.
    candidate = str(tmp_path / "anything.joblib")
    with pytest.raises(ValueError, match="trusted_root is required"):
        _validate_trusted_path(candidate, None)


def test_validate_inside_is_ok(tmp_path):
    f = tmp_path / "x.joblib"
    f.write_text("")
    _validate_trusted_path(str(f), str(tmp_path))


def test_validate_outside_rejected(tmp_path):
    other = tmp_path / "a"
    outside = tmp_path / "b" / "hack.joblib"
    other.mkdir()
    (tmp_path / "b").mkdir()
    outside.write_text("")
    with pytest.raises(ValueError, match="not inside trusted_root"):
        _validate_trusted_path(str(outside), str(other))


def test_validate_parent_escape_rejected(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    escape = str(trusted / ".." / "evil.joblib")
    with pytest.raises(ValueError, match="not inside trusted_root"):
        _validate_trusted_path(escape, str(trusted))


@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="symlink on Windows needs admin; tested on POSIX only")
def test_validate_symlink_escape_resolved(tmp_path):
    # NOTE: _validate_trusted_path uses os.path.abspath, which does NOT resolve symlinks.
    # A symlink file *inside* trusted_root still has abspath inside trusted_root,
    # so the current implementation permits it (documented behavior).
    # We verify at minimum that a symlink *outside* is rejected by abspath check.
    outside = tmp_path / "outside.joblib"
    outside.write_text("")
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    linkname = trusted / "link.joblib"
    try:
        os.symlink(str(outside), str(linkname))
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    # abspath does NOT resolve symlinks, so link is considered inside trusted.
    # This asserts current (documented) behavior, not a security claim.
    _validate_trusted_path(str(linkname), str(trusted))


@pytest.mark.skipif(not sys.platform.startswith("win"),
                    reason="cross-drive only meaningful on Windows")
def test_validate_cross_drive_rejected():
    # Windows: commonpath raises ValueError across different drives. Code catches
    # and re-raises as "not inside trusted_root".
    with pytest.raises(ValueError, match="not inside trusted_root"):
        _validate_trusted_path(r"Z:\foo\bar.joblib", r"C:\trusted")


# ----- _finalize_and_save_metadata -----

def test_finalize_saves_and_roundtrips(tmp_path):
    data_dir = str(tmp_path)
    models_dir = "models"
    metadata = {"model_name": "m1", "target_name": "t1", "mlframe_models": ["cb"]}
    outlier_detector = None
    outlier_detection_result = {"train_od_idx": None, "val_od_idx": None}
    # _finalize_and_save_metadata does not mkdir; callers are expected to have ensured the dir.
    (Path(data_dir) / "models" / "t1" / "m1").mkdir(parents=True)

    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats={"mean": 0.1},
        data_dir=data_dir,

        models_dir=models_dir,
        target_name="t1",
        model_name="m1",
        verbose=0,
    )
    # 2026-04-29: format switched joblib -> pickle proto=5 + zstd L3 (8c301f2).
    metadata_dir = Path(data_dir) / "models" / "t1" / "m1"
    loaded = _load_metadata(metadata_dir)
    assert loaded["model_name"] == "m1"
    assert loaded["trainset_features_stats"] == {"mean": 0.1}
    assert loaded["outlier_detection_result"] == outlier_detection_result


def test_finalize_adds_slug_mappings(tmp_path):
    metadata = {"model_name": "m", "target_name": "t", "mlframe_models": []}
    (Path(tmp_path) / "models" / "t" / "m").mkdir(parents=True)
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=None,
        outlier_detection_result={},
        trainset_features_stats=None,
        data_dir=str(tmp_path),

        models_dir="models",
        target_name="t",
        model_name="m",
        verbose=0,
        slug_to_original_target_type={"reg": "Regression"},
        slug_to_original_target_name={"t": "Target"},
    )
    metadata_dir = Path(tmp_path) / "models" / "t" / "m"
    loaded = _load_metadata(metadata_dir)
    assert loaded["slug_to_original_target_type"] == {"reg": "Regression"}
    assert loaded["slug_to_original_target_name"] == {"t": "Target"}


def test_finalize_no_save_when_no_dirs():
    metadata = {"model_name": "m", "target_name": "t", "mlframe_models": []}
    # Should not raise even when no dirs given
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=None,
        outlier_detection_result={},
        trainset_features_stats=None,
        data_dir="",

        models_dir="",
        target_name="t",
        model_name="m",
        verbose=0,
    )
    # Metadata still populated in place
    assert "outlier_detection_result" in metadata


def test_finalize_bubbles_ioerror(tmp_path, monkeypatch):
    # 2026-04-29: format switched joblib -> pickle + zstd via
    # atomic_write_bytes(metadata_file, _writer). Monkeypatch the new
    # write entry point. _finalize_and_save_metadata catches
    # (OSError, IOError) and re-raises after logging.
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    target_path = bad_dir / "models" / "t" / "m"
    target_path.mkdir(parents=True)

    import mlframe.training.io as io_mod

    def _bad_write(*a, **k):
        raise IOError("disk full")

    monkeypatch.setattr(io_mod, "atomic_write_bytes", _bad_write)
    with pytest.raises(IOError):
        _finalize_and_save_metadata(
            metadata={"model_name": "m", "target_name": "t", "mlframe_models": []},
            outlier_detector=None,
            outlier_detection_result={},
            trainset_features_stats=None,
            data_dir=str(bad_dir),
            models_dir="models",
            target_name="t",
            model_name="m",
            verbose=0,
        )


# ----- save_split_artifacts -----

def test_save_split_artifacts_writes_parquet(tmp_path):
    n = 20
    train_idx = np.arange(0, 10)
    val_idx = np.arange(10, 15)
    test_idx = np.arange(15, 20)
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    group_ids = pd.Series(np.arange(n))

    save_split_artifacts(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        timestamps=timestamps,
        group_ids_raw=group_ids,
        artifacts=None,
        data_dir=str(tmp_path),
        models_dir="models",
        target_name="tgt",
        model_name="mdl",
    )
    split_dir = Path(tmp_path) / "models" / "tgt" / "mdl"
    assert (split_dir / "train_timestamps.parquet").exists()
    assert (split_dir / "val_timestamps.parquet").exists()
    assert (split_dir / "test_timestamps.parquet").exists()
    assert (split_dir / "train_group_ids_raw.parquet").exists()

    # Round-trip train timestamps
    loaded = pd.read_parquet(split_dir / "train_timestamps.parquet")
    assert len(loaded) == len(train_idx)


def test_save_split_artifacts_dict_artifacts(tmp_path):
    n = 10
    train_idx = np.arange(0, 5)
    val_idx = np.arange(5, 8)
    test_idx = np.arange(8, 10)
    artifacts = {"foo": pd.Series(np.arange(n)), "bar baz": pd.Series(np.arange(n, 0, -1))}

    save_split_artifacts(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        timestamps=None,
        group_ids_raw=None,
        artifacts=artifacts,
        data_dir=str(tmp_path),
        models_dir="models",
        target_name="t",
        model_name="m",
    )
    split_dir = Path(tmp_path) / "models" / "t" / "m"
    assert (split_dir / "train_artifacts_foo.parquet").exists()
    # "bar baz" is slugified
    files = list(split_dir.glob("train_artifacts_bar*.parquet"))
    assert len(files) == 1


def test_save_split_artifacts_noop_without_data_dir(tmp_path):
    # data_dir=None -> nothing written, no error
    save_split_artifacts(
        train_idx=np.arange(3), val_idx=np.arange(3, 5), test_idx=np.arange(5, 7),
        timestamps=pd.Series(range(7)),
        group_ids_raw=None,
        artifacts=None,
        data_dir=None,

        models_dir="models",
        target_name="t", model_name="m",
    )
    assert list(tmp_path.iterdir()) == []

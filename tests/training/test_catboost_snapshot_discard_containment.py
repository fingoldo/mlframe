"""Discarding a stale CatBoost snapshot never deletes a file outside train_dir or through a symlink (FSW5-2)."""

import logging
import os

import pytest

from mlframe.training._training_loop_refit import _discard_stale_catboost_snapshot


def test_a_snapshot_inside_train_dir_is_deleted(tmp_path):
    """The normal case still removes the stale snapshot."""
    snap = tmp_path / "td" / "snap.cbsnapshot"
    snap.parent.mkdir()
    snap.write_text("x")
    _discard_stale_catboost_snapshot({"save_snapshot": True, "snapshot_file": "snap.cbsnapshot", "train_dir": str(tmp_path / "td")})
    assert not snap.exists()


def test_a_relative_name_that_climbs_out_of_train_dir_is_not_deleted(tmp_path, caplog):
    """'../victim' next to train_dir is left in place and a warning says why."""
    (tmp_path / "td").mkdir()
    victim = tmp_path / "victim.parquet"
    victim.write_text("keep")
    with caplog.at_level(logging.WARNING):
        _discard_stale_catboost_snapshot({"save_snapshot": True, "snapshot_file": os.path.join("..", "victim.parquet"), "train_dir": str(tmp_path / "td")})
    assert victim.read_text() == "keep"
    assert any("outside train_dir" in r.getMessage() for r in caplog.records)


def test_a_symlink_is_not_followed_for_deletion(tmp_path):
    """A snapshot path that is a symlink is not removed, and its target is untouched."""
    target = tmp_path / "target.txt"
    target.write_text("keep")
    td = tmp_path / "td"
    td.mkdir()
    link = td / "snap.cbsnapshot"
    try:  # audit-exempt: except_skip_masks_call_under_test -- only the symlink creation is guarded, the skip names a host without symlink rights
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    _discard_stale_catboost_snapshot({"save_snapshot": True, "snapshot_file": "snap.cbsnapshot", "train_dir": str(td)})
    assert target.read_text() == "keep" and link.is_symlink()

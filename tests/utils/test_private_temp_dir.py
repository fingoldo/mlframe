"""``private_temp_dir`` hands back a per-user private directory and refuses one a co-tenant could have pre-created (FS-03, FS-04)."""

import os
import stat
import tempfile
from types import SimpleNamespace

from mlframe.utils import private_temp as pt


def _st(mode, uid):
    """A stat-like record with the given mode bits and owner."""
    return SimpleNamespace(st_mode=mode, st_uid=uid)


def test_a_directory_owned_by_the_caller_with_no_group_or_other_bits_is_private():
    """0o700 and our uid pass; a wider mode, another owner and a non-directory do not."""
    d = stat.S_IFDIR
    assert pt._is_private(_st(d | 0o700, 1000), 1000)
    assert not pt._is_private(_st(d | 0o755, 1000), 1000)
    assert not pt._is_private(_st(d | 0o700, 1001), 1000)
    assert not pt._is_private(_st(stat.S_IFLNK | 0o777, 1000), 1000)
    assert pt._is_private(_st(d | 0o777, 0), None)


def test_the_directory_is_named_per_user_and_reused(monkeypatch, tmp_path):
    """Two calls return the same existing directory, and its name carries the user tag."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    first = pt.private_temp_dir("mlframe_caches")
    assert os.path.isdir(first) and os.path.basename(first) == f"mlframe_caches-{pt._user_tag()}"
    assert pt.private_temp_dir("mlframe_caches") == first


def test_an_untrusted_pre_existing_directory_is_replaced_by_a_fresh_one(monkeypatch, tmp_path, caplog):
    """When the existing directory fails the check, the caller gets a different, newly made directory and a warning."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    first = pt.private_temp_dir("mlframe_caches")
    monkeypatch.setattr(pt, "_is_private", lambda st, euid: False)
    with caplog.at_level("WARNING", logger=pt.logger.name):
        second = pt.private_temp_dir("mlframe_caches")
    assert second != first and os.path.isdir(second)
    assert any("not a private directory" in r.getMessage() for r in caplog.records)


def test_crash_dir_fallback_is_the_private_directory_not_the_shared_temp_root(monkeypatch, tmp_path):
    """With no explicit directory, no env override and no log file handler, the crash directory is the per-user one."""
    import logging

    from mlframe.training import crash_diagnostics as cd

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("MLFRAME_CRASH_LOG_DIR", raising=False)
    saved = [(lg, list(lg.handlers)) for lg in (logging.getLogger(), cd.logger)]
    for lg, _ in saved:
        lg.handlers = [h for h in lg.handlers if not hasattr(h, "baseFilename")]
    try:
        got = cd.resolve_crash_dir()
    finally:
        for lg, handlers in saved:
            lg.handlers = handlers
    assert got != str(tmp_path)
    assert os.path.basename(got) == f"mlframe_crash-{pt._user_tag()}"


def test_pipeline_cache_file_lives_in_the_private_directory(monkeypatch, tmp_path):
    """The polars-ds round-trip cache file is under the per-user directory, not directly in the shared temp root."""
    from mlframe.training.core import _setup_helpers_pipeline_cache as pc

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("MLFRAME_PIPELINE_DISK_CACHE_PATH", raising=False)
    monkeypatch.setattr(pc, "_PIPELINE_JSON_DISK_CACHE_PATH", None)
    monkeypatch.setattr(pc, "_parent_attr", lambda name, default=None: None)
    monkeypatch.setattr(pc, "_parent_set", lambda name, value: None)
    got = pc._pipeline_disk_cache_path()
    assert os.path.basename(os.path.dirname(got)) == f"mlframe_caches-{pt._user_tag()}"

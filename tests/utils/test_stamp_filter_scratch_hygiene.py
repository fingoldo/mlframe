"""Two small hygiene fixes of wave 5: the numba stamp is renamed into place, the CatBoost border files go to a private scratch directory (FSW5-3, FSW5-4)."""

import os
import tempfile

from mlframe.utils import numba_cache_guard as guard
from mlframe.utils import private_temp as pt


def test_the_numba_stamp_is_renamed_into_place_not_written_in_place(tmp_path, monkeypatch):
    """The stamp file appears through os.replace from a temporary name, and no temporary file is left behind."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("from numba import njit\n\n\n@njit(cache=True)\ndef f(x):\n    return x + 1\n", encoding="utf-8")
    real_replace = os.replace
    calls = []

    def spy(src, dst):
        """Record the move and delegate."""
        calls.append((os.path.basename(str(src)), os.path.basename(str(dst))))
        return real_replace(src, dst)

    monkeypatch.setattr(guard.os, "replace", spy)
    guard.invalidate_stale_numba_caches(pkg)
    stamp = guard._STAMP_FILE
    assert any(dst == stamp and src != stamp and src.endswith(".tmp") for src, dst in calls), calls
    assert (pkg / guard._STAMP_DIR_NAME / stamp).read_text(encoding="utf-8").strip()
    assert not list((pkg / guard._STAMP_DIR_NAME).glob("*.tmp"))


def test_border_files_are_staged_in_the_private_directory(monkeypatch, tmp_path):
    """Both CatBoost border temp files are created inside the per-user private directory, not the shared temp root."""
    from mlframe.feature_selection.wrappers.rfecv import _cb_border_cache as bc

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = []
    real = tempfile.mkstemp

    def spy(*args, **kwargs):
        """Record the directory argument and delegate."""
        seen.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(bc.tempfile, "mkstemp", spy)

    class _Pool:
        """A pool whose border dump is an empty file."""

        def save_quantization_borders(self, path):
            """Write nothing."""
            open(path, "w", encoding="utf-8").close()

    bc._harvest(_Pool(), [], bc._FoldBorders())
    assert seen and all(d is not None and os.path.basename(d) == f"mlframe_cb_borders-{pt._user_tag()}" for d in seen)

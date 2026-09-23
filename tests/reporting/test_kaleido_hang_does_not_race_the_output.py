"""A kaleido write that outruns its timeout must not be able to overwrite what the recovery wrote."""

import os
import threading
import time

import pytest

from mlframe.reporting.renderers import _kaleido as kal


def test_a_late_writer_lands_on_a_scratch_path_not_the_target(tmp_path, monkeypatch):
    """The abandoned thread used to finish onto the SAME path the recovery writes, so the saved chart was a race."""
    kaleido = pytest.importorskip("kaleido")
    started, wrote = threading.Event(), threading.Event()

    def _slow_write(fig, path, opts=None):
        started.set()
        time.sleep(1.5)  # outlives the timeout below
        with open(path, "w", encoding="utf-8") as f:
            f.write("late writer")
        wrote.set()

    monkeypatch.setattr(kal, "_KALEIDO_PERSISTENT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(kaleido, "write_fig_sync", _slow_write)
    monkeypatch.setattr(kal, "_ensure_kaleido_server_started", lambda: True)
    # A timed-out write is treated as a hung server, so the recovery writes HTML - a path that never touches kaleido.

    target = str(tmp_path / "chart.png")

    class _Fig:
        def write_html(self, path, **kw):
            with open(path, "w", encoding="utf-8") as f:
                f.write("<html>recovery</html>")

    kal.write_image_via_kaleido(_Fig(), target, "png")
    assert started.wait(timeout=10) and wrote.wait(timeout=10), "the late writer must have finished for this to prove anything"

    assert not os.path.exists(target), "the abandoned writer wrote the caller's path"
    assert [p for p in os.listdir(tmp_path) if ".partial-" in p], "it must have landed on a scratch path instead"


def test_abandoning_a_scratch_file_removes_it(tmp_path):
    scratch = tmp_path / "chart.png.partial-abc"
    scratch.write_text("x", encoding="utf-8")
    kal._abandon_scratch_file(str(scratch))
    assert not scratch.exists()


def test_abandoning_a_missing_scratch_file_is_quiet(tmp_path):
    """A write that never got as far as creating its scratch file must not turn cleanup into an error."""
    missing = tmp_path / "never-created.partial-zzz"
    kal._abandon_scratch_file(str(missing))
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == [], "cleanup must not create anything either"

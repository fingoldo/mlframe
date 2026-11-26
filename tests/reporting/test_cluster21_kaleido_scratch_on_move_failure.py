"""Cluster 21 regression: a failed move of the rendered chart abandons its scratch file (audit 2026-11-20, finding 21-F20).

The persistent kaleido path writes to a private scratch file and moves it into place. When the move fails, the partial scratch file must not be
left next to the target. The kaleido server, its restart and the one-shot fallback are stubbed, so the test exercises only the write-and-move.
"""

import os
import sys
from types import SimpleNamespace


from mlframe.reporting.renderers import _kaleido as kd


def _fake_kaleido(monkeypatch):
    """Install a stand-in kaleido whose synchronous write creates the file it is given."""

    def _write_fig_sync(fig, path, opts=None):
        """Create the scratch file as a real write would."""
        with open(path, "wb") as fh:
            fh.write(b"partial image")

    monkeypatch.setitem(sys.modules, "kaleido", SimpleNamespace(write_fig_sync=_write_fig_sync))
    monkeypatch.setattr(kd, "_is_kaleido_persistent_burned", lambda: False)
    monkeypatch.setattr(kd, "_ensure_kaleido_server_started", lambda: True)
    monkeypatch.setattr(kd, "_restart_kaleido_server", lambda: False)

    def _oneshot_stub(fig, path, fmt):
        """Stand-in for the one-shot retry, which is not under test here."""
        raise RuntimeError("one-shot retry stubbed out")

    monkeypatch.setattr(kd, "_oneshot_write_static", _oneshot_stub)


def test_failed_move_leaves_no_scratch_file_next_to_the_target(tmp_path, monkeypatch):
    """When the final move raises, no ``.partial-`` scratch file remains and the target is not created (21-F20)."""
    _fake_kaleido(monkeypatch)

    def _move_fails(src, dst):
        """Fail the move the way a locked or missing destination does."""
        raise OSError("move refused")

    monkeypatch.setattr(os, "replace", _move_fails)
    target = str(tmp_path / "chart.png")
    try:
        kd.write_image_via_kaleido(object(), target, "png")
    except Exception:
        pass  # the caller-visible outcome is not under test; the scratch file is

    assert not os.path.exists(target)
    assert not [name for name in os.listdir(tmp_path) if ".partial-" in name]


def test_successful_move_publishes_the_chart_with_no_scratch_left(tmp_path, monkeypatch):
    """Control: when the move succeeds the chart is published under the target name and no scratch file remains (21-F20)."""
    _fake_kaleido(monkeypatch)
    target = str(tmp_path / "chart.png")
    kd.write_image_via_kaleido(object(), target, "png")

    assert os.path.exists(target)
    assert not [name for name in os.listdir(tmp_path) if ".partial-" in name]

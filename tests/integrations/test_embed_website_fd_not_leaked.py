"""embed_website_to_mlflow closes the temp descriptor when fdopen itself fails (tracker 2026-11-02 wave 8, RES8-3)."""

import os

import pytest

from mlframe.integrations import mlflow as ml


def test_fd_closed_and_temp_file_removed_when_fdopen_fails(monkeypatch, tmp_path):
    """On failure before the file object adopts the descriptor, the descriptor is closed (the unlink also works on Windows)."""
    closed = []
    real_close = os.close
    monkeypatch.setattr(ml.os, "fdopen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(ml.os, "close", lambda fd: (closed.append(fd), real_close(fd))[1])
    with pytest.raises(OSError, match="boom"):
        ml.embed_website_to_mlflow("http://example.com", fname=str(tmp_path / "page"))
    assert len(closed) == 1
    assert list(tmp_path.iterdir()) == []

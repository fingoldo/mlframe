"""A wedged backend render must not block ``render_and_save`` past its per-backend timeout.

The multi-backend path used ``with ThreadPoolExecutor(...)``; the 60s ``result(timeout=...)`` logged the worker
as abandoned, but the block's ``__exit__`` then joined that same worker, so a render that never returned hung the
caller forever. Deep-nightly shards 15/16 sat in exactly that join until the job cap.
"""

from __future__ import annotations

import threading
import time

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import save as S
from mlframe.reporting.spec import FigureSpec


class _Renderer:
    def __init__(self, name, release):
        self.name = name
        self.release = release
        self.saved = []

    def render(self, spec, **kw):
        if self.name == "matplotlib":
            self.release.wait()  # wedged until the test lets it go
        return object()

    def save(self, fig, path, fmt):
        self.saved.append(path)

    def show(self, fig):
        pass

    def close(self, fig):
        pass


def test_wedged_backend_times_out_and_the_other_backend_still_saves(monkeypatch, tmp_path):
    release = threading.Event()
    renderers = {b: _Renderer(b, release) for b in ("plotly", "matplotlib")}
    monkeypatch.setattr(S, "get_renderer", lambda b: renderers[b])
    monkeypatch.setattr(S, "_BACKEND_RENDER_TIMEOUT_S", 0.5)
    try:
        t0 = time.perf_counter()
        S.render_and_save(FigureSpec(), parse_plot_output_dsl("plotly[html] + matplotlib[png]"), str(tmp_path / "chart"), interactive=False)
        elapsed = time.perf_counter() - t0
    finally:
        release.set()
    assert elapsed < 10.0, f"render_and_save waited {elapsed:.1f}s on a wedged backend instead of giving up after the timeout"
    assert renderers["plotly"].saved, "the healthy backend's output was lost"

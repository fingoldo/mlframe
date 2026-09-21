"""A wedged backend render must not block ``render_and_save`` past its per-backend timeout.

The multi-backend path used ``with ThreadPoolExecutor(...)``; the 60s ``result(timeout=...)`` logged the worker
as abandoned, but the block's ``__exit__`` then joined that same worker, so a render that never returned hung the
caller forever. Deep-nightly shards 15/16 sat in exactly that join until the job cap.
"""

from __future__ import annotations

import threading

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import save as S
from mlframe.reporting.spec import FigureSpec


class _Renderer:
    """Renderer stand-in whose matplotlib instance blocks in render until the test releases it."""
    def __init__(self, name, release):
        self.name = name
        self.release = release
        self.saved = []

    def render(self, spec, **kw):
        """Return a dummy figure; the matplotlib instance first waits on the release event."""
        if self.name == "matplotlib":
            self.release.wait()  # wedged until the test lets it go
        return object()

    def save(self, fig, path, fmt):
        """Record the saved path."""
        self.saved.append(path)

    def show(self, fig):
        """No-op show."""
        pass

    def close(self, fig):
        """No-op close."""
        pass


def test_wedged_backend_times_out_and_the_other_backend_still_saves(monkeypatch, tmp_path):
    """A backend wedged in render times out and the other backend's file is still saved."""
    release = threading.Event()
    renderers = {b: _Renderer(b, release) for b in ("plotly", "matplotlib")}
    monkeypatch.setattr(S, "get_renderer", lambda b: renderers[b])
    monkeypatch.setattr(S, "_BACKEND_RENDER_TIMEOUT_S", 0.5)
    caller = threading.Thread(
        target=S.render_and_save,
        args=(FigureSpec(), parse_plot_output_dsl("plotly[html] + matplotlib[png]"), str(tmp_path / "chart")),
        kwargs={"interactive": False},
        daemon=True,
    )
    try:
        caller.start()
        # Returning at all while the matplotlib render is still blocked is the claim; the join bound only stops a
        # regression from hanging the suite, and sits far above the 0.5 s per-backend timeout.
        caller.join(timeout=120)
        returned_while_wedged = not caller.is_alive()
    finally:
        release.set()
    assert returned_while_wedged, "render_and_save never returned while a backend stayed wedged: it is waiting on the abandoned render"
    assert renderers["plotly"].saved, "the healthy backend's output was lost"

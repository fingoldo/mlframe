"""``render_and_save`` / raw-figure saves / combined-HTML index deferred through a ``ReportRenderQueue``: same files as inline rendering."""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pytest

from mlframe.reporting._async_render import ReportRenderQueue
from mlframe.reporting.async_render_hooks import render_queue_scope, resolve_async_render
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers.save import render_and_save, set_format_subfolders
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_DSL = parse_plot_output_dsl("plotly[html] + matplotlib[png]")


def _spec(y: np.ndarray) -> FigureSpec:
    """A one-panel line figure over ``y``."""
    x = np.arange(len(y), dtype=np.float64)
    return FigureSpec(suptitle="t", panels=((LinePanelSpec(x=x, y=y, title="curve", xlabel="i", ylabel="v"),),), figsize=(6.0, 3.0))


def _digests(root: str) -> dict:
    """Relative path -> content hash for every file under ``root``; plotly's random element ids are normalised so HTML is comparable."""
    out = {}
    for base, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            with open(p, "rb") as fh:
                raw = fh.read()
            if f.endswith(".html"):
                raw = _UUID.sub("UUID", raw.decode("utf-8")).encode("utf-8")
            out[os.path.relpath(p, root)] = hashlib.sha1(raw, usedforsecurity=False).hexdigest()
    return out


def test_async_render_matches_inline_render_file_for_file(tmp_path):
    """Queued renders produce the same file set with byte-identical PNGs and structure-identical HTML as inline rendering."""
    rng = np.random.default_rng(0)
    ys = [np.cumsum(rng.standard_normal(300)) for _ in range(4)]
    sync_dir, async_dir = str(tmp_path / "sync"), str(tmp_path / "async")
    set_format_subfolders(True)
    try:
        for i, y in enumerate(ys):
            render_and_save(_spec(y), _DSL, os.path.join(sync_dir, f"fig{i}"), interactive=False)
        with ReportRenderQueue(backend="thread", workers=1) as q, render_queue_scope(q):
            for i, y in enumerate(ys):
                assert render_and_save(_spec(y), _DSL, os.path.join(async_dir, f"fig{i}"), interactive=False) is None
            summary = q.join()
            assert summary.submitted == 4 and summary.failed == 0
    finally:
        set_format_subfolders(None)
    sync, queued = _digests(sync_dir), _digests(async_dir)
    assert sorted(sync) == sorted(queued) and len(sync) == 8
    assert sync == queued, "queued rendering must not change a single byte of the saved artifacts"


def test_layout_override_set_on_the_submitting_thread_reaches_the_worker(tmp_path):
    """The per-format subfolder layout is a thread-local of the suite thread; the task carries it, so the worker writes png/ and html/."""
    set_format_subfolders(True)
    try:
        with ReportRenderQueue(backend="thread", workers=1) as q, render_queue_scope(q):
            render_and_save(_spec(np.arange(20.0)), _DSL, str(tmp_path / "fig"), interactive=False)
    finally:
        set_format_subfolders(None)
    assert (tmp_path / "png" / "fig.png").is_file() and (tmp_path / "html" / "fig.html").is_file()
    assert not (tmp_path / "fig.png").exists()


def test_interactive_and_keep_handles_stay_inline_and_in_order(tmp_path, monkeypatch):
    """Inline display (interactive session) and ``keep_handles`` are never deferred: nothing is queued and the result is returned at once."""
    shown: list = []

    class _Spy:
        """Stands in for a renderer and records ``show`` calls."""

        def render(self, spec, **_kw):
            """Return a dummy figure."""
            return object()

        def save(self, fig, path, fmt):
            """Write an empty file so the path exists."""
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "wb").close()

        def show(self, fig):
            """Record the inline display."""
            shown.append(fig)

    monkeypatch.setattr("mlframe.reporting.renderers.save.get_renderer", lambda backend: _Spy())
    with ReportRenderQueue(backend="thread", workers=1) as q, render_queue_scope(q):
        render_and_save(_spec(np.arange(5.0)), parse_plot_output_dsl("plotly[html]"), str(tmp_path / "a"), interactive=True)
        assert len(shown) == 1, "inline display must happen before render_and_save returns"
        handles = render_and_save(_spec(np.arange(5.0)), parse_plot_output_dsl("plotly[html]"), str(tmp_path / "b"), interactive=False, keep_handles=True)
        assert handles is not None and "plotly" in handles
        assert q.submitted == 0


def test_spec_is_snapshotted_so_later_mutation_of_the_source_array_cannot_change_the_chart(tmp_path):
    """The queued render draws the array as it was at submit time, exactly like an inline render would have."""
    y = np.cumsum(np.random.default_rng(1).standard_normal(200))
    expected_dir, got_dir = str(tmp_path / "expected"), str(tmp_path / "got")
    render_and_save(_spec(y.copy()), parse_plot_output_dsl("matplotlib[png]"), os.path.join(expected_dir, "f"), interactive=False)
    gate_q = ReportRenderQueue(backend="thread", workers=1)
    try:
        import threading

        release = threading.Event()
        gate_q.submit(release.wait, 30, name="occupy-the-only-worker")
        with render_queue_scope(gate_q):
            render_and_save(_spec(y), parse_plot_output_dsl("matplotlib[png]"), os.path.join(got_dir, "f"), interactive=False)
        y[:] = 0.0  # the training loop reuses the buffer while the render is still queued behind the blocked worker
        release.set()
        assert gate_q.close().failed == 0
    finally:
        gate_q.close(wait=False)
    assert _digests(expected_dir) == _digests(got_dir)


def test_raw_figure_save_is_deferred_and_identical(tmp_path):
    """An explicit (non-pyplot) Figure saved through the diagnostics saver is written by the worker with the same bytes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from mlframe.reporting.diagnostics_dispatch import _save_figure

    def _fig():
        """Build a small explicit figure."""
        f = Figure(figsize=(4, 3))
        FigureCanvasAgg(f)
        f.subplots().plot([1, 2, 3], [3, 1, 2])
        return f

    assert _save_figure(_fig(), "matplotlib[png]", str(tmp_path / "sync" / "fig")) is True
    with ReportRenderQueue(backend="thread", workers=1) as q, render_queue_scope(q):
        assert _save_figure(_fig(), "matplotlib[png]", str(tmp_path / "async" / "fig")) is True
        assert q.submitted == 1
    assert (tmp_path / "async" / "fig.png").read_bytes() == (tmp_path / "sync" / "fig.png").read_bytes()


def test_combined_html_index_waits_for_the_charts_it_stitches(tmp_path):
    """The index is queued behind the renders it reads, records its path at once, and equals the inline index once joined."""
    from mlframe.reporting._diagnostics_dispatch_extra import build_combined_html_report

    set_format_subfolders(True)
    try:
        ys = [np.cumsum(np.random.default_rng(i).standard_normal(150)) for i in range(3)]
        results = {}
        for mode in ("sync", "async"):
            base = str(tmp_path / mode / "m_test")
            metrics: dict = {}
            paths = [f"{base}_c{i}" for i in range(3)]

            def _run(base=base, paths=paths, metrics=metrics):
                """Render three charts and stitch them."""
                for p, y in zip(paths, ys):
                    render_and_save(_spec(y), parse_plot_output_dsl("matplotlib[png]"), p, interactive=False)
                build_combined_html_report(base_path=base, chart_paths=paths, plot_outputs="matplotlib[png]", title="T", metrics_dict=metrics)

            if mode == "sync":
                _run()
            else:
                with ReportRenderQueue(backend="thread", workers=3) as q, render_queue_scope(q):
                    _run()
                    assert metrics["charts"]["combined_report"].endswith("m_test_report.html"), "path is recorded before the stitch finishes"
                    q.join()
            results[mode] = (metrics["charts"], _digests(str(tmp_path / mode)))
    finally:
        set_format_subfolders(None)
    sync_files, async_files = results["sync"][1], results["async"][1]
    assert sorted(k.replace("sync", "X") for k in sync_files) == sorted(k.replace("async", "X") for k in async_files)
    reports = {}
    for mode in ("sync", "async"):
        found = [p for p in (tmp_path / mode).rglob("*_report.html")]
        assert len(found) == 1
        reports[mode] = found[0].read_text(encoding="utf-8")
    for text in reports.values():
        assert text.count("<img") == 3 and "data:image/png;base64" in text
    assert re.sub(r"<title>.*?</title>", "", reports["sync"]) == re.sub(r"<title>.*?</title>", "", reports["async"])
    assert results["sync"][0]["saved"] == results["async"][0]["saved"]


def test_worker_failure_is_reported_by_artifact_name_and_later_renders_continue(tmp_path, caplog, monkeypatch):
    """A renderer that raises drops that one artifact with a WARNING naming it; the next queued render still lands."""
    import logging

    real_get = __import__("mlframe.reporting.renderers.save", fromlist=["get_renderer"]).get_renderer

    class _Exploding:
        """Renderer whose save raises for one specific figure."""

        def __init__(self, inner):
            """Wrap the real renderer."""
            self._inner = inner

        def render(self, spec, **kw):
            """Delegate."""
            return self._inner.render(spec, **kw)

        def save(self, fig, path, fmt):
            """Fail for the poisoned file only."""
            if "poison" in path:
                raise OSError("disk on fire")
            return self._inner.save(fig, path, fmt)

    monkeypatch.setattr("mlframe.reporting.renderers.save.get_renderer", lambda backend: _Exploding(real_get(backend)))
    with caplog.at_level(logging.WARNING):
        q = ReportRenderQueue(backend="thread", workers=1)
        with render_queue_scope(q):
            render_and_save(_spec(np.arange(10.0)), parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "poison_fig"), interactive=False)
            render_and_save(_spec(np.arange(10.0)), parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "good_fig"), interactive=False)
        summary = q.close()
    assert summary.failed == 1 and summary.completed == 1
    assert summary.failures[0].name == "poison_fig"
    assert not (tmp_path / "poison_fig.png").exists() and (tmp_path / "good_fig.png").is_file()
    assert any("poison_fig" in r.getMessage() and r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize(
    ("setting", "save_charts", "data_dir", "expected"),
    [
        (False, True, "/x", False),
        (True, True, "/x", True),
        (True, False, "/x", False),
        (True, True, "", False),
        ("auto", False, "/x", False),
        ("auto", True, None, False),
    ],
)
def test_resolve_async_render_needs_saved_charts(setting, save_charts, data_dir, expected):
    """Explicit flags are honoured but nothing is deferred when no chart is saved; 'auto' additionally needs a data_dir."""
    assert resolve_async_render(setting, save_charts=save_charts, data_dir=data_dir) is expected


def test_auto_needs_three_physical_cores(monkeypatch):
    """'auto' stays synchronous on a machine with fewer than three physical cores and turns on at three."""
    monkeypatch.setattr("mlframe.reporting.async_render_hooks.physical_core_count", lambda: 2)
    assert resolve_async_render("auto", save_charts=True, data_dir="/x") is False
    monkeypatch.setattr("mlframe.reporting.async_render_hooks.physical_core_count", lambda: 3)
    assert resolve_async_render("auto", save_charts=True, data_dir="/x") is True


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_decile_table_worker_builds_and_saves_the_same_png_from_handed_over_arrays(tmp_path, backend):
    """The decile table is built AND saved by the worker from (y, score) handed over zero-copy; the PNG equals the inline one byte for byte."""
    from mlframe.reporting._diagnostics_dispatch_extra import render_decile_table_diagnostic

    rng = np.random.default_rng(3)
    n = 70_000  # float64 score > the 64 KiB shared-memory threshold, so the process backend really maps it from a segment
    y = (rng.random(n) > 0.6).astype(np.int64)
    score = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    sync_m: dict = {}
    assert render_decile_table_diagnostic(y_true=y, y_score=score, plot_outputs="matplotlib[png]", base_path=str(tmp_path / "sync" / "m_test"), metrics_dict=sync_m)
    async_m: dict = {}
    own = f"mlframe_rq_{os.getpid()}_"
    before = {n_ for n_ in os.listdir("/dev/shm") if n_.startswith(own)} if os.path.isdir("/dev/shm") else set()  # nosec B108 - POSIX shared-memory mount, not a temp file
    q = ReportRenderQueue(backend=backend, workers=1)
    with render_queue_scope(q):
        assert render_decile_table_diagnostic(y_true=y, y_score=score, plot_outputs="matplotlib[png]", base_path=str(tmp_path / "async" / "m_test"), metrics_dict=async_m)
        y[:] = 0  # the caller is free to reuse its buffers while the worker holds its own snapshot / segment
    summary = q.close()
    assert summary.failed == 0 and summary.completed == 1
    assert (tmp_path / "async" / "m_test_decile_table.png").read_bytes() == (tmp_path / "sync" / "m_test_decile_table.png").read_bytes()
    assert sync_m["charts"]["saved"] == async_m["charts"]["saved"] == ["decile_table"]
    assert sync_m["charts"]["paths"][0].replace("sync", "X") == async_m["charts"]["paths"][0].replace("async", "X")
    after = {n_ for n_ in os.listdir("/dev/shm") if n_.startswith(own)} if os.path.isdir("/dev/shm") else set()  # nosec B108 - POSIX shared-memory mount, not a temp file
    assert after == before


def test_join_stamps_metadata_names_failures_and_logs_the_summary_once(caplog):
    """The suite-level join writes ``metadata["async_render"]``, lists failed artifacts by name, and does not repeat the summary line when nothing new arrived."""
    import logging

    from mlframe.reporting.async_render_hooks import join_suite_render_queue

    def _fail():
        """Raises, to produce one failed artifact."""
        raise OSError("nope")

    q = ReportRenderQueue(backend="thread", workers=1)
    q.submit(lambda: 1, name="fine_chart")
    q.submit(_fail, name="broken_chart")
    meta: dict = {}
    with caplog.at_level(logging.INFO, logger="mlframe.reporting.async_render_hooks"):
        join_suite_render_queue(q, meta)
        join_suite_render_queue(q, meta, final=True)
    summary_lines = [r for r in caplog.records if "render done" in r.getMessage()]
    assert len(summary_lines) == 1 and "broken_chart" in summary_lines[0].getMessage()
    assert meta["async_render"]["failed"] == ["broken_chart"] and meta["async_render"]["artifacts"] == 1
    assert meta["async_render"]["backend"] == "thread"


def test_start_suite_render_queue_never_raises_and_respects_off(monkeypatch):
    """A queue that cannot be built leaves the suite inline (None); ``async_render=False`` and missing charts give None without building anything."""
    from mlframe.reporting import async_render_hooks as hooks

    class _Cfg:
        """Config stand-in."""

        async_render = True
        async_render_backend = "thread"
        async_render_workers = 1
        async_render_max_pending_mb = 16.0

    q = hooks.start_suite_render_queue(_Cfg(), save_charts=True, data_dir="/x", verbose=False)
    assert q is not None and q.workers == 1
    q.close()
    assert hooks.start_suite_render_queue(_Cfg(), save_charts=False, data_dir="/x") is None
    monkeypatch.setattr(hooks, "ReportRenderQueue", lambda **kw: (_ for _ in ()).throw(RuntimeError("cannot build")))
    assert hooks.start_suite_render_queue(_Cfg(), save_charts=True, data_dir="/x", verbose=False) is None

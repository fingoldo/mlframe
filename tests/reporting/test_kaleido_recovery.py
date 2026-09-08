"""Regression sensor: kaleido persistent-server failures must not hang.

History (2026-05-08): a c0031 (lgb+xgb+hgb multiclass + recency
weights) profile run hung for 2+ hours. Post-mortem trace showed
kaleido raised ``KaleidoError: Error 525: Cannot read properties of
undefined (reading 'v')`` on one figure, which asyncio-cancelled the
persistent server's task chain; subsequent ``write_fig_sync`` calls
``await asyncio.gather`` forever.

This test simulates the failure mode and asserts:
1. The failed call returns (does NOT hang).
2. The output file IS written via the oneshot fallback.
3. A subsequent call uses the restarted persistent server.

Without the recovery path in PlotlyRenderer.save(), this test would
hang and the test runner would kill it on timeout.
"""

from __future__ import annotations

import os
import time
import warnings

import pytest

from tests.conftest import perf_time_budget


@pytest.mark.timeout(900)
def test_kaleido_persistent_failure_falls_back_to_oneshot(tmp_path):
    """Synthetic kaleido failure must not hang; output file must exist."""
    pytest.importorskip("kaleido")  # Not in CI [all,dev] extras; sensor needs the real package to patch.
    import plotly.graph_objects as go
    from mlframe.reporting.renderers.plotly import (
        PlotlyRenderer,
        _restart_kaleido_server,
    )
    import kaleido

    r = PlotlyRenderer()
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))

    td = str(tmp_path)
    # Warmup persistent server (so subsequent failure path is exercised).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.save(fig, os.path.join(td, "warmup.png"), "png")

    # Inject a synthetic failure into the persistent path.
    orig = kaleido.write_fig_sync

    def _raise(*args, **kwargs):
        """Helper: Raise."""
        raise RuntimeError("synthetic kaleido failure for test")

    kaleido.write_fig_sync = _raise

    try:
        target = os.path.join(td, "after_error.png")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.save(fig, target, "png")
        elapsed = time.perf_counter() - t0
    finally:
        # Restore for cleanup AND for the next assertion.
        kaleido.write_fig_sync = orig
        # Restart server for clean state across tests.
        _restart_kaleido_server()

    # Assertions:
    # 1. File exists -- oneshot fallback fired and wrote PNG.
    assert os.path.exists(target), "PlotlyRenderer.save did not produce a fallback PNG after the persistent kaleido path failed -- recovery is broken."
    # 2. Did not hang -- elapsed should be bounded by oneshot cost
    # (~30-40s on cold; well under the 60s pytest-timeout).
    # Failure recovery path: server-restart + oneshot fallback. On
    # cold disk cache the Chromium re-spawn is ~30-40s; warmer is
    # ~12-15s. Tolerate up to 90s so test isn't flaky on a busy CI
    # box (the assertion is "did not HANG", not "was fast").
    hang_budget = perf_time_budget(90.0)
    assert elapsed < hang_budget, (
        f"PlotlyRenderer.save took {elapsed:.1f}s after a kaleido "
        f"failure (budget {hang_budget:.0f}s); expected the oneshot fallback -- this looks "
        f"like the deadlock-on-error regression we fixed."
    )
    # 3. File has non-trivial size -- it's a real PNG, not 0-byte stub.
    assert os.path.getsize(target) > 1000, f"Fallback PNG is suspiciously small ({os.path.getsize(target)} bytes)."


@pytest.mark.timeout(900)
def test_kaleido_recovery_restores_persistent_path(tmp_path):
    """After a failure + recovery, the next save should use the
    restarted persistent server (fast path), not stay on oneshot
    forever."""
    pytest.importorskip("kaleido")  # Not in CI [all,dev] extras; sensor needs the real package to patch.
    import plotly.graph_objects as go
    from mlframe.reporting.renderers.plotly import (
        PlotlyRenderer,
        _restart_kaleido_server,
    )
    import kaleido

    r = PlotlyRenderer()
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))

    td = str(tmp_path)
    # Warmup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.save(fig, os.path.join(td, "warmup.png"), "png")

    # Trigger failure + recovery
    orig = kaleido.write_fig_sync

    def _raise(*a, **kw):
        """Helper: Raise."""
        raise RuntimeError("synthetic")

    kaleido.write_fig_sync = _raise
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.save(fig, os.path.join(td, "fail.png"), "png")
    finally:
        kaleido.write_fig_sync = orig

    # Subsequent call -- should hit the restarted persistent server, not the oneshot fallback. The module
    # counts oneshot calls, so ask it: a 15s bound was being asked to separate a cold Chromium restart
    # (~8s, the pass case) from a oneshot save (~13s, the fail case), which it cannot do on a machine even
    # 1.5x slower than the one those numbers came from -- a false green for the exact regression its own
    # message names, and a false red on a cold-disk CI box.
    from mlframe.reporting.renderers._kaleido import get_kaleido_oneshot_stats, reset_kaleido_oneshot_stats

    reset_kaleido_oneshot_stats()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.save(fig, os.path.join(td, "recovered.png"), "png")
    oneshot_calls, _oneshot_seconds = get_kaleido_oneshot_stats()
    # Cleanup for downstream tests
    _restart_kaleido_server()

    assert os.path.exists(os.path.join(td, "recovered.png"))
    assert oneshot_calls == 0, (
        f"After recovery, the save made {oneshot_calls} oneshot call(s); it is still stuck on the oneshot "
        f"fallback instead of reusing the restarted persistent server."
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("decoration", ["plain", "vrect", "annotation"])
def test_a_datetime_axis_figure_exports_a_real_file_under_a_kaleido_version_mismatch(tmp_path, decoration):
    """The installed pair (plotly 5.x, kaleido 1.x) disables ``fig.write_image`` outright.

    Bare ``fig.write_image`` on a datetime axis carrying a shape or an annotation can then fail SILENTLY --
    no exception and no file -- so a caller that only checks for an exception believes it exported a chart it
    did not. The recovery ladder in ``_kaleido.py`` is what makes that survivable, and the contract worth
    pinning is the FILE, not the absence of a raise: a rung that returns quietly having written nothing must
    not read as success. Pinned across the decorations because the shape/annotation-on-a-datetime-axis case
    is the one that breaks the native path while the plain figure still works.
    """
    pytest.importorskip("kaleido")
    import numpy as np
    import plotly.graph_objects as go

    from mlframe.reporting.renderers._kaleido import write_image_via_kaleido

    x = np.array(np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]"))
    fig = go.Figure(go.Scatter(x=x, y=np.arange(len(x), dtype=float)))
    if decoration == "vrect":
        fig.add_vrect(x0=x[5], x1=x[9], fillcolor="red", opacity=0.2)
    elif decoration == "annotation":
        fig.add_annotation(x=x[5], y=1.0, text="marker")

    path = str(tmp_path / f"dt_{decoration}.png")
    write_image_via_kaleido(fig, path, "png")

    # The ladder's last rung deliberately writes interactive HTML instead, which is still a real export.
    produced = [p for p in (path, os.path.splitext(path)[0] + ".html") if os.path.exists(p)]
    assert produced, f"{decoration}: kaleido reported no error and produced no file at all"
    assert os.path.getsize(produced[0]) > 1000, f"{decoration}: wrote a {os.path.getsize(produced[0])}-byte stub"

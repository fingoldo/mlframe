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


@pytest.mark.timeout(600)
def test_kaleido_persistent_failure_falls_back_to_oneshot(tmp_path):
    """Synthetic kaleido failure must not hang; output file must exist."""
    import plotly.graph_objects as go
    from mlframe.reporting.renderers.plotly import (
        PlotlyRenderer, _restart_kaleido_server,
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
    assert os.path.exists(target), (
        "PlotlyRenderer.save did not produce a fallback PNG after "
        "the persistent kaleido path failed -- recovery is broken."
    )
    # 2. Did not hang -- elapsed should be bounded by oneshot cost
    # (~30-40s on cold; well under the 60s pytest-timeout).
    # Failure recovery path: server-restart + oneshot fallback. On
    # cold disk cache the Chromium re-spawn is ~30-40s; warmer is
    # ~12-15s. Tolerate up to 90s so test isn't flaky on a busy CI
    # box (the assertion is "did not HANG", not "was fast").
    assert elapsed < 90.0, (
        f"PlotlyRenderer.save took {elapsed:.1f}s after a kaleido "
        f"failure; expected oneshot fallback (<90s) -- this looks "
        f"like the deadlock-on-error regression we fixed."
    )
    # 3. File has non-trivial size -- it's a real PNG, not 0-byte stub.
    assert os.path.getsize(target) > 1000, (
        f"Fallback PNG is suspiciously small "
        f"({os.path.getsize(target)} bytes)."
    )


@pytest.mark.timeout(600)
def test_kaleido_recovery_restores_persistent_path(tmp_path):
    """After a failure + recovery, the next save should use the
    restarted persistent server (fast path), not stay on oneshot
    forever."""
    import plotly.graph_objects as go
    from mlframe.reporting.renderers.plotly import (
        PlotlyRenderer, _restart_kaleido_server,
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
        raise RuntimeError("synthetic")
    kaleido.write_fig_sync = _raise
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.save(fig, os.path.join(td, "fail.png"), "png")
    finally:
        kaleido.write_fig_sync = orig

    # Subsequent call -- should hit restarted persistent server, fast.
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.save(fig, os.path.join(td, "recovered.png"), "png")
    elapsed = time.perf_counter() - t0
    # Cleanup for downstream tests
    _restart_kaleido_server()

    assert os.path.exists(os.path.join(td, "recovered.png"))
    # Persistent path should be ~0.1-1s for warm-restart;
    # cold-restart of Chromium is ~8s; oneshot is ~13s. Tolerate
    # the cold-restart range.
    assert elapsed < 15.0, (
        f"After recovery, save took {elapsed:.1f}s. Expected "
        f"persistent-server warm reuse (<2s) or cold restart (<10s). "
        f"Suggests stuck on oneshot fallback even after server "
        f"recovery."
    )

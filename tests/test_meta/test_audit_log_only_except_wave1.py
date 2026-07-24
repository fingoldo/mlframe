"""log_only_except wave 1: closes the one genuine gap the fixed scanner heuristic still flags in
``reporting/diagnostics_dispatch.py`` after the pyutilz scanner fix (bare-function "record" helper
calls like ``_record(charts, name, ok)`` are now recognised as a real escalation path, which
cleared 8 of the file's 9 pre-fix findings as false positives). The 9th site
(``render_split_error_diagnostics``'s ``worst_k_table`` except) genuinely never called
``_record(...)`` on failure -- every sibling except block in the same function does -- so a caller
inspecting ``metrics_dict["charts"]["failed"]`` had no way to learn the worst-K table specifically
failed to render.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_render_split_error_diagnostics_records_worst_k_table_failure(monkeypatch):
    """When `worst_k_table` raises, the failure must land in `charts["failed"]`, not just the log."""
    import mlframe.reporting.diagnostics_dispatch as dd

    def _raise(*args, **kwargs):
        """Always raises ``RuntimeError('boom')`` to force the except branch."""
        raise RuntimeError("boom")

    monkeypatch.setattr(dd, "_bounded_sample_idx", lambda n, loss, seed=0: np.arange(min(n, 10)))

    import mlframe.reporting.charts.error_analysis as ea

    monkeypatch.setattr(ea, "worst_k_table", _raise)

    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({"f0": rng.normal(size=n)})
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.1, size=n)
    metrics_dict: dict = {}

    out = dd.render_split_error_diagnostics(
        df=df,
        y_true=y_true,
        y_pred=y_pred,
        task="regression",
        plot_outputs="none",
        base_path="unused",
        metrics_dict=metrics_dict,
    )
    assert out["worst_k_table"] is None
    assert "worst_k_table" in metrics_dict["charts"]["failed"]

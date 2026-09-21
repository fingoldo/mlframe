"""Adversarial validation depends only on the feature frames; it must be fitted once per run, not once per target."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.reporting.diagnostics_dispatch as dd


def test_same_frames_fit_once(monkeypatch, tmp_path):
    calls = []
    import mlframe.reporting.charts.drift as drift

    orig = drift.adversarial_validation

    def _spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(drift, "adversarial_validation", _spy)
    dd._ADVERSARIAL_CACHE.clear()
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    te = pd.DataFrame({"a": rng.normal(1, 1, 300), "b": rng.normal(size=300)})
    for target in ("t1", "t2", "t3"):
        dd.render_target_drift_diagnostics(
            train_frame=tr, test_frame=te, plot_outputs="matplotlib[png]", base_path=str(tmp_path / target),
        )
    assert len(calls) == 1

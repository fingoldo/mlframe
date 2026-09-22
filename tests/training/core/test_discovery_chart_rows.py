"""The winning-spec y-vs-T chart is drawn on train rows with a real T, never on val/test rows or an imputed constant.

It plotted every row of y against a T column in which each domain-violating val/test row held the train median, which drew
a spike at that constant and mixed the test distribution into a train-time selection diagnostic.
"""

from __future__ import annotations

import numpy as np


def test_the_chart_sees_only_train_rows_with_a_finite_t(monkeypatch, tmp_path):
    """``plot_target_distribution`` receives exactly the train rows whose T is finite, and the title counts the rest."""
    import mlframe.training.composite.diagnostics as diag
    from mlframe.training.core._phase_composite_discovery_helpers import _render_composite_discovery_diagnostics

    seen = {}

    def spy(y, t, title="", **kw):
        """Record the arrays the chart would draw."""
        seen["y"], seen["t"], seen["title"] = np.asarray(y), np.asarray(t), title
        import matplotlib.pyplot as plt

        return plt.figure()

    monkeypatch.setattr(diag, "plot_target_distribution", spy)
    n = 100
    y = np.arange(n, dtype=float)
    t = y - 1.0
    t[[5, 80, 90]] = np.nan  # one train row and two holdout rows violate the domain
    train_idx = np.arange(70)
    _render_composite_discovery_diagnostics(data_dir=tmp_path, raw_target_name="y", y_full=y, t_by_spec={"s": t}, specs_export=[], train_idx=train_idx)
    assert seen["y"].size == 69 and seen["t"].size == 69, "train rows only, minus the one without a finite T"
    assert seen["y"].max() < 70, "a val/test row reached the chart"
    assert np.all(np.isfinite(seen["t"]))
    assert "1 train rows without a finite T left out" in seen["title"]

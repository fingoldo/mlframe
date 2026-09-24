"""The columns kept under max_columns must be the most important ones, not the first ones in the frame."""

import numpy as np
import pandas as pd

from mlframe.reporting import _diagnostics_dispatch_extra as dx


def test_truncation_follows_importance_not_frame_order(monkeypatch):
    seen = {}

    def _compose(df, y, features, **kw):
        seen["features"] = list(features)
        raise RuntimeError("stop after capturing the feature list")

    import mlframe.reporting.charts.category_discriminability as cd

    monkeypatch.setattr(cd, "compose_category_discriminability_figure", _compose)
    names = [f"c{i}" for i in range(10)]
    df = pd.DataFrame({n: ["a", "b"] * 5 for n in names})
    importances = np.arange(10, dtype=float)  # the LAST column is the most important
    dx.render_category_discriminability_diagnostic(
        df=df, y_true=np.array([0, 1] * 5), feature_names=names, plot_outputs="png", base_path="x",
        max_columns=3, feature_importances=importances,
    )
    assert seen["features"] == ["c9", "c8", "c7"], "the panel must be built from the most important columns"

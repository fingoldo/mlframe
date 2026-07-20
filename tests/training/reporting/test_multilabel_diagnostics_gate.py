"""Regression test: a 2-D multilabel target must never reach the single-target post-fit diagnostics.

``_render_post_fit_diagnostics`` used to ``np.asarray(targets).ravel()`` before any diagnostic saw the target's
shape. For a genuine multilabel target (n, n_labels), that flattens it to a corrupted length-(n*n_labels) array
AND forces ``ndim`` back to 1 -- so the existing ``y_arr.ndim == 1`` guards downstream could never actually detect
the multilabel case (they only ever saw the post-ravel shape). The corrupted array then reached df-paired
diagnostics (df has n rows) and either crashed natively (bare IndexError in ``separability_panel``) or hit a
length-mismatch guard deep inside them, instead of being skipped at the dispatch gate as originally intended.
Surfaced by profile_fuzz_chains.py on a multilabel_classification combo (3 labels): X=19982 rows, y=59946
= 19982*3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.configs import ReportingConfig
from mlframe.training.reporting._reporting_diagnostics import _render_post_fit_diagnostics


def test_multilabel_target_never_reaches_single_target_diagnostics(monkeypatch, tmp_path):
    """A 2-D multilabel target must skip every diagnostic that pairs a raveled y with a per-row df/y_pred."""
    n, n_labels = 40, 3
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    targets = rng.integers(0, 2, size=(n, n_labels))

    calls = []
    for name in (
        "render_engineered_separability_diagnostic",
        "render_class_structure_diagnostic",
        "render_category_discriminability_diagnostic",
        "render_slice_finder_diagnostic",
    ):
        monkeypatch.setattr(
            "mlframe.reporting.diagnostics_dispatch." + name,
            lambda *a, _n=name, **kw: calls.append(_n),
        )

    _render_post_fit_diagnostics(
        targets=targets,
        model=None,
        df=df,
        columns=list(df.columns),
        preds=None,
        probs=None,
        target_type="multilabel_classification",
        plot_file=str(tmp_path / "plot"),
        plot_outputs="plotly[html]",
        metrics={},
        reporting_config=ReportingConfig(),
    )

    assert calls == [], f"multilabel target reached single-target diagnostics: {calls}"


def test_single_column_target_still_reaches_diagnostics(monkeypatch, tmp_path):
    """A normal 1-D (or (n, 1)) target must still fire the gated diagnostics -- the multilabel guard must not
    over-trigger on ordinary single-target inputs.
    """
    n = 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    targets = rng.integers(0, 2, size=n)

    calls = []
    monkeypatch.setattr(
        "mlframe.reporting.diagnostics_dispatch.render_engineered_separability_diagnostic",
        lambda *a, **kw: calls.append("engineered_separability"),
    )

    _render_post_fit_diagnostics(
        targets=targets,
        model=None,
        df=df,
        columns=list(df.columns),
        preds=None,
        probs=None,
        target_type="binary_classification",
        plot_file=str(tmp_path / "plot"),
        plot_outputs="plotly[html]",
        metrics={},
        reporting_config=ReportingConfig(),
    )

    assert calls == ["engineered_separability"]

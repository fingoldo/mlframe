"""Regression: ``render_prediction_stability_diagnostic`` must not crash when ``member_test_preds`` and
``test_target`` have different row counts.

They come from different upstream slices (e.g. an ensemble-scoring pass over more rows than the target was
subsampled to). The old alignment only ever shrank ``y_true`` down to ``member_preds``'s row count
(``yt[:mp.shape[0]]``), so a y_true SHORTER than member_preds left them mismatched and
``compose_prediction_stability_figure``'s ``abs_error = yt - res.ensemble_mean`` raised a raw
``ValueError: operands could not be broadcast together with shapes (500,) (1500,)`` instead of degrading cleanly.
Surfaced by profiling/bug_hunt_fuzz_chains.py.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.diagnostics_dispatch import render_prediction_stability_diagnostic


def test_shorter_y_true_than_member_preds_does_not_raise(tmp_path):
    """A y_true shorter than member_preds must be aligned down, not left mismatched."""
    rng = np.random.default_rng(0)
    n_mp, n_yt = 1500, 500
    member_preds = rng.random((n_mp, 3))
    y_true = rng.random(n_yt)

    ok = render_prediction_stability_diagnostic(
        member_preds=member_preds,
        y_true=y_true,
        plot_outputs="plotly[html]",
        base_path=str(tmp_path / "chart"),
        metrics_dict={},
    )
    assert ok is True


def test_longer_y_true_than_member_preds_does_not_raise(tmp_path):
    """A y_true longer than member_preds (the previously-handled direction) must keep working."""
    rng = np.random.default_rng(1)
    n_mp, n_yt = 500, 1500
    member_preds = rng.random((n_mp, 3))
    y_true = rng.random(n_yt)

    ok = render_prediction_stability_diagnostic(
        member_preds=member_preds,
        y_true=y_true,
        plot_outputs="plotly[html]",
        base_path=str(tmp_path / "chart"),
        metrics_dict={},
    )
    assert ok is True

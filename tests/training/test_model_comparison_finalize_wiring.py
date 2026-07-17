"""The per-target model-comparison leaderboard must auto-fire from finalize when >=2 models share a target.

Guards the wiring in ``_phase_finalize._render_model_comparison_leaderboards``: the composer is built+tested
elsewhere; this pins that the suite actually CALLS it (a 1-model target must not render, a 2-model target must).
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import numpy as np

from mlframe.training.core._phase_finalize import _render_model_comparison_leaderboards


def _entry(name, y, score):
    """Entry."""
    auc_proxy = float(np.clip(np.mean(score[y == 1]) - np.mean(score[y == 0]) + 0.5, 0.0, 1.0))
    return SimpleNamespace(model_name=name, test_target=y, test_probs=score, metrics={"test": {"roc_auc": auc_proxy}})


def _ctx(tmp_path, models):
    """Ctx."""
    return SimpleNamespace(
        data_dir=str(tmp_path),
        save_charts=True,
        verbose=0,
        target_name="t",
        model_name="m",
        reporting_config=SimpleNamespace(plot_outputs="matplotlib[png]", model_comparison_charts=True),
        models=models,
        metadata={},
        configs=None,
    )


def _saved_files(tmp_path):
    """Saved files."""
    return [p for p in glob.glob(os.path.join(str(tmp_path), "charts", "**", "*"), recursive=True) if os.path.isfile(p)]


def test_leaderboard_fires_for_two_models(tmp_path):
    """Leaderboard fires for two models."""
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.uniform(size=n) < 0.4).astype(int)
    s_good = np.clip(y * 0.6 + rng.normal(0, 0.2, n) + 0.2, 0.0, 1.0)
    s_weak = np.clip(y * 0.2 + rng.normal(0, 0.4, n) + 0.4, 0.0, 1.0)
    _render_model_comparison_leaderboards(_ctx(tmp_path, {"binary_classification": {"t": [_entry("good", y, s_good), _entry("weak", y, s_weak)]}}))
    assert _saved_files(tmp_path), "model-comparison leaderboard not rendered for a 2-model target"


def test_leaderboard_skips_single_model(tmp_path):
    """Leaderboard skips single model."""
    rng = np.random.default_rng(1)
    n = 500
    y = (rng.uniform(size=n) < 0.5).astype(int)
    _render_model_comparison_leaderboards(_ctx(tmp_path, {"binary_classification": {"t": [_entry("only", y, rng.uniform(size=n))]}}))
    assert not _saved_files(tmp_path), "single-model target must not render a comparison leaderboard"


def test_leaderboard_skips_when_no_data_dir(tmp_path):
    """Leaderboard skips when no data dir."""
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.uniform(size=n) < 0.5).astype(int)
    ctx = _ctx(tmp_path, {"binary_classification": {"t": [_entry("a", y, rng.uniform(size=n)), _entry("b", y, rng.uniform(size=n))]}})
    ctx.data_dir = ""  # no persistence -> must short-circuit
    _render_model_comparison_leaderboards(ctx)
    assert not _saved_files(tmp_path)

"""Tests for ``mlframe.training.slicing._slice_pareto_plot.generate_pareto_artifact``.

The plot itself is hard to assert pixel-for-pixel; we verify the contract instead:
  - artefact paths land on disk
  - ctx.metadata["slice_stable_es"][target][model] is populated with the right keys
  - default ``pareto_plot_enabled=True`` produces both backends
  - ``pareto_plot_enabled=False`` and short runs (< min_iterations) skip cleanly
  - ``pareto_persist_shard_history=True`` writes parquet
  - save failures are caught and don't propagate
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest


def _make_callback_with_history(n_iters: int = 30, k: int = 4, seed: int = 0):
    """Synthesize a UniversalCallback with enough slice_shard_score_history for the plot."""
    rng = np.random.default_rng(seed)
    cb = SimpleNamespace()
    # Heteroscedastic-looking trajectory: mean drops first, then std rises late (overfit).
    shard_history = []
    aggregate_history = []
    for t in range(n_iters):
        center = 1.0 - 0.5 * (1.0 - np.exp(-t / 8.0))
        spread = 0.05 + 0.02 * max(0, t - 12)
        shards = (rng.normal(center, spread, k)).tolist()
        shard_history.append(shards)
        aggregate_history.append(float(np.mean(shards)))
    cb.slice_shard_score_history = shard_history
    cb.slice_aggregate_history = aggregate_history
    cb.best_iter = int(np.argmin(aggregate_history))
    return cb


def _make_ctx() -> SimpleNamespace:
    return SimpleNamespace(metadata={})


def _make_config(**overrides):
    """SimpleNamespace stand-in for SliceStableESConfig (the artifact only reads attrs)."""
    base = dict(
        pareto_plot_enabled=True,
        pareto_plot_backends=["matplotlib"],  # plotly default works too but keeps tests fast
        pareto_plot_formats={"matplotlib": ["png"], "plotly": ["html"]},
        pareto_plot_min_iterations=10,
        pareto_plot_show_alt_quantiles=[0.5, 0.9],
        pareto_persist_shard_history=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_pareto_plot_default_on_writes_matplotlib_png(tmp_path) -> None:
    from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact

    cb = _make_callback_with_history(n_iters=30)
    ctx = _make_ctx()
    cfg = _make_config()
    meta = generate_pareto_artifact(
        ctx,
        target_name="y",
        model_name="lgb",
        callback=cb,
        config=cfg,
        output_dir=str(tmp_path),
        direction="min",
    )
    assert meta is not None
    assert "matplotlib" in meta["pareto_plot_paths"]
    png_path = meta["pareto_plot_paths"]["matplotlib"]
    assert os.path.exists(png_path), f"PNG not written: {png_path}"
    assert os.path.getsize(png_path) > 100, "PNG suspiciously empty"
    # Metadata mirrored into ctx
    assert "slice_stable_es" in ctx.metadata
    assert ctx.metadata["slice_stable_es"]["y"]["lgb"] is meta


def test_pareto_plot_alternative_selections_populated(tmp_path) -> None:
    from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact

    cb = _make_callback_with_history(n_iters=40)
    ctx = _make_ctx()
    cfg = _make_config(pareto_plot_show_alt_quantiles=[0.5, 0.7, 0.9, 0.95])
    meta = generate_pareto_artifact(
        ctx,
        target_name="t",
        model_name="cb",
        callback=cb,
        config=cfg,
        output_dir=str(tmp_path),
        direction="min",
    )
    assert meta is not None
    alts = meta["alternative_selections"]
    # All requested quantiles get an entry (unless the frontier is degenerate)
    assert set(alts.keys()) <= {"0.5", "0.7", "0.9", "0.95"}
    assert len(alts) >= 2, f"expected at least 2 alt picks; got {alts}"


def test_pareto_plot_disabled_emits_nothing(tmp_path) -> None:
    from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact

    cb = _make_callback_with_history(n_iters=30)
    ctx = _make_ctx()
    cfg = _make_config(pareto_plot_enabled=False)
    meta = generate_pareto_artifact(
        ctx,
        target_name="y",
        model_name="lgb",
        callback=cb,
        config=cfg,
        output_dir=str(tmp_path),
    )
    assert meta is None
    assert "slice_stable_es" not in ctx.metadata
    assert os.listdir(str(tmp_path)) == [], "directory must remain empty"


def test_pareto_plot_short_run_skip(tmp_path, caplog) -> None:
    import logging
    from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact

    cb = _make_callback_with_history(n_iters=5)  # < min_iterations=10
    ctx = _make_ctx()
    cfg = _make_config()
    with caplog.at_level(logging.INFO):
        meta = generate_pareto_artifact(
            ctx,
            target_name="y",
            model_name="xgb",
            callback=cb,
            config=cfg,
            output_dir=str(tmp_path),
        )
    assert meta is None
    assert any("min" in r.message for r in caplog.records)


def test_pareto_plot_persist_shard_history(tmp_path) -> None:
    from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact

    cb = _make_callback_with_history(n_iters=30, k=5)
    ctx = _make_ctx()
    cfg = _make_config(pareto_persist_shard_history=True)
    meta = generate_pareto_artifact(
        ctx,
        target_name="y",
        model_name="lgb",
        callback=cb,
        config=cfg,
        output_dir=str(tmp_path),
    )
    assert meta is not None
    parquet_path = meta["shard_history_path"]
    assert parquet_path is not None and os.path.exists(parquet_path)
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    # 30 iters * 5 shards = 150 rows
    assert len(df) == 150
    assert set(df.columns) >= {"iter", "shard_idx", "score"}


def test_pareto_plot_plotly_backend_writes_html(tmp_path) -> None:
    pytest.importorskip("plotly")
    from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact

    cb = _make_callback_with_history(n_iters=30)
    ctx = _make_ctx()
    cfg = _make_config(pareto_plot_backends=["plotly"], pareto_plot_formats={"plotly": ["html"]})
    meta = generate_pareto_artifact(
        ctx,
        target_name="y",
        model_name="lgb",
        callback=cb,
        config=cfg,
        output_dir=str(tmp_path),
    )
    assert meta is not None
    html_path = meta["pareto_plot_paths"]["plotly"]
    assert os.path.exists(html_path)
    contents = open(html_path, encoding="utf-8").read()
    assert "Pareto" in contents and "plotly" in contents.lower()


def test_pareto_plot_save_failure_warns_doesnt_raise(tmp_path, caplog, monkeypatch) -> None:
    import logging
    from mlframe.training.slicing import _slice_pareto_plot as mod

    cb = _make_callback_with_history(n_iters=20)
    ctx = _make_ctx()
    cfg = _make_config(pareto_plot_backends=["matplotlib"])

    def _bad_render(*a, **k):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(mod, "_render_matplotlib", _bad_render)
    with caplog.at_level(logging.WARNING):
        meta = mod.generate_pareto_artifact(
            ctx,
            target_name="y",
            model_name="lgb",
            callback=cb,
            config=cfg,
            output_dir=str(tmp_path),
        )
    # meta still returned, but no paths saved.
    assert meta is not None
    assert meta["pareto_plot_paths"] == {}
    assert any("failed" in r.message.lower() for r in caplog.records)

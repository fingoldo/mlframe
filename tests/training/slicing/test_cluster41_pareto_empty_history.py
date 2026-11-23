"""Cluster 41 regression: the Pareto artifact with no shard history returns None instead of raising (audit 2026-11-20, finding 41-F23)."""

from types import SimpleNamespace

from mlframe.training.slicing._slice_pareto_plot import generate_pareto_artifact


def test_empty_shard_history_with_zero_min_iterations_returns_none(tmp_path):
    """min_iterations=0 with an empty history used to raise in argmin / history[0]; it now returns None (41-F23)."""
    config = SimpleNamespace(pareto_plot_enabled=True, pareto_plot_min_iterations=0)
    callback = SimpleNamespace(slice_shard_score_history=[], slice_aggregate_history=[], best_iter=None)
    out = generate_pareto_artifact(
        ctx=SimpleNamespace(metadata={}), target_name="t", model_name="m", callback=callback,
        config=config, output_dir=str(tmp_path),
    )
    assert out is None

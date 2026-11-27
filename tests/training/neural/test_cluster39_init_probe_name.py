"""Cluster 39 regression: the per-fit init probe has a name that says what it does (audit 2026-11-20, finding 39-F62).

It used to be named as a bench-profiling helper although it runs the degenerate-init check on every fit. The name now matches its job.
"""

from mlframe.training.neural import _flat_generate_helpers as helpers


def test_probe_has_the_descriptive_name():
    """The init probe is exported under the name that describes it (39-F62)."""
    assert hasattr(helpers, "_generate_mlp_every_fit_init_probe")
    assert not hasattr(helpers, "_generate_mlp_every_fit_bench_profiling")

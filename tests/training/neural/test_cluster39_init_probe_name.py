"""Cluster 39 regression: the per-fit init probe has a name that says what it does (audit 2026-11-20, finding 39-F62).

It used to be named as a bench-profiling helper although it runs the degenerate-init check on every fit. The name now matches its job, and the
old name is gone from the package.
"""

from pathlib import Path

from mlframe.training.neural import _flat_generate_helpers as helpers


def test_probe_has_the_descriptive_name():
    """The init probe is exported under the name that describes it (39-F62)."""
    assert hasattr(helpers, "_generate_mlp_every_fit_init_probe")
    assert not hasattr(helpers, "_generate_mlp_every_fit_bench_profiling")


def test_old_name_is_gone_from_the_package_sources():
    """No source file still refers to the old bench-profiling name (39-F62)."""
    root = Path(helpers.__file__).resolve().parent
    hits = [p.name for p in root.rglob("*.py") if "_generate_mlp_every_fit_bench_profiling" in p.read_text(encoding="utf-8")]
    assert hits == []

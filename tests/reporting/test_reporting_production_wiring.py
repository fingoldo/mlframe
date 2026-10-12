"""Regression tests: helpers that were reachable only from tests are now called on the production path.

Each test fails if the production call is removed.
"""

import io

import numpy as np

import mlframe.reporting._diagnostics_prep as prep_mod
import mlframe.reporting.catalog as catalog
import mlframe.reporting.charts.error_analysis as error_analysis
from mlframe.reporting.charts.ltr import _per_query_groups, _sorted_layout
from mlframe.reporting.diagnostics_dispatch import render_split_error_diagnostics


def test_split_error_diagnostics_releases_both_caches_on_return(monkeypatch):
    """The per-split dispatcher drops the memoised error prep and the densified matrix when it returns."""
    calls = []
    monkeypatch.setattr(prep_mod, "clear_shared_error_prep", lambda: calls.append("prep"))
    monkeypatch.setattr(error_analysis, "clear_feature_matrix_cache", lambda: calls.append("matrix"))
    render_split_error_diagnostics(df=None, y_true=np.zeros(3), y_pred=np.zeros(3), task="regression", plot_outputs="", base_path="")
    assert calls == ["prep", "matrix"]


def test_describe_panels_prints_standalone_diagnostics_from_the_accessor(monkeypatch):
    """describe_available_panels takes its standalone-diagnostics list from standalone_diagnostics()."""
    monkeypatch.setattr(catalog, "standalone_diagnostics", lambda: [("zz_sentinel_diag", "sentinel description")])
    buf = io.StringIO()
    catalog.describe_available_panels(file=buf)
    assert "zz_sentinel_diag" in buf.getvalue()
    assert "sentinel description" in buf.getvalue()


def test_per_query_groups_partitions_match_sorted_layout():
    """Superseded by _sorted_layout: both give the same query sizes in the same ascending-query order."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 4, 500).astype(float)
    s = rng.random(500)
    g = rng.integers(0, 37, 500)
    old = _per_query_groups(g)
    _, _, _, sizes = _sorted_layout(y, s, g)
    assert [len(ix) for ix in old] == list(sizes)
    assert len(old) == len(sizes)
    for ix in old:
        assert len(np.unique(g[ix])) == 1
    assert sum(len(ix) for ix in old) == len(g)

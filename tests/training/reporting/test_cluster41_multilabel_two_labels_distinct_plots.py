"""Cluster 41 regression: a two-label multilabel run gets one chart file per label (audit 2026-11-20, finding 41-F15)."""

from mlframe.training.reporting._reporting_probabilistic import _class_plot_paths


def test_two_label_multilabel_charts_do_not_share_a_file():
    """Label 0 and label 1 of a two-label multilabel run get distinct paths; the second no longer overwrites the first (41-F15)."""
    p0, _ = _class_plot_paths("rel_stub", "label_a", 0, 2, multilabel=True)
    p1, _ = _class_plot_paths("rel_stub", "label_b", 1, 2, multilabel=True)
    assert p0 != p1


def test_binary_run_keeps_the_bare_perfplot_name():
    """Control: a plain binary run (not multilabel) still writes the bare perfplot name."""
    p, base = _class_plot_paths("rel_stub", "pos", 1, 2)
    assert p == "rel_stub_perfplot.png"
    assert base == "rel_stub_perfplot"

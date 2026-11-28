"""Cluster 41 regression: the report's split label is a known split token or explicit, never an arbitrary last underscore token (audit 2026-11-20, finding 41-F36)."""

from mlframe.training.reporting._reporting_diagnostics import _split_label_of


def test_known_split_suffixes_are_read_and_other_names_give_none():
    """``.../model_val`` is 'val'; ``.../run_v2`` and a bare name are None so no split-restricted diagnostic is skipped; an explicit split wins."""
    assert _split_label_of("out/dir/model_val") == "val"
    assert _split_label_of("out/dir/model_TEST") == "test"
    assert _split_label_of("out/dir/run_v2") is None
    assert _split_label_of("out/dir/run") is None
    assert _split_label_of("") is None
    assert _split_label_of("out/dir/run_v2", explicit="Test") == "test"

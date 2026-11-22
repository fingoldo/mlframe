"""Audit 2026-11-20 findings 34-F25 and 34-F31: skipped-spec wording and the auto-base top-k log count.

34-F31: when the hints already fill top_k, the log reported one MI candidate that was not in the output.
34-F25: the holdout and OOF selection messages said "fell back to its default" where the code skips the spec.
"""

import logging

from mlframe.training.composite.discovery._auto_base_prep import combine_hint_and_ranked


def test_full_hint_list_logs_zero_mi_candidates(caplog):
    """Two hints fill top_k=2: the output is the hints and the log says 0 MI, not 1."""
    ranked = [(0.9, "a"), (0.5, "b"), (0.1, "c")]
    with caplog.at_level(logging.INFO):
        out = combine_hint_and_ranked(ranked, ["h1", "h2"], 2)
    assert out == ["h1", "h2"]
    assert any("(2 hint, 0 MI)" in r.getMessage() for r in caplog.records)


def test_partial_hint_list_still_fills_with_mi():
    """One hint and top_k=3 leaves two MI slots, taken in rank order."""
    ranked = [(0.9, "h1"), (0.5, "b"), (0.1, "c")]
    assert combine_hint_and_ranked(ranked, ["h1"], 3) == ["h1", "b", "c"]

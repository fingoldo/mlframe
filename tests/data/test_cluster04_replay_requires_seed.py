"""Cluster 04 regression: a record without dataset_seed is refused, not replayed as seed 0 (audit 2026-11-20, finding 04-F30)."""

import pytest

from mlframe.data.datasets.replay import replay_from_record


def test_record_without_dataset_seed_is_refused():
    """A missing seed raises rather than silently rebuilding seed 0's bed (04-F30)."""
    with pytest.raises(ValueError, match="no dataset_seed"):
        replay_from_record({"scenario": "anything", "spec_hash": "x"})

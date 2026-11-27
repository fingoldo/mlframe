"""Cluster 37 regression: the group-slice helper has a name that says what it does (audit 2026-11-20, finding 37-F51).

It was named after a source-truth and DID-skip concern it does not have. It is now _group_train_slice.
"""

from mlframe.training.core import _phase_composite_discovery as pcd


def test_group_slice_has_the_descriptive_name():
    """The helper is exported under the descriptive name and the old name is gone (37-F51)."""
    assert callable(pcd._group_train_slice)
    assert not hasattr(pcd, "_run_composite_targ_source_truth_did_skip")

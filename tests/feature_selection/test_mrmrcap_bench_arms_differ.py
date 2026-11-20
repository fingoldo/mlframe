"""The MRMR-cap bench compares two different caps (audit 2026-11-20, finding 12-F19)."""
from mlframe.feature_selection._benchmarks.fs_hybrid import round4_hybrid_mrmrcap_bench as bench


def test_bench_arms_use_distinct_caps():
    """The earlier None-vs-250 pair ran identical fits because MRMR's default is 250; the arms must differ."""
    caps = [cap for _tag, cap in bench.ARMS]
    assert len(set(caps)) == len(caps) == 2
    assert None not in caps

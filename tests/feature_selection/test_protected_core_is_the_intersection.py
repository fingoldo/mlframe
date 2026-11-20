"""The protected-RFECV core is the columns both priors accept, not either (audit 2026-11-20, finding 12-F24)."""
from mlframe.feature_selection._benchmarks.fs_hybrid.round4_prior_protected_rfecv_bench import agreed_core


def test_core_is_the_intersection_of_the_two_selectors():
    """A column only one selector accepted must not be pinned."""
    columns = ["a", "b", "c", "d"]
    assert agreed_core(columns, {"a", "b"}, {"b", "c"}) == ["b"]


def test_empty_prior_gives_an_empty_core():
    """If one selector accepted nothing, nothing is agreed and nothing is pinned."""
    assert agreed_core(["a", "b"], {"a"}, set()) == []

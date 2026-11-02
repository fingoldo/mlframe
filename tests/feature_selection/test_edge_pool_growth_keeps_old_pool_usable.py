"""Growing the shared edge pool must not break a caller still holding the previous pool (tracker 2026-11-02 wave 8, RES8-1)."""

from mlframe.feature_selection.filters import _adaptive_nbins as an


def test_old_pool_still_accepts_work_after_the_pool_grows():
    """A caller that fetched the pool before another thread grew it maps on it afterwards; it used to raise after shutdown."""
    old = an._edge_pool(1)
    new = an._edge_pool(an._EDGE_POOL_WORKERS + 2)
    assert new is not old
    assert list(old.map(lambda v: v + 1, [1, 2, 3])) == [2, 3, 4]

"""Worker threads of the edge pool see the fit's scoring settings (tracker 2026-11-02 wave 14, PAR14-1)."""

from mlframe.feature_selection.filters import _adaptive_nbins, _edge_pool
from mlframe.feature_selection.filters.info_theory._state_and_dispatch import set_su_normalization, use_su_normalization


def test_su_normalization_reaches_pool_workers(monkeypatch):
    """SU is a thread-local flag set by MRMR.fit; a fresh worker read False, so concurrent stages scored with raw MI."""
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda n_jobs, n: 4)
    set_su_normalization(True)
    try:
        seen = _edge_pool.map_on_edge_pool(lambda _item: use_su_normalization(), list(range(8)), min_items=4)
    finally:
        set_su_normalization(False)
    assert seen == [True] * 8
    assert use_su_normalization() is False

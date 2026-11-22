"""Audit 2026-11-20 finding 28-F30: eviction counts and drops entries another process wrote to the cache directory.

Such entries were on disk, counted in the byte total, but absent from this process's ledger, so they were never evicted. The test writes
an entry file directly into the directory, runs the adoption step, and checks it is now in the ledger as the oldest.
"""

from mlframe.training import suite_artefact_cache as sac


def test_foreign_entry_is_adopted_into_the_ledger_as_oldest(tmp_path):
    """A .pkl written outside this cache object is adopted at the oldest end of the LRU."""
    cache = sac.SuiteArtefactCache(str(tmp_path))
    (tmp_path / "foreign_key.pkl").write_bytes(b"x" * 10)
    cache._adopt_unknown_disk_entries()
    assert "foreign_key" in cache._lru
    assert next(iter(cache._lru)) == "foreign_key"

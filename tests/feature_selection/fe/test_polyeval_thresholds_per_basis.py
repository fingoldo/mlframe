"""The polyeval crossovers are read per basis: each basis gets its own region of the cache entry, not the first region for all of them."""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters.hermite_fe import _hermite_oracle as O

_REGIONS = [
    {"basis": "hermite", "n_samples_max": None, "par_threshold": 100_000, "cuda_threshold": 100_000_000},
    {"basis": "legendre", "n_samples_max": None, "par_threshold": 200_000, "cuda_threshold": 3_000_000},
    {"basis_eq": "chebyshev", "basis": "chebyshev", "n_samples_max": None, "par_threshold": 50_000, "cuda_threshold": 100_000_000},
]


class _FakeCache:
    """A cache whose matcher ignores the bare ``basis`` field (as the real one does) and answers with the first region."""

    def __init__(self, regions):
        self._regions = regions

    def get_regions(self, name):
        """The regions of the entry."""
        return list(self._regions)

    def lookup(self, name, **dims):
        """The first region, whatever the dims say (the defect this module works around)."""
        return self._regions[0] if self._regions else None


@pytest.fixture
def cache(monkeypatch):
    """Install a fake cache holding ``_REGIONS``."""
    import pyutilz.performance.kernel_tuning.cache as C

    holder = {"regions": _REGIONS}
    monkeypatch.setattr(C.KernelTuningCache, "load_or_create", classmethod(lambda cls: _FakeCache(holder["regions"])))
    return holder


def test_every_basis_reads_its_own_region(cache):
    """Hermite, legendre and chebyshev (written with and without ``basis_eq``) each get their own thresholds."""
    assert O._lookup_polyeval_thresholds("hermite", 1_000_000) == (100_000, 100_000_000)
    assert O._lookup_polyeval_thresholds("legendre", 1_000_000) == (200_000, 3_000_000)
    assert O._lookup_polyeval_thresholds("chebyshev", 1_000_000) == (50_000, 100_000_000)


def test_a_basis_without_a_region_takes_the_source_defaults_not_the_first_region(cache):
    """Laguerre has no region in this entry: the source defaults apply, hermite's thresholds do not."""
    assert O._lookup_polyeval_thresholds("laguerre", 1_000_000) == (O._PAR_THRESHOLD, O._CUDA_THRESHOLD)


def test_an_entry_without_any_basis_field_still_answers_through_lookup(cache):
    """A legacy entry that carries no basis keeps the matcher's answer."""
    cache["regions"] = [{"n_samples_max": None, "par_threshold": 7, "cuda_threshold": 9}]
    assert O._lookup_polyeval_thresholds("hermite", 10) == (7, 9)


def test_the_sweep_tags_each_region_with_basis_eq():
    """New entries carry ``basis_eq`` so the cache matcher itself separates the bases."""
    import inspect

    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import _auto_tune_sweeps_a as S

    assert '"basis_eq": basis' in inspect.getsource(S._run_sweep_polyeval)

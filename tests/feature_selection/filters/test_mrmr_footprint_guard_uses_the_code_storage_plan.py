"""The RAM-footprint guard counts the bytes per cell of the codes the fit will allocate (one byte for 10 bins), not a fixed int32."""

from __future__ import annotations

import types

import pytest

from mlframe.feature_selection.filters import _mrmr_validate_transform as vt
from tests.feature_selection.binning_cases import binning_kwargs


def test_the_default_settings_store_one_byte_a_cell_for_every_strategy():
    """A default MRMR plans int8 codes whatever the numeric binning strategy (MDLP depth and the adaptive ceiling follow the 127 cap); the plan widens only when the caps are lifted."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    for strategy in ("mdlp", "knuth", "bayesian_blocks", "fixed"):
        assert vt._planned_code_itemsize(MRMR(**binning_kwargs(strategy))) == 1
    assert vt._planned_code_itemsize(MRMR(binning_scheme="mdlp", max_categorical_cardinality=255)) == 2
    assert vt._planned_code_itemsize(MRMR(binning_scheme="uniform", nbins_rule="knuth", nbins_cap=1000, max_adaptive_nbins=1000, max_categorical_cardinality=None)) >= 2


def test_an_object_without_the_binning_settings_is_an_error_not_an_int32_guess():
    """The plan is read from the estimator's own settings; an object that lacks them raises instead of being counted as int32."""
    with pytest.raises(AttributeError):
        vt._planned_code_itemsize(types.SimpleNamespace())


class _Shape:
    """A frame stand-in with only a shape."""

    def __init__(self, n, p):
        self.shape = (n, p)


def test_a_one_gigacell_frame_passes_with_byte_codes_and_is_refused_as_int32(monkeypatch):
    """1e9 cells on a host with 6.7 GB free: 1 GB of byte codes is under half of the free RAM, 4 GB of int32 is not."""
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=int(6.7e9)))
    vt._check_fit_footprint(_Shape(1_000_000, 1000), 1)  # no exception
    with pytest.raises(ValueError, match="refusing to allocate"):
        vt._check_fit_footprint(_Shape(1_000_000, 1000), 4)

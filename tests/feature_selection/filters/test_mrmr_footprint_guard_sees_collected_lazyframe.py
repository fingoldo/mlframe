"""A polars LazyFrame is measured by the RAM-footprint guard once it has been collected."""

from __future__ import annotations

import types
import warnings

import numpy as np
import polars as pl
import pytest

from mlframe.feature_selection.filters import _mrmr_validate_transform as vt


def _stub():
    """The minimal MRMR-like object ``_validate_inputs`` reads."""
    return types.SimpleNamespace(_validate_string_params=lambda: None, quantization_nbins=10, interactions_max_order=2, fe_max_steps=0)


def test_lazyframe_is_checked_against_free_ram_after_collect(monkeypatch):
    """With almost no free RAM a collected lazy frame is refused, as an eager frame of the same size would be."""
    import psutil

    monkeypatch.setattr(vt, "_snapshot_fit_dtypes", lambda self, X: None)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=1000))
    lf = pl.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)}).lazy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="refusing to allocate"):
            vt._validate_inputs(_stub(), lf, np.arange(100) % 2)

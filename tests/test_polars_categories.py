"""categorical_labels returns the label of every physical code on the installed polars, with and without Series.cat.get_categories."""

from __future__ import annotations

import polars as pl

from mlframe._polars_categories import categorical_labels


class _NoGetCategories:
    """A stand-in for the polars 2 ``.cat`` namespace: everything but ``get_categories``."""

    def __init__(self, series: pl.Series) -> None:
        """Wrap the real series so the label lookup has to go through the codes."""
        self._series = series

    def __getattr__(self, name: str):
        """Hide ``get_categories`` and delegate the rest to the real namespace."""
        if name == "get_categories":
            raise AttributeError(name)
        return getattr(self._series.cat, name)


class _SeriesWithoutGetCategories:
    """A Series look-alike whose ``cat`` namespace has no ``get_categories``, as on polars 2."""

    def __init__(self, series: pl.Series) -> None:
        """Keep the real series and expose the restricted namespace."""
        self._series = series
        self.cat = _NoGetCategories(series)

    def to_physical(self) -> pl.Series:
        """Physical codes of the wrapped series."""
        return self._series.to_physical()

    def cast(self, dtype):
        """Cast of the wrapped series."""
        return self._series.cast(dtype)


def test_labels_are_indexed_by_physical_code_with_nulls_ignored() -> None:
    """Every row's label equals ``labels[code]`` for the series' own codes, on the installed polars."""
    s = pl.Series(["b", "a", "b", None, "c"], dtype=pl.Categorical)
    labels = categorical_labels(s)
    codes = s.to_physical().to_list()
    assert [labels[c] for c in codes if c is not None] == ["b", "a", "b", "c"]


def test_labels_are_rebuilt_from_the_codes_when_get_categories_is_missing() -> None:
    """The polars 2 path (no get_categories) gives the same code-to-label mapping, and an unused code gets an empty label."""
    s = pl.Series(["b", "a", "b", None, "c"], dtype=pl.Categorical)
    labels = categorical_labels(_SeriesWithoutGetCategories(s))
    codes = s.to_physical().to_list()
    assert [labels[c] for c in codes if c is not None] == ["b", "a", "b", "c"]
    assert categorical_labels(_SeriesWithoutGetCategories(pl.Series([None, None], dtype=pl.Categorical))) == []

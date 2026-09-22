"""Getting a generated dataset into the frame library a consumer actually uses, without losing the truth.

The generator returns pandas because every consumer in this repository -- the arms, the protocol layer,
the downstream panel -- already takes pandas. Consumers outside it do not, and the conversion has two
traps that are easy to hit and hard to notice.

**Categoricals.** polars' ``Categorical`` resolves its levels through a process-wide string cache, so two
frames built independently can hold codes that mean different things; joining or comparing them gives
wrong answers rather than an error. Every categorical column here therefore becomes a ``pl.Enum`` with an
explicit level list, which the generator knows by construction because the levels are in the spec.

**Missing values.** A masked cell is ``NaN`` in pandas and ``null`` in polars, and they are not the same
thing for a float column: pandas cannot tell a masked value from a computed NaN, polars can. The
conversion keeps them distinct where the source can, and says so in the returned note where it cannot.

Nothing here converts the truth record. The truth is addressed by column NAME, and every conversion below
preserves names exactly, so a truth record built against the pandas frame is valid against the polars one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlframe.data.datasets.spec import DatasetSpec

logger = logging.getLogger(__name__)

__all__ = ["to_polars", "to_numpy", "feature_matrix"]


def to_polars(frame: pd.DataFrame, spec: Optional[DatasetSpec] = None) -> Tuple[Any, List[str]]:
    """Convert to polars, turning every categorical into a ``pl.Enum`` with explicit levels.

    Args:
        frame: The generated frame.
        spec: The specification it came from, which is where the declared level lists live. Without it the
            levels are read off the column, which is correct for one frame and NOT stable across two: a
            level absent from this draw would be absent from this frame's enum and present in another's.

    Returns:
        ``(polars_frame, notes)``. The notes name every conversion that could not be exact, so a caller
        can decide rather than discover.

    Raises:
        ImportError: If polars is not installed, rather than silently handing back pandas under a name
            that promises otherwise.
    """
    import polars as pl

    declared: Dict[str, Tuple[str, ...]] = {}
    if spec is not None:
        declared = {feature.name: tuple(feature.levels or ()) for feature in spec.features if feature.dtype == "category"}

    notes: List[str] = []
    columns: Dict[str, Any] = {}
    for name in frame.columns:
        series = frame[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            levels = list(declared.get(str(name)) or [str(value) for value in series.cat.categories])
            if not declared.get(str(name)):
                notes.append(f"{name!r}: levels read off this draw rather than from a spec, so another draw's enum may differ")
            values = [None if pd.isna(value) else str(value) for value in series]
            columns[str(name)] = pl.Series(str(name), values, dtype=pl.Enum(levels))
        elif str(series.dtype) == "Int64":
            # pandas' nullable integer: the nulls are real and polars can hold them, so they stay null
            # rather than becoming a sentinel or a float NaN.
            columns[str(name)] = pl.Series(str(name), [None if value is pd.NA else int(value) for value in series], dtype=pl.Int64)
        else:
            columns[str(name)] = pl.Series(str(name), series.to_numpy())
    return pl.DataFrame(columns), notes


def to_numpy(frame: pd.DataFrame, spec: Optional[DatasetSpec] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert to a dense float matrix, returning ``(matrix, column_names, notes)``.

    Categoricals become their integer codes, which is what every estimator taking a bare matrix does with
    them anyway -- and the note says so, because an ordinal reading of a nominal column is a real modelling
    choice and not a formatting detail. Masked cells stay ``NaN``: a sentinel would be a value some method
    treats as a very small number.
    """
    notes: List[str] = []
    names: List[str] = []
    columns: List[np.ndarray] = []
    for name in frame.columns:
        series = frame[name]
        names.append(str(name))
        if isinstance(series.dtype, pd.CategoricalDtype):
            notes.append(f"{name!r}: encoded as integer codes, so a consumer reading this matrix treats a NOMINAL column as ordinal")
            columns.append(np.asarray(series.cat.codes, dtype=np.float64))
        else:
            columns.append(np.asarray(pd.to_numeric(series, errors="coerce"), dtype=np.float64))
    if spec is not None and list(spec.feature_names()) != names:
        notes.append("the frame's column order differs from the spec's, which is expected when a bed shuffles its columns per seed")
    return np.column_stack(columns) if columns else np.empty((len(frame), 0)), names, notes


def feature_matrix(frame: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """Return the dense float matrix for a named subset, in the order given.

    The order is the caller's, not the frame's. A selector that returns a ranked list expects the matrix to
    follow that ranking, and silently re-sorting into frame order would change what a downstream model
    with a column-order-sensitive tie-break does.

    Raises:
        KeyError: If a name is not in the frame, so a typo fails here rather than producing a matrix one
            column narrower than the caller believes.
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"these columns are not in the frame: {missing}")
    matrix, _names, _notes = to_numpy(frame[columns])
    return matrix

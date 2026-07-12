"""Rounding-based numeric x categorical interaction builder for Kaggle-style tabular data.

COMPETITION / EXPLORATORY ONLY — NEVER wire this into production defaults.

Source: 3rd_bnp-paribas-cardif-claims-management.md — heavily rounding a numeric
feature to K decimals and concatenating it with an existing categorical column
to build a synthetic composite categorical ("numericalrounded_categorical"),
then feeding the composite through the same frequency/target-likelihood
encoding pipeline used for pure categoricals.

This is a mild variant of standard binning+concat encoding. It is filed under
``mlframe.competition`` (rather than ``mlframe.feature_engineering``) because
the write-up's motivation is squeezing extra leaderboard signal out of a
narrow, high-cardinality composite that is prone to overfitting on small/held-
out competition folds; in production this kind of unconstrained categorical
cross should go through the project's vetted interaction/CMI-gated feature
selection pipeline (``mlframe.feature_selection.filters``), not an ad-hoc
rounding-and-concat step.

Note: the companion idea in the same tracker entry, "IntegerDenominatorRecovery"
(recovering the true integer denominator behind anonymization-scaled floats),
is already implemented by :class:`mlframe.competition.float_precision_denoise.FloatPrecisionDenoiser`
— this module implements only the interaction-builder half.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["RoundedNumericCategoricalInteraction"]


class RoundedNumericCategoricalInteraction:
    """Builds a composite categorical from a rounded numeric column and a categorical column.

    COMPETITION / EXPLORATORY ONLY — NEVER wire into production defaults.

    Discretizes a numeric column to ``decimals`` decimal places, stringifies it,
    and concatenates it (via ``sep``) with a categorical column's string values to
    produce a new synthetic categorical Series. The composite is intended to be
    fed through standard frequency- or target-likelihood-encoding afterward (not
    provided here — this class only builds the composite key), so that any
    interaction-only signal between the rounded numeric value and the categorical
    level (invisible to either column alone) becomes learnable by a downstream
    encoder/model.

    This is a narrow, high-cardinality categorical cross that is prone to
    overfitting on small folds and is not validated against leakage/CMI gates
    the way ``mlframe.feature_selection.filters`` interactions are — use only
    for exploratory Kaggle-style feature hunting.

    Parameters
    ----------
    decimals:
        Number of decimal places the numeric column is rounded to before
        stringification. ``0`` rounds to whole numbers.
    sep:
        Separator string used to join the rounded-numeric and categorical parts.
    missing_token:
        Token substituted for NaN/None entries in either input column, so that
        missing values themselves become a stable joint level rather than
        propagating NaN into the composite.
    """

    __slots__ = ("decimals", "sep", "missing_token")

    def __init__(self, decimals: int = 2, sep: str = "|", missing_token: str = "<NA>") -> None:
        if decimals < 0:
            raise ValueError(f"decimals must be >= 0, got {decimals}")
        if not sep:
            raise ValueError("sep must be a non-empty string")
        self.decimals = decimals
        self.sep = sep
        self.missing_token = missing_token

    def _stringify_numeric(self, numeric: npt.NDArray[np.float64]) -> npt.NDArray[np.str_]:
        rounded = np.round(numeric.astype(np.float64), self.decimals)
        out = np.empty(rounded.shape, dtype=object)
        finite_mask = np.isfinite(rounded)
        fmt = f"%.{self.decimals}f"
        out[finite_mask] = np.char.mod(fmt, rounded[finite_mask])
        out[~finite_mask] = self.missing_token
        return out.astype(str)

    def _stringify_categorical(self, categorical: Union[pd.Series, npt.NDArray]) -> npt.NDArray[np.str_]:
        series = categorical if isinstance(categorical, pd.Series) else pd.Series(categorical)
        filled = series.where(series.notna(), self.missing_token)
        return np.asarray(filled.astype(str).to_numpy())

    def transform(
        self,
        numeric: Union[pd.Series, npt.NDArray, Sequence[float]],
        categorical: Union[pd.Series, npt.NDArray, Sequence],
        name: Optional[str] = None,
    ) -> pd.Series:
        """Build the composite rounded-numeric x categorical Series.

        Parameters
        ----------
        numeric:
            Numeric column to discretize.
        categorical:
            Categorical column to join against.
        name:
            Optional name for the returned Series; defaults to a name derived
            from the two input names when both are ``pd.Series``, else
            ``"rounded_cat_interaction"``.

        Returns
        -------
        pd.Series
            String-typed composite categorical, same length and index (when
            available) as the inputs.
        """
        numeric_arr = np.asarray(numeric, dtype=np.float64)
        cat_arr = self._stringify_categorical(categorical)
        if numeric_arr.shape[0] != cat_arr.shape[0]:
            raise ValueError(f"numeric and categorical must have equal length, got {numeric_arr.shape[0]} vs {cat_arr.shape[0]}")

        numeric_str = self._stringify_numeric(numeric_arr)
        composite = np.char.add(np.char.add(numeric_str.astype(str), self.sep), cat_arr.astype(str))

        index = None
        if isinstance(numeric, pd.Series):
            index = numeric.index
        elif isinstance(categorical, pd.Series):
            index = categorical.index

        if name is None:
            if isinstance(numeric, pd.Series) and isinstance(categorical, pd.Series):
                name = f"{numeric.name}{self.sep}{categorical.name}"
            else:
                name = "rounded_cat_interaction"

        return pd.Series(composite, index=index, name=name, dtype="object")

    def fit_transform(
        self,
        numeric: Union[pd.Series, npt.NDArray, Sequence[float]],
        categorical: Union[pd.Series, npt.NDArray, Sequence],
        name: Optional[str] = None,
    ) -> pd.Series:
        """Stateless alias for :meth:`transform` (kept for encoder-pipeline API symmetry)."""
        return self.transform(numeric, categorical, name=name)

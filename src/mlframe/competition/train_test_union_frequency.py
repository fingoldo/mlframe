"""Train+test-union frequency/groupby-count encoding for drifting categorical data.

*** COMPETITION / EXPLORATORY ONLY - NOT PRODUCTION CODE ***

Source idea (9th place, Microsoft Malware Prediction): "if there are 14 machines with
same versions, we add the number 14 ... Our intention with this strategy is remove
time-dependence." The trick is to compute a categorical value's frequency/occurrence
count over the UNION of train and test rows (not train alone), which decorrelates the
resulting feature from the train/test time split for adversarial-drift-prone columns
such as OS/driver/app version strings that are introduced and retired over time.

Why this is NOT production-safe:
    - You don't have "the test set" in production - only the Kaggle two-file
      offline-batch competition setting has a fixed, fully-observed test frame whose
      feature values (not labels) are legitimately available at feature-engineering
      time. In a real serving system you only ever see past data; there is no
      future-inclusive "union" to count over. Using this trick outside of that
      exact competition setup is a leakage / non-causal-features bug.
    - Even inside a competition, this is only sound when the test frame's feature
      columns are genuinely available offline before scoring (the standard Kaggle
      batch setup), not for online/streaming inference.

This module lives under ``mlframe.competition`` and is NEVER imported by any
production mlframe module, and NEVER re-exported from mlframe's top-level
``__init__.py``. Use it only for exploratory competition work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HIERARCHICAL_LEVEL_NAMES = ("major", "major_minor", "major_minor_patch")

# ``pd.Series.astype("string").str.split(sep)`` maps a missing entry (None/NaN) to ``pd.NA`` rather than
# an empty list, so a naive ``.map(len)``/slice-and-join over the split result crashes on any missing
# version-string row (a realistic input -- older records commonly lack a version string). Missing rows
# are routed to this single sentinel token at every hierarchical level instead, so they still get a
# (shared, consistent) train+test-union frequency rather than crashing or silently dropping out.
_MISSING_LEVEL_TOKEN = "__missing__"


def _union_value_counts(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    """Return value counts computed over the concatenation of the train and test series."""
    union = pd.concat([train_series, test_series], ignore_index=True)
    return union.value_counts()


def _frequency_encode_from_counts(series: pd.Series, counts: pd.Series) -> pd.Series:
    """Map each value in ``series`` to its frequency in ``counts``, returning a float64 series."""
    encoded = series.map(counts).astype("float64")
    return pd.Series(encoded.to_numpy(), index=series.index, name=series.name)


def train_test_union_frequency_encode(
    train_series: pd.Series,
    test_series: pd.Series,
    hierarchical_split_sep: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Frequency-encode a categorical column using counts pooled over train+test.

    *** COMPETITION / EXPLORATORY ONLY - see module docstring. NOT for production. ***

    Computes ``value_counts()`` over ``pd.concat([train_series, test_series])`` and maps
    each row (train and test separately) to that pooled count. This makes the feature
    insensitive to which side of the train/test split a value happens to fall on, which
    helps with adversarial-drift-prone categoricals (e.g. software version strings) whose
    train-only counts would otherwise be a stale/misleading proxy for the value's true
    prevalence at test time.

    If ``hierarchical_split_sep`` is given (e.g. ``"."`` for dotted version strings like
    ``"1.2.3"``), the value is additionally split into hierarchical prefixes
    (``major``, ``major_minor``, ``major_minor_patch``, ... one level per separator-joined
    prefix), each level is frequency-encoded separately over the train+test union, and the
    returned feature is the geometric mean of the per-level frequencies. Coarser levels
    (e.g. ``major``) still recover a useful frequency signal for a full version string that
    is completely novel in train but whose parent version family is not. Use
    ``train_test_union_frequency_encode_hierarchical_components`` to get each level's raw
    encoded series separately instead of the combined geometric mean.

    Parameters
    ----------
    train_series : pd.Series
        Train-side categorical column (string dtype recommended).
    test_series : pd.Series
        Test-side categorical column, same logical column as ``train_series``.
    hierarchical_split_sep : str | None
        If given, treat values as hierarchical strings (e.g. dotted version numbers) and
        combine per-level train+test-union frequencies via a geometric mean.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(train_encoded, test_encoded)``, each aligned to the input series' index, with
        the pooled train+test frequency count (or geometric-mean-of-hierarchical-levels
        frequency, if ``hierarchical_split_sep`` is given) as float values.
    """
    if hierarchical_split_sep is None:
        counts = _union_value_counts(train_series, test_series)
        return (
            _frequency_encode_from_counts(train_series, counts),
            _frequency_encode_from_counts(test_series, counts),
        )

    components = train_test_union_frequency_encode_hierarchical_components(train_series, test_series, hierarchical_split_sep)
    train_levels = [enc[0].to_numpy() for enc in components.values()]
    test_levels = [enc[1].to_numpy() for enc in components.values()]

    train_geomean = np.exp(np.mean(np.log(np.clip(np.stack(train_levels, axis=0), 1.0, None)), axis=0))
    test_geomean = np.exp(np.mean(np.log(np.clip(np.stack(test_levels, axis=0), 1.0, None)), axis=0))

    return (
        pd.Series(train_geomean, index=train_series.index, name=train_series.name),
        pd.Series(test_geomean, index=test_series.index, name=test_series.name),
    )


def train_test_union_frequency_encode_hierarchical_components(
    train_series: pd.Series,
    test_series: pd.Series,
    hierarchical_split_sep: str,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Return each hierarchical version-prefix level's train+test-union frequency encoding separately.

    *** COMPETITION / EXPLORATORY ONLY - see module docstring. NOT for production. ***

    Splits values like ``"1.2.3"`` on ``hierarchical_split_sep`` into cumulative prefixes
    (``"1"``, ``"1.2"``, ``"1.2.3"``, ...) and frequency-encodes each prefix level
    separately over the train+test union, so callers can use ``major``,
    ``major_minor``, ``major_minor_patch`` (etc., for deeper hierarchies) as independent
    sub-features rather than only the combined geometric mean returned by
    ``train_test_union_frequency_encode``.

    Parameters
    ----------
    train_series : pd.Series
        Train-side hierarchical string column (e.g. dotted version strings).
    test_series : pd.Series
        Test-side hierarchical string column, same logical column as ``train_series``.
    hierarchical_split_sep : str
        Separator to split values into hierarchical prefix levels on, e.g. ``"."``.

    Returns
    -------
    dict[str, tuple[pd.Series, pd.Series]]
        Maps a level name (``"level_1"``, ``"level_2"``, ... — using
        ``DEFAULT_HIERARCHICAL_LEVEL_NAMES`` names for the first three levels when
        present) to ``(train_encoded, test_encoded)`` frequency series for that
        prefix depth.
    """
    train_str = train_series.astype("string")
    test_str = test_series.astype("string")

    train_parts = train_str.str.split(hierarchical_split_sep)
    test_parts = test_str.str.split(hierarchical_split_sep)
    # Route missing entries (pd.NA, produced by .str.split on a None/NaN input) to a single sentinel
    # part-list instead of crashing on the .map(len)/slice-and-join calls below.
    train_parts = train_parts.map(lambda parts: parts if isinstance(parts, list) else [_MISSING_LEVEL_TOKEN])
    test_parts = test_parts.map(lambda parts: parts if isinstance(parts, list) else [_MISSING_LEVEL_TOKEN])

    max_depth = 0
    if len(train_parts) > 0:
        max_depth = max(max_depth, int(train_parts.map(len).max()))
    if len(test_parts) > 0:
        max_depth = max(max_depth, int(test_parts.map(len).max()))

    components: dict[str, tuple[pd.Series, pd.Series]] = {}
    for depth in range(1, max_depth + 1):
        level_name = DEFAULT_HIERARCHICAL_LEVEL_NAMES[depth - 1] if depth <= len(DEFAULT_HIERARCHICAL_LEVEL_NAMES) else f"level_{depth}"
        train_level = train_parts.map(lambda parts, d=depth: hierarchical_split_sep.join(parts[:d]))
        test_level = test_parts.map(lambda parts, d=depth: hierarchical_split_sep.join(parts[:d]))
        train_level = pd.Series(train_level.to_numpy(), index=train_series.index)
        test_level = pd.Series(test_level.to_numpy(), index=test_series.index)

        counts = _union_value_counts(train_level, test_level)
        components[level_name] = (
            _frequency_encode_from_counts(train_level, counts),
            _frequency_encode_from_counts(test_level, counts),
        )

    return components

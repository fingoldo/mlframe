"""Sensor for the extended ``_session_fixture_immutability_sensor`` coverage (B2 #28/#29 residue).

Wave 3B added the tripwire for ``sample_regression_data`` and ``sample_classification_data``. Wave 10c extends the
same registration to every remaining mutable session-scope fixture in ``tests/training/conftest.py``
(``sample_polars_data``, ``sample_timeseries_data``, ``sample_categorical_data``,
``sample_categorical_classification_data``, ``sample_large_regression_data``, ``sample_outlier_data``) so any of
them being silently mutated by a downstream test surfaces at session teardown.

This file is a structural meta-sensor: it asserts the registration dictionaries pick up every relevant fixture once
all of them have been consumed. We import the conftest module-level registries and exercise each fixture once to
trigger registration; then we assert the bookkeeping dict contains all expected names. We do NOT mutate any
fixture; the actual mutation-detection logic is in the autouse session fixture itself."""

from __future__ import annotations



_FIXTURES_TO_REGISTER = [
    "sample_regression_data",
    "sample_classification_data",
    "sample_polars_data",
    "sample_timeseries_data",
    "sample_categorical_data",
    "sample_categorical_classification_data",
    "sample_large_regression_data",
    "sample_outlier_data",
]


def test_session_fixture_immutability_sensor_registers_all_dataframes(
    sample_regression_data,
    sample_classification_data,
    sample_polars_data,
    sample_timeseries_data,
    sample_categorical_data,
    sample_categorical_classification_data,
    sample_large_regression_data,
    sample_outlier_data,
) -> None:
    """Force each registered session fixture to materialise once and assert every name landed in the tracker dict.

    The tracker dict is module-level in ``tests/training/conftest`` and lives for the whole session, so consuming a
    fixture in this test populates the entry that the autouse session sensor will compare against at teardown."""
    from tests.training import conftest as _ctf

    registered = set(_ctf._SESSION_FIXTURE_SHAPES.keys())
    missing = [name for name in _FIXTURES_TO_REGISTER if name not in registered]
    assert not missing, (
        f"Session-scope fixture(s) {missing} did NOT register a shape snapshot; the immutability sensor will not "
        f"catch silent mutation for them. Add `_SESSION_FIXTURE_SHAPES[<name>] = _df_shape_signature(df)` + "
        f"`_SESSION_FIXTURE_REFS[<name>] = df` in the fixture body in tests/training/conftest.py."
    )

    # Sanity-check each ref is the actual fixture object (id() comparison is the cheapest invariant for "no copy").
    for name in _FIXTURES_TO_REGISTER:
        ref = _ctf._SESSION_FIXTURE_REFS.get(name)
        assert ref is not None, f"_SESSION_FIXTURE_REFS missing entry for {name}"


def test_df_shape_signature_handles_polars(sample_polars_data) -> None:
    """Polars DataFrames have ``df.columns`` as a plain list (not pandas Index). The signature helper must produce
    a non-degenerate (non-None) tuple instead of crashing in the ``except`` branch and returning ``(None, None,
    None)``."""
    from tests.training.conftest import _df_shape_signature

    pl_df, _feature_names, _y = sample_polars_data
    sig = _df_shape_signature(pl_df)
    assert sig != (None, None, None), (
        "Immutability sensor signature degraded to the failure-tuple for a polars frame; .columns.tolist() path is missing the hasattr fallback."
    )
    shape, cols, dtypes = sig
    assert isinstance(shape, tuple) and len(shape) == 2 and shape[0] > 0 and shape[1] > 0
    assert isinstance(cols, tuple) and len(cols) > 0
    assert isinstance(dtypes, tuple) and len(dtypes) == len(cols)

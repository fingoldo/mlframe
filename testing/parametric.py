"""Thin mlframe-specific wrapper around ``polars.testing.parametric``.

Why this module exists
----------------------
Hand-crafted test frames (``pl.DataFrame({"col": [1,2,3]})``) systematically
miss the shapes that explode in production:

  * nulls inside ``pl.Categorical`` (round 11 — CB fastpath TypeError)
  * high-n_unique + near-all-null text columns (round 12 — CB
    "Dictionary size is 0")
  * constant / all-null / inf / NaN numeric columns
  * dtype permutations (Float32 vs Float64, Int16 vs Int32) triggering
    different C++ code paths in XGB / CatBoost / LightGBM

``polars.testing.parametric.dataframes`` integrates natively with
Hypothesis and already covers ``pl.Categorical``, ``pl.Enum``,
``null_probability`` per column, ``allow_chunks`` for chunked-Arrow
pathologies, etc. Rather than teach every test file to call that module
directly (and migrate again the next time Polars renames parameters —
``null_probability``→``allow_null`` just happened), tests call the
thin helpers here and the migration point stays in one file.

What this module gives you
--------------------------
1. **Named column helpers** for the shapes we keep hitting:
   ``categorical_column``, ``inf_heavy_float_column``, ``constant_column``,
   ``id_column``, ``high_card_text_column``, ``sparse_null_column``.
2. **Named frame profiles**:
   ``adversarial_frame`` — layered pathologies, for fuzz-tests of
   pipeline code that must survive anything;
   ``prod_like_frame`` — schema-matched miniature of prod_jobsdetails,
   for end-to-end smoke tests at small scale.
3. **Hypothesis profiles** tuned for slow frame-generation work —
   ``mlframe-fast`` (default), ``mlframe-ci``, ``mlframe-nightly``.
   Auto-registered at import; select with env var ``MLFRAME_HYP_PROFILE``
   or ``settings.load_profile(...)``.

Usage:

    from hypothesis import given
    from mlframe.testing.parametric import adversarial_frame

    @given(df=adversarial_frame(n_rows=(100, 500)))
    def test_auto_detect_survives_arbitrary_frame(df):
        text, emb = _auto_detect_feature_types(df, cfg, cat_features=[])
        assert isinstance(text, list) and isinstance(emb, list)
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple, Union

from hypothesis import HealthCheck, settings, strategies as st
import polars as pl
from polars.testing.parametric import column, dataframes


# =============================================================================
# Hypothesis profile registration (idempotent)
# =============================================================================

_PROFILES_REGISTERED: bool = False


def register_profiles() -> None:
    """Register mlframe-tuned Hypothesis profiles. Idempotent."""
    global _PROFILES_REGISTERED
    if _PROFILES_REGISTERED:
        return

    common = dict(
        deadline=None,  # frame generation can be slow, deadline isn't meaningful
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.data_too_large,
            HealthCheck.filter_too_much,
            HealthCheck.large_base_example,
        ],
        derandomize=False,
    )
    settings.register_profile("mlframe-fast",    max_examples=10,  **common)
    settings.register_profile("mlframe-ci",      max_examples=50,  **common)
    settings.register_profile("mlframe-nightly", max_examples=500, **common)

    # Users can select via env var at import time.
    chosen = os.environ.get("MLFRAME_HYP_PROFILE", "mlframe-fast")
    if chosen in ("mlframe-fast", "mlframe-ci", "mlframe-nightly"):
        settings.load_profile(chosen)

    _PROFILES_REGISTERED = True


# Register on import so tests that do `from mlframe.testing.parametric import ...`
# get a sane default without boilerplate.
register_profiles()


# =============================================================================
# Column helpers
# =============================================================================
#
# IMPORTANT — why we don't use ``null_probability=``
# --------------------------------------------------
# In polars 1.35 the ``null_probability`` kwarg on ``column`` is
# silently coerced to ``bool(rate)`` inside
# ``_handle_null_probability_deprecation`` (core.py:610). The float rate
# is DISCARDED — any non-zero value just toggles ``allow_null=True`` and
# nulls appear at hypothesis's default frequency (typically very low).
# To control the actual null *rate* we splice nulls into the per-cell
# strategy ourselves via a weighted ``flatmap``.
#
# Trade-off: this bypasses both the deprecated API and the forthcoming
# ``allow_null`` API. When the future Polars adds a real ``null_rate=``
# knob we flip one line here and the rest of the codebase is unchanged.


def _weighted_null(value_strategy: st.SearchStrategy, null_rate: float) -> st.SearchStrategy:
    """Yield ``None`` with probability ``null_rate``, otherwise draw from
    ``value_strategy``.

    Implementation note: Hypothesis's integer strategy is shrink-biased
    toward 0 — under exploration the distribution is weighted toward
    small ints, and shrinking minimizes. So we route small ``i`` to the
    value (what we usually want to see) and large ``i`` to ``None``.
    This gives the correct mix during exploration AND a shrink target
    of "no nulls", which is what the shrinker should converge to.
    """
    if null_rate <= 0.0:
        return value_strategy
    if null_rate >= 1.0:
        return st.none()
    bucket = 10_000
    value_threshold = int(bucket * (1.0 - null_rate))
    return st.integers(0, bucket - 1).flatmap(
        lambda i: value_strategy if i < value_threshold else st.none()
    )


def categorical_column(
    name: str,
    categories: Sequence[str],
    *,
    null_rate: float = 0.05,
    use_enum: bool = True,
):
    """Column with a fixed set of string categories.

    ``use_enum=True`` pins cardinality exactly via ``pl.Enum(categories)`` —
    that's what we want for tests asserting on cardinality thresholds.
    ``use_enum=False`` uses ``pl.Categorical`` with sampled values; the
    category dict is what Polars auto-derives from the sample, so
    cardinality may be smaller than ``len(categories)`` on short frames.

    Nulls are injected at exactly ``null_rate`` (per-cell Bernoulli) —
    this is what round-11's CatBoost fastpath crash required: real
    masked nulls inside a Categorical, not just an empty column.
    """
    cats = list(categories)
    dtype = pl.Enum(cats) if use_enum else pl.Categorical
    value_strategy = st.sampled_from(cats)
    return column(
        name=name, dtype=dtype,
        strategy=_weighted_null(value_strategy, null_rate),
        allow_null=(null_rate > 0.0),
    )


def inf_heavy_float_column(
    name: str,
    *,
    width: int = 32,
    null_rate: float = 0.0,
    specials_rate: float = 0.02,
):
    """Float column with meaningful ``+inf``, ``-inf``, ``NaN`` at
    approximately ``specials_rate`` per cell.

    Default ``st.floats()`` can emit specials but shrinking removes
    them. Weighting them in directly keeps them present in the
    shrinking-minimal example.
    """
    dtype = pl.Float32 if width == 32 else pl.Float64
    normal = st.floats(width=width, allow_nan=False, allow_infinity=False)
    specials = st.sampled_from([float("inf"), float("-inf"), float("nan")])
    # Route small shrink-preferred ``i`` to normal, large ``i`` to specials.
    # Shrinker converges on "no specials", which is correct minimization.
    bucket = 10_000
    normal_threshold = int(bucket * (1.0 - specials_rate))
    mixed = st.integers(0, bucket - 1).flatmap(
        lambda i: normal if i < normal_threshold else specials
    )
    return column(
        name=name, dtype=dtype,
        strategy=_weighted_null(mixed, null_rate),
        allow_null=(null_rate > 0.0),
    )


def constant_column(name: str, dtype: pl.DataType, value):
    """Every row has the same ``value``. Shakes out code paths that
    expect variance in a feature (e.g. scalers dividing by std)."""
    return column(name=name, dtype=dtype, strategy=st.just(value),
                  allow_null=False)


def id_column(name: str, *, width: int = 32):
    """Unique-per-row integer — simulates a ``uid`` or high-card numeric."""
    dtype = pl.Int32 if width == 32 else pl.Int64
    return column(name=name, dtype=dtype, unique=True, allow_null=False)


def high_card_text_column(
    name: str,
    *,
    n_unique_target: int = 1000,
    null_rate: float = 0.0,
):
    """Polars ``Utf8`` column with many unique short strings. Default
    1000 uniques — enough to trigger text-promotion without being
    prohibitively slow."""
    pool = [f"t_{i:06d}" for i in range(n_unique_target)]
    value_strategy = st.sampled_from(pool)
    return column(
        name=name, dtype=pl.Utf8,
        strategy=_weighted_null(value_strategy, null_rate),
        allow_null=(null_rate > 0.0),
    )


def sparse_null_column(
    name: str,
    dtype: pl.DataType,
    *,
    non_null_rate: float = 0.001,
    fill_strategy: Optional[st.SearchStrategy] = None,
):
    """Column that's ~99.9% null — models round-12's trigger where a
    text-ish column was promoted because n_unique > threshold but
    non_null count was tiny."""
    if fill_strategy is None:
        if dtype in (pl.Utf8, pl.String):
            fill_strategy = st.text(min_size=1, max_size=20)
        elif dtype in (pl.Float32, pl.Float64):
            fill_strategy = st.floats(allow_nan=False, allow_infinity=False)
        else:
            fill_strategy = st.integers(-1000, 1000)
    return column(
        name=name, dtype=dtype,
        strategy=_weighted_null(fill_strategy, 1.0 - non_null_rate),
        allow_null=True,
    )


# =============================================================================
# Frame profiles
# =============================================================================

_DEFAULT_CATEGORIES_SMALL = ["v0", "v1", "v2", "v3", "v4", "__MISSING__"]
_DEFAULT_CATEGORIES_MEDIUM = [f"v{i}" for i in range(30)] + ["__MISSING__"]


def _size_tuple(n_rows: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return n_rows if isinstance(n_rows, tuple) else (n_rows, n_rows)


def adversarial_frame(
    n_rows: Union[int, Tuple[int, int]] = (50, 300),
    *,
    include_null_in_cat: bool = True,
    include_inf_in_float: bool = True,
    include_constant_col: bool = True,
    include_sparse_null_col: bool = True,
    include_high_card_cat: bool = False,
    extra_cols: Optional[Iterable] = None,
):
    """Frame strategy layering multiple pathological shapes at once.

    Use for fuzz-tests of pipeline code that promises to survive "any
    reasonable frame" — ``_auto_detect_feature_types``, Polars→pandas
    casts in preprocessing, schema-preserving transformers, etc.

    Do NOT use when the test asserts on specific numeric outputs — the
    shape varies across examples and such assertions won't hold. For
    those, keep hand-built frames.
    """
    cols = [
        column("num_f32", dtype=pl.Float32,
               strategy=_weighted_null(st.floats(width=32), 0.05),
               allow_null=True),
        column("num_i16", dtype=pl.Int16,
               strategy=_weighted_null(st.integers(-32000, 32000), 0.02),
               allow_null=True),
        column("bool_col", dtype=pl.Boolean, allow_null=False),
    ]
    if include_null_in_cat:
        cols.append(categorical_column(
            "cat_null_heavy", _DEFAULT_CATEGORIES_SMALL, null_rate=0.2,
        ))
    if include_inf_in_float:
        cols.append(inf_heavy_float_column(
            "num_specials", null_rate=0.05, specials_rate=0.05,
        ))
    if include_constant_col:
        cols.append(constant_column("const_i16", pl.Int16, 0))
    if include_sparse_null_col:
        cols.append(sparse_null_column(
            "sparse_text", pl.Utf8, non_null_rate=0.005,
        ))
    if include_high_card_cat:
        cols.append(categorical_column(
            "hi_card_cat", _DEFAULT_CATEGORIES_MEDIUM, null_rate=0.1,
        ))
    if extra_cols:
        cols.extend(extra_cols)

    min_size, max_size = _size_tuple(n_rows)
    return dataframes(cols, min_size=min_size, max_size=max_size)


def prod_like_frame(
    n_rows: Union[int, Tuple[int, int]] = (200, 1000),
    *,
    with_target: bool = True,
    with_timestamps: bool = True,
):
    """Schema-matched miniature of ``prod_jobsdetails`` — cats with
    prod-like cardinalities + a few numerics + optional target/timestamps.

    Useful for smoke-fuzzing the training suite at small scale without
    touching real data."""
    cols = [
        categorical_column(
            "category",
            [f"cat_{i}" for i in range(15)] + ["__MISSING__"],
            null_rate=0.05,
        ),
        categorical_column(
            "workload", ["hourly", "fixed", "__MISSING__"],
            null_rate=0.02,
        ),
        categorical_column(
            "contractor_tier",
            ["entry", "inter", "expert", "__MISSING__"],
            null_rate=0.1,
        ),
        column("num_f0", dtype=pl.Float32,
               strategy=_weighted_null(st.floats(width=32, allow_nan=False, allow_infinity=False), 0.02),
               allow_null=True),
        column("num_f1", dtype=pl.Float32,
               strategy=st.floats(width=32, allow_nan=False, allow_infinity=False),
               allow_null=False),
        column("num_f2", dtype=pl.Float32,
               strategy=st.floats(width=32, allow_nan=False, allow_infinity=False),
               allow_null=False),
        column("num_i0", dtype=pl.Int16,
               strategy=_weighted_null(st.integers(-1000, 1000), 0.05),
               allow_null=True),
        column("bool_0", dtype=pl.Boolean, allow_null=False),
    ]
    if with_timestamps:
        # Spread timestamps over ~30 days so wholeday_splitting has enough
        # distinct dates.
        import datetime as dt
        cols.append(column(
            "timestamp", dtype=pl.Datetime,
            strategy=st.datetimes(
                min_value=dt.datetime(2023, 1, 1),
                max_value=dt.datetime(2024, 12, 31),
            ),
            allow_null=False,
        ))
    if with_target:
        cols.append(column(
            "target", dtype=pl.Int8,
            strategy=st.sampled_from([0, 1]),
            allow_null=False,
        ))

    min_size, max_size = _size_tuple(n_rows)
    return dataframes(cols, min_size=min_size, max_size=max_size)


def prod_like_frame_small(
    n_rows: Union[int, Tuple[int, int]] = 200,
):
    """Minimal prod-like frame for suite-level end-to-end fuzzing.

    Used by ``TestTrainSuiteRobustness`` where each example fits CB/XGB
    end-to-end — 200 rows is the sweet spot where a model can actually
    train but frame generation is fast. ``prod_like_frame`` (10 columns
    x 300-500 rows) is too heavy for the hypothesis healthcheck.
    """
    import datetime as dt
    cols = [
        categorical_column("category", ["a", "b", "c", "d", "__MISSING__"], null_rate=0.05),
        column("num_f0", dtype=pl.Float32,
               strategy=st.floats(width=32, allow_nan=False, allow_infinity=False),
               allow_null=False),
        column("num_f1", dtype=pl.Float32,
               strategy=st.floats(width=32, allow_nan=False, allow_infinity=False),
               allow_null=False),
        column("timestamp", dtype=pl.Datetime,
               strategy=st.datetimes(
                   min_value=dt.datetime(2024, 1, 1),
                   max_value=dt.datetime(2024, 2, 1),  # 1 month, enough for wholeday disabled
               ),
               allow_null=False),
        column("target", dtype=pl.Int8,
               strategy=st.sampled_from([0, 1]),
               allow_null=False),
    ]
    min_size, max_size = _size_tuple(n_rows)
    return dataframes(cols, min_size=min_size, max_size=max_size)


__all__ = [
    "register_profiles",
    "categorical_column",
    "inf_heavy_float_column",
    "constant_column",
    "id_column",
    "high_card_text_column",
    "sparse_null_column",
    "adversarial_frame",
    "prod_like_frame",
    "prod_like_frame_small",
]

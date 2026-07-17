"""Regression + biz_value tests for ``_auto_detect_feature_types`` single-pass aggregation fix.

The pre-fix code called ``df[name].n_unique()`` + ``int(df[name].count())`` per candidate column (2 Python -> polars
round-trips per col). On 60 cols that was 50-200 ms. The fix collapses both stats into one lazy ``select`` covering
all candidate columns simultaneously.

Tests cover both the polars branch and the pandas branch.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.configs import FeatureTypesConfig
from mlframe.training.core._misc_helpers import _auto_detect_feature_types


def _synth_polars(n_rows: int, n_cols: int, seed: int) -> pl.DataFrame:
    """Mixed-dtype polars frame: ~half string-like (text candidates), ~quarter mostly-null, ~quarter numeric/low-card."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_cols):
        kind = i % 4
        if kind == 0:
            data[f"c_str_{i}"] = [f"v{int(x)}" for x in rng.integers(0, max(2, n_rows // 50), size=n_rows)]
        elif kind == 1:
            arr = [f"v{int(x)}" for x in rng.integers(0, 500, size=n_rows)]
            mask = rng.random(n_rows) < 0.95
            for j, m in enumerate(mask):
                if m:
                    arr[j] = None
            data[f"c_sparse_{i}"] = arr
        elif kind == 2:
            data[f"c_lowcard_{i}"] = [f"v{int(x)}" for x in rng.integers(0, 8, size=n_rows)]
        else:
            data[f"c_num_{i}"] = rng.normal(size=n_rows).astype(np.float64)
    return pl.DataFrame(data)


def _synth_pandas(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    """Mixed-dtype pandas frame with the same structure as the polars synth."""
    return _synth_polars(n_rows, n_cols, seed).to_pandas()


def _vendored_legacy_polars(df, ftc, cat_features):
    """Vendored pre-fix polars logic, used as the ground-truth oracle for the identical-classification test."""
    import polars as pl

    text_features = list(ftc.text_features or [])
    embedding_features = list(ftc.embedding_features or [])
    user_assigned = set(text_features) | set(embedding_features)
    promote_text = ftc.use_text_features
    threshold = ftc.cat_text_cardinality_threshold
    if ftc.cat_text_cardinality_threshold_pct > 0.0:
        threshold = min(threshold, max(50, int(df.height * ftc.cat_text_cardinality_threshold_pct)))
    min_non_null_abs = max(1, int(round(df.height * ftc.min_non_null_fraction_for_text_promotion)))
    auto_drop: list = []
    for name, dtype in df.schema.items():
        if name in user_assigned:
            continue
        is_text_like = dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(dtype, pl.Enum)
        if not is_text_like:
            continue
        n_unique = df[name].n_unique()
        if n_unique > threshold:
            non_null = int(df[name].count())
            if non_null < min_non_null_abs:
                continue
            if promote_text:
                text_features.append(name)
            else:
                auto_drop.append(name)
    return sorted(text_features), sorted(embedding_features), sorted(auto_drop)


# ---------------------------------------------------------------------------
# Polars branch
# ---------------------------------------------------------------------------


def test_auto_detect_output_identical_pre_vs_post_polars():
    """Single-pass aggregation classification must equal the per-col-call legacy classification (polars branch)."""
    df = _synth_polars(n_rows=10_000, n_cols=40, seed=42)
    ftc = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=100,
        cat_text_cardinality_threshold_pct=0.0,
        min_non_null_fraction_for_text_promotion=0.01,
    )
    legacy_text, legacy_emb, legacy_drop = _vendored_legacy_polars(df, ftc, cat_features=[])
    text, emb, drop = _auto_detect_feature_types(df, ftc, cat_features=[], verbose=False)
    assert sorted(text) == legacy_text, f"text_features diverged: legacy={legacy_text} new={sorted(text)}"
    assert sorted(emb) == legacy_emb, f"embedding_features diverged: legacy={legacy_emb} new={sorted(emb)}"
    assert sorted(drop) == legacy_drop, f"auto-drop list diverged: legacy={legacy_drop} new={sorted(drop)}"


def test_auto_detect_output_identical_pre_vs_post_polars_use_text_false():
    """Same as above but with use_text_features=False -> text candidates land in auto-drop bucket."""
    df = _synth_polars(n_rows=10_000, n_cols=40, seed=7)
    ftc = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=False,
        cat_text_cardinality_threshold=100,
        cat_text_cardinality_threshold_pct=0.0,
        min_non_null_fraction_for_text_promotion=0.01,
    )
    legacy_text, legacy_emb, legacy_drop = _vendored_legacy_polars(df, ftc, cat_features=[])
    text, emb, drop = _auto_detect_feature_types(df, ftc, cat_features=[], verbose=False)
    assert sorted(text) == legacy_text
    assert sorted(emb) == legacy_emb
    assert sorted(drop) == legacy_drop


def test_auto_detect_polars_single_collect_bounded():
    """Instrument ``pl.LazyFrame.collect`` and ``pl.DataFrame.__getitem__``: post-fix must bound them to a small constant.

    Pre-fix did 2 Series accesses (df[name].n_unique() + df[name].count()) per candidate column = 2N Series calls,
    zero lazy collects. Post-fix does ONE lazy collect, zero per-col Series accesses on text-like cols.
    """
    df = _synth_polars(n_rows=5_000, n_cols=40, seed=11)
    ftc = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=100,
        cat_text_cardinality_threshold_pct=0.0,
        min_non_null_fraction_for_text_promotion=0.01,
    )

    orig_collect = pl.LazyFrame.collect
    counts = {"collect": 0}

    def _counting_collect(self, *a, **kw):
        counts["collect"] += 1
        return orig_collect(self, *a, **kw)

    pl.LazyFrame.collect = _counting_collect
    try:
        _auto_detect_feature_types(df, ftc, cat_features=[], verbose=False)
    finally:
        pl.LazyFrame.collect = orig_collect

    # At least 1 (proves the lazy path is in use), at most 4 (slack for any internal polars schema/collect calls).
    assert 1 <= counts["collect"] <= 4, f"post-fix expected 1..4 LazyFrame.collect calls, got {counts['collect']} (regression to per-col path?)"


def test_biz_val_auto_detect_polars_speedup():
    """biz_value: 60 cols x 200k rows must run at most 0.6x the legacy wall-time (bench-of-record: ~0.36x = 2.74x speedup)."""
    n_rows = 200_000
    n_cols = 60
    df = _synth_polars(n_rows=n_rows, n_cols=n_cols, seed=13)
    ftc = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=100,
        cat_text_cardinality_threshold_pct=0.0,
        min_non_null_fraction_for_text_promotion=0.01,
    )

    # Warm both paths.
    _vendored_legacy_polars(df, ftc, cat_features=[])
    _auto_detect_feature_types(df, ftc, cat_features=[], verbose=False)

    t0 = time.perf_counter()
    _vendored_legacy_polars(df, ftc, cat_features=[])
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _auto_detect_feature_types(df, ftc, cat_features=[], verbose=False)
    new_s = time.perf_counter() - t0

    ratio = new_s / max(legacy_s, 1e-9)
    assert ratio <= 0.6, (
        f"auto-detect single-pass regressed: new={new_s * 1000:.1f}ms legacy={legacy_s * 1000:.1f}ms ratio={ratio:.2f} (target<=0.6; bench-of-record ~0.36)"
    )


# ---------------------------------------------------------------------------
# Pandas branch
# ---------------------------------------------------------------------------


def _vendored_legacy_pandas(df, ftc, cat_features):
    """Per-column-call equivalent of the post-single-pass _auto_detect_feature_types.

    Mirrors the SAME dtype-detection contract as the production single-pass code
    (``_string_like_dtype_tokens = ("object", "str", "category")`` plus the
    ``"stringdtype"`` substring rescue) so this test gates ONLY the single-pass
    aggregation refactor, not the broader dtype-name surface that was added to
    handle pandas 3.0 / future.infer_string spellings.
    """
    text_features = list(ftc.text_features or [])
    embedding_features = list(ftc.embedding_features or [])
    user_assigned = set(text_features) | set(embedding_features)
    promote_text = ftc.use_text_features
    threshold = ftc.cat_text_cardinality_threshold
    if ftc.cat_text_cardinality_threshold_pct > 0.0:
        threshold = min(threshold, max(50, int(len(df) * ftc.cat_text_cardinality_threshold_pct)))
    min_non_null_abs = max(1, int(round(len(df) * ftc.min_non_null_fraction_for_text_promotion)))
    _string_like_dtype_tokens = ("object", "str", "category")
    auto_drop: list = []
    for col in df.columns:
        if col in user_assigned:
            continue
        dtype_name = str(df[col].dtype)
        _dtype_lc = dtype_name.lower().lstrip("<")
        _is_string_like = any(_dtype_lc.startswith(tok) for tok in _string_like_dtype_tokens) or "stringdtype" in _dtype_lc
        if _is_string_like:
            if dtype_name.startswith("object"):
                _series = df[col]
                try:
                    _first = next((v for v in _series.head(8) if v is not None), None)
                except Exception:
                    _first = None
                if _first is not None and (hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))):
                    embedding_features.append(col)
                    continue
            n_unique = df[col].nunique()
            if n_unique > threshold:
                non_null = int(df[col].notna().sum())
                if non_null < min_non_null_abs:
                    continue
                if promote_text:
                    text_features.append(col)
                else:
                    auto_drop.append(col)
    return sorted(text_features), sorted(embedding_features), sorted(auto_drop)


@pytest.mark.timeout(600)
def test_auto_detect_output_identical_pre_vs_post_pandas():
    """Single-pass df.agg(['nunique','count']) classification must equal per-col legacy classification (pandas branch).

    Bumped to 600s: the test body itself is fast (n=8000 x 40 synthetic) but
    its fixture / conftest cold-import chain (the whole mlframe.training
    stack) dominates wall-time, and under pytest-xdist parallel-worker
    contention this repeatedly exceeded the default 60s pytest-timeout
    (CI gw7 failure was a timeout, not a content mismatch)."""
    df = _synth_pandas(n_rows=8_000, n_cols=40, seed=42)
    ftc = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=100,
        cat_text_cardinality_threshold_pct=0.0,
        min_non_null_fraction_for_text_promotion=0.01,
    )
    legacy_text, legacy_emb, legacy_drop = _vendored_legacy_pandas(df, ftc, cat_features=[])
    text, emb, drop = _auto_detect_feature_types(df, ftc, cat_features=[], verbose=False)
    assert sorted(text) == legacy_text
    assert sorted(emb) == legacy_emb
    assert sorted(drop) == legacy_drop

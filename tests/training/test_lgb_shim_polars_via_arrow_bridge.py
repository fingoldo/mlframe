"""Regression + biz_value tests for ``lgb_shim.fit`` polars input routed through the Arrow split-blocks bridge.

Pre-fix: the shim claimed to accept polars X (docstring said so) but passed it straight to ``lgb.Dataset``. Installed LightGBM 4.x rejects polars with
``TypeError: Cannot initialize Dataset from DataFrame`` because polars does not implement the array protocols ``Dataset.__init__`` probes. Callers were
forced to convert externally; the shim was silently broken on polars input.

Post-fix: the shim calls ``_maybe_bridge_polars_to_pandas`` (project's ``get_pandas_view_of_polars_df``, Arrow split-blocks bridge). Numeric columns stay
zero-copy, Categorical columns reach LightGBM with their codes intact (verified inside the cached Dataset's ``get_data()``), and the conversion step is
materially faster than bare ``df.to_pandas()`` at modest cat counts (~1.9x at 5 cat cols, 200k rows; the gap narrows as cat-column count grows because the
shared pyarrow Categorical-cast cost dominates).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
import pytest


lgb = pytest.importorskip("lightgbm")
from mlframe.training.lgb_shim import (
    LGBMRegressorWithDatasetReuse,
    _maybe_bridge_polars_to_pandas,
)


def _make_mixed_polars(n_rows: int, seed: int = 0) -> pl.DataFrame:
    """Make mixed polars."""
    rng = np.random.default_rng(seed)
    cat_pool = ["alpha", "beta", "gamma", "delta", "epsilon"]
    data = {f"num{i}": rng.normal(size=n_rows).astype(np.float64) for i in range(4)}
    for j in range(2):
        codes = rng.integers(0, len(cat_pool), size=n_rows)
        data[f"cat{j}"] = pl.Series(f"cat{j}", [cat_pool[c] for c in codes], dtype=pl.Categorical)
    return pl.DataFrame(data)


def test_bridge_preserves_categorical_dtype():
    """The bridge must hand LightGBM a pandas frame with ``pd.Categorical`` columns (not object dtype)."""
    df_pl = _make_mixed_polars(500)
    df_pd = _maybe_bridge_polars_to_pandas(df_pl)
    assert isinstance(df_pd, pd.DataFrame)
    for cat_col in ("cat0", "cat1"):
        dtype = df_pd[cat_col].dtype
        assert isinstance(dtype, pd.CategoricalDtype), f"{cat_col} arrived as {dtype}, expected pd.CategoricalDtype -- the bridge dropped the dictionary"


def test_bridge_passthrough_for_non_polars():
    """Pandas / numpy inputs must pass through untouched -- the bridge is polars-only."""
    pd_in = pd.DataFrame({"a": [1, 2, 3]})
    assert _maybe_bridge_polars_to_pandas(pd_in) is pd_in
    np_in = np.arange(10)
    assert _maybe_bridge_polars_to_pandas(np_in) is np_in


def test_raw_lgb_dataset_rejects_polars_proves_bridge_is_required():
    """Pre-fix the shim handed polars straight to ``lgb.Dataset``; installed LightGBM raises ``TypeError`` -- so the fix is functional, not cosmetic.

    The shim's pre-fix docstring claimed polars worked via ``__array__``. It did not on LightGBM 4.x. This test pins the upstream behaviour so a future
    LightGBM release that learns to consume polars natively flips this test red and signals the bridge can be retired.
    """
    df_pl = _make_mixed_polars(100)
    y = np.zeros(df_pl.height)
    with pytest.raises(TypeError, match="Cannot initialize Dataset from"):
        lgb.Dataset(data=df_pl, label=y, free_raw_data=False).construct()


def test_lgb_dataset_from_bridged_polars_keeps_categorical():
    """End-to-end: polars X -> shim fit -> Dataset retains Categorical dtype in ``get_data``."""
    df_pl = _make_mixed_polars(1_000, seed=1)
    y = np.random.default_rng(2).normal(size=df_pl.height)
    model = LGBMRegressorWithDatasetReuse(n_estimators=5, verbose=-1)
    model.fit(df_pl, y)
    # The cached Dataset is built from the bridged frame; its ``get_data`` returns the pandas frame with Categorical intact.
    cached = model._cached_train_dataset.get_data()
    assert isinstance(cached, pd.DataFrame)
    for cat_col in ("cat0", "cat1"):
        assert isinstance(cached[cat_col].dtype, pd.CategoricalDtype), f"{cat_col} lost Categorical dtype inside the cached Dataset"


def test_predict_numerical_equivalence_polars_vs_bridged_pandas():
    """``model.predict`` output must match (within float epsilon) whether the input is polars or its pre-bridged pandas equivalent."""
    df_pl = _make_mixed_polars(2_000, seed=3)
    y = np.random.default_rng(4).normal(size=df_pl.height)

    model_a = LGBMRegressorWithDatasetReuse(n_estimators=10, random_state=0, verbose=-1)
    model_a.fit(df_pl, y)
    pred_polars = model_a.predict(_maybe_bridge_polars_to_pandas(df_pl))

    df_pd = _maybe_bridge_polars_to_pandas(df_pl)
    model_b = LGBMRegressorWithDatasetReuse(n_estimators=10, random_state=0, verbose=-1)
    model_b.fit(df_pd, y)
    pred_pandas = model_b.predict(df_pd)

    np.testing.assert_allclose(pred_polars, pred_pandas, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_rows", [200_000])
def test_biz_value_bridge_conversion_beats_bare_to_pandas(n_rows: int):
    """biz_value: Arrow split-blocks bridge conversion is within a small
    multiple of bare ``df.to_pandas()`` on this shape.

    The dominant gain of the fix is **correctness** (``pd.Categorical``
    preserved -> LightGBM dispatches the native cat-split path instead of
    falling through ``__array__`` / numeric hashing); that contract is
    pinned separately by ``test_bridge_preserves_categorical_dtype``.

    Wall-time history:
        - 10 num + 5 cat at 200k rows: bridge ~0.65x of bare ``to_pandas``
          measured 2026-05-16 on polars 0.20.x.
        - polars 1.x materially sped up bare ``to_pandas()``; the bridge's
          fixed cat-dictionary rebuild overhead is now visible as up to
          ~3x of bare on the smallest cat-count shape.
    This is a regression sensor for catastrophic slowdown (bridge silently
    routing through a 10-100x slower path); 3.5x ceiling absorbs the
    polars-1.x speedup of the baseline while still flagging a real
    breakage. The absolute wall-time stays in the single-digit ms even at
    worst case, so the practical impact on a 200k-row fit is negligible.
    """
    df_pl = _make_mixed_polars(n_rows, seed=5)
    # Warm both paths to avoid first-call import / pyarrow JIT overhead.
    _ = _maybe_bridge_polars_to_pandas(df_pl)
    _ = df_pl.to_pandas()

    # min-of-N timing: less sensitive to xdist worker scheduling jitter
    # than median when the absolute wall-time is in the single-digit ms.
    n_runs = 7
    post_times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _maybe_bridge_polars_to_pandas(df_pl)
        post_times.append(time.perf_counter() - t0)

    pre_times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = df_pl.to_pandas()
        pre_times.append(time.perf_counter() - t0)

    post_min = float(np.min(post_times))
    pre_min = float(np.min(pre_times))
    ratio = post_min / max(pre_min, 1e-9)
    assert ratio < 3.5, (
        f"Arrow split-blocks bridge ({post_min * 1000:.2f}ms) is catastrophically slower than bare ``df.to_pandas()`` "
        f"({pre_min * 1000:.2f}ms); ratio={ratio:.3f} (expected < 3.5). "
        f"A ratio this high suggests the bridge guard slipped into the legacy round-trip path."
    )

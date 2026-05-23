"""Regression test for iter199: polars group_by + agg cold-start warming.

c0042 binary profile (cb+lgb+linear+xgb, 200k) attributed 2.557s to a single
``group_by(cat).agg(mean, len)`` call in ``_per_group_predict_polars`` on
the first invocation per process. polars' query optimizer / Rust hash-
aggregate kernel has a ~2-3s cold-start cost (verified via bench:
cold=1.9ms tiny + 2.5s production-size; after warm: 0.5ms tiny + 2.5ms big).

The prewarm in ``mlframe.metrics.core._prewarm_numba_cache_body`` runs a
tiny enum-typed group_by + join at module-load time so the first real
call doesn't pay the cold-start. This test asserts the warm completed.
"""
import time

import numpy as np
import pytest


def _ensure_prewarmed():
    from mlframe.metrics.core import prewarm_numba_cache
    prewarm_numba_cache()


def test_polars_group_by_warm_after_prewarm():
    """After prewarm, a production-size polars group_by + agg on Enum-dtype
    must run in well under the ~2.5s fresh-process cold-start cost."""
    import polars as pl

    _ensure_prewarmed()

    n = 200_000
    n_groups = 500
    rng = np.random.default_rng(20260523)
    groups = [f"g{i:06d}" for i in range(n_groups)]
    enum_t = pl.Enum(groups)
    cat = pl.Series("cat", rng.choice(groups, n), dtype=enum_t)
    y = pl.Series("__y__", rng.random(n).astype(np.float64))
    df = pl.DataFrame({"cat": cat, "__y__": y})

    t = time.perf_counter()
    result = df.group_by("cat").agg(
        pl.col("__y__").mean().alias("__mean__"),
        pl.len().alias("__size__"),
    )
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert result.height == n_groups
    # Cold-start without prewarm would be 2500-3000ms; with prewarm, even
    # the production-size aggregation should complete in <100ms (typical
    # ~2-5ms post-warm; 100ms gives plenty of margin for CI noise without
    # masking a missed prewarm).
    assert elapsed_ms < 100.0, (
        f"polars group_by + agg on 200k rows took {elapsed_ms:.1f}ms; "
        f">100ms suggests the cold-start warm-up did NOT fire. Verify "
        f"_prewarm_numba_cache_body includes the polars group_by warm."
    )


def test_polars_join_warm_after_prewarm():
    """The join path is also exercised by ``_per_group_predict_polars._predict``
    (left-joins side_X against stats_df). Warm it alongside the group_by."""
    import polars as pl

    _ensure_prewarmed()

    n = 200_000
    n_groups = 500
    rng = np.random.default_rng(20260523)
    groups = [f"g{i:06d}" for i in range(n_groups)]
    enum_t = pl.Enum(groups)
    cat = pl.Series("cat", rng.choice(groups, n), dtype=enum_t)
    df_left = pl.DataFrame({"cat": cat})
    df_right = pl.DataFrame({
        "cat": pl.Series("cat", groups, dtype=enum_t),
        "value": np.arange(n_groups, dtype=np.float64),
    })

    t = time.perf_counter()
    joined = df_left.join(df_right, on="cat", how="left")
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert joined.height == n
    assert elapsed_ms < 200.0, (
        f"polars join on 200k rows took {elapsed_ms:.1f}ms; >200ms suggests "
        f"the join warm-up did NOT fire."
    )

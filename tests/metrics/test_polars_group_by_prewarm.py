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


def _ensure_prewarmed():
    from mlframe.metrics.core import prewarm_numba_cache

    prewarm_numba_cache()


def test_polars_group_by_warm_after_prewarm():
    """After prewarm, a production-size polars group_by + agg on Enum-dtype
    must run in well under the ~2.5s fresh-process cold-start cost."""
    import os
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
    # the production-size aggregation should complete in <250ms (typical
    # ~2-5ms post-warm; 250ms gives margin for CPU-saturated suite runs
    # where polars contends with concurrent numba/torch work, without
    # masking a missed prewarm which would be >2500ms).
    # GitHub-hosted shared CI runners (verified Windows 3.11 2026-05-24
    # at 585ms) need a wider band -- per-worker CPU contention from
    # parallel xdist jobs blows past the 250ms desktop ceiling without
    # the prewarm having actually missed. Raise to 1500ms on CI: still
    # an order of magnitude below the 2500-3000ms cold-start signature,
    # so the sensor's purpose (catch missed prewarm) is preserved.
    _ceiling_ms = 1500.0 if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") else 250.0
    assert elapsed_ms < _ceiling_ms, (
        f"polars group_by + agg on 200k rows took {elapsed_ms:.1f}ms "
        f"(ceiling {_ceiling_ms:.0f}ms); >ceiling suggests the cold-start "
        f"warm-up did NOT fire. Verify _prewarm_numba_cache_body includes "
        f"the polars group_by warm."
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
    df_right = pl.DataFrame(
        {
            "cat": pl.Series("cat", groups, dtype=enum_t),
            "value": np.arange(n_groups, dtype=np.float64),
        }
    )

    t = time.perf_counter()
    joined = df_left.join(df_right, on="cat", how="left")
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert joined.height == n
    # Same CI band as the group_by sensor: shared GitHub runners flake
    # past the desktop ceiling under concurrent xdist load. Cold-start
    # join would be >2500ms, so the elevated 2000ms CI ceiling still
    # catches a missed prewarm.
    import os as _os

    _ceiling_ms = 2000.0 if _os.environ.get("CI") or _os.environ.get("GITHUB_ACTIONS") else 500.0
    assert elapsed_ms < _ceiling_ms, (
        f"polars join on 200k rows took {elapsed_ms:.1f}ms (ceiling {_ceiling_ms:.0f}ms); >ceiling suggests the join warm-up did NOT fire."
    )

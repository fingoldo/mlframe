"""A/B: per-row summary stats via the pandas/numpy path vs every polars-native formulation tried.

The production log spent 11.3s in ``row_wise_summary_stats`` on a 2.18M x 85 frame, after a polars->pandas
conversion the run did not otherwise need. This measures whether polars can do it natively, and -- because the
first attempt was 8x SLOWER -- which polars construct is the right one.

The answer turned out to be a property of the *engine*, not of the expression. Eagerly, every per-row array
kernel polars offers costs ~2.0-2.6s over 400k x 85 on a 16-core box, while the numba reference does the same
work in 0.26s purely by running ``prange`` across rows. Collecting the same plan with ``engine="streaming"``
splits the frame into morsels and runs them on all cores, which recovers that ~9x.

Beware one trap when re-deriving this: an eager ``select`` holding exactly ONE expression takes a much slower
path than the same expression alongside any second one. Measured at 400k x 85 in a fresh process per case,
``select(concat_arr.arr.median())`` alone is 20.1s, but ``select([concat_arr.arr.median(), mean_horizontal])``
-- strictly more work -- is 2.0s, and two medians together are 2.3s. ``run_single_expression_trap()`` below
reproduces it. Any conclusion drawn from a lone per-row array expression is about that path, not about the
expression, and does not transfer to a real multi-stat call: an early pass through this bench concluded from
exactly such a microbench that ``.arr.median()`` was intrinsically 8x more expensive than reading the median
position off a sorted array. It is not -- in a multi-expression select the two are within 10% of each other.

Every variant below is kept runnable so its number stays reproducible:

``list``
    ``pl.concat_list`` + ``.list.*``, ``.list.eval()`` for quantiles. A ragged List column allocates a per-row
    offset structure and ``list.eval`` runs a sub-expression per row. The naive version, and by far the worst.
``arr``
    ``pl.concat_arr`` + ``.arr.*``, quantiles by NEAREST RANK from ``.arr.sort().arr.get(const)``. Contiguous,
    so merely competitive -- and not exact (nearest-rank is a different definition from ``np.nanquantile``).
``arr_median``
    ``arr`` but with ``.arr.median()`` for the median, the obvious spelling.
``streaming_arr_median``
    The winner, but with the median taken from ``.arr.median()`` instead of off the already-sorted array. The
    two agree to 2.8e-17, so the difference is purely the cost of a second per-row pass.
``horizontal``
    ``pl.mean_horizontal`` / ``min_horizontal`` / ``max_horizontal`` plus std from the horizontal sums of ``x``
    and ``x*x`` -- no intermediate nested column at all. Fast, cannot express order statistics, and its std
    identity ``(s2 - s1^2/k)/k`` is the catastrophically cancellation-prone form this repo has been bitten by
    three times. Kept for the measurement only; never for production.
``eager_interp``
    Exact interpolated quantiles via ``.arr.sort().arr.get(<per-row expr>)``, collected eagerly.
``lazy_default``
    The same plan through ``.lazy().collect()`` on the default engine.
``streaming_rebuild``
    The same plan through ``collect(engine="streaming")``, letting each expression rebuild ``concat_arr``/sort.
``streaming_sorted``
    Streaming, with the sorted row and the per-row valid count materialised once as helper columns.
``streaming_row_sorted``
    Streaming, with ``concat_arr`` materialised first and the sort/count derived from it. **The winner**, and
    what ``row_wise_summary_stats_polars`` now does.
``nan_native``
    Streaming, skipping ``fill_nan(None)`` entirely and relying on NaN sorting after real values under
    ``nulls_last=True``, with the valid count taken from ``is_not_nan``. Fastest of all -- and INEXACT: the
    order statistics are bit-identical, but ``mean``/``std``/``min``/``max`` propagate the NaN they never
    converted (measured max|diff| 6.4 on ``min`` at 2M rows). Reported for the order-statistic cost only.
``inline_fillnan``
    Exact, and the honest version of ``nan_native``: ``fill_nan(None)`` folded into each consumer expression
    instead of run as its own 85-column stage, so no nulled copy of the frame is materialised.
``f32``
    Streaming winner over Float32 inputs -- half the sort bandwidth, but not exact against a float64 reference.
``arr_agg``
    ``.arr.agg(pl.element().quantile(q))``, polars' generic per-row sub-expression escape hatch.
``unpivot_groupby``
    ``with_row_index`` + ``unpivot`` + ``group_by(row).agg(quantile)`` -- order stats over a tall layout
    instead of a per-row array.
``chunked``
    Manual row-chunking into N LazyFrames run through ``pl.collect_all``, the hand-rolled alternative to
    letting the streaming engine do the morsel splitting.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_row_wise_polars``
Add sizes with e.g. ``... bench_row_wise_polars 50000 400000 2000000``.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

# NaN semantics differ between the two worlds: numpy's nan-aware reductions SKIP NaN, while polars treats NaN
# as a float value and null as missing. Every polars variant below converts NaN to null first so both sides
# answer the same question -- except ``nan_native``, whose whole point is skipping that step.
_TO_NULL = [pl.col(pl.Float32, pl.Float64).fill_nan(None)]

FULL_STATS = ["mean", "std", "min", "max", "median", "q10", "q90"]
CHEAP_STATS = ["mean", "std", "min", "max"]


def make_frame(n_rows: int, n_cols: int, nan_frac: float = 0.05, seed: int = 0) -> pl.DataFrame:
    """A float frame with realistic NaN density."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_rows, n_cols))
    data[rng.random(data.shape) < nan_frac] = np.nan
    return pl.DataFrame({f"f{i}": data[:, i] for i in range(n_cols)})


def _interp_quantile(sorted_row: pl.Expr, k: pl.Expr, q: float) -> pl.Expr:
    """``np.nanquantile``'s linear interpolation at position ``q * (k - 1)`` of the row's own valid count."""
    h = (k - 1).cast(pl.Float64) * q
    lo = h.floor().cast(pl.Int64)
    frac = h - h.floor()
    low = sorted_row.arr.get(lo, null_on_oob=True)
    high = sorted_row.arr.get((lo + 1).clip(upper_bound=k - 1), null_on_oob=True)
    return low * (1.0 - frac) + high * frac


def _q_of(stat: str) -> float:
    """Quantile level of a ``"median"``/``"qNN"`` stat name."""
    return 0.5 if stat == "median" else int(stat[1:]) / 100.0


def pandas_path(df: pl.DataFrame, stats) -> dict:
    """What the suite does today: convert, materialise one float64 matrix, reduce with numpy + an njit kernel."""
    from mlframe.feature_engineering.row_wise_summary import row_wise_summary_stats

    out = row_wise_summary_stats(df.to_pandas(), stats=stats)
    return {c.split("_", 2)[-1]: out[c].to_numpy() for c in out.columns}


def production_path(df: pl.DataFrame, stats) -> dict:
    """The shipped ``row_wise_summary_stats_polars``, so the module and the bench cannot drift apart."""
    from mlframe.feature_engineering.row_wise_summary_polars import row_wise_summary_stats_polars

    out = row_wise_summary_stats_polars(df, stats=stats)
    return {c.split("_", 2)[-1]: out[c].to_numpy() for c in out.columns}


def polars_list(df: pl.DataFrame, stats) -> dict:
    """Naive: a ragged List column plus a per-row ``list.eval`` for each quantile."""
    cols = df.columns
    row = pl.concat_list(cols)
    exprs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            exprs.append({"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal}[stat](cols).alias(stat))
        elif stat == "std":
            exprs.append(row.list.std(ddof=0).alias("std"))
        elif stat == "median":
            exprs.append(row.list.median().alias("median"))
        else:
            exprs.append(row.list.eval(pl.element().quantile(_q_of(stat))).list.first().alias(stat))
    out = df.with_columns(_TO_NULL).select(exprs)
    return {c: out[c].to_numpy() for c in out.columns}


def polars_arr(df: pl.DataFrame, stats) -> dict:
    """Fixed-width Array, quantiles by NEAREST RANK position in a sorted row (not exact, kept for the number)."""
    cols = df.columns
    n = len(cols)
    row = pl.concat_arr(cols)
    sorted_row = row.arr.sort(nulls_last=True)
    exprs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            exprs.append({"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal}[stat](cols).alias(stat))
        elif stat == "std":
            exprs.append(row.arr.std(ddof=0).alias("std"))
        else:
            exprs.append(sorted_row.arr.get(min(n - 1, int(_q_of(stat) * (n - 1)))).alias(stat))
    out = df.with_columns(_TO_NULL).select(exprs)
    return {c: out[c].to_numpy() for c in out.columns}


def polars_arr_median(df: pl.DataFrame, stats) -> dict:
    """``polars_arr`` but spelling the median as ``.arr.median()`` -- the obvious spelling, and a 17s trap."""
    cols = df.columns
    n = len(cols)
    row = pl.concat_arr(cols)
    sorted_row = row.arr.sort(nulls_last=True)
    exprs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            exprs.append({"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal}[stat](cols).alias(stat))
        elif stat == "std":
            exprs.append(row.arr.std(ddof=0).alias("std"))
        elif stat == "median":
            exprs.append(row.arr.median().alias("median"))
        else:
            exprs.append(sorted_row.arr.get(min(n - 1, int(_q_of(stat) * (n - 1)))).alias(stat))
    out = df.with_columns(_TO_NULL).select(exprs)
    return {c: out[c].to_numpy() for c in out.columns}


def polars_horizontal(df: pl.DataFrame, stats) -> dict:
    """No nested column at all: horizontal reductions, std from the horizontal sums of x and x*x.

    The std identity here is the cancellation-prone one this repo has been bitten by three times -- measured,
    documented, and deliberately NOT what the production module uses.
    """
    cols = df.columns
    exprs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            exprs.append({"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal}[stat](cols).alias(stat))
        elif stat == "std":
            k = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in cols])
            s1 = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in cols])
            s2 = pl.sum_horizontal([(pl.col(c) * pl.col(c)).fill_null(0.0) for c in cols])
            exprs.append(((s2 - s1 * s1 / k) / k).sqrt().alias("std"))
        else:
            raise ValueError("polars_horizontal cannot express an order statistic")
    out = df.with_columns(_TO_NULL).select(exprs)
    return {c: out[c].to_numpy() for c in out.columns}


def _exact_exprs(cols, stats, row: pl.Expr, sorted_row: pl.Expr, k: pl.Expr) -> list:
    """The exact stat expressions, parameterised by how ``row``/``sorted_row``/``k`` are obtained."""
    exprs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            exprs.append({"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal}[stat](cols).alias(stat))
        elif stat == "std":
            exprs.append(row.arr.std(ddof=0).alias("std"))
        else:
            exprs.append(_interp_quantile(sorted_row, k, _q_of(stat)).alias(stat))
    return exprs


def _valid_count(cols) -> pl.Expr:
    """Per-row count of non-null values."""
    return pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in cols])


def polars_eager_interp(df: pl.DataFrame, stats) -> dict:
    """Exact interpolated quantiles, collected eagerly -- one thread per expression, no morsel parallelism."""
    cols = df.columns
    row = pl.concat_arr(cols)
    out = df.with_columns(_TO_NULL).select(_exact_exprs(cols, stats, row, row.arr.sort(nulls_last=True), _valid_count(cols)))
    return {c: out[c].to_numpy() for c in out.columns}


def polars_lazy_default(df: pl.DataFrame, stats) -> dict:
    """The same plan on the default lazy engine, which materialises the whole sorted Array column."""
    cols = df.columns
    row = pl.concat_arr(cols)
    lf = df.lazy().with_columns(_TO_NULL).select(_exact_exprs(cols, stats, row, row.arr.sort(nulls_last=True), _valid_count(cols)))
    out = lf.collect()
    return {c: out[c].to_numpy() for c in out.columns}


def polars_streaming_rebuild(df: pl.DataFrame, stats) -> dict:
    """Streaming, but every expression rebuilds ``concat_arr`` and its sort independently."""
    cols = df.columns
    row = pl.concat_arr(cols)
    lf = df.lazy().with_columns(_TO_NULL).select(_exact_exprs(cols, stats, row, row.arr.sort(nulls_last=True), _valid_count(cols)))
    out = lf.collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_streaming_sorted(df: pl.DataFrame, stats) -> dict:
    """Streaming, with only the sorted row and the valid count promoted to helper columns."""
    cols = df.columns
    row = pl.concat_arr(cols)
    lf = (
        df.lazy()
        .with_columns(_TO_NULL)
        .with_columns([row.arr.sort(nulls_last=True).alias("_s"), _valid_count(cols).alias("_k")])
        .select(_exact_exprs(cols, stats, row, pl.col("_s"), pl.col("_k")))
    )
    out = lf.collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_streaming_row_sorted(df: pl.DataFrame, stats) -> dict:
    """Streaming, with ``concat_arr`` materialised first and the sort/count derived from it. The winner."""
    cols = df.columns
    lf = (
        df.lazy()
        .with_columns(_TO_NULL)
        .with_columns(pl.concat_arr(cols).alias("_r"))
        .with_columns([pl.col("_r").arr.sort(nulls_last=True).alias("_s"), _valid_count(cols).alias("_k")])
        .select(_exact_exprs(cols, stats, pl.col("_r"), pl.col("_s"), pl.col("_k")))
    )
    out = lf.collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_nan_native(df: pl.DataFrame, stats) -> dict:
    """Streaming winner without ``fill_nan(None)``: NaN sorts after real values under ``nulls_last=True``."""
    cols = df.columns
    k = pl.sum_horizontal([pl.col(c).is_not_nan().fill_null(False).cast(pl.Int64) for c in cols])
    lf = (
        df.lazy()
        .with_columns(pl.concat_arr(cols).alias("_r"))
        .with_columns([pl.col("_r").arr.sort(nulls_last=True).alias("_s"), k.alias("_k")])
        .select(_exact_exprs(cols, stats, pl.col("_r"), pl.col("_s"), pl.col("_k")))
    )
    out = lf.collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_inline_fillnan(df: pl.DataFrame, stats) -> dict:
    """Exact, with ``fill_nan(None)`` folded into each consumer so no nulled copy of the frame is materialised."""
    nulled_cols = [pl.col(c).fill_nan(None).alias(c) for c in df.columns]
    k = pl.sum_horizontal([pl.col(c).fill_nan(None).is_not_null().cast(pl.Int64) for c in df.columns])
    lf = (
        df.lazy()
        .with_columns(pl.concat_arr(nulled_cols).alias("_r"))
        .with_columns([pl.col("_r").arr.sort(nulls_last=True).alias("_s"), k.alias("_k")])
        .select(_exact_exprs(nulled_cols, stats, pl.col("_r"), pl.col("_s"), pl.col("_k")))
    )
    out = lf.collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_streaming_arr_median(df: pl.DataFrame, stats) -> dict:
    """The winner, but paying a second per-row pass for the median instead of reading it off the sorted array."""
    cols = df.columns
    exprs = []
    for stat in stats:
        if stat == "median":
            exprs.append(pl.col("_r").arr.median().alias("median"))
        else:
            exprs.extend(_exact_exprs(cols, [stat], pl.col("_r"), pl.col("_s"), pl.col("_k")))
    lf = (
        df.lazy()
        .with_columns(_TO_NULL)
        .with_columns(pl.concat_arr(cols).alias("_r"))
        .with_columns([pl.col("_r").arr.sort(nulls_last=True).alias("_s"), _valid_count(cols).alias("_k")])
        .select(exprs)
    )
    out = lf.collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_f32(df: pl.DataFrame, stats) -> dict:
    """The winner over Float32: half the sort bandwidth, but no longer exact against a float64 reference."""
    return polars_streaming_row_sorted(df.with_columns(pl.col(pl.Float64).cast(pl.Float32)), stats)


def polars_arr_agg(df: pl.DataFrame, stats) -> dict:
    """Quantiles via ``.arr.agg(pl.element().quantile(q))``, polars' generic per-row sub-expression."""
    cols = df.columns
    row = pl.concat_arr(cols)
    exprs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            exprs.append({"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal}[stat](cols).alias(stat))
        elif stat == "std":
            exprs.append(row.arr.std(ddof=0).alias("std"))
        else:
            exprs.append(row.arr.agg(pl.element().quantile(_q_of(stat), interpolation="linear")).alias(stat))
    out = df.lazy().with_columns(_TO_NULL).select(exprs).collect(engine="streaming")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_unpivot_groupby(df: pl.DataFrame, stats) -> dict:
    """Order stats over a tall layout: ``unpivot`` to one row per (row, column), then ``group_by`` the row id."""
    cols = df.columns
    aggs = []
    for stat in stats:
        if stat in ("mean", "min", "max"):
            aggs.append(getattr(pl.col("value"), stat)().alias(stat))
        elif stat == "std":
            aggs.append(pl.col("value").std(ddof=0).alias("std"))
        else:
            aggs.append(pl.col("value").quantile(_q_of(stat), interpolation="linear").alias(stat))
    lf = df.lazy().with_columns(_TO_NULL).with_row_index("_i").unpivot(index="_i", on=cols).group_by("_i").agg(aggs).sort("_i")
    out = lf.collect(engine="streaming").drop("_i")
    return {c: out[c].to_numpy() for c in out.columns}


def polars_chunked(df: pl.DataFrame, stats, n_chunks: int = 22) -> dict:
    """Hand-rolled row chunking through ``pl.collect_all`` instead of letting the streaming engine split."""
    cols = df.columns
    step = (df.height + n_chunks - 1) // n_chunks
    lfs = [
        df.slice(i * step, step)
        .lazy()
        .with_columns(_TO_NULL)
        .with_columns(pl.concat_arr(cols).alias("_r"))
        .with_columns([pl.col("_r").arr.sort(nulls_last=True).alias("_s"), _valid_count(cols).alias("_k")])
        .select(_exact_exprs(cols, stats, pl.col("_r"), pl.col("_s"), pl.col("_k")))
        for i in range(n_chunks)
    ]
    out = pl.concat(pl.collect_all(lfs))
    return {c: out[c].to_numpy() for c in out.columns}


# Order-statistic variants: every one of these can answer the FULL stat set.
FULL_PATHS = [
    ("pandas+numpy", pandas_path),
    ("production (polars)", production_path),
    ("polars streaming row+sorted", polars_streaming_row_sorted),
    ("polars streaming sorted", polars_streaming_sorted),
    ("polars streaming rebuild", polars_streaming_rebuild),
    ("polars inline fill_nan", polars_inline_fillnan),
    ("polars streaming arr.median()", polars_streaming_arr_median),
    ("polars nan-native (inexact)", polars_nan_native),
    ("polars f32 (inexact)", polars_f32),
    ("polars arr.agg", polars_arr_agg),
    ("polars chunked collect_all", polars_chunked),
    ("polars unpivot+group_by", polars_unpivot_groupby),
    ("polars lazy default engine", polars_lazy_default),
    ("polars eager interp", polars_eager_interp),
    ("polars arr (nearest-rank)", polars_arr),
    ("polars arr.median()", polars_arr_median),
    ("polars list", polars_list),
]

# No-order-statistic variants, where a horizontal reduction is enough.
CHEAP_PATHS = [
    ("pandas+numpy", pandas_path),
    ("production (polars)", production_path),
    ("polars streaming row+sorted", polars_streaming_row_sorted),
    ("polars eager interp", polars_eager_interp),
    ("polars horizontal (unstable std)", polars_horizontal),
]

# Variants whose runtime is measured in tens of seconds; skipped above this row count unless --all is passed.
SLOW_PATHS = frozenset({"polars list", "polars arr.median()", "polars lazy default engine", "polars eager interp"})
SLOW_PATH_MAX_ROWS = 400_000


def _time(fn, df, stats) -> float:
    """Wall time of one call."""
    t0 = time.perf_counter()
    fn(df, stats)
    return time.perf_counter() - t0


def _run(df, stats, paths, repeats):
    """Warm each path, time best-of-N, print the ratio against the first path, and return the first's output."""
    reference = None
    times = {}
    for name, fn in paths:
        if name in SLOW_PATHS and df.height > SLOW_PATH_MAX_ROWS:
            print(f"{name:34s} (skipped above {SLOW_PATH_MAX_ROWS:,} rows)")
            continue
        fn(df.head(2000), stats)
        best = min(_time(fn, df, stats) for _ in range(repeats))
        times[name] = best
        if reference is None:
            reference = fn(df, stats)
        print(f"{name:34s} {best:8.3f}s   {times[paths[0][0]] / best:6.2f}x")
    return reference


def _exactness(df, ref, stats, fn, label) -> None:
    """Print max|diff| per stat against the numpy reference, plus whether NaN placement agrees."""
    got = fn(df, stats)
    print(f"\nexactness vs numpy -- {label}:")
    for stat in stats:
        same_nan = np.array_equal(np.isnan(ref[stat]), np.isnan(got[stat]))
        print(f"  {stat:7s} max|diff| = {np.nanmax(np.abs(ref[stat] - got[stat])):.3e}   nan placement match = {same_nan}")


def run_single_expression_trap(n_rows: int = 400_000, n_cols: int = 85, repeats: int = 3) -> None:
    """Reproduce the lone-expression eager-select trap that makes any single-kernel microbench unusable here.

    Run this in a FRESH process per case for the clean number -- that is how the 20.1s-vs-2.0s figure quoted in
    the module docstring was obtained. Run in one process, as here, the allocator is already warm and the gap
    narrows to roughly 4.9s vs 2.4s, but the direction is the same: adding a trivial second expression makes a
    strictly larger select several times faster.
    """
    df = make_frame(n_rows, n_cols).with_columns(_TO_NULL)
    cols = df.columns
    row = pl.concat_arr(cols)
    sorted_row = row.arr.sort(nulls_last=True)
    cases = {
        "median alone": lambda: df.select(row.arr.median().alias("m")).height,
        "median + mean_horizontal": lambda: df.select([row.arr.median().alias("m"), pl.mean_horizontal(cols).alias("h")]).height,
        "median + median": lambda: df.select([row.arr.median().alias("m"), row.arr.median().alias("m2")]).height,
        "sort().get() alone": lambda: df.select(sorted_row.arr.get(42).alias("a")).height,
        "sort() materialised alone": lambda: df.select(sorted_row.alias("s")).height,
        "median alone, streaming": lambda: df.lazy().select(row.arr.median().alias("m")).collect(engine="streaming").height,
    }
    print(f"\n--- single-expression trap, {n_rows:,} x {n_cols} ---")
    for name, fn in cases.items():
        fn()
        print(f"{name:34s} {min(_time(lambda _df, _stats: fn(), df, None) for _ in range(repeats)):8.3f}s")


def bench(n_rows: int = 400_000, n_cols: int = 85, repeats: int = 3) -> None:
    """Warm each path once, report best-of-N for both stat sets, and check the values against numpy."""
    df = make_frame(n_rows, n_cols)
    print(f"\n================ {n_rows:,} rows x {n_cols} cols ================")

    print(f"--- all stats {FULL_STATS} ---")
    ref = _run(df, FULL_STATS, FULL_PATHS, repeats)

    print(f"\n--- no median/quantile {CHEAP_STATS} ---")
    _run(df, CHEAP_STATS, CHEAP_PATHS, repeats)

    _exactness(df, ref, FULL_STATS, production_path, "production (polars)")
    # The two fastest-looking variants are the two that do not actually answer the question; print the proof
    # next to their timings so the numbers above are never quoted on their own.
    _exactness(df, ref, FULL_STATS, polars_nan_native, "nan-native (order stats only)")
    _exactness(df, ref, FULL_STATS, polars_f32, "f32")


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [400_000]
    for size in sizes:
        bench(size)
    run_single_expression_trap()

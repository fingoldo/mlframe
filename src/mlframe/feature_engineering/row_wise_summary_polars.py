"""``row_wise_summary_stats_polars``: the per-row cross-sectional summary, computed natively in polars.

The pandas implementation converts the frame, materialises one contiguous float64 matrix and reduces it with
nan-aware numpy calls plus an njit row-quantile kernel -- which is genuinely fast, and a naive polars rewrite is
far SLOWER. Measured at 400,000 x 85 (5% NaN, polars 1.41.2, 16 physical cores), mean/std/min/max/median/q10/q90:

===========================================================  =========  ==================
construct                                                    time       vs pandas+numpy
===========================================================  =========  ==================
``pl.concat_list`` + ``.list.eval()``                        19.315s    0.10x
eager ``select``, interpolated ``.arr.sort().arr.get()``      3.702s    0.50x
``.lazy().select(...).collect()``, default engine             3.095s    0.60x
``pl.concat_arr`` + ``.arr.*``, nearest-rank (inexact)        2.188s    0.85x
pandas + numpy (the reference)                                1.862s    --
**this module** (streaming engine, pre-materialised row)      0.396s    **4.70x faster**
===========================================================  =========  ==================

and, for the subset that needs no ordering (mean/std/min/max):

===========================================================  =========  ==================
construct                                                    time       vs pandas+numpy
===========================================================  =========  ==================
pandas + numpy                                                1.546s    --
eager horizontal reductions (unstable std)                    0.431s    3.59x
**this module** (streaming engine)                            0.194s    **7.96x faster**
===========================================================  =========  ==================

The win grows with size, because the numpy path pays a whole-frame conversion that streaming morsels never
materialise: 3.43x / 4.70x / 6.01x on the full stat set at 50k / 400k / 2M rows, and 5.42x / 7.96x / 11.93x
without order statistics. Three findings drive the shape of the module, all reproducible from
``_benchmarks/bench_row_wise_polars.py``:

1. The engine matters far more than the expression. Eagerly, any per-row array kernel over 400k x 85 costs
   ~2.0-2.6s on a 16-core box -- ``.arr.median()``, ``.arr.sort().arr.get(i)``, all of them -- while the numba
   reference does the same work in 0.26s purely by running ``prange`` across rows. ``collect(engine="streaming")``
   splits the frame into morsels and runs the pipeline on all cores, and the same expressions drop to ~0.23-0.30s.
   That ~9x is the entire reason polars can win here at all.
2. Under the streaming engine, materialising ``pl.concat_arr`` and its sorted twin ONCE into (per-morsel) helper
   columns beats letting each expression rebuild them: 0.385s vs 0.640s at 400k, because the six ``.arr.get()``
   calls a three-quantile request makes would otherwise each pay their own partial selection over the row. This
   is the opposite of the eager result, where forcing the same intermediate is catastrophic (13.8s just to
   materialise the sorted column) -- the morsel-sized materialisation is what makes it cheap, so the two-stage
   plan below is only a good plan as a *streaming* plan.
3. Median is read off that shared sorted array as the q=0.5 quantile rather than through ``.arr.median()``:
   0.52s vs 0.75s at 400k for the full stat set. The two agree to 2.8e-17, so this is purely about not paying a
   second per-row pass when a sorted array is already on hand for q10/q90.

A measurement trap worth knowing before re-deriving any of this: an eager ``select`` holding exactly ONE
expression takes a ~10x slower path than the same expression alongside any second one -- a lone
``.arr.median()`` measures 20.1s, and 2.0s the moment a trivial ``pl.mean_horizontal`` is selected next to it.
Microbenchmarking a single per-row array kernel on its own therefore reports a number that never occurs in a
real multi-stat call.

Exactness against the numpy reference is a requirement, not a hope. Measured at 400k x 85: every order statistic
(min/max/median/q10/q90) is bit-identical, ``mean`` differs by at most 3.9e-16 and ``std`` by 8.9e-16 (float
summation order only). ``std`` needs ``ddof=0`` to match ``np.nanstd``'s default, and quantiles need the same
linear interpolation ``np.nanquantile`` uses -- ``.arr.get()`` accepts a per-row expression index, so the
interpolation position varies with each row's own count of non-null values.

``std`` deliberately goes through ``.arr.std()`` (a centred two-pass reduction) rather than the horizontal
``(s2 - s1^2/k)/k`` identity: that identity is catastrophically cancellation-prone on large-offset columns and
has already produced three live correctness bugs in this repo.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import polars as pl

# Shared with the pandas implementation on purpose: this module is a drop-in for it, and a different
# default stat set would mean the two produce different COLUMNS for the same call.
from mlframe.feature_engineering.row_wise_summary import _DEFAULT_STATS

# Stats expressible as a horizontal reduction, which needs no intermediate row array at all.
_HORIZONTAL_STATS = frozenset({"mean", "min", "max", "sum"})

_HORIZONTAL_REDUCERS = {"mean": pl.mean_horizontal, "min": pl.min_horizontal, "max": pl.max_horizontal, "sum": pl.sum_horizontal}

# Helper columns live only inside the streaming plan and are dropped by the final projection; the prefix keeps
# them from colliding with a caller's own column names.
_HELPER_PREFIX = "__rws"


def _stat_kind(stat: Union[str, float]) -> Tuple[str, Union[str, float]]:
    """Classify one requested stat as ``("horizontal"|"std"|"quantile", payload)``, raising on anything else."""
    if isinstance(stat, str):
        if stat in _HORIZONTAL_STATS:
            return ("horizontal", stat)
        if stat == "std":
            return ("std", "std")
        if stat == "median":
            return ("quantile", 0.5)
        if stat.startswith("q") and stat[1:].isdigit():
            return ("quantile", int(stat[1:]) / 100.0)
    elif isinstance(stat, (int, float)) and not isinstance(stat, bool) and 0.0 <= float(stat) <= 1.0:
        return ("quantile", float(stat))
    raise ValueError(f"row_wise_summary_stats_polars: unrecognized stat {stat!r}")


def _stat_name(stat: Union[str, float]) -> str:
    """Output-column suffix for one requested stat."""
    return stat if isinstance(stat, str) else f"q{round(float(stat) * 100)}"


def _quantile_expr(sorted_row: pl.Expr, k: pl.Expr, q: float) -> pl.Expr:
    """Linear-interpolated quantile at ``q`` over a row sorted with nulls last.

    Mirrors ``np.nanquantile``'s default: position ``h = q * (k - 1)`` over the row's own ``k`` non-null
    values, then interpolate between the two neighbouring order statistics.
    """
    h = (k - 1).cast(pl.Float64) * q
    lo = h.floor().cast(pl.Int64)
    frac = h - h.floor()
    low = sorted_row.arr.get(lo, null_on_oob=True)
    high = sorted_row.arr.get((lo + 1).clip(upper_bound=k - 1), null_on_oob=True)
    return low * (1.0 - frac) + high * frac


def _stats_block(
    cols: Sequence[str],
    stats: Sequence[Union[str, float]],
    kinds: Sequence[Tuple[str, Union[str, float]]],
    prefix: str,
    tag: str,
) -> list[pl.Expr]:
    """Output expressions for one column group, reading the helper columns named by ``tag``."""
    row = pl.col(f"{_HELPER_PREFIX}_row_{tag}")
    sorted_row = pl.col(f"{_HELPER_PREFIX}_sorted_{tag}")
    count = pl.col(f"{_HELPER_PREFIX}_count_{tag}")

    exprs: list[pl.Expr] = []
    for stat, (kind, payload) in zip(stats, kinds):
        alias = f"{prefix}_{_stat_name(stat)}"
        if kind == "horizontal":
            exprs.append(_HORIZONTAL_REDUCERS[str(payload)](cols).alias(alias))
        elif kind == "std":
            # ddof=0 to match np.nanstd's default; polars defaults to 1 and would differ in the third decimal.
            exprs.append(row.arr.std(ddof=0).alias(alias))
        else:
            exprs.append(_quantile_expr(sorted_row, count, float(payload)).alias(alias))
    return exprs


def row_wise_summary_stats_polars(
    X: pl.DataFrame,
    columns: Optional[Sequence[str]] = None,
    stats: Sequence[Union[str, float]] = _DEFAULT_STATS,
    column_prefix: str = "row_summary",
    groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> pl.DataFrame:
    """Per-row summary statistics ACROSS a block of columns, without leaving polars.

    Same contract as :func:`mlframe.feature_engineering.row_wise_summary.row_wise_summary_stats`: one output
    column per stat (or per ``group x stat``), same row count and order as ``X``, NaN ignored rather than
    propagated. NaN is converted to null up front because polars treats NaN as an ordinary float value while
    the numpy reference skips it -- without that the two implementations answer different questions.

    The whole thing is planned lazily and collected through the streaming engine: the per-row array kernels are
    single-threaded per expression, so morsel parallelism is what makes this beat the numpy path rather than
    lose to it by 2x (see the module docstring for the measurements).
    """
    kinds = [_stat_kind(stat) for stat in stats]

    if groups is not None:
        blocks = [(str(i), list(group_cols), f"{column_prefix}_{group_name}") for i, (group_name, group_cols) in enumerate(groups.items())]
    else:
        cols = list(columns) if columns is not None else [c for c, dtype in zip(X.columns, X.dtypes) if dtype.is_numeric()]
        blocks = [("0", cols, column_prefix)]

    lf = X.lazy().with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))

    needs_row = any(kind in ("std", "quantile") for kind, _ in kinds)
    needs_order = any(kind == "quantile" for kind, _ in kinds)
    if needs_row:
        lf = lf.with_columns([pl.concat_arr(block_cols).alias(f"{_HELPER_PREFIX}_row_{tag}") for tag, block_cols, _ in blocks])
    if needs_order:
        # Sorted once per group and reused by every quantile: the six .arr.get() calls a three-quantile request
        # makes would otherwise each pay their own partial selection over the row.
        lf = lf.with_columns(
            [pl.col(f"{_HELPER_PREFIX}_row_{tag}").arr.sort(nulls_last=True).alias(f"{_HELPER_PREFIX}_sorted_{tag}") for tag, _, _ in blocks]
            + [
                pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in block_cols]).alias(f"{_HELPER_PREFIX}_count_{tag}")
                for tag, block_cols, _ in blocks
            ]
        )

    exprs: list[pl.Expr] = []
    for tag, block_cols, prefix in blocks:
        exprs.extend(_stats_block(block_cols, stats, kinds, prefix, tag))
    return lf.select(exprs).collect(engine="streaming")


__all__ = ["row_wise_summary_stats_polars"]

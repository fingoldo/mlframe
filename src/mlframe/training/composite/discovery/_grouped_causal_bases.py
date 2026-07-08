"""Strictly-causal PER-GROUP (per-entity) base-column engineering for composite-target discovery.

On a group-sequential target (a wellbore log ordered by measured-depth MD within each well, a per-entity panel ordered by
an index -- NO timestamp needed, only a real within-group order) the most valuable base is a strictly-causal lag / rolling
feature of the target COMPUTED WITHIN EACH GROUP: at predict time the previous in-group value ``y_{t-1}`` is already
observed, so it is a legitimate predictor, NOT leakage, and the additive inverse ``y = T_hat + y_prev`` is in-range BY
CONSTRUCTION (a per-row REAL previous value) -- it does not collapse on unseen groups the way a between-group base does.

This module materialises that family. It differs from :mod:`_base_engineering` (which lags the WHOLE frame in one global
time order) by respecting group boundaries: every engineered value at row ``i`` uses ONLY target rows strictly before ``i``
*within the same group*. A group's first row has no in-group predecessor -> filled with the group's first observed value
(``first_fill="group_first"``, the default, keeps every base finite so the inverse stays in-range) or ``NaN``
(``first_fill="nan"``, honest "no causal history yet"; downstream screening masks non-finite rows pairwise).

Ordering uses a SEGMENT sort (mirrors ``linear_residual_grouped`` in ``transforms/linear.py``): rows are sorted so each
group is a contiguous segment, ordered within the group by ``time_column`` (a monotone within-group key such as MD; when
``None`` the caller-guaranteed per-group frame row order is used), the causal bases are computed over the contiguous
segments, then mapped BACK to the original frame row order so each returned ndarray aligns positionally with ``df``.

Memory (100 GB-frame rule): the frame is never copied. Only ``target_col`` / ``group_field`` / ``time_column`` are pulled
as single ndarrays; :func:`attach_grouped_causal_bases` adds the engineered columns via polars ``with_columns``
(zero-copy on existing buffers) or a pandas shallow ``copy(deep=False)`` + column assign (shares existing blocks, never a
deep copy, never mutates the caller's frame).

cProfile (200k rows / 500 groups, ``lags=(1,2)`` + ``tmean3`` + ``expmean``; run ``python -m
mlframe.training.composite.discovery._grouped_causal_bases``): after a warm-up the wall is ~6 ms, dominated by the two
segment argsorts (``np.lexsort`` / ``np.argsort``, ~55%) and the back-map fancy-index gathers (~25%); the numba kernels
themselves are ~15%. Verdict: NO ACTIONABLE SPEEDUP -- the O(n log n) sort is the floor for an ordered-within-group
computation and numpy's sort already dispatches to native code; the per-group causal loops are @njit with O(n) running
sums (no per-row O(window) rescan). The sort is not worth a GPU hand-off at these shapes (transfer dwarfs the 6 ms).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._base_engineering import _extract_column, add_engineered_bases_to_pool

try:  # numba is a core mlframe dep; the pure-numpy fallback keeps the module importable in a stripped CI env.
    import numba

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover -- numba always present in prod
    _HAVE_NUMBA = False


_VALID_OPS = ("lag", "trailing_mean", "expanding_mean")


def _njit(func):
    """Apply ``numba.njit(cache=True)`` when available, else return the plain Python function (correctness-preserving)."""
    if _HAVE_NUMBA:
        return numba.njit(cache=True)(func)
    return func


def _grouped_lag_impl(y_sorted, offsets, k, fill_group):
    n = y_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_groups = offsets.shape[0] - 1
    for g in range(n_groups):
        lo = offsets[g]
        hi = offsets[g + 1]
        if hi <= lo:
            continue
        first = y_sorted[lo]
        for j in range(lo, hi):
            if j - lo >= k:
                out[j] = y_sorted[j - k]
            else:
                out[j] = first if fill_group else np.nan
    return out


def _grouped_expanding_impl(y_sorted, offsets, fill_group):
    n = y_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_groups = offsets.shape[0] - 1
    for g in range(n_groups):
        lo = offsets[g]
        hi = offsets[g + 1]
        if hi <= lo:
            continue
        first = y_sorted[lo]
        run_sum = 0.0
        for j in range(lo, hi):
            pos = j - lo
            if pos == 0:
                out[j] = first if fill_group else np.nan
            else:
                out[j] = run_sum / pos
            run_sum += y_sorted[j]
    return out


def _grouped_trailing_impl(y_sorted, offsets, window, fill_group):
    # Trailing mean over the up-to-``window`` in-group rows STRICTLY BEFORE each row, via an O(n) running-window sum
    # (add the current element, drop the one leaving the window) -- no per-row O(window) rescan.
    n = y_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_groups = offsets.shape[0] - 1
    for g in range(n_groups):
        lo = offsets[g]
        hi = offsets[g + 1]
        if hi <= lo:
            continue
        first = y_sorted[lo]
        win_sum = 0.0
        count = 0
        for j in range(lo, hi):
            if count == 0:
                out[j] = first if fill_group else np.nan
            else:
                out[j] = win_sum / count
            win_sum += y_sorted[j]
            count += 1
            if count > window:
                win_sum -= y_sorted[j - window]
                count -= 1
    return out


_grouped_lag_kernel = _njit(_grouped_lag_impl)
_grouped_expanding_kernel = _njit(_grouped_expanding_impl)
_grouped_trailing_kernel = _njit(_grouped_trailing_impl)


def _extract_raw(df: Any, col: str) -> np.ndarray:
    """Pull a single column as a raw 1-D ndarray (polars or pandas) WITHOUT a float cast -- group keys may be strings."""
    getcol = getattr(df, "get_column", None)
    if callable(getcol):  # polars
        return np.asarray(df.get_column(col).to_numpy())
    return df[col].to_numpy() if hasattr(df[col], "to_numpy") else np.asarray(df[col])


def _group_codes(df: Any, group_field: str, n: int) -> tuple[np.ndarray, int]:
    """Dense int64 group codes (0..K-1) for ``group_field``; second element is K. Works on string / numeric keys."""
    raw = _extract_raw(df, group_field).reshape(-1)
    if raw.shape[0] != n:
        raise ValueError(f"group column '{group_field}' length {raw.shape[0]} != target length {n}")
    _, codes = np.unique(raw, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    return codes, (int(codes.max()) + 1 if n else 0)


def _segment_order(df: Any, codes: np.ndarray, time_column: str | None, n: int) -> np.ndarray:
    """Row order making each group a contiguous ascending-code segment, ordered within the group by ``time_column``.

    ``time_column=None`` uses a STABLE argsort of the group codes, preserving the caller-guaranteed original within-group
    row order. With a time key, ``np.lexsort((t, codes))`` sorts primarily by group, secondarily by time (lexsort is
    stable, so equal-time rows keep original order).
    """
    if time_column is None:
        return np.argsort(codes, kind="stable")
    t = _extract_column(df, time_column)
    if t.shape[0] != n:
        raise ValueError(f"time column '{time_column}' length {t.shape[0]} != target length {n}")
    return np.asarray(np.lexsort((t, codes)))


def engineer_grouped_causal_bases(
    df: Any,
    target_col: str,
    group_field: str,
    time_column: str | None = None,
    *,
    lags: Sequence[int] = (1,),
    trailing_windows: Sequence[int] = (3,),
    ops: Sequence[str] = ("lag", "trailing_mean", "expanding_mean"),
    first_fill: str = "group_first",
) -> dict[str, np.ndarray]:
    """Build strictly-causal PER-GROUP base columns from the target series, aligned to the frame's original row order.

    Parameters
    ----------
    df : polars or pandas frame (NOT copied -- only ``target_col`` / ``group_field`` / ``time_column`` are pulled).
    target_col : target column ``y`` the bases are derived from.
    group_field : per-entity key column (e.g. well id). Bases never cross a group boundary.
    time_column : monotone within-group order key (e.g. MD). ``None`` -> use the existing per-group frame row order.
    lags : lag offsets ``k >= 1`` for the ``"lag"`` op (lag 0 would be the current target = leakage).
    trailing_windows : trailing window sizes ``w >= 1`` for ``"trailing_mean"`` (window excludes the current row).
    ops : subset of ``{"lag", "trailing_mean", "expanding_mean"}``.
    first_fill : ``"group_first"`` fills a group's history-less head rows with the group's first observed value (base
        stays finite / in-range); ``"nan"`` leaves them NaN (honest "no causal history yet").

    Returns
    -------
    dict ``{engineered_base_name -> ndarray}`` in original-frame row order. Names: ``"<target>__gcausal_lag<k>"``,
    ``"<target>__gcausal_tmean<w>"``, ``"<target>__gcausal_expmean"``.
    """
    bad_ops = [o for o in ops if o not in _VALID_OPS]
    if bad_ops:
        raise ValueError(f"unknown op(s) {bad_ops}; valid: {_VALID_OPS}")
    if any(int(k) < 1 for k in lags):
        raise ValueError("lags must be >= 1 (lag 0 is the current target = leakage)")
    if any(int(w) < 1 for w in trailing_windows):
        raise ValueError("trailing_windows must be >= 1")
    if first_fill not in ("group_first", "nan"):
        raise ValueError("first_fill must be 'group_first' or 'nan'")

    y = _extract_column(df, target_col)
    n = y.shape[0]
    codes, n_groups = _group_codes(df, group_field, n)
    order = _segment_order(df, codes, time_column, n)
    y_sorted = y[order]

    counts = np.bincount(codes, minlength=n_groups)
    offsets = np.empty(n_groups + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)
    fill_group = first_fill == "group_first"

    out: dict[str, np.ndarray] = {}
    if "lag" in ops:
        for k in lags:
            sorted_vec = _grouped_lag_kernel(y_sorted, offsets, int(k), fill_group)
            out[f"{target_col}__gcausal_lag{int(k)}"] = sorted_vec[inv]
    if "trailing_mean" in ops:
        for w in trailing_windows:
            sorted_vec = _grouped_trailing_kernel(y_sorted, offsets, int(w), fill_group)
            out[f"{target_col}__gcausal_tmean{int(w)}"] = sorted_vec[inv]
    if "expanding_mean" in ops:
        sorted_vec = _grouped_expanding_kernel(y_sorted, offsets, fill_group)
        out[f"{target_col}__gcausal_expmean"] = sorted_vec[inv]
    return out


def _is_polars(df: Any) -> bool:
    return callable(getattr(df, "with_columns", None)) and callable(getattr(df, "get_column", None))


def _frame_columns(df: Any) -> list[str]:
    cols = getattr(df, "columns", None)
    return list(cols) if cols is not None else []


def _attach_columns(df: Any, cols: Mapping[str, np.ndarray]) -> Any:
    """Return ``df`` augmented with ``cols`` WITHOUT copying existing data or mutating the caller's frame."""
    if _is_polars(df):
        import polars as pl

        return df.with_columns([pl.Series(name, np.asarray(arr, dtype=np.float32)) for name, arr in cols.items()])
    # pandas: shallow copy shares existing blocks (no data copy); assigning new columns touches only the copy.
    df2 = df.copy(deep=False)
    for name, arr in cols.items():
        df2[name] = np.asarray(arr, dtype=np.float32)
    return df2


def attach_grouped_causal_bases(
    df: Any,
    target_col: str,
    group_field: str,
    time_column: str | None = None,
    *,
    existing_columns: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, list[str]]:
    """Engineer grouped causal bases and attach them to ``df`` as columns (memory-safe). Returns (augmented_df, names).

    Names already present in ``df`` (or ``existing_columns``) are skipped so a user-supplied causal lag is never
    overwritten. ``**kwargs`` forward to :func:`engineer_grouped_causal_bases`.
    """
    engineered = engineer_grouped_causal_bases(df, target_col, group_field, time_column, **kwargs)
    if not engineered:
        return df, []
    existing = set(existing_columns) if existing_columns is not None else set(_frame_columns(df))
    to_add = {name: arr for name, arr in engineered.items() if name not in existing}
    if not to_add:
        return df, []
    return _attach_columns(df, to_add), list(to_add.keys())


def maybe_add_grouped_causal_bases(
    self: Any,
    df: Any,
    target_col: str,
    usable_features: Sequence[str],
    base_candidates: Sequence[str],
    train_idx: np.ndarray,
) -> tuple[Any, list[str], list[str]]:
    """Discovery wire point: when grouped causal engineering is enabled AND a group key is available, materialise the
    strictly-causal per-group bases, attach them to ``df``, and register them as both first-class features (so the MI
    baseline sees them) and base candidates. Returns ``(df, usable_features, base_candidates)`` (unchanged when disabled).

    Config toggles are read defensively via ``getattr`` (the parent registers the fields later):
    ``engineer_causal_bases`` (default True), ``engineer_causal_group_column`` / ``group_column`` (the group key),
    ``time_column`` (optional within-group order), ``engineer_causal_lags`` / ``engineer_causal_trailing_windows`` /
    ``engineer_causal_ops`` / ``engineer_causal_first_fill``.
    """
    config = getattr(self, "config", None)
    usable_list = list(usable_features)
    base_list = list(base_candidates)
    if config is None or not getattr(config, "engineer_causal_bases", True):
        return df, usable_list, base_list

    group_field = getattr(config, "engineer_causal_group_column", None) or getattr(config, "group_column", None)
    if not group_field or group_field not in _frame_columns(df):
        return df, usable_list, base_list

    time_column = getattr(config, "time_column", None)
    lags = getattr(config, "engineer_causal_lags", (1,))
    trailing_windows = getattr(config, "engineer_causal_trailing_windows", (3,))
    ops = getattr(config, "engineer_causal_ops", ("lag", "trailing_mean", "expanding_mean"))
    first_fill = getattr(config, "engineer_causal_first_fill", "group_first")

    # Route through the (formerly dormant) pool helper so grouped bases flow via the one engineered-base entry point.
    engineered = add_engineered_bases_to_pool(
        None, df, target_col, time_column,
        group_field=group_field, lags=lags, trailing_windows=trailing_windows, ops=ops, first_fill=first_fill,
    )
    if not engineered:
        return df, usable_list, base_list

    existing = set(_frame_columns(df))
    to_add = {name: arr for name, arr in engineered.items() if name not in existing}
    if not to_add:
        return df, usable_list, base_list

    df = _attach_columns(df, to_add)
    for name in to_add:
        if name not in usable_list:
            usable_list.append(name)
        if name not in base_list:
            base_list.append(name)
    return df, usable_list, base_list


def _profile_main() -> None:  # pragma: no cover -- manual cProfile harness (see module docstring for the recorded verdict)
    import cProfile
    import pstats

    rng = np.random.default_rng(0)
    n, n_groups = 200_000, 500
    groups = rng.integers(0, n_groups, size=n)
    md = rng.random(n)
    y = rng.normal(size=n).astype(np.float64)
    try:
        import polars as pl

        df = pl.DataFrame({"y": y, "well": groups, "md": md})
    except Exception:
        import pandas as pd

        df = pd.DataFrame({"y": y, "well": groups, "md": md})

    engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1, 2), trailing_windows=(3,))  # warm numba
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1, 2), trailing_windows=(3,))
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":  # pragma: no cover
    _profile_main()

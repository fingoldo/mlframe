"""Offset-product pair FE: ``(u + s) * (v + t)`` with ``u = f(x_a)``, ``v = g(x_b)`` and closed-form shifts.

A product whose factor changes sign inside the data range (``ln 2 + ln c`` crosses zero at ``c = 0.5``) is an interaction the fixed unary x binary pair table cannot express, because every
preset form puts the sign change at 0. The shifts come from one 4x4 interaction regression on rank(y) (``_offset_product_kernels.ols2_shift``), so there is no grid; a fused scan
(``scan_offset_products``) scores every column pair x unary pair without storing a candidate.

Acceptance (see audits/2026-10-09/offset_product_fe/FOLLOWUP.md): for each column pair the unary pair with the best gain on the even rows is chosen, and it is kept only when its MI on the
odd (held-out) rows beats the best of five shift-free baselines (``u*v``, ``u+v``, ``u``, ``v`` and their least-squares weighted sum) by ``ACCEPT_MARGIN_C / n``. The shifts of the kept candidates are then refitted on every row.
Replay (kind ``offset_product``) is the pure function ``clip((u + s) * (v + t))`` of the two source columns; the target is never used at transform time.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Callable, Optional, cast, Sequence

import numpy as np
import pandas as pd

from ._offset_product_kernels import N_BASELINES, ols2_shift, scan_offset_products

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

__all__ = ["hybrid_offset_product_fe", "build_offset_product_recipe", "apply_offset_product_recipe", "OFFSET_UNARIES"]

# Unary maps scanned on each side. ``neg`` is omitted (the sign is absorbed by the regression) as are the non-monotone-in-the-wrong-way duplicates of ``abs``/``sqr`` pairs that the
# minimal preset already makes redundant.
OFFSET_UNARIES = ("identity", "abs", "sqr", "reciproc", "sqrt", "log", "sin")
ACCEPT_MARGIN_C = 40.0  # held-out MI gain must exceed ACCEPT_MARGIN_C / n_scan nats; the largest null gain seen over 6 seeds x 15 pairs x n in 5k..300k was 24 / n (_benchmarks/offset_product/null_gain.py), the stat study used 26 / n
DEFAULT_SCAN_ROWS = 100_000  # the scan decides on at most this many rows (stat study: decisions are stable from n=5k); accepted shifts are refitted on all rows
N_MI_BINS = 10
_WINSOR_Q = (0.01, 0.99)  # winsorisation of the unary outputs for the regression only
_OUTPUT_CLIP_Q = (0.001, 0.999)  # replay clips the product to its fit-time quantiles so heavy tails cannot dominate a linear consumer
_MIN_ROWS = 200


def _op_label(unary: str, col: str) -> str:
    """Display form of ``unary(col)``."""
    return col if unary == "identity" else f"{unary}({col})"


def _shift_label(v: float) -> str:
    """Signed compact number for a column name."""
    return f"{v:+.4g}"


@functools.cache
def _unary_funcs(preset: str) -> "dict[str, Callable]":
    """Memoised unary registry of ``preset`` (the same registry the ``unary_binary`` replay uses; read-only)."""
    from .feature_engineering import create_unary_transformations

    return cast("dict[str, Callable]", create_unary_transformations(preset=preset))


def _unary_column(fn: Callable, x: np.ndarray) -> np.ndarray:
    """``fn(x)`` as float64 with the invalid-value warnings of the registry's domain-restricted maps silenced."""
    with np.errstate(all="ignore"):
        return np.asarray(fn(x), dtype=np.float64)


def _finite_fill(u: np.ndarray) -> float:
    """Mean of the finite entries of ``u`` (0.0 when there are none): the value a non-finite unary output is replaced with, at fit and at replay."""
    ok = np.isfinite(u)
    return float(u[ok].mean()) if ok.any() else 0.0


def _rank_scaled(y: np.ndarray) -> np.ndarray:
    """Average ranks of ``y`` scaled into ``(0, 1]``."""
    from scipy.stats import rankdata

    return np.asarray(rankdata(y, method="average") / float(len(y)), dtype=np.float64)


def build_offset_product_recipe(
    *, name: str, src_names: Sequence[str], unary_names: Sequence[str], unary_preset: str, shifts: Sequence[float], fills: Sequence[float], out_clip: Sequence[float]
) -> "EngineeredRecipe":
    """Frozen recipe of one offset-product column; stores only the shifts, the non-finite fills and the output clip (no target)."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="offset_product",
        src_names=tuple(str(c) for c in src_names),
        unary_names=tuple(str(u) for u in unary_names),
        unary_preset=unary_preset,
        extra={
            "s": float(shifts[0]),
            "t": float(shifts[1]),
            "fill_u": float(fills[0]),
            "fill_v": float(fills[1]),
            "clip_lo": float(out_clip[0]),
            "clip_hi": float(out_clip[1]),
        },
    )


def _product(u: np.ndarray, v: np.ndarray, s: float, t: float, fill_u: float, fill_v: float) -> np.ndarray:
    """``(u + s) * (v + t)`` with non-finite unary outputs replaced by their fit-time fills."""
    u = np.where(np.isfinite(u), u, fill_u)
    v = np.where(np.isfinite(v), v, fill_v)
    return (u + s) * (v + t)


def apply_offset_product_recipe(recipe, X) -> np.ndarray:
    """Replay one offset-product column from the stored shifts; a pure function of the two source columns."""
    from .engineered_recipes.shared import extract_column

    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        raise ValueError(f"offset_product recipe '{recipe.name}' needs 2 src_names and 2 unary_names; got {recipe.src_names} / {recipe.unary_names}")
    ex = recipe.extra
    funcs = _unary_funcs(recipe.unary_preset)
    ua, ub = recipe.unary_names
    for un in (ua, ub):
        if un not in funcs:
            raise KeyError(f"Unary function '{un}' not in '{recipe.unary_preset}' preset. Replay requires the same preset that was active at fit time.")
    u = _unary_column(funcs[ua], np.asarray(extract_column(X, recipe.src_names[0]), dtype=np.float64))
    v = _unary_column(funcs[ub], np.asarray(extract_column(X, recipe.src_names[1]), dtype=np.float64))
    out = _product(u, v, ex["s"], ex["t"], ex["fill_u"], ex["fill_v"])
    return np.asarray(np.clip(out, ex["clip_lo"], ex["clip_hi"]))


def _scan_inputs(cols: "list[np.ndarray]", funcs: "dict[str, Callable]", unaries: Sequence[str], rows: np.ndarray):
    """Unary outputs of every pooled column on the scan rows, non-finite entries filled, plus the winsorisation bounds fitted on the even (training) rows."""
    m, nu, n = len(cols), len(unaries), len(rows)
    U = np.empty((m, nu, n), dtype=np.float64)
    clips = np.empty((m, nu, 2), dtype=np.float64)
    for ci, x in enumerate(cols):
        xs = x[rows]
        for ui, un in enumerate(unaries):
            u = _unary_column(funcs[un], xs)
            u = np.where(np.isfinite(u), u, _finite_fill(u))
            U[ci, ui] = u
            clips[ci, ui] = np.quantile(u[::2], _WINSOR_Q)
    return U, clips


def _scan(U: np.ndarray, tasks: np.ndarray, yr: np.ndarray, codes: np.ndarray, ky: int, clips: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(out_shift, out_mi)`` of the scan: the fused CUDA kernel in strict-resident GPU mode (``None`` from it, or any device failure, falls back), else the njit/prange kernel."""
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled

        if fe_gpu_strict_resident_enabled():
            from ._offset_product_gpu import scan_offset_products_gpu

            res = scan_offset_products_gpu(U, tasks, yr, codes, ky, N_MI_BINS, clips)
            if res is not None:
                return res
    except Exception as e:  # device unavailable or kernel fault: the CPU scan below gives the same selection
        logger.debug("offset-product GPU scan fell back to the CPU kernel: %r", e)
    out_shift = np.empty((len(tasks), 2))
    out_mi = np.empty((len(tasks), 2, 1 + N_BASELINES))
    scan_offset_products(U, tasks, yr, codes, ky, N_MI_BINS, clips, out_shift, out_mi)
    return out_shift, out_mi


def hybrid_offset_product_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    max_pair_cols: int = 6,
    top_k: int = 3,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    unary_preset: str = "minimal",
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """Scan every pair of the top-``max_pair_cols`` raw numeric columns x unary pair, keep at most ``top_k`` held-out-validated offset products.

    Returns ``(X_aug, appended, recipes, enc_df)``. ``y`` only steers the scan and the acceptance; recipes carry the shifts, never ``y``."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_offset_product_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    empty: tuple[pd.DataFrame, list, list, pd.DataFrame] = (X, [], [], pd.DataFrame())
    cols = [c for c in (num_cols if num_cols else X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    n = len(X)
    if len(cols) < 2 or n < _MIN_ROWS or y is None:
        return empty
    from ._extra_fe_families import _top_mi_num_cols
    from ._y_encoding import encode_y_for_classif_mi

    cols = _top_mi_num_cols(X, cols, y, int(max_pair_cols))
    if len(cols) < 2:
        return empty
    y_arr = np.asarray(y).ravel()
    codes = np.asarray(encode_y_for_classif_mi(y_arr), dtype=np.int64)
    ky = int(codes.max()) + 1
    rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
    funcs = _unary_funcs(unary_preset)
    unaries = tuple(u for u in OFFSET_UNARIES if u in funcs)
    xs = [np.asarray(X[c].to_numpy(), dtype=np.float64) for c in cols]
    U, clips = _scan_inputs(xs, funcs, unaries, rows)
    nu = len(unaries)
    pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    tasks = np.array([(i, j, a, b) for (i, j) in pairs for a in range(nu) for b in range(nu)], dtype=np.int64)
    _shifts, out_mi = _scan(U, tasks, _rank_scaled(y_arr[rows]), codes[rows], ky, clips)
    n_scan = len(rows)
    margin = ACCEPT_MARGIN_C / n_scan
    accepted = []
    per_pair = len(unaries) ** 2
    for p, (i, j) in enumerate(pairs):
        rows_p = slice(p * per_pair, (p + 1) * per_pair)
        with np.errstate(invalid="ignore", all="ignore"):
            # The winner is picked on the even rows against the baselines of the SAME unary pair; the held-out test then compares it with the best shift-free baseline of ANY unary
            # pair of this column pair (the preset's own products and sums included), so a product that is only a better additive fit than some other unary choice is not accepted.
            train_gain = out_mi[rows_p, 0, 0] - np.nanmax(out_mi[rows_p, 0, 1:], axis=1)
            if not np.isfinite(train_gain).any():
                continue
            k = p * per_pair + int(np.nanargmax(train_gain))
            g_ho = float(out_mi[k, 1, 0] - np.nanmax(out_mi[rows_p, 1, 1:]))
        if g_ho > margin:
            accepted.append((g_ho, i, j, int(tasks[k, 2]), int(tasks[k, 3])))
        elif reject_sink is not None:
            reject_sink(gate="offset_product_heldout_margin", candidate=f"{cols[i]}x{cols[j]}", operand_names=f"{cols[i]},{cols[j]}", operator="offmul", observed=g_ho, threshold=margin, reason="held-out MI gain below margin")
    accepted.sort(key=lambda r: -r[0])
    yr_full = _rank_scaled(y_arr)
    new_cols: "dict[str, np.ndarray]" = {}
    recipes = []
    for _, i, j, a, b in accepted[: int(top_k)]:
        ua, ub = unaries[a], unaries[b]
        u = _unary_column(funcs[ua], xs[i])
        v = _unary_column(funcs[ub], xs[j])
        fills = (_finite_fill(u), _finite_fill(v))
        u = np.where(np.isfinite(u), u, fills[0])
        v = np.where(np.isfinite(v), v, fills[1])
        qu = np.quantile(u, _WINSOR_Q)
        qv = np.quantile(v, _WINSOR_Q)
        shifts = np.empty(4)
        ols2_shift(u, v, yr_full, 0, 1, qu[0], qu[1], qv[0], qv[1], shifts)
        if not np.isfinite(shifts[:2]).all():
            continue
        col = _product(u, v, shifts[0], shifts[1], fills[0], fills[1])
        lo, hi = np.quantile(col, _OUTPUT_CLIP_Q)
        if not hi > lo:
            continue
        name = f"offmul({_op_label(ua, cols[i])}{_shift_label(shifts[0])},{_op_label(ub, cols[j])}{_shift_label(shifts[1])})"
        if name in X.columns or name in new_cols:
            continue
        new_cols[name] = np.clip(col, lo, hi)
        recipes.append(
            build_offset_product_recipe(
                name=name, src_names=(cols[i], cols[j]), unary_names=(ua, ub), unary_preset=unary_preset, shifts=(float(shifts[0]), float(shifts[1])), fills=fills, out_clip=(lo, hi)
            )
        )
    if not new_cols:
        return empty
    enc_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, enc_df], axis=1), list(new_cols), recipes, enc_df

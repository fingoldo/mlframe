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
from typing import TYPE_CHECKING, Callable, Optional, Sequence

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
ACCEPT_MIN_RELATIVE_GAIN = 0.03  # and the gain must be at least this fraction of the best baseline held-out MI: at 100k+ rows 40 / n is under 0.001 nats, a gain no consumer can use
DEFAULT_SCAN_ROWS = 100_000  # the scan decides on at most this many rows (stat study: decisions are stable from n=5k); accepted shifts are refitted on all rows
N_MI_BINS = 10
_WINSOR_Q = (0.01, 0.99)  # winsorisation of the unary outputs for the regression only
_OUTPUT_CLIP_Q = (0.001, 0.999)  # replay clips the product to its fit-time quantiles so heavy tails cannot dominate a linear consumer
_MIN_ROWS = 200
SYNERGY_MIN = 0.02  # a pair is scanned only if its joint 10 x 10 Miller-Madow MI exceeds the larger marginal MI by this many nats; the 25 planted pairs of the pre-filter study had >= 0.041


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

    return create_unary_transformations(preset=preset)


def _unary_column(fn: Callable, x: np.ndarray) -> np.ndarray:
    """``fn(x)`` as float64 with the invalid-value warnings of the registry's domain-restricted maps silenced."""
    with np.errstate(all="ignore"):
        return np.asarray(fn(x), dtype=np.float64)


def _log_anchor(x: np.ndarray) -> float:
    """The shift ``smart_log`` applies to the fit column (``1e-5 - min`` for a column with non-positive values, else 0), frozen so replay on another batch cannot shift differently."""
    lo = float(np.nanmin(x)) if np.isfinite(x).any() else 1.0
    return 0.0 if lo > 0 else 1e-5 - lo


def _apply_unary(funcs: "dict[str, Callable]", name: str, x: np.ndarray, log_shift: float = 0.0) -> np.ndarray:
    """Unary map ``name`` of ``x``; ``log`` uses the frozen ``log_shift`` instead of the batch-dependent shift of the registry's ``smart_log``."""
    if name == "log":
        with np.errstate(all="ignore"):
            return np.asarray(np.log(x + log_shift) if log_shift != 0.0 else np.log(x), dtype=np.float64)
    return _unary_column(funcs[name], x)


def _finite_fill(u: np.ndarray) -> float:
    """Mean of the finite entries of ``u`` (0.0 when there are none): the value a non-finite unary output is replaced with, at fit and at replay."""
    ok = np.isfinite(u)
    return float(u[ok].mean()) if ok.any() else 0.0


def _rank_all_rows(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Average ranks of every row of ``y`` scaled into ``(0, 1]``, taken against the sorted scan rows when ``y`` has more rows than that (a binary search per row instead of a full argsort:
    0.04 s against 0.26 s at 1M rows, with a rank error of 1 / len(rows), far below what the shift regression resolves)."""
    if len(rows) >= len(y):
        return _rank_scaled(y)
    ref = np.sort(y[rows])
    left = np.searchsorted(ref, y, side="left")
    right = np.searchsorted(ref, y, side="right") if (ref[1:] == ref[:-1]).any() else left
    return (left + right + 1) / (2.0 * len(ref))


def _rank_scaled(y: np.ndarray) -> np.ndarray:
    """Average ranks of ``y`` scaled into ``(0, 1]``."""
    from scipy.stats import rankdata

    return rankdata(y, method="average") / float(len(y))


def build_offset_product_recipe(
    *, name: str, src_names: Sequence[str], unary_names: Sequence[str], unary_preset: str, shifts: Sequence[float], fills: Sequence[float], out_clip: Sequence[float],
    log_shifts: Sequence[float] = (0.0, 0.0),
) -> "EngineeredRecipe":
    """Frozen recipe of one offset-product column; stores only the shifts, the non-finite fills, the frozen log anchors and the output clip (no target)."""
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
            "log_shift_u": float(log_shifts[0]),
            "log_shift_v": float(log_shifts[1]),
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
    u = _apply_unary(funcs, ua, np.asarray(extract_column(X, recipe.src_names[0]), dtype=np.float64), ex["log_shift_u"])
    v = _apply_unary(funcs, ub, np.asarray(extract_column(X, recipe.src_names[1]), dtype=np.float64), ex["log_shift_v"])
    out = _product(u, v, ex["s"], ex["t"], ex["fill_u"], ex["fill_v"])
    return np.clip(out, ex["clip_lo"], ex["clip_hi"])


def _scan_inputs(cols: "list[np.ndarray]", funcs: "dict[str, Callable]", unaries: Sequence[str], rows: np.ndarray, log_shifts: Optional[Sequence[float]] = None):
    """Unary outputs of every pooled column on the scan rows, non-finite entries filled, plus the winsorisation bounds fitted on the even (training) rows.

    ``log_shifts``: the frozen ``log`` anchor of each column (default: computed from the whole column, which is what the recipe stores)."""
    anchors = list(log_shifts) if log_shifts is not None else [_log_anchor(x) for x in cols]
    m, nu, n = len(cols), len(unaries), len(rows)
    U = np.empty((m, nu, n), dtype=np.float64)
    clips = np.empty((m, nu, 2), dtype=np.float64)
    for ci, x in enumerate(cols):
        xs = x[rows]
        for ui, un in enumerate(unaries):
            u = _apply_unary(funcs, un, xs, anchors[ci])
            u = np.where(np.isfinite(u), u, _finite_fill(u))
            U[ci, ui] = u
        clips[ci] = np.quantile(U[ci][:, ::2], _WINSOR_Q, axis=1).T
    return U, clips


def _mm_mi(codes: np.ndarray, n_codes: int, y_codes: np.ndarray, ky: int) -> float:
    """Miller-Madow corrected plug-in MI (nats) of integer ``codes`` in ``[0, n_codes)`` and ``y_codes`` in ``[0, ky)``."""
    n = len(codes)
    joint = np.bincount(codes * ky + y_codes, minlength=n_codes * ky).reshape(n_codes, ky).astype(np.float64)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] / n * np.log(joint[nz] * n / (px @ py)[nz])).sum())
    return mi - (int((px > 0).sum()) - 1) * (int((py > 0).sum()) - 1) / (2.0 * n)


def _quantile_codes(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bin codes of ``x`` in ``[0, n_bins)`` (ties share a bin; non-finite values go to the lowest bin)."""
    x = np.where(np.isfinite(x), x, -np.inf)
    edges = np.quantile(x[np.isfinite(x)], np.linspace(0.0, 1.0, n_bins + 1)[1:-1]) if np.isfinite(x).any() else np.zeros(n_bins - 1)
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def synergy_pairs(xs: "list[np.ndarray]", rows: np.ndarray, codes: np.ndarray, ky: int, min_synergy: float = SYNERGY_MIN) -> "list[tuple[int, int]]":
    """Column pairs whose joint MI with the target exceeds the larger marginal MI by more than ``min_synergy``: the only pairs a shifted product can help (the shifted forms carry no
    information beyond the joint distribution of the two columns). Costs one bincount per pair on the scan rows, a few ms against the scan's O(unary pairs) passes."""
    q = [_quantile_codes(x[rows], N_MI_BINS) for x in xs]
    marg = [_mm_mi(c, N_MI_BINS, codes, ky) for c in q]
    return [
        (i, j)
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if _mm_mi(q[i] * N_MI_BINS + q[j], N_MI_BINS * N_MI_BINS, codes, ky) - max(marg[i], marg[j]) > min_synergy
    ]


def _scan_all(
    xs: "list[np.ndarray]", funcs: "dict[str, Callable]", unaries: Sequence[str], rows: np.ndarray, tasks: np.ndarray, yr: np.ndarray, codes: np.ndarray, ky: int, anchors: "list[float]"
) -> "tuple[np.ndarray, np.ndarray]":
    """``(out_mi, clips)`` of the scan over the pooled columns ``xs``: the fused CUDA path with device-built inputs when the kernel tuning cache (or, untuned, the strict-resident GPU mode) says
    so, else the njit/prange kernel on host-built inputs. A ``None`` from the device path, or any device failure, falls back to the host path (same selection)."""
    try:
        from .._benchmarks.kernel_tuning_cache.dispatch import lookup_offset_scan_backend
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled

        if lookup_offset_scan_backend(len(rows), int(tasks.shape[0]), strict_resident=fe_gpu_strict_resident_enabled()) == "gpu":
            from ._offset_product_gpu import scan_offset_products_device

            res = scan_offset_products_device(xs, tuple(unaries), rows, tasks, yr, codes, ky, N_MI_BINS, anchors, _WINSOR_Q)
            if res is not None:
                return res
    except Exception as e:  # device unavailable or kernel fault: the CPU scan below gives the same selection
        logger.debug("offset-product GPU scan fell back to the CPU kernel: %r", e)
    U, clips = _scan_inputs(xs, funcs, unaries, rows, anchors)
    out_shift = np.empty((len(tasks), 2))
    out_mi = np.empty((len(tasks), 2, 1 + N_BASELINES))
    scan_offset_products(U, tasks, yr, codes, ky, N_MI_BINS, clips, out_shift, out_mi)
    return out_mi, clips


def hybrid_offset_product_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    max_pair_cols: int = 6,
    top_k: int = 3,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    min_synergy: Optional[float] = SYNERGY_MIN,
    unary_preset: str = "minimal",
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """Scan every pair of the top-``max_pair_cols`` raw numeric columns x unary pair, keep at most ``top_k`` held-out-validated offset products.

    Returns ``(X_aug, appended, recipes, enc_df)``. ``y`` only steers the scan and the acceptance; recipes carry the shifts, never ``y``."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_offset_product_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    empty = (X, [], [], pd.DataFrame())
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
    rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
    codes = np.asarray(encode_y_for_classif_mi(y_arr[rows]), dtype=np.int64)
    ky = int(codes.max()) + 1
    funcs = _unary_funcs(unary_preset)
    unaries = tuple(u for u in OFFSET_UNARIES if u in funcs)
    xs = [np.asarray(X[c].to_numpy(), dtype=np.float64) for c in cols]
    nu = len(unaries)
    if min_synergy is None:
        pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    else:
        pairs = synergy_pairs(xs, rows, codes, ky, float(min_synergy))
    if not pairs:
        return empty
    used = sorted({c for pair in pairs for c in pair})
    slot = {c: k for k, c in enumerate(used)}
    tasks = np.array([(slot[i], slot[j], a, b) for (i, j) in pairs for a in range(nu) for b in range(nu)], dtype=np.int64)
    used_xs = [xs[c] for c in used]
    out_mi, clips = _scan_all(used_xs, funcs, unaries, rows, tasks, _rank_scaled(y_arr[rows]), codes, ky, [_log_anchor(x) for x in used_xs])
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
            base_ho = float(np.nanmax(out_mi[rows_p, 1, 1:]))
            g_ho = float(out_mi[k, 1, 0]) - base_ho
        if g_ho > margin and g_ho > ACCEPT_MIN_RELATIVE_GAIN * base_ho:
            accepted.append((g_ho, i, j, int(tasks[k, 2]), int(tasks[k, 3])))
        elif reject_sink is not None:
            reject_sink(gate="offset_product_heldout_margin", candidate=f"{cols[i]}x{cols[j]}", operand_names=f"{cols[i]},{cols[j]}", operator="offmul", observed=g_ho, threshold=margin, reason="held-out MI gain below margin")
    accepted.sort(key=lambda r: -r[0])
    yr_full = _rank_all_rows(y_arr, rows) if accepted else None  # ranking every row is only needed to refit the shifts of an accepted candidate
    new_cols: "dict[str, np.ndarray]" = {}
    recipes = []
    for _, i, j, a, b in accepted[: int(top_k)]:
        ua, ub = unaries[a], unaries[b]
        log_shifts = (_log_anchor(xs[i]) if ua == "log" else 0.0, _log_anchor(xs[j]) if ub == "log" else 0.0)
        u = _apply_unary(funcs, ua, xs[i], log_shifts[0])
        v = _apply_unary(funcs, ub, xs[j], log_shifts[1])
        fills = (_finite_fill(u), _finite_fill(v))
        u = np.where(np.isfinite(u), u, fills[0])
        v = np.where(np.isfinite(v), v, fills[1])
        qu = clips[slot[i], a]  # winsorisation bounds fitted on the scan rows, the same ones the scan regressed with
        qv = clips[slot[j], b]
        shifts = np.empty(4)
        ols2_shift(u, v, yr_full, 0, 1, qu[0], qu[1], qv[0], qv[1], shifts)
        if not np.isfinite(shifts[:2]).all():
            continue
        col = _product(u, v, shifts[0], shifts[1], fills[0], fills[1])
        lo, hi = np.quantile(col[rows], _OUTPUT_CLIP_Q)
        if not hi > lo:
            continue
        name = f"offmul({_op_label(ua, cols[i])}{_shift_label(shifts[0])},{_op_label(ub, cols[j])}{_shift_label(shifts[1])})"
        if name in X.columns or name in new_cols:
            continue
        new_cols[name] = np.clip(col, lo, hi)
        recipes.append(
            build_offset_product_recipe(
                name=name, src_names=(cols[i], cols[j]), unary_names=(ua, ub), unary_preset=unary_preset, shifts=shifts, fills=fills, out_clip=(lo, hi), log_shifts=log_shifts
            )
        )
    if not new_cols:
        return empty
    enc_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, enc_df], axis=1), list(new_cols), recipes, enc_df

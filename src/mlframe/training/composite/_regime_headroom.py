"""Regime-headroom map: WHERE (in data-space) does a composite target pay off, and where should the failsafe deploy?

The composite VALUE report (:mod:`_value_report`) answers "did the composite help, and per which GROUP?". This module
answers the orthogonal question: bin the rows by a chosen REGIME AXIS -- base-value quantiles, a feature, a group-level
score -- and per bin show the HEADROOM = how much the best achievable composite beats the ``min(raw, lag)`` failsafe. An
operator reads the map to see the regions where composites earn their keep (positive headroom) and the regions where the
lag/raw failsafe should be deployed instead (negative headroom, e.g. the extrapolated-base tail).

Per bin (rows sorted then quantile-cut on ``axis_values``): RMSE(raw), RMSE(composite), RMSE(lag, if given), the winner
(lowest RMSE), and ``headroom = (min(rmse_raw, rmse_lag) - rmse_composite) / min(rmse_raw, rmse_lag)`` -- positive means
the composite beats the best failsafe by that fraction; negative means the failsafe wins and the composite is a net loss
there. Without a lag baseline the failsafe reference is ``rmse_raw`` alone. All RMSEs in a bin are on the SAME rows:
rows non-finite in ANY supplied series (y, raw, composite, lag, weight) are dropped, so every comparison is matched.

The structured dict is JSON-serializable (no-data cells are ``None``, never NaN; serialize with ``sort_keys=True``). The
rendered text is ASCII-only (Windows cp1251-safe).

Performance (cProfile, 1M rows / 10 bins, ``_benchmarks/bench_regime_headroom.py``, best-of-5 full map ~193 ms): the two
costs are the per-bin SSE reduction and the quantile binning. The reduction was A/B'd -- the fused ``numba.njit``
single-pass kernel (:func:`_bin_stats_njit`) is ~12x faster than the four ``np.bincount`` passes (3.9 ms vs 47 ms at
n=1M) because it folds the finite-select + counts + SSE(raw/comp/lag) into one sweep with no length-n squared-error
temporaries, so it is the DEFAULT; :func:`_bin_stats_bincount` is the no-numba fallback. The remaining cost is the
``np.quantile`` (partition, ~44 ms) + ``np.searchsorted`` (~42 ms) binning, both intrinsic to exact quantile-cut and
already the fastest available exact primitives; verdict there: no actionable speedup.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import numba

    @numba.njit(cache=True)
    def _bin_stats_njit(codes, w, y, raw, comp, lag, has_lag, k):
        """Fused single-pass bin reduction: counts + weighted SSE(raw/comp/lag) in ONE sweep, skipping code -1 rows.

        Avoids the boolean-select + the several length-n squared-error temporaries the bincount path allocates; measured
        ~12x on the isolated reduction at n=1M / k=10 (3.9 ms vs 47 ms), so this is the DEFAULT.
        """
        rows = np.zeros(k, dtype=np.int64)
        W = np.zeros(k)
        sse_raw = np.zeros(k)
        sse_comp = np.zeros(k)
        sse_lag = np.zeros(k)
        for i in range(codes.shape[0]):
            b = codes[i]
            if b < 0:
                continue
            wi = w[i]
            yi = y[i]
            er = raw[i] - yi
            ec = comp[i] - yi
            rows[b] += 1
            W[b] += wi
            sse_raw[b] += wi * er * er
            sse_comp[b] += wi * ec * ec
            if has_lag:
                el = lag[i] - yi
                sse_lag[b] += wi * el * el
        return rows, W, sse_raw, sse_comp, sse_lag

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional accelerator here
    _HAVE_NUMBA = False

__all__ = [
    "regime_headroom_map",
    "render_regime_headroom_map",
]


def _as1d(a: Any) -> np.ndarray:
    """Coerce any array-like input to a contiguous 1-D float64 array (accepts pandas/polars Series, lists, ndarrays)."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _ascii(s: Any) -> str:
    """Force ASCII for printed/logged strings (cp1251 crashes on non-ASCII)."""
    return str(s).encode("ascii", "replace").decode("ascii")


def _quantile_bin_codes(axis: np.ndarray, valid: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign each VALID row a bin code in [0, k) by quantiles of ``axis``; returns (codes, edges).

    Ties are handled by unique-ing the quantile edges: a near-constant axis collapses to a single bin. Rows outside
    ``valid`` get code -1 (excluded). Edges are the k+1 boundaries of the realized bins (len == k+1).
    """
    codes = np.full(axis.shape[0], -1, dtype=np.int64)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return codes, np.asarray([], dtype=np.float64)
    av = axis[idx]
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(av, qs)
    edges = np.unique(raw_edges)  # collapse degenerate/duplicate edges -> fewer, well-defined bins
    if edges.shape[0] < 2:  # constant axis -> a single bin holding every valid row
        codes[idx] = 0
        lo = float(edges[0]) if edges.shape[0] else float(av[0])
        return codes, np.asarray([lo, lo], dtype=np.float64)
    k = edges.shape[0] - 1
    # np.searchsorted with the interior edges maps values into [0, k); clip guards the max edge and FP drift.
    c = np.searchsorted(edges[1:-1], av, side="right")
    codes[idx] = np.clip(c, 0, k - 1).astype(np.int64)
    return codes, edges


def _bin_stats_bincount(codes, w, y, raw, comp, lag, has_lag, k):
    """Vectorized fallback (no numba): select valid rows once (bincount rejects negatives), then O(n) bincount passes."""
    sel = codes >= 0
    c = codes[sel]
    wv = w[sel]
    yv = y[sel]
    er = raw[sel] - yv
    ec = comp[sel] - yv
    rows = np.bincount(c, minlength=k).astype(np.int64)
    W = np.bincount(c, weights=wv, minlength=k)
    sse_raw = np.bincount(c, weights=wv * er * er, minlength=k)
    sse_comp = np.bincount(c, weights=wv * ec * ec, minlength=k)
    sse_lag = np.bincount(c, weights=wv * (lag[sel] - yv) ** 2, minlength=k) if has_lag else None
    return rows, W, sse_raw, sse_comp, sse_lag


def _bin_stats(codes, w, y, raw, comp, lag, has_lag, k):
    """Per-bin (rows, W, sse_raw, sse_comp, sse_lag|None) reductions; excluded rows carry code -1.

    Dispatches to the fused njit single-pass kernel (default, ~12x faster at prod shape) or the bincount fallback.
    """
    if _HAVE_NUMBA:
        lag_arr = lag if has_lag else y  # placeholder; njit never reads it when has_lag is False
        codes = np.ascontiguousarray(codes, dtype=np.int64)
        rows, W, sse_raw, sse_comp, sse_lag = _bin_stats_njit(codes, w, y, raw, comp, lag_arr, has_lag, k)
        return rows, W, sse_raw, sse_comp, (sse_lag if has_lag else None)
    return _bin_stats_bincount(codes, w, y, raw, comp, lag, has_lag, k)


def regime_headroom_map(
    y: Any,
    y_pred_raw: Any,
    y_pred_composite: Any,
    y_pred_lag: Any = None,
    *,
    axis_values: Any,
    n_bins: int = 10,
    sample_weight: Any = None,
    group_ids: Any = None,
    help_rtol: float = 1e-9,
) -> dict:
    """Bin rows by ``axis_values`` quantiles and map, per bin, the composite's HEADROOM over the ``min(raw, lag)`` failsafe.

    Parameters
    ----------
    y, y_pred_raw, y_pred_composite
        True target, raw-y model prediction, composite ``y_hat`` prediction (aligned 1-D).
    y_pred_lag
        Optional AR-failsafe / lag baseline. When given, the failsafe reference per bin is ``min(rmse_raw, rmse_lag)``;
        otherwise it is ``rmse_raw`` alone.
    axis_values
        The regime axis to bin by (base value, a feature, a group-level score). Quantile-binned. Required keyword.
    n_bins
        Target number of quantile bins; the realized count can be smaller when the axis has ties / low cardinality.
    sample_weight
        Optional per-row weight (default None == unit weight). Non-finite / non-positive weights drop the row.
    group_ids
        Optional per-row group label; when given, each bin also reports the number of DISTINCT groups it spans (an
        operator signal that a bin's verdict is not driven by a single group). Purely informational; does not affect binning.
    help_rtol
        Relative band around zero headroom treated as "tied" for the summary's helped/hurt fraction (default ~0).

    Returns
    -------
    dict
        JSON-serializable (``None`` for no-data cells, never NaN; serialize with ``sort_keys=True``). Keys: ``n_bins``
        (realized), ``n_bins_requested``, ``n_rows`` (valid), ``has_lag``, ``help_rtol``, ``bins`` (list, ascending by
        axis), and ``summary``. Render with :func:`render_regime_headroom_map`.
    """
    y = _as1d(y)
    raw = _as1d(y_pred_raw)
    comp = _as1d(y_pred_composite)
    n = y.shape[0]
    has_lag = y_pred_lag is not None
    lag = _as1d(y_pred_lag) if has_lag else np.empty(0)

    for name, arr in (("y_pred_raw", raw), ("y_pred_composite", comp)):
        if arr.shape[0] != n:
            raise ValueError(f"regime_headroom_map: {name} length {arr.shape[0]} != y length {n}")
    if has_lag and lag.shape[0] != n:
        raise ValueError(f"regime_headroom_map: y_pred_lag length {lag.shape[0]} != y length {n}")

    if axis_values is None:
        raise ValueError("regime_headroom_map: axis_values is required")
    axis = _as1d(axis_values)
    if axis.shape[0] != n:
        raise ValueError(f"regime_headroom_map: axis_values length {axis.shape[0]} != y length {n}")

    if n_bins < 1:
        raise ValueError(f"regime_headroom_map: n_bins must be >= 1, got {n_bins}")

    w = np.ones(n, dtype=np.float64) if sample_weight is None else _as1d(sample_weight)
    if w.shape[0] != n:
        raise ValueError(f"regime_headroom_map: sample_weight length {w.shape[0]} != y length {n}")

    gids = None if group_ids is None else np.asarray(group_ids).reshape(-1)
    if gids is not None and gids.shape[0] != n:
        raise ValueError(f"regime_headroom_map: group_ids length {gids.shape[0]} != y length {n}")

    # Matched finite mask: every RMSE in a bin is on the SAME rows, so per-bin verdicts are like-for-like comparisons.
    valid = np.isfinite(y) & np.isfinite(raw) & np.isfinite(comp) & np.isfinite(axis) & np.isfinite(w) & (w > 0)
    if has_lag:
        valid = valid & np.isfinite(lag)

    if not valid.any():
        return _empty_report(int(n_bins), has_lag, float(help_rtol))

    codes, edges = _quantile_bin_codes(axis, valid, int(n_bins))
    k = max(int(codes.max()) + 1, 1)
    rows, W, sse_raw, sse_comp, sse_lag = _bin_stats(codes, w, y, raw, comp, lag if has_lag else y, has_lag, k)

    n_groups_per_bin = _distinct_groups_per_bin(codes, gids, k) if gids is not None else None

    bins: list[dict] = []
    for b in range(k):
        if W[b] <= 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = float(np.sqrt(sse_raw[b] / W[b]))
            rc = float(np.sqrt(sse_comp[b] / W[b]))
            rl = float(np.sqrt(sse_lag[b] / W[b])) if has_lag else None
        failsafe = rr if not has_lag else min(rr, rl if rl is not None else rr)
        headroom = ((failsafe - rc) / failsafe) if failsafe > 0 else None
        cands = [("raw", rr), ("composite", rc)]
        if has_lag and rl is not None:
            cands.append(("lag", rl))
        cands.sort(key=lambda kv: kv[1])
        winner = cands[0][0]
        bins.append(
            {
                "bin": b,
                "axis_lo": float(edges[b]) if b < edges.shape[0] else None,
                "axis_hi": float(edges[b + 1]) if b + 1 < edges.shape[0] else None,
                "n": int(rows[b]),
                "weight": float(W[b]),
                "n_groups": None if n_groups_per_bin is None else int(n_groups_per_bin[b]),
                "rmse_raw": rr,
                "rmse_composite": rc,
                "rmse_lag": rl,
                "failsafe_rmse": float(failsafe),
                "headroom": headroom,
                "winner": winner,
            }
        )

    summary = _summarize(bins, float(help_rtol))
    return {
        "n_bins": len(bins),
        "n_bins_requested": int(n_bins),
        "n_rows": int(rows.sum()),
        "has_lag": has_lag,
        "help_rtol": float(help_rtol),
        "bins": bins,
        "summary": summary,
    }


def _distinct_groups_per_bin(codes: np.ndarray, gids: np.ndarray, k: int) -> np.ndarray:
    """Count distinct group labels per bin. O(n log n) via one sort of (bin, group) pairs; narrow-op, no frame copy."""
    out = np.zeros(k, dtype=np.int64)
    valid = codes >= 0
    if not valid.any():
        return out
    cv = codes[valid]
    # Factorize group labels to compact integer codes so (bin, group) pairs sort/compare cheaply for any label dtype.
    _, ginv = np.unique(gids[valid], return_inverse=True)
    order = np.lexsort((ginv, cv))
    cs = cv[order]
    gs = ginv[order]
    # A pair starts a new distinct (bin, group) run when its bin OR group differs from the previous row.
    new_run = np.empty(cs.shape[0], dtype=bool)
    new_run[0] = True
    new_run[1:] = (cs[1:] != cs[:-1]) | (gs[1:] != gs[:-1])
    np.add.at(out, cs[new_run], 1)
    return out


def _summarize(bins: list[dict], help_rtol: float) -> dict:
    """Fraction of bins where the composite helps, and the single worst-hurt bin (most-negative headroom)."""
    scored = [b for b in bins if b["headroom"] is not None]
    if not scored:
        return {
            "n_bins_scored": 0,
            "frac_bins_helped": None,
            "frac_bins_hurt": None,
            "median_headroom": None,
            "worst_hurt_bin": None,
        }
    hr = np.asarray([b["headroom"] for b in scored], dtype=np.float64)
    helped = int(np.count_nonzero(hr > help_rtol))
    hurt = int(np.count_nonzero(hr < -help_rtol))
    total = len(scored)
    worst = min(scored, key=lambda b: b["headroom"])
    return {
        "n_bins_scored": total,
        "frac_bins_helped": helped / total,
        "frac_bins_hurt": hurt / total,
        "median_headroom": float(np.median(hr)),
        "worst_hurt_bin": {
            "bin": worst["bin"],
            "axis_lo": worst["axis_lo"],
            "axis_hi": worst["axis_hi"],
            "headroom": worst["headroom"],
            "winner": worst["winner"],
        },
    }


def _empty_report(n_bins: int, has_lag: bool, help_rtol: float) -> dict:
    """The zero-rows/zero-valid-bins report shape: same schema as a real report but with empty ``bins`` and an
    all-``None`` summary, so downstream consumers (JSON serialization, ``render_regime_headroom_map``) never branch
    on a missing key."""
    return {
        "n_bins": 0,
        "n_bins_requested": int(n_bins),
        "n_rows": 0,
        "has_lag": has_lag,
        "help_rtol": float(help_rtol),
        "bins": [],
        "summary": {
            "n_bins_scored": 0,
            "frac_bins_helped": None,
            "frac_bins_hurt": None,
            "median_headroom": None,
            "worst_hurt_bin": None,
        },
    }


def _pct(x: Optional[float]) -> str:
    """Format a fraction as a signed percentage string for the ASCII report, or ``"n/a"`` for a ``None`` (no-data) cell."""
    return "n/a" if x is None else f"{100.0 * x:+.2f}%"


def _num(x: Optional[float]) -> str:
    """Format a float to 6 significant digits for the ASCII report, or ``"n/a"`` for a ``None`` (no-data) cell."""
    return "n/a" if x is None else f"{x:.6g}"


def render_regime_headroom_map(report: dict, *, max_bins: int = 20) -> str:
    """Render the regime-headroom map as a compact ASCII table (cp1251-safe)."""
    L: list[str] = []
    L.append("# Regime headroom map")
    L.append("")
    L.append(f"Bins: {report['n_bins']} (requested {report['n_bins_requested']}). "
             f"Rows: {report['n_rows']}. Failsafe: {'min(raw, lag)' if report['has_lag'] else 'raw'}.")
    L.append("")

    s = report["summary"]
    L.append("## Summary")
    L.append("")
    L.append(f"- Bins where composite helps: {_pct(s['frac_bins_helped'])} " f"(hurt {_pct(s['frac_bins_hurt'])}, scored {s['n_bins_scored']})")
    L.append(f"- Median headroom: {_pct(s['median_headroom'])}")
    wh = s["worst_hurt_bin"]
    if wh is not None:
        L.append(f"- Worst-hurt bin: #{wh['bin']} [{_num(wh['axis_lo'])}, {_num(wh['axis_hi'])}] "
                 f"headroom {_pct(wh['headroom'])} -> deploy failsafe ({_ascii(wh['winner'])} wins)")
    L.append("")

    bins = report["bins"]
    if bins:
        L.append("## Per-bin headroom (ascending by axis)")
        L.append("")
        L.append("| bin | axis range | n | winner | headroom |")
        L.append("|-----|------------|---|--------|----------|")
        L.extend(
            f"| {b['bin']} | [{_num(b['axis_lo'])}, {_num(b['axis_hi'])}] | {b['n']} | " f"{_ascii(b['winner'])} | {_pct(b['headroom'])} |"
            for b in bins[:max_bins]
        )
        if len(bins) > max_bins:
            L.append("")
            L.append(f"_({len(bins) - max_bins} more bins omitted.)_")
        L.append("")

    return "\n".join(L)

"""Composite-target VALUE report: did the residual transform earn its keep, and where?

A composite deploys ``T = transform(y, base)`` / ``y_hat = inverse(T_hat, base)``. On grouped real
data (e.g. a wellbore regression with hundreds of wells) the net RMSE delta over the raw-y model can sit
near zero while the composite HELPS some groups and HURTS others. This module answers "did it help, and
where?" with a per-GROUP breakdown plus an aggregate verdict:

* per group: RMSE(raw), RMSE(composite), RMSE(lag failsafe, if given), the winner, and the realized lift
  of the composite over raw and over lag;
* aggregate: fraction of groups helped / hurt / tied (vs raw, and vs lag), the ROW-WEIGHTED net lift so
  big groups dominate correctly, and the count of groups where the composite is WORSE than the lag
  failsafe (the "should not have deployed here" signal);
* an EXPECTED-vs-REALIZED block when the selector's predicted lift / reconstruction RMSE is supplied.

All metrics are computed on the SAME rows per group: rows non-finite in ANY supplied series (y, raw,
composite, lag, weight) are dropped, so every RMSE and every helped/hurt call is a matched comparison.

The structured dict is JSON-serializable (missing/no-data values are ``None``, never NaN; serialize with
``sort_keys=True`` per project convention). The rendered text block is ASCII-only (Windows cp1251-safe).

Performance (cProfile, 1M rows / 773 groups, ``_benchmarks/bench_composite_value_report.py``): the hot
step is the per-group SSE reduction. The isolated reduction is 8.9x faster as a fused single-pass
``numba.njit`` kernel than as 3-4 ``np.bincount`` passes (3.9 ms vs 34.6 ms), and folding the finite-mask
INTO that one sweep also drops the several length-n squared-error / masked-copy temporaries the numpy path
allocates. So the njit fused reduction (:func:`_grouped_stats_njit`) is the DEFAULT; the vectorized
``np.bincount`` path (:func:`_grouped_stats_bincount`) is the fallback when numba is unavailable, and stays
bit-close (sequential vs bincount summation differ ~1e-12, never a verdict flip). After the reduction the
next cost is ``pd.factorize`` of the group ids (~22 ms, O(n)); everything else is O(n_groups).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import pandas as pd

    _HAVE_PANDAS = True
except Exception:  # pragma: no cover - pandas is a hard dep in practice
    _HAVE_PANDAS = False

try:
    import numba

    @numba.njit(cache=True)
    def _grouped_stats_njit(codes, w, y, raw, comp, lag, has_lag, n_groups):
        """Fused single-pass grouped reduction: finite-gate + counts + weighted SSE(raw/comp/lag) in ONE sweep.

        A row is valid iff its group code is >= 0, its weight is finite and > 0, and y / raw / comp (and lag
        when present) are all finite -- matching the numpy fallback exactly so both compute the same matched
        comparison. Avoids the finite boolean array and the length-n squared-error / masked-copy temporaries.
        """
        rows = np.zeros(n_groups, dtype=np.int64)
        W = np.zeros(n_groups)
        sse_raw = np.zeros(n_groups)
        sse_comp = np.zeros(n_groups)
        sse_lag = np.zeros(n_groups)
        for i in range(codes.shape[0]):
            g = codes[i]
            if g < 0:
                continue
            wi = w[i]
            if not (wi > 0.0) or not np.isfinite(wi):
                continue
            yi = y[i]
            ri = raw[i]
            ci = comp[i]
            if not (np.isfinite(yi) and np.isfinite(ri) and np.isfinite(ci)):
                continue
            if has_lag:
                li = lag[i]
                if not np.isfinite(li):
                    continue
                el = li - yi
                sse_lag[g] += wi * el * el
            er = ri - yi
            ec = ci - yi
            rows[g] += 1
            W[g] += wi
            sse_raw[g] += wi * er * er
            sse_comp[g] += wi * ec * ec
        return rows, W, sse_raw, sse_comp, sse_lag

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional accelerator here
    _HAVE_NUMBA = False

__all__ = [
    "build_composite_value_report",
    "render_composite_value_report",
]


def _as1d(a: Any) -> np.ndarray:
    """Coerce any array-like (list, pandas Series, polars Series, ndarray) to a flat float64 ndarray for the reduction kernels."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _to_native(v: Any) -> Any:
    """JSON-native group label (numpy scalars -> python; bytes -> ascii; else str)."""
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, np.generic):
        n = v.item()
        return n if isinstance(n, (str, bool, int, float)) else str(n)
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    return str(v)


def _ascii(s: Any) -> str:
    """Force ASCII for printed/logged strings (cp1251 crashes on non-ASCII)."""
    return str(s).encode("ascii", "replace").decode("ascii")


def _factorize(group_ids: Any) -> tuple[np.ndarray, list]:
    """(codes, unique_labels). NaN / null labels map to code -1 (excluded downstream)."""
    if _HAVE_PANDAS:
        codes, uniq = pd.factorize(np.asarray(group_ids), sort=False)
        return np.asarray(codes, dtype=np.int64), list(uniq)
    arr = np.asarray(group_ids)
    uniq, codes = np.unique(arr, return_inverse=True)
    return np.asarray(codes, dtype=np.int64), list(uniq)


def _verdict(comp: float, ref: float, rtol: float) -> str:
    """Classify the composite's RMSE against a reference (raw or lag) RMSE for ONE group: "helped" if comp is below
    ref by more than ``rtol``, "hurt" if above by more than ``rtol``, else "tied". A non-positive/zero-data ``ref``
    (no valid rows) reads as "tied" unless the composite itself is positive, in which case it is "hurt" (there is
    no baseline to beat, but a non-zero composite error still counts against it).
    """
    if not (ref > 0):
        return "tied" if comp <= 0 else "hurt"
    if comp < ref * (1.0 - rtol):
        return "helped"
    if comp > ref * (1.0 + rtol):
        return "hurt"
    return "tied"


def _grouped_stats_bincount(codes, w, y, raw, comp, lag, has_lag, n_groups):
    """Vectorized fallback: build the matched finite mask, then O(n) bincount reductions."""
    finite = np.isfinite(y) & np.isfinite(raw) & np.isfinite(comp) & np.isfinite(w) & (w > 0) & (codes >= 0)
    if has_lag:
        finite = finite & np.isfinite(lag)
    cv = codes[finite]
    wv = w[finite]
    yv = y[finite]
    er = raw[finite] - yv
    ec = comp[finite] - yv
    rows = np.bincount(cv, minlength=n_groups).astype(np.int64)
    W = np.bincount(cv, weights=wv, minlength=n_groups)
    sse_raw = np.bincount(cv, weights=wv * er * er, minlength=n_groups)
    sse_comp = np.bincount(cv, weights=wv * ec * ec, minlength=n_groups)
    sse_lag = np.bincount(cv, weights=wv * (lag[finite] - yv) ** 2, minlength=n_groups) if has_lag else None
    return rows, W, sse_raw, sse_comp, sse_lag


def _grouped_stats(codes, w, y, raw, comp, lag, has_lag, n_groups):
    """Dispatch the per-group reduction to the fused njit kernel (default) or the bincount fallback."""
    if _HAVE_NUMBA:
        lag_arr = lag if has_lag else y  # placeholder; njit never reads it when has_lag is False
        rows, W, sse_raw, sse_comp, sse_lag = _grouped_stats_njit(codes, w, y, raw, comp, lag_arr, has_lag, n_groups)
        return rows, W, sse_raw, sse_comp, (sse_lag if has_lag else None)
    return _grouped_stats_bincount(codes, w, y, raw, comp, lag, has_lag, n_groups)


def build_composite_value_report(
    y_true: Any,
    y_pred_raw: Any,
    y_pred_composite: Any,
    group_ids: Any,
    y_pred_lag: Any = None,
    *,
    expected_lift: Optional[float] = None,
    expected_rmse: Optional[float] = None,
    sample_weight: Any = None,
    tie_rtol: float = 1e-3,
    expected_tol: float = 0.02,
) -> dict:
    """Per-group + aggregate value report for a composite vs its raw-y and lag-failsafe baselines.

    Parameters
    ----------
    y_true, y_pred_raw, y_pred_composite
        True target, the raw-y model prediction, and the composite ``y_hat`` prediction (aligned 1-D).
    group_ids
        Group label per row (e.g. well id). Any dtype; null labels are dropped.
    y_pred_lag
        Optional AR-failsafe / lag baseline prediction. Enables the vs-lag columns and the
        worse-than-lag signal.
    expected_lift, expected_rmse
        Optional selector estimates: the relative lift over raw and/or the honest-OOF reconstruction
        RMSE it predicted. When given, an expected-vs-realized calibration block is added.
    sample_weight
        Optional per-row weight. Default ``None`` == row-count weighting (big groups dominate the net).
    tie_rtol
        Relative band around equal RMSE that counts as a tie (default 0.1%).
    expected_tol
        Relative-lift band for the expected-vs-realized calibration verdict (default 2 percentage-pts).

    Returns
    -------
    dict
        JSON-serializable (``None`` for no-data cells, never NaN). Keys: ``n_groups``, ``n_groups_no_data``,
        ``n_rows``, ``has_lag``, ``tie_rtol``, ``per_group`` (list), ``aggregate`` (dict), and
        ``expected_vs_realized`` (dict or ``None``). Render with :func:`render_composite_value_report`.
    """
    y = _as1d(y_true)
    raw = _as1d(y_pred_raw)
    comp = _as1d(y_pred_composite)
    n = y.shape[0]
    has_lag = y_pred_lag is not None
    lag = _as1d(y_pred_lag) if has_lag else None

    for name, arr in (("y_pred_raw", raw), ("y_pred_composite", comp)):
        if arr.shape[0] != n:
            raise ValueError(f"build_composite_value_report: {name} length {arr.shape[0]} != y_true length {n}")
    if has_lag and lag is not None and lag.shape[0] != n:
        raise ValueError(f"build_composite_value_report: y_pred_lag length {lag.shape[0]} != y_true length {n}")

    codes, uniq = _factorize(group_ids)
    if codes.shape[0] != n:
        raise ValueError(f"build_composite_value_report: group_ids length {codes.shape[0]} != y_true length {n}")
    n_groups_total = len(uniq)

    w = np.ones(n, dtype=np.float64) if sample_weight is None else _as1d(sample_weight)
    if w.shape[0] != n:
        raise ValueError(f"build_composite_value_report: sample_weight length {w.shape[0]} != y_true length {n}")

    empty = _empty_report(n, has_lag, tie_rtol, n_groups_total)
    if n == 0 or n_groups_total == 0:
        return empty

    codes = np.ascontiguousarray(codes, dtype=np.int64)
    rows, W, sse_raw, sse_comp, sse_lag = _grouped_stats(codes, w, y, raw, comp, lag, has_lag, n_groups_total)
    if int(rows.sum()) == 0:
        return empty

    valid = W > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse_raw = np.where(valid, np.sqrt(sse_raw / W), np.nan)
        rmse_comp = np.where(valid, np.sqrt(sse_comp / W), np.nan)
        rmse_lag = np.where(valid, np.sqrt(sse_lag / W), np.nan) if has_lag else None

    per_group: list[dict] = []
    for g in np.nonzero(valid)[0]:
        rr = float(rmse_raw[g])
        rc = float(rmse_comp[g])
        lift_raw = (rr - rc) / rr if rr > 0 else 0.0
        entry: dict[str, Any] = {
            "group": _to_native(uniq[g]),
            "n": int(rows[g]),
            "weight": float(W[g]),
            "rmse_raw": rr,
            "rmse_composite": rc,
            "rmse_lag": None,
            "lift_over_raw": lift_raw,
            "lift_over_lag": None,
            "verdict_vs_raw": _verdict(rc, rr, tie_rtol),
            "verdict_vs_lag": None,
            "worse_than_lag": False,
            "winner": None,
        }
        cands = [("raw", rr), ("composite", rc)]
        if has_lag and rmse_lag is not None:
            rl = float(rmse_lag[g])
            entry["rmse_lag"] = rl
            entry["lift_over_lag"] = (rl - rc) / rl if rl > 0 else 0.0
            entry["verdict_vs_lag"] = _verdict(rc, rl, tie_rtol)
            entry["worse_than_lag"] = rc > rl * (1.0 + tie_rtol)
            cands.append(("lag", rl))
        cands.sort(key=lambda kv: kv[1])
        best_label, best = cands[0]
        entry["winner"] = "tie" if len(cands) > 1 and cands[1][1] <= best * (1.0 + tie_rtol) else best_label
        per_group.append(entry)

    per_group.sort(key=lambda e: str(e["group"]))

    aggregate = _aggregate(rmse_raw, rmse_comp, rmse_lag, W, sse_raw, sse_comp, sse_lag, valid, uniq, has_lag, tie_rtol)

    expected_vs_realized = _expected_vs_realized(aggregate, expected_lift=expected_lift, expected_rmse=expected_rmse, expected_tol=expected_tol)

    return {
        "n_groups": int(valid.sum()),
        "n_groups_no_data": int(n_groups_total - int(valid.sum())),
        "n_rows": int(rows.sum()),
        "has_lag": has_lag,
        "tie_rtol": float(tie_rtol),
        "per_group": per_group,
        "aggregate": aggregate,
        "expected_vs_realized": expected_vs_realized,
    }


def _empty_report(n: int, has_lag: bool, tie_rtol: float, n_groups_total: int) -> dict:
    """Build the degenerate report returned when there are zero input rows, zero groups, or every group has zero
    valid (finite, matched) rows. Preserves the same schema as a normal :func:`build_composite_value_report` result
    so downstream code (JSON consumers, :func:`render_composite_value_report`) never needs a separate empty-case
    branch; ``n_groups_no_data`` is set to the full group count and ``net_verdict`` reads "no data".
    """
    return {
        "n_groups": 0,
        "n_groups_no_data": int(n_groups_total),
        "n_rows": 0,
        "has_lag": has_lag,
        "tie_rtol": float(tie_rtol),
        "per_group": [],
        "aggregate": {
            "vs_raw": {"helped": 0, "hurt": 0, "tied": 0, "helped_frac": None, "hurt_frac": None, "tied_frac": None},
            "vs_lag": None,
            "net_weighted_lift_over_raw": None,
            "net_weighted_lift_over_lag": None,
            "pooled_rmse_raw": None,
            "pooled_rmse_composite": None,
            "pooled_rmse_lag": None,
            "pooled_lift_over_raw": None,
            "pooled_lift_over_lag": None,
            "n_worse_than_lag": 0,
            "worse_than_lag_groups": [],
            "net_verdict": "no data",
        },
        "expected_vs_realized": None,
    }


def _tally(rmse_comp: np.ndarray, rmse_ref: np.ndarray, valid: np.ndarray, rtol: float) -> dict:
    """Vectorized aggregate counterpart to :func:`_verdict`: bucket every valid group into helped/hurt/tied (via the
    same ``rtol`` band) against a reference RMSE array (raw or lag), and return counts plus their fractions of the
    valid-group total (``None`` fractions when there are zero valid groups, to keep the report JSON-null-safe).
    """
    helped = valid & (rmse_comp < rmse_ref * (1.0 - rtol))
    hurt = valid & (rmse_comp > rmse_ref * (1.0 + rtol))
    tied = valid & ~helped & ~hurt
    total = int(valid.sum())
    denom = float(total) if total else None
    return {
        "helped": int(helped.sum()),
        "hurt": int(hurt.sum()),
        "tied": int(tied.sum()),
        "helped_frac": (int(helped.sum()) / denom) if denom else None,
        "hurt_frac": (int(hurt.sum()) / denom) if denom else None,
        "tied_frac": (int(tied.sum()) / denom) if denom else None,
    }


def _net_lift(rmse_ref: np.ndarray, rmse_comp: np.ndarray, W: np.ndarray, valid: np.ndarray) -> Optional[float]:
    """Row-weighted (not group-averaged) relative lift of composite over a reference RMSE across all valid groups
    with positive reference RMSE. Weighting by group total weight ``W`` means a group with many rows dominates the
    net verdict correctly, rather than letting a small group with a big per-group lift skew a simple mean. Returns
    ``None`` when no group qualifies (all-zero reference RMSE or nothing valid).
    """
    idx = np.nonzero(valid & (rmse_ref > 0))[0]
    if idx.size == 0:
        return None
    lift = (rmse_ref[idx] - rmse_comp[idx]) / rmse_ref[idx]
    wsum = float(W[idx].sum())
    return float(np.dot(lift, W[idx]) / wsum) if wsum > 0 else None


def _pooled_rmse(sse: np.ndarray, W: np.ndarray, valid: np.ndarray) -> Optional[float]:
    """Single overall RMSE pooled across all valid groups: sqrt(total weighted SSE / total weight), i.e. as if every
    row from every group were one big dataset. Distinct from averaging per-group RMSEs (which would over-weight
    small/noisy groups); ``None`` when total weight is zero (no valid rows anywhere).
    """
    wsum = float(W[valid].sum())
    return float(np.sqrt(float(sse[valid].sum()) / wsum)) if wsum > 0 else None


def _aggregate(rmse_raw, rmse_comp, rmse_lag, W, sse_raw, sse_comp, sse_lag, valid, uniq, has_lag, rtol) -> dict:
    """Roll the per-group RMSE arrays into the top-level aggregate block: helped/hurt/tied tallies (vs raw and,
    when available, vs lag), pooled RMSE and pooled lift per baseline, the row-weighted net lift, the list of groups
    where the composite is WORSE than the lag failsafe (sorted by label, for the "should not have deployed here"
    signal), and the overall ``net_verdict`` string derived from whether the net lift over raw clears ``rtol``.
    """
    vs_raw = _tally(rmse_comp, rmse_raw, valid, rtol)
    vs_lag = _tally(rmse_comp, rmse_lag, valid, rtol) if has_lag else None

    pooled_raw = _pooled_rmse(sse_raw, W, valid)
    pooled_comp = _pooled_rmse(sse_comp, W, valid)
    pooled_lag = _pooled_rmse(sse_lag, W, valid) if has_lag else None
    pooled_lift_raw = ((pooled_raw - pooled_comp) / pooled_raw) if (pooled_raw and pooled_raw > 0 and pooled_comp is not None) else None
    pooled_lift_lag = ((pooled_lag - pooled_comp) / pooled_lag) if (pooled_lag and pooled_lag > 0 and pooled_comp is not None) else None

    net_raw = _net_lift(rmse_raw, rmse_comp, W, valid)
    net_lag = _net_lift(rmse_lag, rmse_comp, W, valid) if has_lag else None

    worse_groups: list = []
    if has_lag:
        worse = valid & (rmse_comp > rmse_lag * (1.0 + rtol))
        worse_groups = [_to_native(uniq[g]) for g in np.nonzero(worse)[0]]
        worse_groups.sort(key=str)

    if net_raw is None:
        verdict = "no data"
    elif net_raw > rtol:
        verdict = "composite helped overall"
    elif net_raw < -rtol:
        verdict = "composite hurt overall"
    else:
        verdict = "net neutral"

    return {
        "vs_raw": vs_raw,
        "vs_lag": vs_lag,
        "net_weighted_lift_over_raw": net_raw,
        "net_weighted_lift_over_lag": net_lag,
        "pooled_rmse_raw": pooled_raw,
        "pooled_rmse_composite": pooled_comp,
        "pooled_rmse_lag": pooled_lag,
        "pooled_lift_over_raw": pooled_lift_raw,
        "pooled_lift_over_lag": pooled_lift_lag,
        "n_worse_than_lag": len(worse_groups),
        "worse_than_lag_groups": worse_groups,
        "net_verdict": verdict,
    }


def _expected_vs_realized(aggregate, *, expected_lift, expected_rmse, expected_tol) -> Optional[dict]:
    """Compare the target selector's PREDICTED lift/RMSE against what the report actually realized, so a caller can
    judge whether the selector's estimate was calibrated. Returns ``None`` when neither ``expected_lift`` nor
    ``expected_rmse`` was supplied (selector calibration not requested). The lift gap is bucketed into "optimistic"
    (selector over-promised, realized fell short by more than ``expected_tol``), "pessimistic" (realized beat the
    promise by more than ``expected_tol``), or "on-target"; the RMSE gap is reported as a raw delta only (no verdict
    bucket, since "expected RMSE" units/scale vary more than lift fractions do).
    """
    if expected_lift is None and expected_rmse is None:
        return None
    realized_lift = aggregate["net_weighted_lift_over_raw"]
    realized_rmse = aggregate["pooled_rmse_composite"]
    out: dict[str, Any] = {
        "expected_lift": None if expected_lift is None else float(expected_lift),
        "realized_lift": realized_lift,
        "expected_rmse": None if expected_rmse is None else float(expected_rmse),
        "realized_rmse": realized_rmse,
        "lift_gap": None,
        "rmse_gap": None,
        "calibration": "unknown",
    }
    if expected_lift is not None and realized_lift is not None:
        gap = realized_lift - float(expected_lift)
        out["lift_gap"] = gap
        if gap < -expected_tol:
            out["calibration"] = "optimistic (selector over-promised)"
        elif gap > expected_tol:
            out["calibration"] = "pessimistic (selector under-promised)"
        else:
            out["calibration"] = "on-target"
    if expected_rmse is not None and realized_rmse is not None:
        out["rmse_gap"] = realized_rmse - float(expected_rmse)
    return out


def _pct(x: Optional[float]) -> str:
    """Format a fraction as a signed percentage for the rendered text block (``None`` -> ``"n/a"``)."""
    return "n/a" if x is None else f"{100.0 * x:+.2f}%"


def _num(x: Optional[float]) -> str:
    """Format a metric (RMSE, gap) at 6 significant digits for the rendered text block (``None`` -> ``"n/a"``)."""
    return "n/a" if x is None else f"{x:.6g}"


def render_composite_value_report(report: dict, *, max_groups: int = 20) -> str:
    """Render the value-report dict as an ASCII-only Markdown/text block (cp1251-safe)."""
    L: list[str] = []
    L.append("# Composite target value report")
    L.append("")
    L.append(f"Groups: {report['n_groups']} ({report['n_groups_no_data']} no-data excluded). "
             f"Rows: {report['n_rows']}. Tie band: {100.0 * report['tie_rtol']:.3g}%.")
    L.append("")

    agg = report["aggregate"]
    L.append("## Verdict")
    L.append("")
    L.append(f"- Net row-weighted lift over raw: {_pct(agg['net_weighted_lift_over_raw'])} " f"-> {_ascii(agg['net_verdict'])}")
    L.append(
        f"- Pooled RMSE: raw {_num(agg['pooled_rmse_raw'])} -> composite {_num(agg['pooled_rmse_composite'])} " f"(lift {_pct(agg['pooled_lift_over_raw'])})"
    )
    vr = agg["vs_raw"]
    L.append(f"- vs raw: {vr['helped']} helped / {vr['hurt']} hurt / {vr['tied']} tied "
             f"({_pct(vr['helped_frac'])} / {_pct(vr['hurt_frac'])} / {_pct(vr['tied_frac'])})")
    if report["has_lag"]:
        vl = agg["vs_lag"]
        L.append(f"- Net row-weighted lift over lag: {_pct(agg['net_weighted_lift_over_lag'])} " f"(pooled lag RMSE {_num(agg['pooled_rmse_lag'])})")
        if vl is not None:
            L.append(f"- vs lag: {vl['helped']} helped / {vl['hurt']} hurt / {vl['tied']} tied "
                     f"({_pct(vl['helped_frac'])} / {_pct(vl['hurt_frac'])} / {_pct(vl['tied_frac'])})")
        L.append(f"- Groups where composite is WORSE than the lag failsafe: {agg['n_worse_than_lag']} "
                 f"<- should not have deployed here")
        if agg["worse_than_lag_groups"]:
            shown = ", ".join(_ascii(g) for g in agg["worse_than_lag_groups"][:max_groups])
            more = "" if len(agg["worse_than_lag_groups"]) <= max_groups else f", (+{len(agg['worse_than_lag_groups']) - max_groups} more)"
            L.append(f"  - {shown}{more}")
    L.append("")

    evr = report.get("expected_vs_realized")
    if evr is not None:
        L.append("## Expected vs realized")
        L.append("")
        L.append(f"- Selector expected lift {_pct(evr['expected_lift'])}, realized {_pct(evr['realized_lift'])} "
                 f"-> {_ascii(evr['calibration'])} (gap {_pct(evr['lift_gap'])})")
        if evr["expected_rmse"] is not None:
            L.append(f"- Selector expected reconstruction RMSE {_num(evr['expected_rmse'])}, "
                     f"realized {_num(evr['realized_rmse'])} (gap {_num(evr['rmse_gap'])})")
        L.append("")

    per_group = report["per_group"]
    if per_group:
        L.append("## Per-group breakdown (top by weighted impact)")
        L.append("")
        ranked = sorted(per_group, key=lambda e: e["weight"] * abs(e["lift_over_raw"]), reverse=True)
        if report["has_lag"]:
            L.append("| group | n | rmse_raw | rmse_composite | rmse_lag | lift_raw | lift_lag | winner | vs_raw |")
            L.append("|-------|---|----------|----------------|----------|----------|----------|--------|--------|")
            L.extend(
                f"| {_ascii(e['group'])} | {e['n']} | {_num(e['rmse_raw'])} | {_num(e['rmse_composite'])} | "
                f"{_num(e['rmse_lag'])} | {_pct(e['lift_over_raw'])} | {_pct(e['lift_over_lag'])} | "
                f"{e['winner']} | {e['verdict_vs_raw']} |"
                for e in ranked[:max_groups]
            )
        else:
            L.append("| group | n | rmse_raw | rmse_composite | lift_raw | winner | vs_raw |")
            L.append("|-------|---|----------|----------------|----------|--------|--------|")
            L.extend(
                f"| {_ascii(e['group'])} | {e['n']} | {_num(e['rmse_raw'])} | {_num(e['rmse_composite'])} | "
                f"{_pct(e['lift_over_raw'])} | {e['winner']} | {e['verdict_vs_raw']} |"
                for e in ranked[:max_groups]
            )
        if len(ranked) > max_groups:
            L.append("")
            L.append(f"_({len(ranked) - max_groups} more groups omitted.)_")
        L.append("")

    return "\n".join(L)

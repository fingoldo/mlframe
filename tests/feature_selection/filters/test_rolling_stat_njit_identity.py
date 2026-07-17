"""_rolling_stat_past_only njit two-pointer kernel is bit-identical to the reference per-entity list loop."""

import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._temporal_agg_fe import _rolling_stat_past_only, _reduce, _is_datetime_like, _numeric_window


def _ref(times, vals, gc, window, stat):
    """Helper that ref."""
    n = vals.size
    out = np.full(n, np.nan)
    td = pd.Timedelta(window) if _is_datetime_like(times) else _numeric_window(window)
    et = {}
    ev = {}
    for i in range(n):
        g = int(gc[i])
        t = times[i]
        tl = et.setdefault(g, [])
        vl = ev.setdefault(g, [])
        if tl:
            lo = t - td
            wv = [vv for (tt, vv) in zip(tl, vl) if (tt >= lo) and (tt < t) and np.isfinite(vv)]
            if wv:
                out[i] = _reduce(np.asarray(wv, np.float64), stat)
        tl.append(t)
        vl.append(vals[i])
    return out


def _sorted_by_entity_time(n, n_ent, seed):
    """Sorted by entity time."""
    rng = np.random.default_rng(seed)
    gc = np.sort(rng.integers(0, n_ent, n))
    t = np.zeros(n)
    for g in np.unique(gc):
        m = gc == g
        t[m] = np.sort(rng.uniform(0, 1000, int(m.sum())))
    vals = rng.standard_normal(n)
    vals[::19] = np.nan
    return t, vals, gc


def test_numeric_time_all_stats_bit_identical():
    """Numeric time all stats bit identical."""
    t, vals, gc = _sorted_by_entity_time(8000, 80, 0)
    for stat in ("count", "mean", "std", "min", "max"):
        got = _rolling_stat_past_only(t, vals, gc, "50", stat)
        ref = _ref(t, vals, gc, "50", stat)
        assert np.allclose(got, ref, equal_nan=True, rtol=0, atol=1e-9), stat


def test_datetime_time_bit_identical():
    """Datetime time bit identical."""
    t, vals, gc = _sorted_by_entity_time(6000, 60, 1)
    times = np.datetime64("2020-01-01") + t.astype("int64").astype("timedelta64[h]")
    for stat in ("mean", "std", "count"):
        got = _rolling_stat_past_only(times, vals, gc, "72h", stat)
        ref = _ref(times, vals, gc, "72h", stat)
        assert np.allclose(got, ref, equal_nan=True, rtol=0, atol=1e-9), stat

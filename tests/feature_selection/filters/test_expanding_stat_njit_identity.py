"""_expanding_stat_past_only njit kernel is bit-identical to the reference per-entity dict loop (incl NaN)."""
import numpy as np
from mlframe.feature_selection.filters._temporal_agg_fe import _expanding_stat_past_only


def _ref(sorted_vals, group_codes, stat):
    n = sorted_vals.size; out = np.full(n, np.nan)
    ns = {}; rs = {}; rss = {}; rmn = {}; rmx = {}
    for i in range(n):
        g = int(group_codes[i]); cnt = ns.get(g, 0)
        if cnt > 0:
            if stat == "count": out[i] = float(cnt)
            elif stat == "mean": out[i] = rs[g] / cnt
            elif stat == "std":
                if cnt > 1:
                    m = rs[g] / cnt; var = (rss[g] - cnt * m * m) / (cnt - 1)
                    out[i] = float(np.sqrt(var)) if var > 0 else 0.0
                else: out[i] = 0.0
            elif stat == "min": out[i] = rmn[g]
            elif stat == "max": out[i] = rmx[g]
        v = sorted_vals[i]
        if np.isfinite(v):
            ns[g] = cnt + 1; rs[g] = rs.get(g, 0.0) + v; rss[g] = rss.get(g, 0.0) + v * v
            rmn[g] = v if g not in rmn else min(rmn[g], v)
            rmx[g] = v if g not in rmx else max(rmx[g], v)
        else: ns.setdefault(g, cnt)
    return out


def test_all_stats_bit_identical_incl_nan():
    rng = np.random.default_rng(0)
    n = 20000
    gc = np.sort(rng.integers(0, 150, n))
    sv = rng.standard_normal(n); sv[::17] = np.nan
    for stat in ("count", "mean", "std", "min", "max"):
        got = _expanding_stat_past_only(sv, gc, stat)
        ref = _ref(sv, gc, stat)
        assert np.array_equal(got, ref, equal_nan=True), stat


def test_empty_and_single_entity():
    assert _expanding_stat_past_only(np.empty(0), np.empty(0, dtype=np.int64), "mean").size == 0
    sv = np.array([1.0, 2.0, 3.0]); gc = np.zeros(3, dtype=np.int64)
    got = _expanding_stat_past_only(sv, gc, "mean")
    assert np.isnan(got[0]) and got[1] == 1.0 and got[2] == 1.5

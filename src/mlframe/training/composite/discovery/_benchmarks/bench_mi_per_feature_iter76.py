"""iter76 A/B: ``_mi_per_feature_prebinned`` matrix-level sentinel gate vs the pre-fix per-column scan.

OLD = the pre-fix body (per-column ``col_b >= 0`` + ``.sum()`` O(n) scan for every column), rebuilt
verbatim here and run against the SAME inputs; NEW = the shipped matrix-level ``(fb < 0).any()`` gate.
Paired interleaved best-of-N + bit-identity on the per-feature MI vector across all-finite, sentinel-laced,
and excluded-column inputs.

Run:
    python -m mlframe.training.composite.discovery._benchmarks.bench_mi_per_feature_iter76
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from timeit import default_timer as timer

import scipy.stats  # noqa: F401
import numba  # noqa: F401
import numpy as np

from mlframe.training.composite.discovery import screening as S
from mlframe.training.composite.discovery.screening import (
    _mi_per_feature_prebinned,
    _prebin_feature_columns,
    _mi_from_binned_pair,
)

_N = 200_000
_F = 100
_NBINS = 16


def _old_mi_per_feature_prebinned(feature_binned, target, *, nbins, exclude_col=None):
    if feature_binned.shape[0] == 0 or feature_binned.shape[1] == 0:
        return None
    if feature_binned.shape[0] != target.shape[0]:
        return None
    n_cols_in = feature_binned.shape[1]
    drop = exclude_col if (exclude_col is not None and 0 <= exclude_col < n_cols_in) else None
    if drop is not None and n_cols_in == 1:
        return None
    finite = np.isfinite(target)
    n_fin = int(finite.sum())
    if n_fin < 5 * nbins:
        return None
    if n_fin == finite.shape[0]:
        t_f = target
        fb_f = feature_binned
    else:
        t_f = target[finite]
        fb_f = feature_binned[finite]
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    t_edges = np.nanquantile(t_f, qs)
    t_idx = np.searchsorted(t_edges, t_f, side="right").astype(np.int64)
    np.clip(t_idx, 0, nbins - 1, out=t_idx)
    out_len = fb_f.shape[1] - (1 if drop is not None else 0)
    per_feat = np.empty(out_len, dtype=np.float64)
    out_j = 0
    for j in range(fb_f.shape[1]):
        if drop is not None and j == drop:
            continue
        col_b = fb_f[:, j]
        col_valid = col_b >= 0
        n_cv = int(col_valid.sum())
        if n_cv < 5 * nbins:
            per_feat[out_j] = 0.0
            out_j += 1
            continue
        if n_cv == col_b.shape[0]:
            per_feat[out_j] = _mi_from_binned_pair(col_b, t_idx, nbins=nbins)
        else:
            per_feat[out_j] = _mi_from_binned_pair(col_b[col_valid], t_idx[col_valid], nbins=nbins)
        out_j += 1
    return per_feat


def _build(seed=0, sentinel=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((_N, _F)).astype(np.float32)
    if sentinel:
        # inject NaN into a few columns so the prebin emits -1 sentinels
        for c in (3, 17, 88):
            mask = rng.random(_N) < 0.1
            x[mask, c] = np.nan
    y = np.abs(0.6 * x[:, 0] + 0.3 * x[:, 1]).astype(np.float64) + rng.standard_normal(_N) * 0.1 + 0.5
    pb = _prebin_feature_columns(x, nbins=_NBINS)
    return pb, y


def _identity():
    ok = True
    for sentinel in (False, True):
        pb, y = _build(seed=1, sentinel=sentinel)
        for excl in (None, 5):
            new = _mi_per_feature_prebinned(pb, y, nbins=_NBINS, exclude_col=excl)
            old = _old_mi_per_feature_prebinned(pb, y, nbins=_NBINS, exclude_col=excl)
            md = float(np.max(np.abs(new - old)))
            print(f"  identity sentinel={sentinel} exclude={excl}: maxdiff={md:.3e}")
            ok = ok and (md == 0.0)
    return ok


def main():
    print(f"mi_per_feature A/B  n={_N} F={_F} nbins={_NBINS} py={sys.version.split()[0]}")
    identical = _identity()
    print(f"  bit-identical: {identical}")

    pb, y = _build(seed=0, sentinel=False)
    # warm
    _mi_per_feature_prebinned(pb, y, nbins=_NBINS)
    _old_mi_per_feature_prebinned(pb, y, nbins=_NBINS)

    trials = 40
    new_ts, old_ts = [], []
    new_wins = 0
    for _ in range(trials):
        t0 = timer(); _mi_per_feature_prebinned(pb, y, nbins=_NBINS); nt = timer() - t0
        t0 = timer(); _old_mi_per_feature_prebinned(pb, y, nbins=_NBINS); ot = timer() - t0
        new_ts.append(nt); old_ts.append(ot)
        if nt < ot:
            new_wins += 1
    new_ts.sort(); old_ts.sort()
    nmin, nmed = new_ts[0] * 1e3, new_ts[len(new_ts) // 2] * 1e3
    omin, omed = old_ts[0] * 1e3, old_ts[len(old_ts) // 2] * 1e3
    print(f"  OLD min {omin:.2f}ms med {omed:.2f}ms -> NEW min {nmin:.2f}ms med {nmed:.2f}ms")
    print(f"  speedup min {omin / nmin:.2f}x med {omed / nmed:.2f}x ; NEW faster {new_wins}/{trials}")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mi_per_feature_iter76.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "ts": datetime.now().isoformat(), "n": _N, "F": _F, "nbins": _NBINS,
            "bit_identical": identical, "new_wins": new_wins, "trials": trials,
            "old_min_ms": omin, "old_med_ms": omed, "new_min_ms": nmin, "new_med_ms": nmed,
        }, fh, indent=2)


if __name__ == "__main__":
    main()

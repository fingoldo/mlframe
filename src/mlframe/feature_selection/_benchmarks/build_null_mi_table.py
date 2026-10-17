"""Offline build of the null-MI threshold table used by the pair search.

For every fingerprint (n rows, quantile bins, target classes and their entropy, number of candidates N) the script draws target
vectors with that class prior, pure-noise candidate columns, quantile-bins them, measures each column's plug-in MI with the target,
keeps the maximum over the N candidates, and records its 95th percentile over repeated draws. The table is what the pair search
uses as its significance floor at fit time, so no permutations run per fit.

The MI is computed on the GPU with cupy (one histogram per draw for all candidates at once). Run it once; the output JSON is
checked in under ``filters/data``. Usage::

    python -m mlframe.feature_selection._benchmarks.build_null_mi_table --out src/mlframe/feature_selection/filters/data/null_mi_table.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

N_GRID = (500, 1000, 2000, 5000, 20000)
BINS_GRID = (8, 10, 12, 16)
CANDIDATE_GRID = (1, 5, 20, 50, 200, 1000)
# (name, class prior). Binary priors vary the entropy; multiclass ones are balanced.
PRIOR_GRID = (
    ("bin_0.5", (0.5, 0.5)),
    ("bin_0.3", (0.3, 0.7)),
    ("bin_0.1", (0.1, 0.9)),
    ("k3_bal", (1 / 3, 1 / 3, 1 / 3)),
    ("k5_bal", (0.2, 0.2, 0.2, 0.2, 0.2)),
)
QUANTILE = 0.95


def entropy_nats(prior) -> float:
    """Shannon entropy of a class prior, in nats (the same unit as the plug-in MI)."""
    p = np.asarray(prior, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def max_null_mi_gpu(cp, n: int, n_bins: int, prior, n_candidates: int, reps: int, seed: int) -> np.ndarray:
    """For ``reps`` draws: the maximum plug-in MI over ``n_candidates`` pure-noise columns, all generated and scored on the GPU.

    Target and candidates are drawn on the device (no host-to-device copy per draw), ranks come from one argsort plus a scatter,
    and the contingency counts are one bincount per draw. ``seed`` makes a configuration reproducible on its own.
    """
    prior_g = cp.asarray(np.asarray(prior, dtype=np.float64))
    k = int(prior_g.shape[0])
    cum = cp.cumsum(prior_g)
    gen = cp.random.default_rng(seed)
    out = np.empty(reps, dtype=np.float64)
    cols = cp.arange(n_candidates, dtype=cp.int64)[None, :]
    for r in range(reps):
        u = gen.random(n, dtype=cp.float64)
        y_g = cp.minimum(cp.searchsorted(cum, u, side="right"), k - 1).astype(cp.int64)
        x = gen.standard_normal((n, n_candidates), dtype=cp.float32)
        order = cp.argsort(x, axis=0)
        ranks = cp.empty_like(order)
        ranks[order, cols] = cp.arange(n, dtype=cp.int64)[:, None]
        codes = (ranks * n_bins // n).astype(cp.int64)
        flat = (cols * n_bins + codes) * k + y_g[:, None]
        counts = cp.bincount(flat.ravel(), minlength=n_candidates * n_bins * k).reshape(n_candidates, n_bins, k).astype(cp.float64)
        pxy = counts / n
        px = pxy.sum(axis=2, keepdims=True)
        py = pxy.sum(axis=1, keepdims=True)
        term = cp.where(pxy > 0, pxy * (cp.log(cp.where(pxy > 0, pxy, 1.0)) - cp.log(px) - cp.log(py)), 0.0)
        out[r] = float(cp.max(term.sum(axis=(1, 2))).get())
    return out


def build(reps: int, seed: int, log=None) -> dict:
    """The full table keyed by ``n|bins|prior|N`` with its entropy and the 95th percentile of the max-null MI."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    table = {}
    total = len(N_GRID) * len(BINS_GRID) * len(PRIOR_GRID) * len(CANDIDATE_GRID)
    done = 0
    t0 = time.perf_counter()
    for n in N_GRID:
        for n_bins in BINS_GRID:
            for prior_name, prior in PRIOR_GRID:
                for n_candidates in CANDIDATE_GRID:
                    # Large fingerprints dominate the cost (n x N per draw): fewer draws there keep the build to minutes; the 95th percentile still
                    # rests on at least 40 maxima, which is enough for a stable floor at this scale.
                    reps_here = reps if n * n_candidates <= 2_000_000 else max(40, reps // 5)
                    cfg_seed = int(rng.integers(0, 2**31 - 1))
                    t_cfg = time.perf_counter()
                    draws = max_null_mi_gpu(cp, n, n_bins, prior, n_candidates, reps_here, cfg_seed)
                    cp.cuda.Device().synchronize()
                    done += 1
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done * (total - done)
                    if log is not None:
                        log(
                            f"[{done}/{total}] n={n} bins={n_bins} prior={prior_name} N={n_candidates} reps={reps_here} "
                            f"cfg={time.perf_counter() - t_cfg:.2f}s elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
                        )
                    key = f"{n}|{n_bins}|{prior_name}|{n_candidates}"
                    table[key] = {
                        "n": n,
                        "bins": n_bins,
                        "prior": list(prior),
                        "entropy": round(entropy_nats(prior), 6),
                        "candidates": n_candidates,
                        "threshold": round(float(np.quantile(draws, QUANTILE)), 8),
                        "mean_max": round(float(np.mean(draws)), 8),
                    }
    table_meta = {"reps": reps, "seed": seed, "quantile": QUANTILE, "built_seconds": round(time.perf_counter() - t0, 1)}
    return {"meta": table_meta, "entries": table}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    result = build(args.reps, args.seed, log=lambda s: print(s, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(result['entries'])} entries to {args.out} in {result['meta']['built_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

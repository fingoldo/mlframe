"""Profile the clustering of correlated factors (cluster_aggregate and DCD) on the biz_val synthetic generators, at a size given on the command line.

The biz_val tests fix their size (n=6000 at each call site) and assert AUC thresholds tuned for it, so this driver reuses the SAME generators (``tests/feature_selection/_synth/biz_val_synth.py``) and the same MRMR settings, with the row count as an argument.

Usage::

    python profile_cluster_correlated.py --n 20000 --generator latent --k 5 --variant both --repeats 3 --profile

``--variant``: ``on`` (cluster_aggregate and DCD enabled, the defaults), ``off`` (both disabled), ``dcd_off`` (only DCD disabled), ``both`` (on and off, alternating). ``--repeats`` runs each variant that many times and reports every wall: a single fit carries cold-start cost, so only repeated fits are comparable. ``--profile`` prints cProfile sorted by cumulative time for the last fit of each variant. The selected names are printed so a speed-up can be checked against unchanged selection.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
import warnings

warnings.simplefilter("ignore")
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1"}.items():
    os.environ.setdefault(_k, _v)

VARIANTS = {
    "on": {},
    "off": {"cluster_aggregate_enable": False, "dcd_enable": False},
    "dcd_off": {"dcd_enable": False},
}


def make_data(generator: str, n: int, k: int, seed: int):
    """The biz_val generator named by ``generator`` at ``n`` rows (``latent``: one hidden factor reflected in ``k`` noisy columns; ``two_groups``: two hidden factors, ``k`` columns each)."""
    from tests.feature_selection._synth.biz_val_synth import as_df, make_latent_reflections, make_two_latent_groups

    if generator == "latent":
        X, y, _info = make_latent_reflections(n=n, loadings=(1.0,) * k, noise_sd=(0.85,) * k, n_noise=3, seed=seed)
    elif generator == "two_groups":
        X, y, _info = make_two_latent_groups(n=n, k1=k, k2=k, noise=0.85, n_noise=3, seed=seed)
    else:
        raise SystemExit(f"unknown generator {generator!r}; expected latent or two_groups")
    return as_df(X), y


def fit_once(X, y, variant: str, seed: int):
    """One MRMR fit with the variant's overrides; returns (wall seconds, selected names, the estimator)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    t0 = time.perf_counter()
    m = MRMR(verbose=0, random_seed=seed, **VARIANTS[variant]).fit(X, y)
    wall = time.perf_counter() - t0
    return wall, sorted(str(v) for v in m.get_feature_names_out()), m


def main() -> None:
    """Run the requested variants, print every wall and the selected names, optionally the cProfile of the last fit of each variant."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5, help="noisy reflections per hidden factor")
    ap.add_argument("--generator", default="latent", choices=("latent", "two_groups"))
    ap.add_argument("--variant", default="both", choices=("on", "off", "dcd_off", "both"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    X, y = make_data(args.generator, args.n, args.k, args.seed)
    variants = ["on", "off"] if args.variant == "both" else [args.variant]
    print(f"DATA generator={args.generator} n={args.n} k={args.k} columns={X.shape[1]}", flush=True)
    fit_once(X, y, variants[0], args.seed)  # warm-up: cold start (numba cache, cupy) is not what is measured
    for rep in range(args.repeats):
        for v in variants:
            prof = cProfile.Profile() if (args.profile and rep == args.repeats - 1) else None
            if prof:
                prof.enable()
            wall, names, _m = fit_once(X, y, v, args.seed)
            if prof:
                prof.disable()
            print(f"WALL variant={v} rep={rep} wall_s={wall:.2f} selected={len(names)} names={names[:6]}", flush=True)
            if prof:
                buf = io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
                print(f"PROFILE variant={v}\n{buf.getvalue()}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))  # repo root: profiling -> _benchmarks -> feature_selection -> mlframe -> src -> root
    main()

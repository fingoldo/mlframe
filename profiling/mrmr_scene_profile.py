"""MRMR.fit profiling harness on the `scene` wide bed (2407x299).

Reused/extended from _benchmarks/fs_hybrid/round4_scene_cprofile.py:
 - pre-imports networkx (irrelevant warm import, keep it out of the profile),
 - bypasses the kernel-tuning DISK + SWEEP via an in-memory KernelTuningCache + forced-fallback get_or_tune,
 - runs BOTH cProfile (deterministic tottime attribution) AND a main-thread
   sampling profiler (sys._current_frames) so GIL-bound C/CUDA time that
   cProfile under-attributes still shows up,
 - captures the SELECTED feature set + engineered recipe so we have a
   bit-identity baseline to compare against after each optimization.

Usage:
  PYTHONPATH=<worktree>/src python mrmr_scene_profile.py [--rows N] [--njobs J]
                                                         [--dataset scene|gina_agnostic|Bioresponse]
                                                         [--cprofile-out path] [--selection-out path]
                                                         [--no-sampler]
"""
from __future__ import annotations
import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import threading
import time
import collections

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Pre-import networkx so its (irrelevant) one-time import never lands in the profile.
try:
    import networkx  # noqa: F401
except Exception:
    pass


def _disable_kernel_tuning_sweep():
    """Force the measurement-backed fallback (no cross-process sweep, no disk filelock)
    and an in-memory cache so we profile COMPUTE, not the cache/disk path."""
    try:
        import pyutilz.performance.kernel_tuning.cache as _M

        def _no_sweep(self, kernel_name, *, dims, tuner, axes, fallback, **kw):
            return fallback() if callable(fallback) else fallback

        _M.KernelTuningCache.get_or_tune = _no_sweep
        _inmem = _M.KernelTuningCache(in_memory=True)
        _M.KernelTuningCache.load_or_create = classmethod(lambda cls: _inmem)
        print("[kernel-tuning sweep+disk DISABLED -> in-memory fallback]", flush=True)
    except Exception as e:
        print(f"[no-sweep patch failed: {e}]", flush=True)


_disable_kernel_tuning_sweep()


def load_dataset(name: str, n_rows: int):
    from sklearn.datasets import fetch_openml
    if name == "scene":
        d = fetch_openml(name="scene", version=1, as_frame=True, parser="auto")
    elif name == "gina_agnostic":
        d = fetch_openml(name="gina_agnostic", version=1, as_frame=True, parser="auto")
    elif name == "Bioresponse":
        d = fetch_openml(name="Bioresponse", version=1, as_frame=True, parser="auto")
    else:
        raise ValueError(f"unknown dataset {name!r}")
    X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    y = pd.Series(pd.factorize(d.target)[0])
    y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if n_rows and n_rows < len(X):
        idx = np.random.default_rng(0).choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y


class MainThreadSampler(threading.Thread):
    """Cheap statistical sampler of the MAIN thread's Python stack. Captures
    GIL-bound time that cProfile under-counts (C calls show as their Python caller)."""

    def __init__(self, main_ident: int, interval: float = 0.005):
        super().__init__(daemon=True)
        self.main_ident = main_ident
        self.interval = interval
        # NB: must NOT be named ``_stop`` -- that shadows threading.Thread._stop()
        # which Thread.join() calls internally (-> 'Event' object is not callable).
        self._stop_evt = threading.Event()
        self.leaf_counts = collections.Counter()   # innermost frame file:line:func
        self.stack_counts = collections.Counter()   # full collapsed stack (folded)
        self.n_samples = 0

    def run(self):
        import sys as _sys
        while not self._stop_evt.is_set():
            frames = _sys._current_frames()
            fr = frames.get(self.main_ident)
            if fr is not None:
                self.n_samples += 1
                # leaf
                co = fr.f_code
                leaf = f"{os.path.basename(co.co_filename)}:{fr.f_lineno}:{co.co_name}"
                self.leaf_counts[leaf] += 1
                # folded stack
                stack = []
                f = fr
                depth = 0
                while f is not None and depth < 60:
                    c = f.f_code
                    stack.append(f"{os.path.basename(c.co_filename)}:{c.co_name}")
                    f = f.f_back
                    depth += 1
                self.stack_counts[";".join(reversed(stack))] += 1
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()


def selection_fingerprint(model, X):
    """Bit-identity reference: selected columns (raw + engineered) and the
    engineered-feature recipe, as a JSON-serializable dict."""
    try:
        cols = list(model.transform(X.iloc[:5]).columns)
    except Exception as e:
        cols = [f"<transform-failed: {e}>"]
    fp = {"selected_columns": cols, "n_selected": len(cols)}
    # engineered recipe / provenance, if exposed
    for attr in ("_engineered_features_", "engineered_features_", "selected_features_",
                 "support_features_", "_fe_recipe_", "fe_recipe_"):
        v = getattr(model, attr, None)
        if v is None:
            continue
        try:
            if isinstance(v, dict):
                fp[attr] = {str(k): str(v[k]) for k in v}
            elif isinstance(v, (list, tuple, np.ndarray)):
                fp[attr] = [str(x) for x in list(v)]
            else:
                fp[attr] = str(v)
        except Exception:
            fp[attr] = "<unserializable>"
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=int(os.environ.get("SCENE_N", "0")))  # 0 = full
    ap.add_argument("--njobs", type=int, default=-1)
    ap.add_argument("--dataset", type=str, default="scene")
    ap.add_argument("--cprofile-out", type=str, default="")
    ap.add_argument("--selection-out", type=str, default="")
    ap.add_argument("--no-sampler", action="store_true")
    ap.add_argument("--fe-max-steps", type=int, default=1)
    ap.add_argument("--no-profile", action="store_true", help="just time + selection, no cProfile/sampler")
    ap.add_argument("--sampler-only", action="store_true", help="run the low-overhead sampling profiler WITHOUT cProfile")
    args = ap.parse_args()

    X, y = load_dataset(args.dataset, args.rows)
    print(f"{args.dataset} shape={X.shape} pos={float(y.mean()):.3f} njobs={args.njobs} fe_max_steps={args.fe_max_steps}", flush=True)

    from mlframe.feature_selection.filters import MRMR
    import mlframe
    print(f"[mlframe from: {mlframe.__file__}]", flush=True)

    # WARM the JIT + imports on a tiny fit FIRST (discarded) so the profiled fit is free of one-time numba-compile cost.
    tw = time.time()
    MRMR(verbose=0, fe_max_steps=args.fe_max_steps, n_jobs=args.njobs, random_seed=0).fit(X.iloc[:150], y.iloc[:150])
    print(f"[warm-up fit {time.time()-tw:.1f}s]", flush=True)

    m = MRMR(verbose=0, fe_max_steps=args.fe_max_steps, n_jobs=args.njobs, random_seed=0)

    sampler = None
    if (args.sampler_only or not args.no_profile) and not args.no_sampler:
        sampler = MainThreadSampler(threading.get_ident(), interval=0.005)
        sampler.start()

    if args.no_profile or args.sampler_only:
        t0 = time.time()
        m.fit(X, y)
        dt = time.time() - t0
    else:
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
        m.fit(X, y)
        pr.disable()
        dt = time.time() - t0
        # Dump the raw stats IMMEDIATELY (before any cleanup that could raise) so a
        # downstream error never loses the (expensive) profile.
        if args.cprofile_out:
            try:
                pr.dump_stats(args.cprofile_out)
                print(f"[cProfile stats -> {args.cprofile_out}]", flush=True)
            except Exception as _e:
                print(f"[cProfile dump failed: {_e}]", flush=True)

    if sampler is not None:
        sampler.stop()
        try:
            sampler.join(timeout=2.0)
        except Exception as _e:
            print(f"[sampler join failed: {_e}]", flush=True)

    fp = selection_fingerprint(m, X)
    print(f"\n=== FIT WALL = {dt:.2f}s ; n_selected={fp['n_selected']} ; "
          f"engineered={sum(1 for c in fp['selected_columns'] if c not in set(X.columns))} ===", flush=True)

    if args.selection_out:
        fp["fit_wall_s"] = dt
        with open(args.selection_out, "w") as fh:
            json.dump(fp, fh, indent=2, sort_keys=True)
        print(f"[selection fingerprint -> {args.selection_out}]", flush=True)

    if not args.no_profile and not args.sampler_only:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats("tottime").print_stats(40)
        print("\n========== cProfile TOP 40 by TOTTIME ==========")
        print(s.getvalue())
        s2 = io.StringIO()
        ps2 = pstats.Stats(pr, stream=s2)
        ps2.sort_stats("cumulative").print_stats(30)
        print("\n========== cProfile TOP 30 by CUMTIME ==========")
        print(s2.getvalue())

    if sampler is not None and sampler.n_samples > 0:
        print(f"\n========== SAMPLING PROFILER (main thread, {sampler.n_samples} samples @5ms) ==========")
        print("--- TOP 30 LEAF frames (where the main thread actually sits) ---")
        for name, cnt in sampler.leaf_counts.most_common(30):
            print(f"{cnt:6d} ({100.0*cnt/sampler.n_samples:5.1f}%)  {name}")
        print("\n--- TOP 15 folded stacks ---")
        for name, cnt in sampler.stack_counts.most_common(15):
            print(f"{cnt:6d} ({100.0*cnt/sampler.n_samples:5.1f}%)  ...{name[-220:]}")


if __name__ == "__main__":
    main()

"""RFECV.fit profiling/bit-identity harness on the `scene` wide bed (2407x299).

Mirrors profiling/mrmr_scene_profile.py but drives mlframe.feature_selection.wrappers.RFECV.
RFECV is model-fit-bound (each MBH iter = n_splits inner fits + per-fold feature importances),
so the inner estimator dominates wall. Two estimator presets:
  --est logreg : LogisticRegression(max_iter=400) -- cheap, stable timings, fast iteration
  --est lgbm   : LightGBMClassifier -- the estimator the 2026-06-05 numpy-mirror opt targeted;
                 closer to a realistic GBDT RFECV run (slower).

Captures:
  - full fit wall (warm-up fit FIRST so JIT/import cost is excluded),
  - the SELECTED feature set (support_ -> column names) as the bit-identity golden,
  - cProfile tottime/cumtime top-N (RFECV has far fewer dispatches than MRMR's FE,
    so cProfile overhead is acceptable -- verified by --no-profile wall vs profiled wall),
  - optional main-thread sampler for GIL-bound C time.

Usage:
  PYTHONPATH=<worktree>/src python rfecv_scene_profile.py [--rows N] [--est logreg|lgbm]
        [--cv K] [--max-refits M] [--njobs J] [--no-profile] [--selection-out path]
        [--cprofile-out path] [--no-sampler]
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


def load_dataset(name: str, n_rows: int):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
    X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    y = pd.Series(pd.factorize(d.target)[0])
    y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if n_rows and n_rows < len(X):
        idx = np.random.default_rng(0).choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y


def make_estimator(kind: str):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=400, random_state=0)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, num_leaves=31, random_state=0, n_jobs=1, verbosity=-1)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=1)
    raise ValueError(f"unknown estimator {kind!r}")


class MainThreadSampler(threading.Thread):
    def __init__(self, main_ident: int, interval: float = 0.005):
        super().__init__(daemon=True)
        self.main_ident = main_ident
        self.interval = interval
        self._stop_evt = threading.Event()
        self.leaf_counts = collections.Counter()
        self.stack_counts = collections.Counter()
        self.n_samples = 0

    def run(self):
        import sys as _sys
        while not self._stop_evt.is_set():
            frames = _sys._current_frames()
            fr = frames.get(self.main_ident)
            if fr is not None:
                self.n_samples += 1
                co = fr.f_code
                leaf = f"{os.path.basename(co.co_filename)}:{fr.f_lineno}:{co.co_name}"
                self.leaf_counts[leaf] += 1
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
    """Bit-identity reference: selected columns as a sorted list."""
    feat_in = list(getattr(model, "feature_names_in_", X.columns))
    support = getattr(model, "support_", None)
    if support is None or len(support) == 0:
        sel = []
    elif isinstance(support[0], (bool, np.bool_)):
        sel = [feat_in[i] for i, s in enumerate(support) if s]
    else:
        sel = [feat_in[int(i)] for i in support]
    fp = {
        "selected_columns": sorted(map(str, sel)),
        "n_selected": len(sel),
        "n_features_": int(getattr(model, "n_features_", len(sel))),
    }
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=int(os.environ.get("SCENE_N", "0")))
    ap.add_argument("--dataset", type=str, default="scene")
    ap.add_argument("--est", type=str, default="logreg")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--max-refits", type=int, default=20)
    ap.add_argument("--njobs", type=int, default=1)
    ap.add_argument("--cprofile-out", type=str, default="")
    ap.add_argument("--selection-out", type=str, default="")
    ap.add_argument("--no-sampler", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--sampler-only", action="store_true")
    args = ap.parse_args()

    X, y = load_dataset(args.dataset, args.rows)
    print(f"{args.dataset} shape={X.shape} pos={float(y.mean()):.3f} est={args.est} cv={args.cv} "
          f"max_refits={args.max_refits} njobs={args.njobs}", flush=True)

    from mlframe.feature_selection.wrappers import RFECV
    import mlframe
    print(f"[mlframe from: {mlframe.__file__}]", flush=True)

    def build():
        return RFECV(
            estimator=make_estimator(args.est),
            cv=args.cv,
            max_refits=args.max_refits,
            n_jobs=args.njobs,
            verbose=0,
            random_state=0,
        )

    # WARM-up fit on a tiny slice (discarded) so JIT/import one-time cost is excluded.
    tw = time.time()
    build().fit(X.iloc[:200], y.iloc[:200])
    print(f"[warm-up fit {time.time()-tw:.1f}s]", flush=True)

    m = build()

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
        except Exception:
            pass

    fp = selection_fingerprint(m, X)
    print(f"\n=== FIT WALL = {dt:.2f}s ; n_selected={fp['n_selected']} ===", flush=True)

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
        print(f"\n========== SAMPLING PROFILER ({sampler.n_samples} samples @5ms) ==========")
        print("--- TOP 30 LEAF frames ---")
        for name, cnt in sampler.leaf_counts.most_common(30):
            print(f"{cnt:6d} ({100.0*cnt/sampler.n_samples:5.1f}%)  {name}")


if __name__ == "__main__":
    main()

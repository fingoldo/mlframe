"""Bench CPX6: knn_label_dispersion_features per-query loop vs vectorized.

The prior code looped over every query row computing np.bincount (regression) /
np.unique (classification) to build per-row class/bin counts. This is a Python
loop of n_q iterations each doing a small array op. We replace it with dense
label codes + np.add.at over an (n_q, k) index, building the full (n_q, n_codes)
count matrix in one vectorized pass, then reducing for entropy/majority_share.

OLD baseline loaded from `git show HEAD:<spatial.py>`.

Run:
    CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_engineering/_benchmarks/bench_knn_dispersion_vectorize.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SPATIAL_REL = "src/mlframe/feature_engineering/spatial.py"


def _load_baseline_module():
    src = subprocess.check_output(
        ["git", "show", f"HEAD:{SPATIAL_REL}"], cwd=REPO, text=True
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="spatial_baseline_"))
    p = tmpdir / "spatial_baseline.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("spatial_baseline", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_data(n_ref, n_q, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((n_ref, 2))
    q = rng.standard_normal((n_q, 2))
    reg_labels = rng.standard_normal(n_ref)
    cls_labels = rng.integers(0, n_classes, size=n_ref)
    return ref, q, reg_labels, cls_labels


def _best_of(fn, reps=5):
    best = float("inf"); out = None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return best, out


def _cmp(r_old, r_new):
    max_abs = 0.0
    for kk in r_old:
        a, b = np.asarray(r_old[kk], float), np.asarray(r_new[kk], float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"NaN mask mismatch {kk}"
        m = np.isfinite(a)
        if m.any():
            d = np.max(np.abs(a[m] - b[m]))
            max_abs = max(max_abs, d)
    return max_abs


def main():
    import mlframe.feature_engineering.spatial as new_mod
    base = _load_baseline_module()

    for n_q in (5_000, 50_000):
        ref, q, reg_labels, cls_labels = _make_data(20_000, n_q)
        for task, labels in (("regression", reg_labels), ("classification", cls_labels)):
            kw = dict(k=10, task=task)
            t_old, r_old = _best_of(lambda: base.knn_label_dispersion_features(q, ref, labels, **kw))
            t_new, r_new = _best_of(lambda: new_mod.knn_label_dispersion_features(q, ref, labels, **kw))
            max_abs = _cmp(r_old, r_new)
            print(f"task={task:<14} n_q={n_q:>6}  OLD={t_old*1e3:8.2f}ms  "
                  f"NEW={t_new*1e3:8.2f}ms  speedup={t_old/t_new:5.2f}x  max_abs_diff={max_abs:.2e}")


if __name__ == "__main__":
    main()

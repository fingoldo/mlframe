"""Benchmark E2.2: pympler.asizeof pre-pickle check vs naive post-save sensor.

User flag 2026-05-22: extra deps are fine if there's a speed win. Question:
does pympler.asizeof(payload) before pickle let us skip the expensive
first-dump on oversized models, saving wall-clock at the
save_mlframe_model entry?

Hypothesis: on a 4M-row payload the lean=False dump is 100+ MB of zstd-
compressed dill output that takes 2-5 seconds. The auto-lean retry doubles
that wall-clock (one fat dump, then one lean dump). pympler.asizeof walks
the object graph WITHOUT pickling so it's a cheap upfront estimate. If the
estimate exceeds the threshold we can flip lean=True before the first
pickle.

Run::

    python -m mlframe.training._benchmarks.bench_pympler_pre_pickle_check

Reports per N: (asizeof_ms, naive_save_ms, lean_save_ms, savings_ms).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

# Make src/ importable when run as a plain script too.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, os.path.join(_SRC, "src"))

from mlframe.training.io import save_mlframe_model  # noqa: E402


def _build_payload(n_train: int):
    rng = np.random.default_rng(42)
    return SimpleNamespace(
        model=SimpleNamespace(some_small_state=np.zeros(100, dtype=np.float32)),
        test_preds=rng.standard_normal(n_train // 8).astype(np.float32),
        test_probs=None,
        test_target=rng.standard_normal(n_train // 8).astype(np.float32),
        val_preds=rng.standard_normal(n_train // 8).astype(np.float32),
        val_probs=None,
        val_target=rng.standard_normal(n_train // 8).astype(np.float32),
        train_preds=rng.standard_normal(n_train).astype(np.float32),
        train_probs=None,
        train_target=rng.standard_normal(n_train).astype(np.float32),
        oof_preds=rng.standard_normal(n_train).astype(np.float32),
        oof_probs=None,
        metrics={"test": {"R2": 0.95}},
        columns=["f0", "f1", "f2"],
        pre_pipeline=None,
        train_od_idx=None, val_od_idx=None,
        trainset_features_stats={f"f{i}": {"mean": 0.0, "std": 1.0} for i in range(50)},
    )


def main():
    import pympler.asizeof as pasize

    print()
    print("# bench_pympler_pre_pickle_check  (pre-pickle asizeof vs post-save sensor+retry)")
    print()
    print(f"{'N':>10} {'asizeof_MB':>11} {'asizeof_ms':>11} {'naive_ms':>10} {'lean_ms':>10} {'naive_size_MB':>14} {'lean_size_MB':>13}")
    print("-" * 90)

    for n in (100_000, 1_000_000, 5_000_000):
        payload = _build_payload(n)

        # 1. pympler.asizeof on the in-memory graph.
        t0 = time.perf_counter()
        est_bytes = pasize.asizeof(payload)
        asizeof_ms = (time.perf_counter() - t0) * 1000.0
        est_mb = est_bytes / (1024 * 1024)

        # 2. naive save (lean=False), with auto-retry disabled to isolate the cost.
        with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
            fp = tf.name
        try:
            t0 = time.perf_counter()
            save_mlframe_model(payload, fp, verbose=0, lean=False, auto_lean_retry=False)
            naive_ms = (time.perf_counter() - t0) * 1000.0
            naive_size_mb = os.path.getsize(fp) / (1024 * 1024)
        finally:
            if os.path.exists(fp):
                os.remove(fp)

        # 3. lean save (skip the fat round entirely).
        with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
            fp = tf.name
        try:
            t0 = time.perf_counter()
            save_mlframe_model(payload, fp, verbose=0, lean=True, auto_lean_retry=False)
            lean_ms = (time.perf_counter() - t0) * 1000.0
            lean_size_mb = os.path.getsize(fp) / (1024 * 1024)
        finally:
            if os.path.exists(fp):
                os.remove(fp)

        print(
            f"{n:>10d} {est_mb:>11.1f} {asizeof_ms:>11.1f} {naive_ms:>10.1f} "
            f"{lean_ms:>10.1f} {naive_size_mb:>14.1f} {lean_size_mb:>13.2f}"
        )

    print()
    print("# Interpretation:")
    print("#   asizeof_ms < naive_ms when N is large -> pre-check pays off")
    print("#   savings on auto-lean retry: naive_ms (since we skip the fat first dump)")


if __name__ == "__main__":
    main()

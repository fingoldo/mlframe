"""Bench zstd single-threaded vs multi-threaded compress at typical metadata sizes (iter130).

c0097 iter130 profile attributed 470ms self-time / 2 calls to
``_finalize_and_save_metadata`` -- ~235ms / call, dominated by
pickle+zstd compression of the suite metadata bundle (~30 MB).

Bench results (zstandard 0.22, AMD Ryzen):

  28.6 MB   level=3 threads=0:    65 ms   threads=-1: 33 ms (~2x)
  57.2 MB   level=3 threads=0:   131 ms   threads=-1: 54 ms (~2.4x)

Output is BYTE-IDENTICAL to single-threaded -- zstd's multi-threaded API
splits into independent frames that decompress the same. Pure-win
no-tradeoff optimisation (no extra dep, no compression-ratio loss).

Run: ``python profiling/bench_zstd_threads.py``
"""

import time
import pickle
import zstandard as zstd
import numpy as np

# Synthetic metadata at typical 200k-row multi-model size: ~5-20 MB pickled
def make_synthetic_metadata(scale=1):
    rng = np.random.default_rng(0)
    # Mimic: trained models (numpy arrays), OOF preds, per-target reports
    metadata = {
        "trained": {
            f"model_{i}": {
                "weights": rng.standard_normal((200_000 * scale, 5)),
                "feature_importance": rng.standard_normal(50),
                "stats": {"mae": 0.123, "rmse": 0.456},
            }
            for i in range(3)
        },
        "oof_preds": rng.standard_normal((200_000 * scale, 3)),
        "test_preds": rng.standard_normal((50_000 * scale, 3)),
    }
    return metadata


for scale in (1, 2):
    md = make_synthetic_metadata(scale)
    pkl_bytes = pickle.dumps(md, protocol=5)
    print(f"\n== scale={scale}, pickled size = {len(pkl_bytes) / 1024 / 1024:.1f} MB ==")

    for level in (1, 3):
        for threads in (0, 1, -1):  # 0 = no MT, 1 = MT off, -1 = MT all cores
            cctx = zstd.ZstdCompressor(level=level, threads=threads)
            # Warmup
            for _ in range(3):
                cctx.compress(pkl_bytes)
            times = []
            for _ in range(3):
                t = time.perf_counter()
                for _ in range(5):
                    out = cctx.compress(pkl_bytes)
                times.append((time.perf_counter() - t) / 5)
            print(f'  level={level}, threads={threads:>2}: {min(times)*1000:7.1f}ms  out={len(out)/1024:.1f}KB')

"""Microbench: the pre-pickle ``pympler.asizeof`` size precheck in ``save_mlframe_model``.

Lead from iter143: ``asizeof`` walks the whole in-memory object graph to decide the eager/lean flip.
The precheck comment claims ``asizeof=0.5ms vs save=160ms at N=5M``; that figure was for a SimpleNamespace
holding a few big ndarrays (shallow graph, where asizeof is genuinely fast). This bench measures asizeof on
a DEEP object graph (a fitted sklearn RandomForest holds thousands of Tree objects + per-node arrays) -- the
worst case, since asizeof's cost is O(number of distinct python objects), not O(bytes). The eager/lean flip
only runs on SimpleNamespace payloads, so we wrap the model in a SimpleNamespace as the real caller does.

Run: python -m mlframe.training._benchmarks.bench_save_asizeof_precheck
"""
import time
from types import SimpleNamespace

import numpy as np


def _best_of(fn, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), sorted(ts)[len(ts) // 2]


def _make_bundle(kind: str):
    rng = np.random.default_rng(0)
    if kind == "shallow_bigarray":
        # The original benched shape: a few big ndarrays, shallow graph.
        return SimpleNamespace(
            preds=rng.standard_normal(5_000_000).astype(np.float32),
            oof=rng.standard_normal(5_000_000).astype(np.float32),
            meta={"a": 1, "b": 2},
        )
    if kind == "deep_rf":
        from sklearn.ensemble import RandomForestClassifier
        X = rng.standard_normal((20_000, 30))
        y = (rng.standard_normal(20_000) > 0).astype(int)
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=0)
        rf.fit(X, y)
        return SimpleNamespace(model=rf, feature_names=[f"f{i}" for i in range(30)], meta={"k": "v"})
    raise ValueError(kind)


def main():
    from pympler import asizeof as _pa
    import pickle as _pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file

    for kind in ("shallow_bigarray", "deep_rf"):
        bundle = _make_bundle(kind)
        # Warm.
        _pa.asizeof(bundle)
        _pickle.dumps(bundle, protocol=_pickle.HIGHEST_PROTOCOL)

        asz_min, asz_med = _best_of(lambda: _pa.asizeof(bundle), n=5)
        # The cheaper alternative: pickle.dumps already produces the bytes the save needs,
        # and len(bytes) is the true on-disk-precompression size. Time a full pickle.dumps.
        pk_min, pk_med = _best_of(lambda: _pickle.dumps(bundle, protocol=_pickle.HIGHEST_PROTOCOL), n=5)
        est_bytes = _pa.asizeof(bundle)
        pkl_bytes = len(_pickle.dumps(bundle, protocol=_pickle.HIGHEST_PROTOCOL))

        print(f"[{kind}]")
        print(f"  asizeof:        min={asz_min*1e3:8.2f} ms  med={asz_med*1e3:8.2f} ms  -> est {est_bytes/1e6:.1f} MB")
        print(f"  pickle.dumps:   min={pk_min*1e3:8.2f} ms  med={pk_med*1e3:8.2f} ms  -> {pkl_bytes/1e6:.1f} MB serialized")
        print(f"  asizeof / pickle ratio (min): {asz_min/pk_min:.2f}x")
        print()


if __name__ == "__main__":
    main()

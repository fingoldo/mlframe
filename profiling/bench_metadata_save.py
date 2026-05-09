"""Benchmark metadata save: joblib.dump vs pickle protocol=5 (+ optional zstd).

Models the structure of mlframe's ``metadata`` dict that
``_finalize_and_save_metadata`` writes via ``atomic_write_bytes(... lambda f: joblib.dump(metadata, f))``.

Realistic synthetic content:
  - small dict items (strings, ints, small lists)
  - sklearn fitted Pipeline (StandardScaler + LogisticRegression, real fitted on synthetic data)
  - sklearn IsolationForest fitted on a 5k x 10 numeric frame
  - large numpy arrays (trainset_features_stats per-feature: mean/std/quantiles)
  - per-model schemas (~40 entries, each a dict with 100-element list)
  - selected_features lists
"""
from __future__ import annotations

import io
import os
import pickle
import time
from typing import Callable, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


def build_synthetic_metadata(n_rows: int = 5000, n_cols: int = 50) -> dict:
    """Build a metadata dict matching the shape mlframe saves."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    y = rng.integers(0, 2, n_rows)

    # Real fitted sklearn objects so pickle-time work is realistic.
    scaler = StandardScaler().fit(X)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]).fit(X, y)
    od = IsolationForest(n_estimators=20, random_state=0).fit(X)

    cols = [f"feat_{i}" for i in range(n_cols)]
    cat_features = cols[:5]
    text_features = cols[5:7]

    trainset_features_stats = {
        col: {
            "mean": float(X[:, i].mean()),
            "std": float(X[:, i].std()),
            "min": float(X[:, i].min()),
            "max": float(X[:, i].max()),
            "quantiles": np.quantile(X[:, i], np.linspace(0.05, 0.95, 19)).astype(np.float32),
        }
        for i, col in enumerate(cols)
    }

    model_schemas = {
        f"model__{w}__sch_{i:04x}": {
            "schema_hash": f"{i:08x}",
            "input_schema": [
                {"name": col, "dtype": "Float32", "role": "numeric"} for col in cols
            ],
            "mlframe_model": "hgb" if i % 2 else "xgb",
            "weight_name": w,
        }
        for w in ("uniform", "recency")
        for i in range(20)
    }

    fairness_report = {
        f"binary__y__model_{i}__val": {
            "subgroup_metrics": {
                f"group_{j}": {"roc_auc": float(rng.random()), "n": int(rng.integers(50, 500))}
                for j in range(8)
            }
        }
        for i in range(4)
    }

    selected_features_per_model = {
        f"binary/y/model_{i}": [f"feat_{j}" for j in rng.choice(n_cols, size=20, replace=False)]
        for i in range(8)
    }

    metadata = {
        "pipeline": pipeline,
        "extensions_pipeline": None,
        "cat_features": cat_features,
        "text_features": text_features,
        "embedding_features": [],
        "columns": cols,
        "outlier_detection": {"applied": True, "n_dropped": 47, "fraction_dropped": 0.0094},
        "outlier_detector": od,
        "trainset_features_stats": trainset_features_stats,
        "model_schemas": model_schemas,
        "fairness_report": fairness_report,
        "selected_features": sorted({c for v in selected_features_per_model.values() for c in v}),
        "selected_features_per_model": selected_features_per_model,
        "slug_to_original_target_type": {"binary_classification": "BinaryClassification"},
        "slug_to_original_target_name": {"y": "y"},
    }
    return metadata


def time_writer(name: str, dump_fn: Callable[[object, io.BufferedWriter], None], obj: object, n: int = 5) -> Tuple[float, int]:
    """Time ``n`` writes to a temp BytesIO and return (avg_ms, payload_bytes)."""
    # Warmup
    buf = io.BytesIO()
    dump_fn(obj, buf)
    payload = len(buf.getvalue())  # works even after writer closes wrapped streams

    times = []
    for _ in range(n):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        dump_fn(obj, buf)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  {name:<45} avg {avg:6.2f}ms  payload {payload/1024:8.1f} KiB")
    return avg, payload


def time_reader(name: str, load_fn: Callable[[io.BufferedReader], object], blob: bytes, n: int = 5) -> float:
    """Time ``n`` reads from BytesIO; sanity-check load returns a dict."""
    obj = load_fn(io.BytesIO(blob))
    assert isinstance(obj, dict), f"{name}: load returned {type(obj)}, expected dict"
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        load_fn(io.BytesIO(blob))
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  {name:<45} avg {avg:6.2f}ms")
    return avg


def build_numpy_heavy_metadata(n_rows: int = 50000, n_cols: int = 200) -> dict:
    """Numpy-heavy variant: large quantile arrays per feature + a few
    fitted IsolationForests carrying internal numpy state. Models the
    pathological case where joblib's numpy fast-path could pay off.
    """
    base = build_synthetic_metadata(n_rows, n_cols)
    rng = np.random.default_rng(1)
    base["large_numpy_arrays"] = {
        f"big_{i}": rng.standard_normal(50000).astype(np.float32) for i in range(20)
    }
    return base


def main() -> int:
    for label, builder in (
        ("typical metadata (n=5000, k=50)", lambda: build_synthetic_metadata(5000, 50)),
        ("numpy-heavy metadata (n=50000, k=200, +20 large arrays)", lambda: build_numpy_heavy_metadata(50000, 200)),
    ):
        print(f"\n############################################################")
        print(f"### {label}")
        print(f"############################################################")
        run_one(builder())
    return 0


def run_one(metadata: dict) -> int:
    obj_repr_len = len(repr(metadata))
    print(f"  repr() length: {obj_repr_len:,} chars\n")

    print("=== WRITE ===")
    write_results = {}

    def joblib_dump_default(o, f):
        joblib.dump(o, f)
    write_results["joblib.dump (default)"] = time_writer("joblib.dump (default compress=0)", joblib_dump_default, metadata)

    def joblib_dump_compress3(o, f):
        joblib.dump(o, f, compress=3)
    write_results["joblib.dump compress=3"] = time_writer("joblib.dump compress=3 (zlib)", joblib_dump_compress3, metadata)

    def pickle_proto5(o, f):
        pickle.dump(o, f, protocol=5)
    write_results["pickle.dump proto=5"] = time_writer("pickle.dump protocol=5 (no compress)", pickle_proto5, metadata)

    try:
        import zstandard as zstd
        # zstd's ``stream_writer`` closes the wrapped stream on __exit__,
        # which breaks our ``buf.getvalue()`` accounting. Use one-shot
        # ``ZstdCompressor.compress(pickled)`` to keep the BytesIO open.
        def pickle_proto5_zstd(o, f, level=3):
            cctx = zstd.ZstdCompressor(level=level)
            f.write(cctx.compress(pickle.dumps(o, protocol=5)))
        write_results["pickle.dump proto=5 + zstd L3"] = time_writer("pickle.dump protocol=5 + zstd L3", pickle_proto5_zstd, metadata)
        def pickle_proto5_zstd1(o, f):
            return pickle_proto5_zstd(o, f, level=1)
        write_results["pickle.dump proto=5 + zstd L1"] = time_writer("pickle.dump protocol=5 + zstd L1", pickle_proto5_zstd1, metadata)
    except ImportError:
        print("  (zstandard not installed; skipping zstd benchmarks)")
        zstd = None

    print("\n=== READ (roundtrip sanity) ===")
    # Build payloads
    buf = io.BytesIO(); joblib.dump(metadata, buf); blob_joblib_default = buf.getvalue()
    buf = io.BytesIO(); joblib.dump(metadata, buf, compress=3); blob_joblib_c3 = buf.getvalue()
    buf = io.BytesIO(); pickle.dump(metadata, buf, protocol=5); blob_pickle = buf.getvalue()

    time_reader("joblib.load (default)", joblib.load, blob_joblib_default)
    time_reader("joblib.load compress=3", joblib.load, blob_joblib_c3)
    time_reader("pickle.load proto=5", pickle.load, blob_pickle)

    if zstd is not None:
        for level in (1, 3):
            cctx = zstd.ZstdCompressor(level=level)
            blob_zstd = cctx.compress(pickle.dumps(metadata, protocol=5))

            def load_zstd(f):
                dctx = zstd.ZstdDecompressor()
                return pickle.loads(dctx.decompress(f.read()))
            time_reader(f"pickle.load proto=5 + zstd L{level}", load_zstd, blob_zstd)

    print("\n=== SUMMARY ===")
    baseline = write_results["joblib.dump (default)"][0]
    for name, (ms, sz) in write_results.items():
        print(f"  {name:<45} {ms:6.2f}ms  ({ms/baseline:.2f}x baseline)  {sz/1024:8.1f} KiB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

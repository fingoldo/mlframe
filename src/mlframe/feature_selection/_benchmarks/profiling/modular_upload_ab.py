"""A/B of the source-block upload of modular_family_codes_gpu: two host float64 copies then upload, against per-column upload in the native dtype widened on the device."""
import time, numpy as np, pandas as pd, cupy as cp, psutil
me = psutil.Process(); rng = np.random.default_rng(0); n = 100000
for dt in (np.float32, np.float64):
    for k in (8, 30):
        X = pd.DataFrame({f"c{i}": rng.normal(size=n).astype(dt) for i in range(k)}); part = list(X.columns)
        def old():
            return cp.asarray(np.ascontiguousarray(X[part].to_numpy(dtype=np.float64).T))
        def new():
            src = cp.empty((len(part), n), dtype=cp.float64)
            for j, c in enumerate(part):
                col = X[c].to_numpy()
                src[j] = cp.asarray(np.ascontiguousarray(col)) if col.dtype in (np.float32, np.float64) else cp.asarray(X[c].to_numpy(dtype=np.float64, na_value=np.nan))
            return src
        assert bool(cp.array_equal(old(), new()))
        res = {}
        for name, f in (("old", old), ("new", new), ("old", old), ("new", new)):
            f(); cp.cuda.Stream.null.synchronize(); pf0 = me.memory_info().num_page_faults; t = time.perf_counter()
            for _ in range(10): f()
            cp.cuda.Stream.null.synchronize(); res[name] = (round((time.perf_counter() - t) / 10 * 1000, 1), int((me.memory_info().num_page_faults - pf0) / 10))
        print(f"{np.dtype(dt).name} k={k:3d}  old {res['old'][0]} ms {res['old'][1]} faults | new {res['new'][0]} ms {res['new'][1]} faults")

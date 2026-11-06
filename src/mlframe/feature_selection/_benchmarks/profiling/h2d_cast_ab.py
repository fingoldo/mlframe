"""A/B: host float64 copy then upload, against upload as float32 and cast on the device (the candidate-matrix upload of _strict_resident_mi)."""
import time, numpy as np, cupy as cp, psutil
me = psutil.Process()
rng = np.random.default_rng(0)
n = 100000
for k in (5, 14, 40):
    X = rng.normal(size=(n, k)).astype(np.float32)
    def old():
        return cp.asarray(np.ascontiguousarray(np.asarray(X, dtype=np.float64)))
    def new():
        return cp.asarray(np.ascontiguousarray(X)).astype(cp.float64)
    assert bool(cp.array_equal(old(), new()))
    res = {}
    for name, f in (("old", old), ("new", new), ("old", old), ("new", new)):
        f(); cp.cuda.Stream.null.synchronize()
        pf0 = me.memory_info().num_page_faults; t = time.perf_counter()
        for _ in range(20): f()
        cp.cuda.Stream.null.synchronize()
        dt = (time.perf_counter() - t) / 20 * 1000; pf = (me.memory_info().num_page_faults - pf0) / 20
        res[name] = (round(dt, 2), int(pf))
    print(f"k={k:3d} MB={X.nbytes/1e6:6.1f}  old {res['old'][0]} ms {res['old'][1]} faults/call | new {res['new'][0]} ms {res['new'][1]} faults/call")

"""cProfile of save_mlframe_model + load_mlframe_model end-to-end at a representative bundle.

Re-profile after iter143 (lib-version memoization). Surfaces the real save/load hotspots so the iter144
leads (asizeof precheck, sha256 reopen) can be judged against the actual wall breakdown.

Run: python -m mlframe.training._benchmarks.bench_save_load_profile
"""
import cProfile
import io as _io
import pstats
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _make_bundle():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20_000, 30))
    y = (rng.standard_normal(20_000) > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    rf.fit(X, y)
    return SimpleNamespace(model=rf, feature_names=[f"f{i}" for i in range(30)], meta={"k": "v"})


def main():
    from mlframe.training.io import save_mlframe_model, load_mlframe_model

    bundle = _make_bundle()
    tmp = Path(tempfile.mkdtemp()) / "bundle.dump"

    # Warm (lib-version memo, numba, zstd ctor).
    save_mlframe_model(bundle, str(tmp), verbose=0)
    load_mlframe_model(str(tmp))

    def _best(fn, n=7):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3, sorted(ts)[len(ts) // 2] * 1e3

    s_min, s_med = _best(lambda: save_mlframe_model(bundle, str(tmp), verbose=0))
    l_min, l_med = _best(lambda: load_mlframe_model(str(tmp)))
    print(f"save: min={s_min:.2f} ms  med={s_med:.2f} ms")
    print(f"load: min={l_min:.2f} ms  med={l_med:.2f} ms")
    print()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        save_mlframe_model(bundle, str(tmp), verbose=0)
        load_mlframe_model(str(tmp))
    pr.disable()
    s = _io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()

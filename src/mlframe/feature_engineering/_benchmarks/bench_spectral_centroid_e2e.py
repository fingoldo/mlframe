"""End-to-end A/B + identity for the spec@k centroid reduction in spectral.py.

OLD side loaded from git (HEAD spectral.py) into a temp module; NEW is the working tree.
Confirms the full-function wall improves and output is identical to ~1e-9.

Run: CUDA_VISIBLE_DEVICES="" python bench_spectral_centroid_e2e.py
"""
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[3]  # .../src
SPECTRAL_REL = "mlframe/feature_engineering/spectral.py"


def _load_old_module():
    src = subprocess.run(
        ["git", "show", f"HEAD:src/{SPECTRAL_REL}"],
        cwd=PKG_ROOT.parent, capture_output=True, text=True, check=True,
    ).stdout
    # rewrite the relative import to absolute so it loads standalone
    src = src.replace("from .grouped import", "from mlframe.feature_engineering.grouped import")
    tmp = Path(tempfile.mkdtemp()) / "spectral_old.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("spectral_old", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sys.path.insert(0, str(PKG_ROOT))
    import mlframe.feature_engineering.spectral as new
    old = _load_old_module()

    rng = np.random.default_rng(7)
    n = 200_000
    values = np.cumsum(rng.standard_normal(n))
    group_ids = rng.integers(0, 5, n)

    for name in ("rolling_spectral_centroid", "rolling_spectral_bandwidth"):
        fo = getattr(old, name); fn = getattr(new, name)
        ro = fo(values, group_ids, window_K=100)
        rn = fn(values, group_ids, window_K=100)
        m = np.nanmax(np.abs(ro - rn))
        # warm + time
        for _ in range(2):
            fo(values, group_ids, window_K=100); fn(values, group_ids, window_K=100)
        to = min((lambda: (lambda t0: (fo(values, group_ids, window_K=100), time.perf_counter() - t0)[1])(time.perf_counter()))() for _ in range(7))
        tn = min((lambda: (lambda t0: (fn(values, group_ids, window_K=100), time.perf_counter() - t0)[1])(time.perf_counter()))() for _ in range(7))
        print(f"{name:32} OLD {to*1e3:8.2f}ms  NEW {tn*1e3:8.2f}ms  speedup {to/tn:4.2f}x  max_abs_diff {m:.2e}")


if __name__ == "__main__":
    main()

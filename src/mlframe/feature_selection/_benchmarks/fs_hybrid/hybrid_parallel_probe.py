"""Probe: do the two independent expensive hybrid members (_run_shap, _run_boruta_premerge) overlap under a
threading backend, and is their output bit-identical to the sequential path? Decides Lever-1 (member parallelism).

We fit the hybrid up to Stage 2's shared state (by calling the real fit once), then re-run JUST the two members
sequentially vs via a 2-thread pool, comparing wall AND the returned column lists. Threading (not loky) because the
members are sklearn/njit/numpy heavy (GIL-releasing) and a thread pool keeps RNG/seed state byte-identical (same
process, no pickling of fitted MRMR member / no re-seeded child RNG) -> the combine result stays deterministic.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_ROWS = int(os.environ.get("HYB_N", "1500"))
SEED = int(os.environ.get("HYB_SEED", "0"))


def _disable_kernel_tuning_sweep():
    try:
        import pyutilz.performance.kernel_tuning.cache as _M
        def _no_sweep(self, kernel_name, *, dims, tuner, axes, fallback, **kw):
            if not callable(fallback):
                return fallback
            try:
                return fallback()
            except TypeError:
                try:
                    return fallback(**dims)
                except TypeError:
                    return fallback(*dims.values())
        _M.KernelTuningCache.get_or_tune = _no_sweep
        _inmem = _M.KernelTuningCache(in_memory=True)
        _M.KernelTuningCache.load_or_create = classmethod(lambda cls: _inmem)
    except Exception:  # nosec B110 - best-effort path
        pass
# NOTE: do NOT call _disable_kernel_tuning_sweep() at import -- discover_tuners imports this module and an
# import-time monkeypatch of get_or_tune breaks ``refresh-all``. It is invoked under __main__ below.


def main() -> None:
    """Fit the hybrid once for shared state, then re-run the two independent members sequentially vs in a 2-thread
    pool, printing the wall speedup and whether each member's output is bit-identical across the two paths."""
    from hard_synth import make_hard_dataset
    from mlframe.feature_selection import HybridSelector
    X, y, _ = make_hard_dataset(n_samples=N_ROWS, seed=SEED)
    print(f"shape={X.shape}", flush=True)

    # warm
    HybridSelector(random_state=SEED).fit(X.iloc[:150], y.iloc[:150])

    # Fit once to populate shared state (members_, fi_, relevant_, X_aug etc.) -- we re-run the two members below.
    h = HybridSelector(random_state=SEED); h.fit(X, y)
    X_aug, relevant, artifacts = h._Xaug_, h.relevant_, h.artifacts_

    # sequential
    t0 = time.time(); shap_s = h._run_shap(X_aug, y, relevant, artifacts); boruta_s = h._run_boruta_premerge(X_aug, y, relevant)
    seq = time.time() - t0
    print(f"SEQUENTIAL shap+boruta: {seq:.2f}s", flush=True)

    # threaded (2 workers)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_shap = ex.submit(h._run_shap, X_aug, y, relevant, artifacts)
        f_boruta = ex.submit(h._run_boruta_premerge, X_aug, y, relevant)
        shap_t, boruta_t = f_shap.result(), f_boruta.result()
    par = time.time() - t0
    print(f"THREADED   shap+boruta: {par:.2f}s  (speedup {seq/par:.2f}x)", flush=True)

    print(f"shap bit-identical:   {shap_s == shap_t}", flush=True)
    print(f"boruta bit-identical: {boruta_s == boruta_t}", flush=True)
    if shap_s != shap_t:
        print(f"  seq  shap={shap_s}\n  par  shap={shap_t}", flush=True)
    if boruta_s != boruta_t:
        print(f"  seq  boruta={boruta_s}\n  par  boruta={boruta_t}", flush=True)


if __name__ == "__main__":
    _disable_kernel_tuning_sweep()
    main()

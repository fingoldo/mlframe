"""Bench: hoist the y/z-invariant block out of the greedy-CMI noise-floor scan and the step-0 marginal scan.

Two unfused callers of ``_renumber_joint`` / ``_entropy_from_classes`` remained in
``greedy_cmi_fe_construct`` after the per-candidate main scan was fused via ``cmi_from_binned_fixed_yz``:

1. ``_noise_floor_for_current_z`` scored 24 sampled candidates against a FIXED ``y_shuf`` / ``z_joint`` via
   the plain ``_cmi_from_binned`` -- recomputing ``renumber(y_shuf, z)`` + ``H(Z)`` + ``H(Y,Z)`` (conditional)
   or ``H(Y)`` (marginal) on every one of the 24 samples, all invariant across them.
2. The step-0 (empty-Z) main scan scored EVERY candidate via ``_cmi_from_binned(cand, y, None)`` --
   recomputing ``_entropy_from_classes(y)`` per candidate though y is fit-constant.

The fix routes both through the existing hoist helpers (``precompute_cmi_yz_terms`` +
``cmi_from_binned_fixed_yz`` / ``precompute_marginal_y_terms`` + ``marginal_mi_binned_fixed_y``), so the
invariant block is computed ONCE per step. Bit-identical (same MM plug-in CMI, ~1e-15 fp order).

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_cmi_greedy_noisefloor_marginal_hoist
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as m


def _bench(n=30_000, n_samples=24, n_cand=200, nbins=10, reps=20):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, n).astype(np.int64)
    z = rng.integers(0, 30, n).astype(np.int64)
    cands = [rng.integers(0, nbins, n).astype(np.int64) for _ in range(n_cand)]
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    sample = cands[:n_samples]

    # warm numba
    _ = m._cmi_from_binned(cands[0], y_shuf, z)
    yi, zi, h_yz, h_z, k_yz, k_z, nf = m.precompute_cmi_yz_terms(y_shuf, z)
    yzd, _ = m._renumber_joint(yi, zi)
    _ = m.cmi_from_binned_fixed_yz(cands[0], yi, zi, h_yz, h_z, k_yz, k_z, nf, yz_i=yzd)
    yt = m.precompute_marginal_y_terms(y)
    _ = m.marginal_mi_binned_fixed_y(cands[0], *yt)

    def old_noise():
        return [m._cmi_from_binned(c, y_shuf, z) for c in sample]

    def new_noise():
        _yi, _zi, _hyz, _hz, _kyz, _kz, _nf = m.precompute_cmi_yz_terms(y_shuf, z)
        _yzd, _ = m._renumber_joint(_yi, _zi)
        return [m.cmi_from_binned_fixed_yz(c, _yi, _zi, _hyz, _hz, _kyz, _kz, _nf, yz_i=_yzd) for c in sample]

    def old_marg():
        return [m._cmi_from_binned(c, y, None) for c in cands]

    def new_marg():
        _yt = m.precompute_marginal_y_terms(y)
        return [m.marginal_mi_binned_fixed_y(c, *_yt) for c in cands]

    # identity check
    assert np.allclose(old_noise(), new_noise(), atol=1e-9)
    assert np.allclose(old_marg(), new_marg(), atol=1e-9)

    def timeit(fn, inner=5):
        best = np.inf
        for _ in range(reps):
            t = time.perf_counter()
            for _ in range(inner):
                fn()
            best = min(best, (time.perf_counter() - t) / inner)
        return best

    on, nn = timeit(old_noise), timeit(new_noise)
    om, nm = timeit(old_marg), timeit(new_marg)
    print(f"n={n} samples={n_samples} cand={n_cand}")
    print(f"  noise-floor (per step): OLD {on*1e3:8.2f}ms  NEW {nn*1e3:8.2f}ms  speedup {on/nn:.2f}x")
    print(f"  step-0 marginal scan:   OLD {om*1e3:8.2f}ms  NEW {nm*1e3:8.2f}ms  speedup {om/nm:.2f}x")


if __name__ == "__main__":
    _bench()

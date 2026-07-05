"""Warm best-of-N process_time A/B for the fused densify+entropy kernel in the greedy-CMI-FE hot path.

Baseline (OLD): the per-candidate CMI scorer computes ``xz = _renumber_joint(x,z)`` and
``xyz = _renumber_joint(x,y,z)`` (a 3-column factorize = 1 factorize + 2 combine walks), then feeds each
dense-label array to ``_entropy_from_classes`` (a second bincount pass) and discards the labels.

NEW: ``_joint_entropy_two`` histograms the joint partition inline (O(n+k), no length-n relabel array, no
second pass) and returns (H, k) directly; the round-fixed dense ``(y,z)`` codes are reused so ``H(X,Y,Z)``
is a 2-array densify against ``yz_dense`` (partition(x, part(y,z)) == partition(x,y,z)) instead of the
3-column renumber. Bit-identical to ~1e-9 (entropy is over the count multiset, label-permutation-invariant).

This benches the per-candidate scorer ``cmi_from_binned_fixed_yz`` (OLD 3-col path vs NEW yz_i path) at the
production shape (n=1e6, realistic nbins / support cardinality), paired + process_time so concurrent-agent
wall noise cancels. Run:  python -m mlframe.feature_selection.filters._benchmarks.bench_cmi_fused_joint_entropy
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as m


def _bench(n=1_000_000, nbins=10, z_card=200, ncand=40, trials=7, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n).astype(np.int64)  # target classes
    z = rng.integers(0, z_card, n).astype(np.int64)  # conditioning support codes
    cands = [rng.integers(0, nbins, n).astype(np.int64) for _ in range(ncand)]
    yi, zi, h_yz, h_z, k_yz, k_z, nf = m.precompute_cmi_yz_terms(y, z)
    yzd, _ = m._renumber_joint(yi, zi)

    def run_old():
        s = 0.0
        for x in cands:
            s += m.cmi_from_binned_fixed_yz(x, yi, zi, h_yz, h_z, k_yz, k_z, nf)
        return s

    def run_new():
        s = 0.0
        for x in cands:
            s += m.cmi_from_binned_fixed_yz(x, yi, zi, h_yz, h_z, k_yz, k_z, nf, yz_i=yzd)
        return s

    # warm (numba JIT + cache)
    a = run_old(); b = run_new()
    assert abs(a - b) < 1e-6, (a, b)

    old_t, new_t, wins = [], [], 0
    for _ in range(trials):
        t0 = time.process_time(); run_old(); old_t.append(time.process_time() - t0)
        t0 = time.process_time(); run_new(); new_t.append(time.process_time() - t0)
        if new_t[-1] < old_t[-1]:
            wins += 1
    om, nm = min(old_t), min(new_t)
    omed, nmed = sorted(old_t)[trials // 2], sorted(new_t)[trials // 2]
    print(f"n={n} nbins={nbins} z_card={z_card} ncand={ncand} trials={trials}")
    print(f"  OLD (3-col renumber+entropy): min={om*1e3:8.2f}ms  median={omed*1e3:8.2f}ms")
    print(f"  NEW (fused joint entropy)   : min={nm*1e3:8.2f}ms  median={nmed*1e3:8.2f}ms")
    print(f"  speedup: min {om/nm:.2f}x  median {omed/nmed:.2f}x  new-faster {wins}/{trials}")


if __name__ == "__main__":
    _bench(n=1_000_000, nbins=10, z_card=200)
    _bench(n=1_000_000, nbins=10, z_card=2000)
    _bench(n=200_000, nbins=8, z_card=100)

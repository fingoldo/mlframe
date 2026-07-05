"""Benchmark: does Miller-Madow debiasing of the SU edge reduce over-clustering
of high-cardinality features? (audit dcd-core-5 / integration-defaults-8 /
friend-graph-2 / shap-proxy-clustering-1).

Raw plug-in SU = 2*(H_a + H_b - H_ab)/(H_a + H_b). Plug-in entropy is biased
LOW, and H_ab (k_a*k_b bins) is biased most -> MI over-estimated -> SU
over-estimated -> a fixed tau (0.7) spuriously links INDEPENDENT high-cardinality
pairs. Miller-Madow adds (k-1)/(2n) per entropy, correcting H_ab most -> SU down
for high-card independent pairs, while a genuine near-duplicate (which really
shares information) keeps a high SU.

We compute SU both ways (raw vs MM) for:
  (1) a genuine redundant pair (b = a with 10% flips)   -> should stay HIGH
  (2) an INDEPENDENT high-cardinality pair               -> raw inflated, MM low
across cardinalities and n, and report how often each crosses tau=0.7.
Standalone (uses info_theory primitives) so we decide BEFORE wiring it in.

RESULT (2026-06-03): NO ACTIONABLE WIN -> do NOT wire MM into the SU/MI edges.
At realistic settings (DCD caps binning at ~10 bins; n>=1500) independent
high-cardinality pairs already score SU 0.002-0.028, FAR below tau=0.7 -- there
is no over-clustering to fix. MM only shaves ~0.01 off true-redundancy SU
(harmless) but corrects nothing. The plug-in bias only bites at cardinality
>> n which binning prevents, and the friend-graph edge already carries a
cardinality-aware G-test floor (edge_significance*(na-1)(nb-1)/(2n)). Shipping
MM here would be a risky no-value change.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory import (
    entropy,
    entropy_miller_madow,
    merge_vars,
)


def _su(fd, a, b, fn, n, *, mm: bool):
    idx_a = np.array([a], dtype=np.int64)
    idx_b = np.array([b], dtype=np.int64)
    idx_ab = np.array([a, b], dtype=np.int64)
    _, fa, _ = merge_vars(fd, idx_a, None, fn)
    _, fb, _ = merge_vars(fd, idx_b, None, fn)
    _, fab, _ = merge_vars(fd, idx_ab, None, fn)
    if mm:
        ha = entropy_miller_madow(fa, n)
        hb = entropy_miller_madow(fb, n)
        hab = entropy_miller_madow(fab, n)
    else:
        ha, hb, hab = entropy(fa), entropy(fb), entropy(fab)
    denom = ha + hb
    if denom <= 0:
        return 0.0
    mi = ha + hb - hab
    return max(0.0, min(1.0, 2.0 * mi / denom))


def main():
    tau = 0.7
    print(f"tau={tau}  (SU > tau => the pair is CLUSTERED/pruned)")
    print(f"{'n':>6} {'card':>5} | {'redundant raw':>13} {'redundant mm':>12} | " f"{'indep raw':>10} {'indep mm':>9}")
    for n in (1500, 5000):
        for card in (4, 10, 16):
            red_raw = red_mm = ind_raw = ind_mm = 0.0
            reps = 8
            for seed in range(reps):
                rng = np.random.default_rng(seed * 131 + n + card)
                a = rng.integers(0, card, n).astype(np.int32)
                flip = rng.random(n) < 0.10
                b = np.where(flip, rng.integers(0, card, n), a).astype(np.int32)  # redundant
                c = rng.integers(0, card, n).astype(np.int32)  # independent of a
                fd = np.column_stack([a, b, c])
                fn = np.array([card, card, card], dtype=np.int64)
                red_raw += _su(fd, 0, 1, fn, n, mm=False)
                red_mm += _su(fd, 0, 1, fn, n, mm=True)
                ind_raw += _su(fd, 0, 2, fn, n, mm=False)
                ind_mm += _su(fd, 0, 2, fn, n, mm=True)
            red_raw /= reps; red_mm /= reps; ind_raw /= reps; ind_mm /= reps
            def mark(v):
                return f"{v:.3f}{'*' if v > tau else ' '}"

            print(f"{n:>6} {card:>5} | {mark(red_raw):>13} {mark(red_mm):>12} | " f"{mark(ind_raw):>10} {mark(ind_mm):>9}   (* = clustered)")
    print("\nWANT: redundant stays >tau (clustered) under BOTH; independent " "high-card should be >tau under raw (false cluster) but <tau under MM.")


if __name__ == "__main__":
    main()

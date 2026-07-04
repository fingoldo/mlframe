"""Evidence for critique N-F6: is the Chao-Shen (CS) MI estimator matched by its permutation null?

N-F6 was filed as the CS analogue of N-F1 (a permutation null must use the SAME estimator as the observed statistic).
N-F1 was a real bug because Miller-Madow IS wired into the observed relevance (set_mi_miller_madow) and threaded into
compute_relevance_score, while the null used plug-in -> mismatch.

This script establishes, with evidence, that CS is NOT in the same situation:

1. chao_shen_mi / chao_shen_entropy are STANDALONE estimators. No production relevance/redundancy/null path calls them
   (grep: the only callers are tests + a micro-bench). The njit relevance kernel compute_relevance_score dispatches
   only use_su / use_mm / plug-in; there is no CS branch.

2. mi_correction='chao_shen' is an ACCEPTED option string (_VALID_MI_CORRECTIONS) but MRMR.fit only wires
   miller_madow: `_mm_on = mi_correction == 'miller_madow'; set_mi_miller_madow(_mm_on)`. So mi_correction='chao_shen'
   silently degrades to the plug-in 'none' behaviour for BOTH the observed relevance and its permutation null.

Consequence: there is NO observed-vs-null estimator MISMATCH to fix -- under mi_correction='chao_shen' the observed and
the null are ALREADY the same estimator (both plug-in). N-F6 therefore collapses to DOC (no null-using CS path exists).

The separate, genuine finding this surfaces (tracked FUTURE, not fixed here because wiring CS into relevance is a
selection-changing feature needing its own biz-value bench + a joint-count CS relevance kernel): mi_correction='chao_shen'
is accepted-but-silently-ignored. This script demonstrates that silent degrade so a future wiring PR has a pinned baseline.

Run:
  python -m mlframe.feature_selection.filters._benchmarks.bench_nf6_chao_shen_null_status
"""
import numpy as np


def _demo_chao_shen_is_standalone():
    from mlframe.feature_selection.filters._chao_shen import chao_shen_mi
    from mlframe.feature_selection.filters.info_theory._class_mi_kernels import compute_mi_from_classes

    rng = np.random.default_rng(0)
    n, kx = 2000, 8
    x = rng.integers(0, kx, n).astype(np.int32)
    y = np.where(rng.random(n) < 0.2, x % 2, rng.integers(0, 2, n)).astype(np.int32)
    fx = np.bincount(x, minlength=kx).astype(np.float64) / n
    fy = np.bincount(y, minlength=2).astype(np.float64) / n
    plugin = compute_mi_from_classes(x, fx, y, fy, dtype=np.int32)
    cs = chao_shen_mi(x, y)
    print(f"standalone estimators: plug-in MI={plugin:.5f}  chao_shen MI={cs:.5f}  (CS is a direct call, not routed through any null)")


def _demo_mi_correction_chao_shen_silently_degrades():
    # mi_correction='chao_shen' passes validation but MRMR.fit only sets the Miller-Madow toggle,
    # so the observed relevance AND its null both stay plug-in -> matched estimators, no mismatch.
    from mlframe.feature_selection.filters.info_theory._state_and_dispatch import use_mi_miller_madow, set_mi_miller_madow

    for corr in ("none", "miller_madow", "chao_shen"):
        mm_on = corr == "miller_madow"      # exactly MRMR.fit's wiring
        set_mi_miller_madow(mm_on)
        print(f"mi_correction={corr!r:14s} -> use_mi_miller_madow()={use_mi_miller_madow()}  "
              f"(chao_shen and none produce the SAME plug-in observed+null: no estimator mismatch)")
    set_mi_miller_madow(False)


if __name__ == "__main__":
    _demo_chao_shen_is_standalone()
    print()
    _demo_mi_correction_chao_shen_silently_degrades()

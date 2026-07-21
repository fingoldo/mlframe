"""A/B bench + bit-identity verification for the vectorized ``tune_decision_threshold`` rewrite.

Confirms the vectorized searchsorted/suffix-sum sweep selects the EXACT same threshold as the original
O(n_candidates * n) per-candidate scorer loop (kept here verbatim as ``_reference_tune_decision_threshold``
for comparison only), across random-valued AND heavily-tied/discrete ``p`` inputs, then measures wall time
at the sizes the audit flagged as the realistic production scale.
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.training.core._setup_helpers import DEFAULT_PROBABILITY_THRESHOLD, tune_decision_threshold


def _reference_tune_decision_threshold(y_true, pos_proba, *, metric="balanced_accuracy", default=DEFAULT_PROBABILITY_THRESHOLD, n_candidates=200):
    """The ORIGINAL O(n_candidates * n) implementation, kept verbatim for A/B comparison only."""
    from sklearn.metrics import f1_score

    from mlframe.metrics.core import balanced_accuracy_binary

    y = np.asarray(y_true).ravel()
    p = np.asarray(pos_proba, dtype=np.float64).ravel()
    if y.shape[0] == 0 or y.shape[0] != p.shape[0] or not np.all(np.isfinite(p)):
        return float(default)
    classes = np.unique(y)
    if classes.shape[0] < 2:
        return float(default)

    if metric == "f1":
        def scorer(yt, yp):
            return f1_score(yt, yp, zero_division=0)
    else:
        scorer = balanced_accuracy_binary

    candidates = np.linspace(0.0, 1.0, n_candidates + 2)[1:-1]
    best_thr = float(default)
    best_score = scorer(y, (p >= default).astype(np.int8))
    for thr in candidates:
        s = scorer(y, (p >= thr).astype(np.int8))
        if s > best_score:
            best_score = s
            best_thr = float(thr)
    return best_thr


def _verify_bit_identical():
    rng = np.random.default_rng(0)
    cases = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        n = r.integers(50, 5000)
        y = r.integers(0, 2, size=n)
        p = r.uniform(0, 1, size=n)
        cases.append((f"random_seed{seed}", y, p))
    # Heavily-tied/discrete p: only 5 distinct probability values -- the case that would break a
    # rank-based approximation but must still be bit-identical under the exact searchsorted approach.
    n_tied = 3000
    y_tied = rng.integers(0, 2, size=n_tied)
    p_tied = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=n_tied)
    cases.append(("heavily_tied_p", y_tied, p_tied))
    # All-identical p (degenerate).
    cases.append(("constant_p", rng.integers(0, 2, size=500), np.full(500, 0.42)))

    n_mismatches = 0
    for name, y, p in cases:
        for metric in ("balanced_accuracy", "f1"):
            ref = _reference_tune_decision_threshold(y, p, metric=metric)
            new = tune_decision_threshold(y, p, metric=metric)
            match = ref == new
            print(f"{name:20s} metric={metric:18s} ref={ref:.6f} new={new:.6f} match={match}")
            if not match:
                n_mismatches += 1
    if n_mismatches:
        raise AssertionError(f"{n_mismatches} bit-identity mismatch(es) -- see above.")
    print("\nAll cases bit-identical.")


def _bench():
    rng = np.random.default_rng(0)
    for n in (10_000, 100_000, 1_000_000):
        y = rng.integers(0, 2, size=n).astype(np.int8)
        p = rng.uniform(0, 1, size=n)
        for metric in ("balanced_accuracy", "f1"):
            t0 = time.perf_counter()
            tune_decision_threshold(y, p, metric=metric)
            t1 = time.perf_counter()
            print(f"n={n:>9,}  {metric:18s} vectorized: {t1 - t0:.4f}s")


if __name__ == "__main__":
    _verify_bit_identical()
    print()
    _bench()

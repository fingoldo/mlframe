"""Isolated bench: ensemble-flavour selection metric (ICE-first vs AUC-first).

Production default lives in ``training/core/_ensemble_chooser.py``:
``_CLASSIFICATION_METRICS`` probes ``("ice","lower")`` FIRST, so on a
classification run ``_choose_ensemble_flavour`` picks the flavour that
minimises integral calibration error on OOF, ignoring discrimination.
The chosen flavour is persisted to metadata and REPLAYED verbatim at
predict-time -- so this metric order is the production ensemble-combination
default.

Lever: does ranking flavours by OOF ICE-first beat ranking by OOF ROC-AUC-first
on the HONEST (held-out test) ROC-AUC of the chosen flavour?

Method (fully isolated, no suite):
  - 5 synthetic binary scenarios x 3 seeds = 15 cells.
  - In each cell build K diverse member-probability vectors (varied skill +
    miscalibration), split into OOF (selection surface) and TEST (honest).
  - Build all 6 SIMPLE_ENSEMBLING_METHODS flavours via the REAL ``combine_probs``.
  - Score each flavour's OOF ICE (real ``compute_probabilistic_multiclass_error``)
    and OOF/TEST ROC-AUC.
  - Run the REAL chooser logic (mirrored, configurable metric order) under
    policy A = ICE-first (production) vs policy B = ROC-AUC-first.
  - Compare HONEST test ROC-AUC of each policy's chosen flavour.

Honest metric: held-out test ROC-AUC of the production-chosen flavour.
Run: python -m mlframe.models.ensembling._benchmarks.bench_ensemble_chooser_rank_metric
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.models.ensembling.base import SIMPLE_ENSEMBLING_METHODS, combine_probs
from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error


def _two_col(p1: np.ndarray) -> np.ndarray:
    """(N,) positive-class prob -> (N,2) prob matrix the ICE metric expects."""
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p1, p1])


def _make_member(rng, y, skill, bias, temp):
    """A member prob = sigmoid(temp*(logit_signal*skill) + bias) with noise."""
    n = y.shape[0]
    signal = np.where(y == 1, rng.normal(1.2, 1.0, n), rng.normal(-1.2, 1.0, n))
    z = temp * (signal * skill) + bias + rng.normal(0, 0.3, n)
    return 1.0 / (1.0 + np.exp(-z))


def _scenarios():
    # (name, n, K, base_rate, member_specs[(skill,bias,temp)])
    return [
        ("balanced_5mem", 4000, 5, 0.5,
         [(1.0, 0.0, 1.0), (0.8, 0.6, 1.4), (1.1, -0.4, 0.7), (0.6, 0.2, 1.0), (0.9, 0.0, 1.2)]),
        ("imbalanced_10pct", 5000, 5, 0.1,
         [(1.0, 0.0, 1.0), (0.9, 0.8, 1.5), (1.2, -0.6, 0.6), (0.5, 0.3, 1.0), (1.0, 0.1, 1.3)]),
        ("imbalanced_3pct", 6000, 4, 0.03,
         [(1.1, 0.0, 1.0), (0.7, 1.0, 1.6), (1.0, -0.8, 0.6), (0.9, 0.2, 1.1)]),
        ("strong_overconfident", 4000, 6, 0.4,
         [(1.0, 0.0, 2.0), (1.0, 0.0, 2.2), (0.9, 0.0, 1.8), (1.1, 0.0, 2.1), (0.8, 0.0, 1.9), (1.0, 0.0, 2.0)]),
        ("mixed_skill", 5000, 5, 0.25,
         [(1.4, 0.0, 1.0), (0.4, 0.5, 1.2), (1.0, -0.3, 0.9), (0.7, 0.1, 1.1), (0.3, 0.0, 1.0)]),
        ("hard_lowsignal", 5000, 5, 0.3,
         [(0.35, 0.0, 1.0), (0.30, 0.4, 1.3), (0.40, -0.3, 0.8), (0.25, 0.1, 1.0), (0.32, 0.0, 1.1)]),
        ("hard_imbal_5pct", 6000, 5, 0.05,
         [(0.45, 0.0, 1.0), (0.30, 0.7, 1.5), (0.50, -0.6, 0.7), (0.28, 0.2, 1.0), (0.40, 0.0, 1.2)]),
    ]


# Two candidate metric probe orders. Policy A is the PRODUCTION default
# (ice first). Policy B ranks discrimination (roc_auc) first.
_ICE_FIRST = (("ice", "lower"), ("roc_auc", "higher"))
_AUC_FIRST = (("roc_auc", "higher"), ("ice", "lower"))


def _choose(flavour_metrics: dict, order, split="oof") -> str:
    """Mirror of _choose_ensemble_flavour ranking logic for a given probe order."""
    for metric, direction in order:
        scored = [(k, m[split].get(metric)) for k, m in flavour_metrics.items()]
        scored = [(k, s) for k, s in scored if s is not None and np.isfinite(s)]
        if not scored:
            continue
        if direction == "lower":
            scored.sort(key=lambda kv: (kv[1], kv[0]))
        else:
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[0][0]
    return next(iter(flavour_metrics))


def run():
    wins_auc_first = 0
    wins_ice_first = 0
    ties = 0
    deltas = []
    print(f"{'scenario':22s} {'seed':>4s} {'ice_pick':>8s} {'auc_pick':>8s} "
          f"{'test_auc_ice':>12s} {'test_auc_auc':>12s} {'delta':>8s}")
    for name, n, K, base, specs in _scenarios():
        for seed in (0, 1, 2):
            rng = np.random.default_rng(1000 + seed)
            y = (rng.random(n) < base).astype(int)
            if y.sum() < 5 or (n - y.sum()) < 5:
                continue
            members = np.stack([_make_member(rng, y, *s) for s in specs])  # (K,N)
            # split rows into OOF (selection) and TEST (honest), stratified-ish
            idx = rng.permutation(n)
            cut = n // 2
            oof_i, test_i = idx[:cut], idx[cut:]
            y_oof, y_test = y[oof_i], y[test_i]
            if y_oof.sum() < 3 or y_test.sum() < 3:
                continue

            fmetrics = {}
            for flav in SIMPLE_ENSEMBLING_METHODS:
                combined = combine_probs(members[:, oof_i], flav)  # OOF positive-class probs
                combined_t = combine_probs(members[:, test_i], flav)
                ice = compute_probabilistic_multiclass_error(
                    y_true=y_oof, y_score=_two_col(combined), method="multicrit", nbins=10,
                )
                try:
                    auc_oof = roc_auc_score(y_oof, combined)
                    auc_test = roc_auc_score(y_test, combined_t)
                except ValueError:
                    auc_oof = auc_test = None
                fmetrics[flav] = {
                    "oof": {"ice": float(ice), "roc_auc": auc_oof},
                    "test": {"roc_auc": auc_test},
                }

            ice_pick = _choose(fmetrics, _ICE_FIRST)
            auc_pick = _choose(fmetrics, _AUC_FIRST)
            t_ice = fmetrics[ice_pick]["test"]["roc_auc"]
            t_auc = fmetrics[auc_pick]["test"]["roc_auc"]
            if t_ice is None or t_auc is None:
                continue
            delta = t_auc - t_ice  # >0 => AUC-first wins honest test AUC
            deltas.append(delta)
            if delta > 1e-4:
                wins_auc_first += 1
            elif delta < -1e-4:
                wins_ice_first += 1
            else:
                ties += 1
            print(f"{name:22s} {seed:>4d} {ice_pick:>8s} {auc_pick:>8s} "
                  f"{t_ice:>12.5f} {t_auc:>12.5f} {delta:>+8.5f}")

    arr = np.asarray(deltas)
    print("\n=== SUMMARY (honest held-out test ROC-AUC of chosen flavour) ===")
    print(f"cells evaluated      : {len(arr)}")
    print(f"AUC-first WINS        : {wins_auc_first}")
    print(f"ICE-first WINS        : {wins_ice_first}")
    print(f"ties (|delta|<1e-4)   : {ties}")
    print(f"mean test-AUC delta   : {arr.mean():+.6f}  (positive favours AUC-first)")
    print(f"median test-AUC delta : {np.median(arr):+.6f}")
    maj = "AUC-first" if wins_auc_first > wins_ice_first else (
        "ICE-first (production)" if wins_ice_first > wins_auc_first else "TIE")
    print(f"MAJORITY              : {maj}")


if __name__ == "__main__":
    run()

"""Isolated bench: LeakageSafeEncoder ``smoothing`` default for target_mean.

The smoothing constant pulls rare-category encodings toward the global prior:

    enc(c) = (n_c * mean_c + smoothing * prior) / (n_c + smoothing)

Too small -> rare categories overfit their noisy in-sample mean; too large ->
informative high-count categories get washed toward the prior. The honest
metric here is HELD-OUT predictive quality of the encoded feature on rows the
encoder never saw: we fit on a train split, ``transform`` a disjoint test
split, and score (a) ROC-AUC of the encoding vs the true binary label and
(b) MSE between the encoding and the TRUE per-category posterior used to
generate the data. Both are leak-free (test rows were in no fold).

Scenarios sweep cardinality and the rare-category mass -- the regime where
smoothing matters most. 5 synthetic generators x 3 seeds = 15 cells per
candidate. A challenger flips the default only if it wins the MAJORITY of
cells on the headline metric (mean held-out AUC).

VERDICT (deferred, NOT flipped): smoothing=3.0 beats the default 10.0 on held-out
log-loss (17/25) and posterior-MSE (17/25) but only ties on AUC (14/25). The default
is NOT flipped because ``smoothing`` is a single field shared by target_mean /
m_estimate / james_stein / loo and only target_mean is benched here, and the suite
passes ``smoothing`` explicitly so an estimator-default flip would be a no-op. A real
flip needs a per-method bench + a coherent TargetEncodeParams default. Bench kept for
that follow-up (REJECTED != DELETED).

Run:  python bench_target_encoder_smoothing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[4]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from mlframe.training.feature_handling.target_encoders import (  # noqa: E402
    LeakageSafeEncoder,
)

CANDIDATES = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
SEEDS = [0, 1, 2, 3, 4]


def _make_data(rng, kind):
    """Return (cats, y, true_posterior) for a binary target with a known
    per-category positive rate. ``kind`` controls cardinality + rarity."""
    if kind == "lowcard_balanced":
        K, n = 20, 4000
        true_rate = rng.uniform(0.2, 0.8, size=K)
        counts = rng.integers(50, 400, size=K)
    elif kind == "highcard_rare":
        K, n = 400, 6000
        true_rate = rng.uniform(0.1, 0.9, size=K)
        counts = rng.integers(2, 30, size=K)  # many rare cats
    elif kind == "mixed_zipf":
        K, n = 200, 8000
        true_rate = rng.uniform(0.05, 0.95, size=K)
        w = 1.0 / np.arange(1, K + 1)
        counts = np.maximum(1, (w / w.sum() * n).astype(int))
    elif kind == "weak_signal":
        K, n = 100, 5000
        true_rate = 0.5 + rng.uniform(-0.12, 0.12, size=K)  # near-prior
        counts = rng.integers(10, 120, size=K)
    elif kind == "strong_rare":
        K, n = 150, 5000
        true_rate = rng.choice([0.05, 0.95], size=K)  # strong but rare
        counts = rng.integers(3, 25, size=K)
    else:
        raise ValueError(kind)

    cat_ids = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(cat_ids)
    cats = np.array([f"c{int(i)}" for i in cat_ids], dtype=object)
    post = true_rate[cat_ids]
    y = (rng.uniform(size=len(cat_ids)) < post).astype(np.float64)
    return cats, y, post


def _split(rng, n, frac=0.5):
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


SCENARIOS = [
    "lowcard_balanced",
    "highcard_rare",
    "mixed_zipf",
    "weak_signal",
    "strong_rare",
]


def main():
    # cell_auc[sm] = list of held-out AUCs; cell_mse[sm] = list of post-MSEs
    cell_auc = {sm: [] for sm in CANDIDATES}
    cell_mse = {sm: [] for sm in CANDIDATES}
    cell_ll = {sm: [] for sm in CANDIDATES}
    wins = {sm: 0 for sm in CANDIDATES}       # by AUC
    wins_ll = {sm: 0 for sm in CANDIDATES}    # by log-loss (calibration)

    for kind in SCENARIOS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            cats, y, post = _make_data(rng, kind)
            tr, te = _split(rng, len(cats))
            best_sm, best_auc = None, -1.0
            best_sm_ll, best_ll = None, np.inf
            for sm in CANDIDATES:
                enc = LeakageSafeEncoder(
                    method="target_mean", smoothing=sm, cv=5, random_state=seed
                )
                enc.fit(cats[tr], y[tr])
                pred = enc.transform(cats[te])
                yte = y[te]
                try:
                    auc = roc_auc_score(yte, pred)
                except ValueError:
                    auc = 0.5
                mse = float(np.mean((pred - post[te]) ** 2))
                # target_mean encoding IS a probability estimate; clip and score
                # log-loss directly -- calibration-sensitive headline.
                p = np.clip(pred, 1e-6, 1 - 1e-6)
                ll = log_loss(yte, p, labels=[0.0, 1.0])
                cell_auc[sm].append(auc)
                cell_mse[sm].append(mse)
                cell_ll[sm].append(ll)
                if auc > best_auc:
                    best_auc, best_sm = auc, sm
                if ll < best_ll:
                    best_ll, best_sm_ll = ll, sm
            wins[best_sm] += 1
            wins_ll[best_sm_ll] += 1

    print("=== held-out target-mean encoding: smoothing sweep ===")
    print(f"{'smoothing':>10} {'mean_AUC':>10} {'mean_LogLoss':>13} "
          f"{'mean_MSEpost':>13} {'AUCwins':>8} {'LLwins':>8}")
    for sm in CANDIDATES:
        print(
            f"{sm:>10.1f} {np.mean(cell_auc[sm]):>10.4f} "
            f"{np.mean(cell_ll[sm]):>13.4f} {np.mean(cell_mse[sm]):>13.5f} "
            f"{wins[sm]:>8d} {wins_ll[sm]:>8d}"
        )
    n_cells = len(SCENARIOS) * len(SEEDS)
    best_by_auc = max(CANDIDATES, key=lambda s: np.mean(cell_auc[s]))
    best_by_ll = min(CANDIDATES, key=lambda s: np.mean(cell_ll[s]))
    best_ll_wins = max(CANDIDATES, key=lambda s: wins_ll[s])
    print(f"\nn_cells={n_cells}  best_mean_AUC sm={best_by_auc}  "
          f"best_mean_LogLoss sm={best_by_ll}  "
          f"majority_LLwins sm={best_ll_wins} ({wins_ll[best_ll_wins]}/{n_cells})")

    # Head-to-head: challenger=3.0 vs incumbent default=10.0, per-cell.
    chal, inc = 3.0, 10.0
    ll_better = sum(a < b for a, b in zip(cell_ll[chal], cell_ll[inc]))
    auc_better = sum(a > b for a, b in zip(cell_auc[chal], cell_auc[inc]))
    mse_better = sum(a < b for a, b in zip(cell_mse[chal], cell_mse[inc]))
    print(f"\nH2H sm={chal} vs incumbent sm={inc} (per-cell, n={n_cells}):")
    print(f"  log-loss better: {ll_better}/{n_cells}   "
          f"AUC better: {auc_better}/{n_cells}   "
          f"MSE-post better: {mse_better}/{n_cells}")


if __name__ == "__main__":
    main()

"""Honest measurement: arithmetic mean vs geometric (log-average) mean for blending member probabilities.

Lever (qual-13): the ensemble/chain predict path blends per-member class probabilities with an ARITHMETIC mean
(``stacked.mean(axis=0)`` in ``_classif_helpers._ChainEnsemble.predict_proba`` and equal-weight strategies elsewhere).
The hypothesis under the log-loss proper scoring rule is that a GEOMETRIC mean / log-average (then renormalised to the
simplex) is better for diverse probabilistic members -- this is the classic "logarithmic opinion pool" vs "linear
opinion pool" question. But it is NOT a slam-dunk: the arithmetic (linear) pool is generally better-calibrated; the
geometric (log) pool sharpens and can hurt Brier / calibration (ECE).

This bench builds synthetic ensembles of K diverse probabilistic members on a known binary target, and measures
honest-holdout NLL (log-loss), Brier, and ECE for both aggregations across >=5 seeds and several scenarios. Run:

    python -m mlframe.training.composite.ensemble._benchmarks.bench_proba_aggregation_arith_vs_geom

Verdict gate (project rule): a flip only happens if one aggregation wins the MAJORITY of cells on the headline honest
metric (NLL) WITHOUT materially regressing Brier / ECE, across the majority of seeds AND scenarios.

Measured verdict (2026-06-15, 8 scenarios x 7 seeds = 56 cells): GEOMETRIC wins NLL 50/56, Brier 50/56, ECE 53/56.
Only S7 (extreme contamination, 40% confidently-wrong members) flips to arithmetic on NLL/Brier -- the known log-pool
fragility, gated as the opt-in ``proba_aggregation="arithmetic"`` path. Default flipped to "geometric".

cProfile: ``aggregate_member_probas`` is one vectorised ``np.log``/``np.exp`` over an (M, N, K) array, called once per
``predict`` (M ~ 3 chains). It is <1% of predict wall (the per-chain ``predict_proba`` dominates); no actionable
speedup, and the kernel-acceleration ladder is not warranted (single call per predict, not a hot loop).
"""

from __future__ import annotations

import numpy as np


def _arith_mean(stacked: np.ndarray) -> np.ndarray:
    """Linear opinion pool: arithmetic mean over members. ``stacked`` is (M, N, K)."""
    return stacked.mean(axis=0)


def _geom_mean(stacked: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Logarithmic opinion pool: geometric mean over members, renormalised to the simplex.

    geo_k = exp(mean_m log p_mk); then divide by sum_k geo_k so each row sums to 1. ``eps`` floors probabilities
    before the log so a single zero from one member does not annihilate the whole class (the standard log-pool guard).
    """
    clipped = np.clip(stacked, eps, 1.0)
    log_mean = np.log(clipped).mean(axis=0)
    geo = np.exp(log_mean)
    geo /= geo.sum(axis=-1, keepdims=True)
    return geo


def _nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Multiclass log-loss (NLL). ``p`` is (N, K), ``y`` integer labels."""
    pc = np.clip(p, eps, 1.0)
    n = y.shape[0]
    return float(-np.log(pc[np.arange(n), y]).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error against one-hot)."""
    k = p.shape[1]
    onehot = np.eye(k)[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the predicted top class (standard confidence-binned ECE)."""
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y.shape[0]
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _make_members(rng, n, k, n_members, diversity, miscal, balance, wrong_frac=0.0):
    """Build a known target + K diverse, optionally miscalibrated member probability matrices.

    diversity: stddev of per-member logit noise (higher = more diverse / decorrelated members).
    miscal: temperature exponent applied to member probabilities (>1 => overconfident/sharpened members; <1 => underconfident).
    balance: class prior vector (length k).
    wrong_frac: fraction of members that are CONFIDENTLY WRONG (a rogue/adversarial member putting high mass on a wrong
                class). This is the classic regime where the LINEAR (arithmetic) pool is robust and the LOG pool is
                catastrophic -- one confident-wrong member drives the geometric mean's correct-class mass toward the eps
                floor, blowing up NLL. Including it is mandatory to avoid the "geometric always wins" synthetic artefact.
    Returns y (N,), stacked (M, N, K) honest-holdout member probabilities.
    """
    prior = np.array(balance, dtype=float)
    prior /= prior.sum()
    y = rng.choice(k, size=n, p=prior)
    base_logits = np.zeros((n, k))
    base_logits[np.arange(n), y] = rng.uniform(1.2, 2.2)  # separable-ish true signal strength
    n_wrong = int(round(wrong_frac * n_members))
    members = []
    for mi in range(n_members):
        if mi < n_wrong:
            # Confidently-wrong rogue member: sharp mass on a non-true class for a subset of rows.
            wrong_cls = (y + 1 + rng.integers(0, max(1, k - 1), size=n)) % k
            logits = rng.normal(0.0, 0.3, size=(n, k))
            logits[np.arange(n), wrong_cls] += rng.uniform(2.5, 4.0)
        else:
            noise = rng.normal(0.0, diversity, size=(n, k))
            logits = base_logits + noise + rng.normal(0.0, 0.3, size=(1, k))  # per-member bias
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        if miscal != 1.0:
            p = p**miscal
            p /= p.sum(axis=1, keepdims=True)
        members.append(p)
    return y, np.stack(members, axis=0)


SCENARIOS = {
    # name: dict(k, n_members, diversity, miscal, balance)
    "S1_diverse_calibrated_binary": dict(k=2, n_members=5, diversity=1.0, miscal=1.0, balance=[1, 1]),
    "S2_overconfident_members_binary": dict(k=2, n_members=5, diversity=0.8, miscal=2.0, balance=[1, 1]),
    "S3_underconfident_diverse_3cls": dict(k=3, n_members=7, diversity=1.4, miscal=0.6, balance=[1, 1, 1]),
    "S4_imbalanced_binary": dict(k=2, n_members=5, diversity=1.0, miscal=1.3, balance=[9, 1]),
    "S5_lowK_highdiversity_4cls": dict(k=4, n_members=3, diversity=1.8, miscal=1.0, balance=[1, 1, 1, 1]),
    # Adversarial regimes where the LINEAR pool is expected to win (robustness to confidently-wrong members):
    "S6_one_rogue_member_binary": dict(k=2, n_members=5, diversity=1.0, miscal=1.0, balance=[1, 1], wrong_frac=0.2),
    "S7_two_rogue_of_five_3cls": dict(k=3, n_members=5, diversity=1.0, miscal=1.0, balance=[1, 1, 1], wrong_frac=0.4),
    "S8_calibrated_lowdiversity_binary": dict(k=2, n_members=4, diversity=0.4, miscal=1.0, balance=[1, 1], wrong_frac=0.0),
}

SEEDS = [11, 23, 37, 51, 67, 83, 99]
N = 4000


def run():
    results = {}
    wins = {"NLL": {"arith": 0, "geom": 0}, "Brier": {"arith": 0, "geom": 0}, "ECE": {"arith": 0, "geom": 0}}
    print(f"{'scenario':<34}{'seed':>5}{'NLL_a':>9}{'NLL_g':>9}{'Bri_a':>9}{'Bri_g':>9}{'ECE_a':>9}{'ECE_g':>9}")
    for sname, cfg in SCENARIOS.items():
        agg = {m: {"arith": [], "geom": []} for m in ("NLL", "Brier", "ECE")}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            y, stacked = _make_members(rng, N, cfg["k"], cfg["n_members"], cfg["diversity"], cfg["miscal"], cfg["balance"], cfg.get("wrong_frac", 0.0))
            pa = _arith_mean(stacked)
            pg = _geom_mean(stacked)
            ma = (_nll(y, pa), _brier(y, pa), _ece(y, pa))
            mg = (_nll(y, pg), _brier(y, pg), _ece(y, pg))
            for metric, va, vg in zip(("NLL", "Brier", "ECE"), ma, mg):
                agg[metric]["arith"].append(va)
                agg[metric]["geom"].append(vg)
                wins[metric]["arith" if va <= vg else "geom"] += 1
            print(f"{sname:<34}{seed:>5}{ma[0]:>9.4f}{mg[0]:>9.4f}{ma[1]:>9.4f}{mg[1]:>9.4f}{ma[2]:>9.4f}{mg[2]:>9.4f}")
        results[sname] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in agg.items()}
    print("\n=== per-scenario MEAN over seeds ===")
    for sname, r in results.items():
        print(f"{sname}")
        for m in ("NLL", "Brier", "ECE"):
            a, g = r[m]["arith"], r[m]["geom"]
            tag = "ARITH" if a <= g else "GEOM"
            print(f"   {m:<6} arith={a:.4f}  geom={g:.4f}  winner={tag}  delta={(g - a):+.4f}")
    print("\n=== per-cell win counts (lower is better; #seeds*#scen = " f"{len(SEEDS) * len(SCENARIOS)} cells) ===")
    for m in ("NLL", "Brier", "ECE"):
        print(f"   {m:<6} arith_wins={wins[m]['arith']:>3}  geom_wins={wins[m]['geom']:>3}")
    return results, wins


if __name__ == "__main__":
    run()

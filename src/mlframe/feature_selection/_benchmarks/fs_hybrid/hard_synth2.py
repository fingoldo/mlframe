"""Adversarial FS-comparison datasets (round-5 hard beds).

Each preset is engineered so that ONE individual feature selector provably LOSES (misses signal / keeps
noise / collapses on cost) while a NAMED rescuing member recovers it -- so HybridSelector's vote=1 union
over members beats every individual. Returns ``(X: DataFrame, y: Series, truth: dict)`` with the recovery
contract keys consumed by run_experiment.recovery():

  base                 - causal latent feature names (recovery target)
  relevant             - base + redundant copies
  noise                - pure-noise feature names
  interaction_operands - operands of multiplicative/XOR terms (marginal MI ~ 0; synergy blindspot)
  quadratic_operands   - operands of squared terms (kept for contract symmetry; may be empty)

Design notes are in each generator docstring: which individual is defeated and which member rescues.
All seed-parameterized + deterministic via np.random.default_rng(seed). Classification beds are kept
balanced unless the spec is explicitly rare (D3).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _finalize(cols: dict, rng, base_names, redundant_names, noise_names, interaction_operands, quadratic_operands, y):
    """Assemble DataFrame, shuffle columns (position carries no info), build truth dict."""
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order); X = X[order]
    truth = dict(
        base=base_names,
        relevant=base_names + redundant_names,
        noise=noise_names,
        interaction_operands=interaction_operands,
        quadratic_operands=quadratic_operands,
    )
    return X, pd.Series(y, name="target"), truth


# --------------------------------------------------------------------------- D1
def pure_xor_zeromain(seed: int = 0, n: int = 8000, p: int = 120):
    """3 XOR pairs (y ~ sign(a)*sign(b)): operands have EXACTLY zero marginal effect, plus 4 linear anchors.

    DEFEATS: MRMR-marginal + Boruta -- both score features by univariate / marginal relevance, and each XOR
             operand alone is independent of y (0 main effect, 0 marginal MI), so neither operand survives.
    RESCUED BY: tree-member (LightGBM splits capture the joint sign*sign interaction) + MRMR-FE synergy
             (engineered products expose the XOR). HybridSelector's tree/FE members catch the 6 operands.
    """
    rng = np.random.default_rng(seed)
    n_xor = 3; n_lin = 4
    n_base = 2 * n_xor + n_lin
    z = rng.standard_normal((n, n_base))
    logit = np.zeros(n)
    inter_idx = []
    idx = 0
    for _ in range(n_xor):
        a, b = idx, idx + 1; idx += 2
        logit += 2.2 * np.sign(z[:, a]) * np.sign(z[:, b])  # pure XOR: each operand 0 marginal, 0 main
        inter_idx += [a, b]
    lin_idx = list(range(idx, idx + n_lin)); idx += n_lin
    for k, j in enumerate(lin_idx):
        logit += (1.1 - 0.1 * k) * z[:, j]
    pr = 1.0 / (1.0 + np.exp(-logit / 1.4))
    y = (rng.random(n) < pr).astype(int)

    cols, base_names = {}, []
    for i in range(n_base):
        cols[name := f"inf_{i}"] = z[:, i]; base_names.append(name)
    noise_names = []
    n_noise = p - n_base
    for i in range(n_noise):
        cols[name := f"noise_{i}"] = rng.standard_normal(n)
        noise_names.append(name)
    return _finalize(cols, rng, base_names, [], noise_names, [f"inf_{i}" for i in inter_idx], [], y)


# --------------------------------------------------------------------------- D2
def heavytail_linear(seed: int = 0, n: int = 6000, p: int = 60):
    """6 linear signals pushed through sign(z)*|z|^3 with Student-t(df=2) operand noise.

    DEFEATS: MRMR-marginal + Boruta -- the cubic warp + heavy-tailed (infinite-variance) contamination wrecks
             rank/MI-based marginal scoring and Boruta's shadow comparison (a few extreme tail draws dominate
             impurity), so genuine weak-to-moderate signals get rejected as indistinguishable from shadows.
    RESCUED BY: RFECV-logit (linear backward elimination on a StandardScaler'd, clipped axis is robust to the
             monotone warp) + MRMR prewarp FE (rank/quantile prewarp restores the linear-usable axis).
    """
    rng = np.random.default_rng(seed)
    n_lin = 6
    z = rng.standard_normal((n, n_lin))
    # heavy-tailed contamination on the operands (df=2 -> infinite variance)
    z = z + 0.6 * rng.standard_t(df=2, size=(n, n_lin))
    warp = np.sign(z) * np.abs(z) ** 3  # severe monotone distortion
    logit = np.zeros(n)
    for k in range(n_lin):
        col = warp[:, k]
        col = col / (np.std(col) + 1e-9)  # normalize the warped contribution
        logit += (1.0 - 0.1 * k) * col
    pr = 1.0 / (1.0 + np.exp(-logit / 1.5))
    y = (rng.random(n) < pr).astype(int)

    cols, base_names = {}, []
    for i in range(n_lin):
        cols[name := f"inf_{i}"] = warp[:, i]; base_names.append(name)
    noise_names = []
    for i in range(p - n_lin):
        cols[name := f"noise_{i}"] = np.sign(g := rng.standard_normal(n)) * np.abs(g) ** 3
        noise_names.append(name)
    return _finalize(cols, rng, base_names, [], noise_names, [], [], y)


# --------------------------------------------------------------------------- D3
def rare_class_imbalance(seed: int = 0, n: int = 12000, p: int = 80, pos_rate: float = 0.02):
    """pos_rate ~ 0.02: 5 linear + 1 interaction + noise. Positives are scarce (~240 of 12000).

    DEFEATS: ShapProxied + Boruta -- both rely on internal CV / shadow trees; with ~2% positives the per-fold
             positive count is tiny, SHAP/impurity estimates are high-variance, and shadows beat real features
             by chance, so signals are dropped (low recall on the minority-driving features).
    RESCUED BY: MRMR (global MI over all 12k rows, no per-fold positive starvation) + RFECV with stratified CV
             (preserves the ~2% rate per fold -> stable elimination).
    """
    rng = np.random.default_rng(seed)
    n_lin = 5; n_inter = 1
    n_base = n_lin + 2 * n_inter
    z = rng.standard_normal((n, n_base))
    logit = np.zeros(n)
    for k in range(n_lin):
        logit += (1.4 - 0.12 * k) * z[:, k]
    a, b = n_lin, n_lin + 1
    logit += 1.6 * z[:, a] * z[:, b]
    inter_idx = [a, b]
    # shift the intercept so the marginal positive rate hits pos_rate
    from math import log
    # binary search the bias for the requested prevalence (deterministic, no randomness)
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rate = np.mean(1.0 / (1.0 + np.exp(-(logit + mid))))
        if rate < pos_rate:
            lo = mid
        else:
            hi = mid
    bias = 0.5 * (lo + hi)
    pr = 1.0 / (1.0 + np.exp(-(logit + bias)))
    y = (rng.random(n) < pr).astype(int)

    cols, base_names = {}, []
    for i in range(n_base):
        cols[name := f"inf_{i}"] = z[:, i]; base_names.append(name)
    noise_names = []
    for i in range(p - n_base):
        cols[name := f"noise_{i}"] = rng.standard_normal(n)
        noise_names.append(name)
    return _finalize(cols, rng, base_names, [], noise_names, [f"inf_{i}" for i in inter_idx], [], y)


# --------------------------------------------------------------------------- D4
def categorical_highcard(seed: int = 0, n: int = 8000, p: int = 70):
    """4 high-card categoricals (50-200 levels, target-correlated per-level group means) + 3 linear.

    DEFEATS: RFECV-logit -- raw integer-coded high-card categoricals are NOT linearly usable (the per-level
             effect is non-monotone in the code), so the logit estimator's coefficients are ~0 and backward
             elimination prunes the categorical signals.
    RESCUED BY: tree-member (LightGBM splits on the integer code recover the per-level group means natively)
             + MRMR-binned (MI over the binned/categorical code captures the level->target association).
    """
    rng = np.random.default_rng(seed)
    cardinalities = [50, 100, 150, 200]
    n_cat = len(cardinalities); n_lin = 3
    logit = np.zeros(n)
    cols, base_names = {}, []
    for ci, card in enumerate(cardinalities):
        codes = rng.integers(0, card, size=n)
        level_effect = rng.standard_normal(card)  # non-monotone per-level target effect
        logit += 0.9 * level_effect[codes]
        cols[name := f"inf_cat_{ci}"] = codes.astype(np.int64)
        base_names.append(name)
    z = rng.standard_normal((n, n_lin))
    for k in range(n_lin):
        logit += (1.0 - 0.1 * k) * z[:, k]
        cols[name := f"inf_lin_{k}"] = z[:, k]; base_names.append(name)
    pr = 1.0 / (1.0 + np.exp(-logit / 1.5))
    y = (rng.random(n) < pr).astype(int)

    noise_names = []
    n_used = n_cat + n_lin
    n_noise = p - n_used
    for i in range(n_noise):
        if i % 2 == 0:  # mix of high-card categorical noise and gaussian noise
            cols[name := f"noise_{i}"] = rng.integers(0, 120, size=n).astype(np.int64)
        else:
            cols[name := f"noise_{i}"] = rng.standard_normal(n)
        noise_names.append(name)
    return _finalize(cols, rng, base_names, [], noise_names, [], [], y)


# --------------------------------------------------------------------------- D5
def synth_pgg_n(seed: int = 0, n: int = 300, p: int = 2000):
    """n=300, p=2000 (p >> n): 15 sparse linear + 5 redundant clusters + 1980 noise. KNOWN truth.

    DEFEATS: RFECV -- with p>>n, recursive elimination is both prohibitively costly (thousands of features to
             rank with tiny n) and unstable (importance ranks are near-random at n=300, p=2000), so it prunes
             true signals along with noise (high variance, low recall).
    RESCUED BY: MRMR-filter (cheap univariate MI ranking scales to p=2000 and ranks the 15 strong sparse
             signals at the top) + Boruta-premerge (shadow test is reliable for STRONG sparse signals even at
             small n, confirming the MRMR shortlist).
    """
    rng = np.random.default_rng(seed)
    n_lin = 15
    z = rng.standard_normal((n, n_lin))
    logit = np.zeros(n)
    for k in range(n_lin):
        logit += 2.0 * z[:, k]  # strong sparse signals (recoverable by MI even at small n)
    pr = 1.0 / (1.0 + np.exp(-logit / 2.5))
    y = (rng.random(n) < pr).astype(int)

    cols, base_names = {}, []
    for i in range(n_lin):
        cols[name := f"inf_{i}"] = z[:, i]; base_names.append(name)
    # 5 redundant clusters: noisy copies of the first 5 signals
    redundant_names = []
    for parent in range(5):
        for j in range(3):
            cols[name := f"red_{parent}_{j}"] = z[:, parent] + 0.30 * rng.standard_normal(n)
            redundant_names.append(name)
    noise_names = []
    n_noise = p - n_lin - len(redundant_names)
    for i in range(n_noise):
        cols[name := f"noise_{i}"] = rng.standard_normal(n); noise_names.append(name)
    return _finalize(cols, rng, base_names, redundant_names, noise_names, [], [], y)


# --------------------------------------------------------------------------- D6
def noise_dominated_weaksparse(seed: int = 0, n: int = 10000, p: int = 300):
    """10 weak linear signals (coef ~ 0.25, individually sub-shadow; ~1.0 collectively) + 290 noise.

    DEFEATS: MRMR + Boruta recall -- each weak signal's marginal MI / shadow-importance is below the rejection
             threshold individually (coef ~0.25 is sub-shadow), so marginal/shadow methods reject most of them
             (low recall) despite their large JOINT contribution.
    RESCUED BY: RFECV -- multivariate backward elimination evaluates features in the joint model, where the 10
             weak signals collectively dominate, so elimination keeps them (recovers the weak-sparse block).
    """
    rng = np.random.default_rng(seed)
    n_lin = 10
    z = rng.standard_normal((n, n_lin))
    coef = 0.25
    logit = coef * z.sum(axis=1)  # collective ~ sqrt(10)*0.25 ~ 0.79 sd; each alone weak
    pr = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < pr).astype(int)

    cols, base_names = {}, []
    for i in range(n_lin):
        cols[name := f"inf_{i}"] = z[:, i]; base_names.append(name)
    noise_names = []
    for i in range(p - n_lin):
        cols[name := f"noise_{i}"] = rng.standard_normal(n); noise_names.append(name)
    return _finalize(cols, rng, base_names, [], noise_names, [], [], y)


# Registry mirrors scenarios.SCENARIOS so importance_shootout-style loops work unchanged.
HARD_SCENARIOS = {
    "D1_pure_xor_zeromain": pure_xor_zeromain,
    "D2_heavytail_linear": heavytail_linear,
    "D3_rare_class_imbalance": rare_class_imbalance,
    "D4_categorical_highcard": categorical_highcard,
    "D5_synth_pgg_n": synth_pgg_n,
    "D6_noise_dominated_weaksparse": noise_dominated_weaksparse,
}


def make_hard(scenario: str, seed: int = 0):
    return HARD_SCENARIOS[scenario](seed)


if __name__ == "__main__":
    for name, fn in HARD_SCENARIOS.items():
        X, y, t = fn(0)
        print(f"{name:32s} shape={X.shape} pos={float(y.mean()):.3f} base={len(t['base'])} "
              f"inter={len(t['interaction_operands'])} relevant={len(t['relevant'])} noise={len(t['noise'])}")

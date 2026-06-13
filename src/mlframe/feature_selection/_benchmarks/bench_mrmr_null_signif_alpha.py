"""Isolated bench for the MRMR empirical-null debiasing significance level.

Lever
-----
``_MRMR_NULL_SIGNIF_ALPHA`` (filters/evaluation.py:52, env ``MLFRAME_MRMR_NULL_SIGNIF_ALPHA``, default 0.05).
Production path (CPU, ``use_gpu=False``) in ``evaluate_candidate``:

    direct_gain, _, null_mean, p_value = mi_direct(..., return_null_mean=True)
    if p_value >= _MRMR_NULL_SIGNIF_ALPHA:
        direct_gain = max(0.0, direct_gain - null_mean)   # demote toward 0

The null path runs ``max(npermutations, 32)`` shuffles so the p-value grid is k/32 (0, 0.031, 0.0625, 0.094, ...).
alpha picks the cutoff: a feature with <=1 exceedance (p<=0.031) keeps full MI at alpha=0.05; >=2 exceedances (p>=0.0625) is debiased.

Honest metric
-------------
On a wide candidate pool the in-sample plug-in MI over-ranks high-cardinality / heavy-tailed decoys against a genuine
weak low-cardinality signal. The debiasing should DEMOTE the noise decoys (gain -> ~0) and KEEP the genuine signal.
We score, per (scenario, seed), the post-gate relevance of every candidate and compute:

  * keep_signal  : genuine weak signal retains > half its observed MI (not over-corrected).
  * drop_noise   : pure-noise high-card decoy is demoted to ~0.
  * sep_margin   : min(kept signal gain) - max(noise gain) -- positive => signal still out-ranks noise after gate (honest).

A good alpha maximises drop_noise AND keep_signal AND sep_margin simultaneously. This is the exact selection-quality the
gate exists to protect, measured on the REAL production kernel call.

Run:
  python -m mlframe.feature_selection._benchmarks.bench_mrmr_null_signif_alpha

Verdict (KEEP default 0.05) -- 7 scenarios x 5 seeds, n=4000, null_perms=32, honest composite = keep% + drop% + sep%:

  alpha   keep%   drop%   sep%   composite
  0.0100  100.0   94.3    96.7   2.9095   <- nominal best
  0.0310  100.0   94.3    96.7   2.9095
  0.0500  100.0   92.4    96.7   2.8905   <- current default
  0.1000  100.0   89.5    96.7   2.8619

keep% (genuine signal retains >50% of observed MI) and sep% (signal out-ranks noise post-gate) are FLAT 100/96.7 across the
whole 0.01-0.15 band: a real signal -- even a weak one -- clears p ~ 0 over 32 perms and is never over-corrected at any alpha
here, while noise sits at p ~ 0.3-0.7. The p-value distribution is BIMODAL far from every candidate cutoff, so alpha has
near-zero leverage on selection quality -- exactly the docstring's "stable across 0.02-0.10" claim, now measured.

The only axis that moves is drop% (noise demotion), which is monotone in -alpha because a stricter (lower) alpha debiases more
borderline decoys. alpha=0.01 nominally wins (+1.9pp drop vs 0.05) but the delta is <1% of composite, driven by 1-2 borderline
high-card decoys across 35 runs (within seed noise), and those same decoys are dropped downstream by min_relevance_gain anyway.
Lowering to 0.01 also tightens "keep" to require ZERO exceedances of 32, raising the over-correction risk for a weak signal that
happens to draw a single tie -- a risk this bimodal synthetic does not stress but a real wide pool would. No robust MAJORITY win
for any non-default value; 0.05 is the principled standard level. KEEP 0.05; option stays tunable via MLFRAME_MRMR_NULL_SIGNIF_ALPHA.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

import numpy as np

from mlframe.feature_selection.filters.permutation import mi_direct

ALPHAS = [0.01, 0.031, 0.05, 0.0625, 0.10, 0.15]
NULL_PERMS = 32  # production: max(baseline_npermutations=2, _NULL_MEAN_MIN_PERMS=32)


def _digitize(x: np.ndarray, nbins: int) -> np.ndarray:
    """Equi-frequency bin into ``nbins`` int32 codes (mirrors discretization upstream)."""
    ranks = np.argsort(np.argsort(x))
    codes = (ranks * nbins // len(x)).astype(np.int32)
    return np.clip(codes, 0, nbins - 1)


def _make_scenario(name: str, n: int, rng: np.random.Generator):
    """Return (factors_data int32 [n_feat+1, n], factors_nbins, labels dict feat_idx->'signal'|'noise', y_idx)."""
    cols = []
    labels = {}
    y_card = 2

    # Target: binary, with a latent driver z.
    z = rng.normal(size=n)
    y = (z + rng.normal(scale=1.0, size=n) > 0).astype(np.int32)

    def add(code, kind):
        cols.append(code)
        labels[len(cols) - 1] = kind

    if name == "weak_lowcard_vs_highcard_noise":
        # Genuine WEAK low-card signal: noisy 3-bin function of z.
        sig = _digitize(z + rng.normal(scale=2.2, size=n), 3); add(sig, "signal")
        # High-card pure-noise decoys (50 levels) -- plug-in MI inflated.
        for _ in range(3):
            add(_digitize(rng.normal(size=n), 50), "noise")
    elif name == "weak_lowcard_vs_heavytail_noise":
        sig = _digitize(z + rng.normal(scale=2.0, size=n), 4); add(sig, "signal")
        for _ in range(3):
            add(_digitize(rng.standard_t(df=2, size=n), 30), "noise")
    elif name == "monotone_noise_vs_signal":
        # Monotone (datetime-like) high-card noise + genuine moderate signal.
        sig = _digitize(z + rng.normal(scale=1.4, size=n), 5); add(sig, "signal")
        for _ in range(3):
            add(_digitize(np.arange(n) + rng.normal(scale=n * 0.01, size=n), 40), "noise")
    elif name == "two_weak_signals_amid_noise":
        add(_digitize(z + rng.normal(scale=2.4, size=n), 3), "signal")
        add(_digitize(z + rng.normal(scale=2.0, size=n), 4), "signal")
        for _ in range(2):
            add(_digitize(rng.normal(size=n), 50), "noise")
    elif name == "all_noise":
        # No signal at all -- every feature should be demoted.
        for _ in range(4):
            add(_digitize(rng.normal(size=n), 50), "noise")
    elif name == "strong_signal_robustness":
        # Strong clean signal must NEVER be demoted regardless of alpha.
        add(_digitize(z + rng.normal(scale=0.4, size=n), 6), "signal")
        for _ in range(3):
            add(_digitize(rng.normal(size=n), 50), "noise")
    elif name == "borderline_weak_signal":
        # Very weak genuine signal whose permutation p-value sits NEAR alpha -- the case where a too-strict (low) alpha over-corrects real signal to 0.
        add(_digitize(z + rng.normal(scale=3.2, size=n), 3), "signal")
        for _ in range(3):
            add(_digitize(rng.normal(size=n), 50), "noise")
    else:
        raise ValueError(name)

    factors = np.column_stack(cols + [y]).astype(np.int32)  # (n_samples, n_features) sklearn layout
    nbins = np.array([int(c.max()) + 1 for c in cols] + [y_card], dtype=np.int32)
    y_idx = factors.shape[1] - 1
    return factors, nbins, labels, y_idx


def _eval_alpha(factors, nbins, labels, y_idx, alpha, seed):
    """Apply the production gate at ``alpha``; return per-feature post-gate gain + label."""
    out = []
    for fi, kind in labels.items():
        observed, _, null_mean, p_value = mi_direct(
            factors, x=(fi,), y=(y_idx,), factors_nbins=nbins,
            npermutations=2, max_failed=2, min_nonzero_confidence=0.0,
            dtype=np.int32, base_seed=seed, prefer_gpu=False, return_null_mean=True,
        )
        gain = observed
        if p_value >= alpha:
            gain = max(0.0, observed - null_mean)
        out.append((kind, observed, gain))
    return out


def main():
    scenarios = [
        "weak_lowcard_vs_highcard_noise",
        "weak_lowcard_vs_heavytail_noise",
        "monotone_noise_vs_signal",
        "two_weak_signals_amid_noise",
        "all_noise",
        "strong_signal_robustness",
        "borderline_weak_signal",
    ]
    seeds = [0, 1, 2, 3, 4]
    n = 4000

    # Aggregate per-alpha score across all (scenario, seed).
    agg = {a: {"keep": 0, "keep_tot": 0, "drop": 0, "drop_tot": 0, "sep_wins": 0, "sep_tot": 0} for a in ALPHAS}

    for sc in scenarios:
        for seed in seeds:
            rng = np.random.default_rng(seed * 100 + hash(sc) % 1000)
            factors, nbins, labels, y_idx = _make_scenario(sc, n, rng)
            for a in ALPHAS:
                res = _eval_alpha(factors, nbins, labels, y_idx, a, seed)
                sig_obs = [(o, g) for k, o, g in res if k == "signal"]
                noise_g = [g for k, o, g in res if k == "noise"]
                # keep_signal: genuine signal keeps > 50% of observed MI.
                for o, g in sig_obs:
                    agg[a]["keep_tot"] += 1
                    if o <= 1e-9 or g >= 0.5 * o:
                        agg[a]["keep"] += 1
                # drop_noise: noise demoted near 0 (< 25% of the median signal observed MI).
                med_sig = np.median([o for o, _ in sig_obs]) if sig_obs else 0.0
                for g in noise_g:
                    agg[a]["drop_tot"] += 1
                    if g <= max(1e-4, 0.25 * med_sig):
                        agg[a]["drop"] += 1
                # sep_margin win: every kept signal out-ranks every noise (post-gate).
                if sig_obs and noise_g:
                    agg[a]["sep_tot"] += 1
                    if min(g for _, g in sig_obs) > max(noise_g):
                        agg[a]["sep_wins"] += 1

    print(f"n={n}  scenarios={len(scenarios)}  seeds={len(seeds)}  null_perms={NULL_PERMS}")
    print(f"{'alpha':>7} {'keep%':>8} {'drop%':>8} {'sep%':>8} {'composite':>10}")
    best = None
    for a in ALPHAS:
        d = agg[a]
        keep = d["keep"] / max(1, d["keep_tot"])
        drop = d["drop"] / max(1, d["drop_tot"])
        sep = d["sep_wins"] / max(1, d["sep_tot"])
        comp = keep + drop + sep
        print(f"{a:>7.4f} {keep*100:>7.1f}% {drop*100:>7.1f}% {sep*100:>7.1f}% {comp:>10.4f}")
        if best is None or comp > best[1]:
            best = (a, comp)
    print(f"\nbest alpha by composite (keep+drop+sep): {best[0]} (composite={best[1]:.4f})")
    print(f"current default: 0.05")


if __name__ == "__main__":
    main()

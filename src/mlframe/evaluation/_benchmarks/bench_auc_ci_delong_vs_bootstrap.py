"""qual-9 bench: DeLong closed-form AUC standard-error / CI vs bootstrap-of-AUC.

Lever: a single-sample ROC-AUC uncertainty estimate. mlframe shipped the PAIRED DeLong test
(``delong_test``) but no single-sample AUC SE / CI -- the de-facto path was bootstrapping
``roc_auc_score`` (qual-5 BCa). This bench adds ``auc_variance`` / ``auc_ci`` and asks which estimate
is closer to the KNOWN ground truth.

Ground truth: the TRUE standard error of the AUC estimator is the standard deviation of the AUC computed
on repeated INDEPENDENT test draws from the same generative model. We estimate it by Monte-Carlo over
``N_TRUTH`` fresh draws per scenario (the gold SD). For each of ``N_SEEDS`` separate single test sets we
then compute (a) the DeLong closed-form SE and (b) the bootstrap SD of AUC, and score each by
``|SE_estimate - SD_truth|``. We ALSO score empirical CI coverage of a nominal-95% interval over many
trials per cell.

Scenarios sweep the regimes where bootstrap-of-AUC is known to struggle: small n and AUC near the 1.0
ceiling (left-skewed, capped sampling distribution). Run:

    python -m mlframe.evaluation._benchmarks.bench_auc_ci_delong_vs_bootstrap
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.stats

try:
    import numba  # noqa: F401
except Exception:
    pass
sys.modules.setdefault("cupy", None)

from sklearn.metrics import roc_auc_score

from mlframe.evaluation.bootstrap import auc_ci, auc_variance, bootstrap_metric


def _draw(rng, n, auc_target):
    """Binary labels + scores with a controllable population AUC via a normal shift model.

    Positives ~ N(mu, 1), negatives ~ N(0, 1); population AUC = Phi(mu / sqrt(2)). Solve mu for target.
    """
    mu = scipy.stats.norm.ppf(auc_target) * np.sqrt(2.0)
    y = (rng.random(n) < 0.5).astype(int)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos < 2 or n_neg < 2:
        # force at least 2/2
        y[:2] = 1
        y[-2:] = 0
        n_pos = int(y.sum())
    s = rng.standard_normal(n)
    s[y == 1] += mu
    return y, s


def _bootstrap_se(y, s, rng_seed, n_boot=2000):
    res = bootstrap_metric(
        y, s, lambda a, b: float(roc_auc_score(a, b)),
        n_bootstrap=n_boot, alpha=0.05, random_state=rng_seed, stratify=y, method="bca",
    )
    return res


SCENARIOS = [
    ("auc0.97_n150", 0.97, 150),
    ("auc0.97_n400", 0.97, 400),
    ("auc0.90_n100", 0.90, 100),
    ("auc0.85_n200", 0.85, 200),
    ("auc0.99_n300", 0.99, 300),
]
N_SEEDS = 8
N_TRUTH = 4000
N_COVER = 300
COVER_BOOT = 300


def main():
    print("=== qual-9: DeLong AUC SE/CI vs bootstrap-of-AUC ===")
    print(f"{N_TRUTH} MC draws for truth-SD; {N_SEEDS} seeds for SE-bias; {N_COVER} trials/cell for coverage\n")
    se_wins_delong = 0
    se_cells = 0
    scen_majorities = []
    cover_rows = []
    for name, auc_t, n in SCENARIOS:
        truth_rng = np.random.default_rng(20260615)
        aucs = np.empty(N_TRUTH)
        for i in range(N_TRUTH):
            y, s = _draw(truth_rng, n, auc_t)
            aucs[i] = roc_auc_score(y, s)
        sd_truth = float(aucs.std(ddof=1))
        mean_auc = float(aucs.mean())

        # SE-bias per seed
        delong_abserr = []
        boot_abserr = []
        delong_se = []
        boot_se = []
        seed_wins = 0
        for sd in range(N_SEEDS):
            rng = np.random.default_rng(1000 + sd)
            y, s = _draw(rng, n, auc_t)
            d_se = auc_variance(y, s)["se"]
            b = _bootstrap_se(y, s, rng_seed=1000 + sd)
            b_se = float(b["samples"].std(ddof=1)) if "samples" in b else float("nan")
            de = abs(d_se - sd_truth)
            be = abs(b_se - sd_truth)
            delong_abserr.append(de)
            boot_abserr.append(be)
            delong_se.append(d_se)
            boot_se.append(b_se)
            if de < be:
                seed_wins += 1
            se_cells += 1
            if de < be:
                se_wins_delong += 1
        scen_majorities.append((name, seed_wins, N_SEEDS))
        print(f"[{name}] truth SD(AUC)={sd_truth:.4f} mean_auc={mean_auc:.4f}")
        print(f"   DeLong  mean|SE-SD|={np.mean(delong_abserr):.5f}  (mean SE={np.mean(delong_se):.4f})")
        print(f"   Bootstrp mean|SE-SD|={np.mean(boot_abserr):.5f}  (mean SE={np.mean(boot_se):.4f})")
        print(f"   DeLong SE closer in {seed_wins}/{N_SEEDS} seeds")

        # Coverage of nominal-95% CI
        cov_rng = np.random.default_rng(7777)
        d_hit = 0
        b_hit = 0
        for _ in range(N_COVER):
            y, s = _draw(cov_rng, n, auc_t)
            dci = auc_ci(y, s, alpha=0.05, method="delong")
            if np.isfinite(dci["lo"]) and dci["lo"] <= mean_auc <= dci["hi"]:
                d_hit += 1
            bci = auc_ci(y, s, alpha=0.05, method="bootstrap", n_bootstrap=COVER_BOOT, random_state=int(cov_rng.integers(1 << 30)))
            if np.isfinite(bci["lo"]) and bci["lo"] <= mean_auc <= bci["hi"]:
                b_hit += 1
        d_cov = d_hit / N_COVER
        b_cov = b_hit / N_COVER
        cover_rows.append((name, d_cov, b_cov))
        print(f"   coverage(target 0.95): DeLong={d_cov:.3f} (gap {abs(d_cov-0.95):.3f})  Bootstrap={b_cov:.3f} (gap {abs(b_cov-0.95):.3f})\n")

    print("=== SUMMARY ===")
    print(f"DeLong SE closer-to-truth in {se_wins_delong}/{se_cells} (scenario x seed) cells")
    for name, w, t in scen_majorities:
        print(f"   {name}: {w}/{t} {'MAJORITY' if w*2 > t else 'no-majority'}")
    print("Coverage |gap-to-0.95| (lower better):")
    d_better_cover = 0
    for name, dc, bc in cover_rows:
        better = "DeLong" if abs(dc - 0.95) < abs(bc - 0.95) else "Bootstrap"
        if better == "DeLong":
            d_better_cover += 1
        print(f"   {name}: DeLong gap {abs(dc-0.95):.3f}  Bootstrap gap {abs(bc-0.95):.3f}  -> {better}")
    print(f"DeLong better coverage in {d_better_cover}/{len(cover_rows)} scenarios")


if __name__ == "__main__":
    main()

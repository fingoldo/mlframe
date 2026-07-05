"""Isolated bench: absolute-residual vs normalized (locally-adaptive) split-conformal.

Question (kernel-lever): the conformal default in ``conformal.py`` uses the plain
absolute-residual nonconformity score (constant-width band). For heteroscedastic
targets a normalized score s_i = |y_i - yhat_i| / sigma_hat(x_i) yields variable-width
bands. Should normalized be the default?

Both schemes attain marginal coverage >= 1-alpha by construction (finite-sample split
conformal via ``conformal_quantile``). So marginal coverage does NOT discriminate them.
The HONEST discriminators are:
  * conditional coverage: max abs deviation of per-quantile-bin coverage from nominal
    (a good interval covers EVERYWHERE, not just on average -- absolute under-covers
    the high-variance region and over-covers the low-variance region).
  * mean interval width at matched marginal coverage (efficiency / sharpness).

Decision rule: normalized wins a scenario+seed if it has BOTH lower worst-bin coverage
gap (better conditional coverage) AND not-worse mean width by >2%. Flip the default only
if normalized wins the MAJORITY of scenario x seed cells.

Run:
  python -m mlframe.training.composite._benchmarks.bench_conformal_normalized_vs_absolute
"""

import json
import math
import os

import numpy as np

from mlframe.training.composite.conformal import conformal_quantile


def _sigma_hat_from_residuals(yhat_cal, abs_res_cal, yhat_eval, n_bins=20):
    """Cheap conditional-scale estimate: bin by yhat, mean |residual| per bin.

    Mirrors the standard normalized-conformal recipe (Lei et al.): a residual-magnitude
    model fit on calibration data, here a non-parametric binned mean so the bench has no
    extra dependency on the base learner. Floored to avoid div-by-zero.
    """
    edges = np.quantile(yhat_cal, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx_cal = np.clip(np.searchsorted(edges, yhat_cal, side="right") - 1, 0, n_bins - 1)
    sig = np.empty(n_bins)
    for b in range(n_bins):
        m = idx_cal == b
        sig[b] = abs_res_cal[m].mean() if m.any() else abs_res_cal.mean()
    floor = max(1e-9, 0.05 * abs_res_cal.mean())
    sig = np.maximum(sig, floor)
    idx_e = np.clip(np.searchsorted(edges, yhat_eval, side="right") - 1, 0, n_bins - 1)
    return sig[idx_cal], sig[idx_e]


def _conditional_gap(covered, scale_var, n_bins=10):
    """Worst-bin |empirical coverage - target|, binning by the heteroscedastic driver."""
    edges = np.quantile(scale_var, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, scale_var, side="right") - 1, 0, n_bins - 1)
    gaps = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 20:
            gaps.append(covered[m].mean())
    return gaps


def _make_scenario(name, n, rng):
    """Return (x_driver, yhat, y) for a heteroscedastic regression-residual setup.

    We work directly in residual space: ``yhat`` is the point prediction and the noise
    scale varies with a driver. This isolates the conformal scoring choice from any base
    learner, which is exactly what the lever is about.
    """
    x = rng.uniform(0, 1, n)
    base = 10.0 * x
    if name == "linear_scale":
        sigma = 0.2 + 3.0 * x
    elif name == "quadratic_scale":
        sigma = 0.2 + 5.0 * x**2
    elif name == "step_scale":
        sigma = np.where(x > 0.5, 4.0, 0.4)
    elif name == "u_shape_scale":
        sigma = 0.3 + 6.0 * (x - 0.5) ** 2
    elif name == "homoscedastic":
        sigma = np.full(n, 1.5)
    elif name == "exp_scale":
        sigma = 0.2 + np.exp(2.5 * x)
    else:
        raise ValueError(name)
    noise = rng.normal(0, 1, n) * sigma
    y = base + noise
    yhat = base  # oracle point prediction; residual = noise (the heteroscedastic part)
    return x, yhat, y


def run(alpha=0.1, n_cal=2000, n_eval=4000, seeds=(0, 1, 2, 3, 4)):
    scenarios = ["linear_scale", "quadratic_scale", "step_scale", "u_shape_scale", "exp_scale", "homoscedastic"]
    target = 1.0 - alpha
    rows = []
    for sc in scenarios:
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + hash(sc) % 997)
            xc, yhc, yc = _make_scenario(sc, n_cal, rng)
            xe, yhe, ye = _make_scenario(sc, n_eval, rng)
            res_c = yc - yhc
            abs_c = np.abs(res_c)

            # --- ABSOLUTE (current default) ---
            q_abs = conformal_quantile(res_c, alpha)
            lo_a, hi_a = yhe - q_abs, yhe + q_abs
            cov_a = (ye >= lo_a) & (ye <= hi_a)
            marg_a = cov_a.mean()
            width_a = float(np.mean(hi_a - lo_a))
            gaps_a = _conditional_gap(cov_a, xe)
            condgap_a = max(abs(g - target) for g in gaps_a)

            # --- NORMALIZED (challenger) ---
            sig_c, sig_e = _sigma_hat_from_residuals(yhc, abs_c, yhe)
            q_norm = conformal_quantile(res_c / sig_c, alpha)
            lo_n, hi_n = yhe - q_norm * sig_e, yhe + q_norm * sig_e
            cov_n = (ye >= lo_n) & (ye <= hi_n)
            marg_n = cov_n.mean()
            width_n = float(np.mean(hi_n - lo_n))
            gaps_n = _conditional_gap(cov_n, xe)
            condgap_n = max(abs(g - target) for g in gaps_n)

            norm_wins = (condgap_n < condgap_a) and (width_n <= width_a * 1.02)
            rows.append(dict(scenario=sc, seed=seed, target=target,
                             marg_abs=round(marg_a, 4), marg_norm=round(marg_n, 4),
                             width_abs=round(width_a, 3), width_norm=round(width_n, 3),
                             condgap_abs=round(condgap_a, 4), condgap_norm=round(condgap_n, 4),
                             norm_wins=bool(norm_wins)))
    return rows


def main():
    rows = run()
    het = [r for r in rows if r["scenario"] != "homoscedastic"]
    wins = sum(r["norm_wins"] for r in rows)
    het_wins = sum(r["norm_wins"] for r in het)
    print(f"{'scenario':16} {'seed':>4} {'marg_a':>7} {'marg_n':>7} " f"{'w_abs':>7} {'w_norm':>7} {'cg_abs':>7} {'cg_nrm':>7} win")
    for r in rows:
        print(f"{r['scenario']:16} {r['seed']:>4} {r['marg_abs']:>7} {r['marg_norm']:>7} "
              f"{r['width_abs']:>7} {r['width_norm']:>7} {r['condgap_abs']:>7} "
              f"{r['condgap_norm']:>7} {'Y' if r['norm_wins'] else '.'}")
    print(f"\nnormalized wins {wins}/{len(rows)} cells overall; "
          f"{het_wins}/{len(het)} heteroscedastic cells")
    avg_cg_a = np.mean([r["condgap_abs"] for r in het])
    avg_cg_n = np.mean([r["condgap_norm"] for r in het])
    print(f"avg conditional-coverage gap (het): abs={avg_cg_a:.4f} norm={avg_cg_n:.4f}")
    out = os.path.join(os.path.dirname(__file__), "_results", "conformal_normalized_vs_absolute.json")
    with open(out, "w") as f:
        json.dump(dict(rows=rows, het_wins=het_wins, het_total=len(het),
                       avg_condgap_abs=float(avg_cg_a),
                       avg_condgap_norm=float(avg_cg_n)), f, indent=2, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

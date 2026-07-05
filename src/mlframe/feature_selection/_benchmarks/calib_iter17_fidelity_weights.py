"""iter17 calibration study: pick the default ``fidelity_weights`` for ``proxy_trust_guard`` by
correlating each component metric (Spearman, recall@k) with downstream selector RECOVERY across a
small handful of synthetic regimes. The metric with the higher across-regime correlation to recovery
should dominate the weighted composite; the chosen weights are then registered as the
``proxy_trust_guard`` / ``ShapProxiedFS.trust_guard_fidelity_weights`` default.

Watchdog-safe: 3-4 regimes, width<=3000, prints per regime, hard <=90s per cell.

Run with the worktree on PYTHONPATH:
  $env:PYTHONPATH='<worktree>\\src'
  D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.calib_iter17_fidelity_weights
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def _hb(msg: str) -> None:
    print(f"[calib-iter17 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


REGIMES = [
    # Calibrated to PRODUCE recovery + recall@k spread across regimes (the first pass at width 400 had
    # recall@k pinned at 5/6 and recovery near-ceiling, so neither component varied). Knobs that move
    # the needle here: smaller n_samples shrinks the SHAP signal, larger n_noise grows the prefilter
    # haystack, lower SNR puts the proxy under genuine stress, and distinct random_state per regime
    # so anchor sets don't collide on the same default rng.
    # Calibrated for MAXIMUM regime spread on recovery + recall. The first pass at width 400 had
    # everything pinned near-ceiling; the second had ONE failure regime dominating both correlations.
    # We add graduated difficulty: easy / medium-redundancy / medium-noise / hard-interaction / hard-
    # interaction-with-XOR so the regression of recovery on each metric has real degrees of freedom.
    # 1. Additive high-SNR: marginal signal dominates, the easy regime.
    dict(name="additive_highSNR",
         kwargs=dict(n_samples=2000, n_informative=8, n_redundant=0, n_noise=400,
                     interaction_order=0, interaction_strength=0.0, snr=5.0,
                     task="regression", seed=0),
         random_state=0),
    # 2. Redundancy-heavy at moderate SNR: SHAP marginal still solid but cluster pressure on subset.
    dict(name="redundancy_heavy",
         kwargs=dict(n_samples=1200, n_informative=8, n_redundant=24, redundancy_rho=0.9,
                     n_noise=400, snr=2.5, task="regression", seed=1),
         random_state=1),
    # 3. Interaction-heavy (order=2, strength=0.7): hard for marginal SHAP proxy, expect mid recovery.
    dict(name="interaction_heavy",
         kwargs=dict(n_samples=1200, n_informative=8, n_redundant=0, n_noise=400,
                     interaction_order=2, interaction_strength=0.7, snr=3.0,
                     task="regression", seed=2),
         random_state=2),
    # 4. XOR interaction: pure non-monotone, SHAP marginal proxy is structurally weakest. Often
    # produces the lowest recovery.
    dict(name="xor_interaction",
         kwargs=dict(n_samples=1500, n_informative=6, n_redundant=0, n_noise=400,
                     interaction_order="xor", interaction_strength=0.9, snr=3.0,
                     task="regression", seed=4),
         random_state=4),
    # 5. Noise-heavy at low SNR: large haystack stresses prefilter + trust guard.
    dict(name="noise_heavy", kwargs=dict(n_samples=1500, n_informative=8, n_redundant=0, n_noise=1200, snr=2.0, task="regression", seed=3), random_state=3),
]


def run_one(regime: dict, time_budget_s: float = 90.0) -> dict:
    name = regime["name"]
    kwargs = regime["kwargs"]
    rs = int(regime.get("random_state", 0))
    _hb(f"regime {name}: building dataset ({kwargs.get('n_samples')}s x "
        f"{kwargs.get('n_informative') + kwargs.get('n_redundant', 0) + kwargs.get('n_noise')} cols)")
    X, y, roles = make_regime_dataset(**kwargs)
    informatives = {c for c, r in roles.items() if r == "informative"}

    # ShapProxiedFS with the cheap defaults the calibration cares about: prefilter funnels to 400,
    # holdout 0.3, 30 anchors. Keep trust_guard ON to populate the trust report, default fidelity
    # weights (0.5/0.5) since we read the raw spearman / recall_at_k components, NOT the composite.
    fs = ShapProxiedFS(
        classification=False, metric="rmse",
        min_features=2, max_features=8, top_n=5,
        n_models=1, n_splits=3, holdout_size=0.3,
        trust_guard=True, n_anchors=30, fidelity_floor=0.6,
        revalidate=True, n_revalidation_models=1,
        run_importance_ablation=False,
        use_bias_corrector=True,
        prefilter_top=200, prefilter_method="auto",
        cluster_features="auto",
        random_state=rs, verbose=False, tqdm=False, n_jobs=-1,
    )

    t0 = time.time()
    fs.fit(X, y)
    elapsed = time.time() - t0

    trust = fs.shap_proxy_report_.get("trust", {})
    sp = float(trust.get("spearman", float("nan")))
    rc = float(trust.get("recall_at_k", float("nan")))
    selected = set(fs.selected_features_)
    recovery = len(informatives & selected)
    selected_n = len(selected)
    _hb(f"regime {name}: spearman={sp:.4f} recall@k={rc:.4f} " f"recovery={recovery}/{len(informatives)} (|S|={selected_n}) elapsed={elapsed:.1f}s")
    if elapsed > time_budget_s:
        _hb(f"  WARNING: regime {name} exceeded soft budget {time_budget_s:.0f}s " f"(actual {elapsed:.1f}s); consider narrower kwargs")
    return dict(name=name, spearman=sp, recall_at_k=rc, recovery=recovery, n_informative=len(informatives), selected_n=selected_n, elapsed=elapsed)


def main():
    rows = []
    overall_t0 = time.time()
    for reg in REGIMES:
        rows.append(run_one(reg))
        _hb(f"  cumulative wall: {time.time() - overall_t0:.1f}s")

    print("\nper-regime table:", flush=True)
    print(f"{'regime':<20} {'spearman':>10} {'recall@k':>10} " f"{'recovery':>10} {'sel|S|':>8} {'sec':>7}", flush=True)
    for r in rows:
        print(f"{r['name']:<20} {r['spearman']:>10.4f} {r['recall_at_k']:>10.4f} "
              f"{r['recovery']:>5}/{r['n_informative']:<4} {r['selected_n']:>8} "
              f"{r['elapsed']:>7.1f}", flush=True)

    sp = np.array([r["spearman"] for r in rows], dtype=np.float64)
    rc = np.array([r["recall_at_k"] for r in rows], dtype=np.float64)
    # Use recovery RATE (recovered / planted), not absolute count, so regimes with different
    # n_informative are comparable on the same [0,1] scale. The first pass mixed n_informative=6 vs 8
    # which makes absolute-count correlations regime-dependent.
    rec = np.array([r["recovery"] / max(1, r["n_informative"]) for r in rows], dtype=np.float64)

    # Pearson correlations across regimes. Spearman-corr would be too coarse on 4 points.
    def _corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    corr_sp = _corr(sp, rec)
    corr_rc = _corr(rc, rec)
    print(f"\ncorr(spearman, recovery)   = {corr_sp:+.4f}", flush=True)
    print(f"corr(recall@k, recovery)   = {corr_rc:+.4f}", flush=True)

    # Translate correlations -> weights. Use max(0, corr) since a negative correlation means the
    # metric is anti-predictive (NOT zero weight: still informative, just inverted), but for a
    # convex composite we floor at 0. If both are nonpositive, fall back to 0.5/0.5 (the symmetric
    # default we are trying to replace) and report inconclusive.
    sp_pos = max(0.0, corr_sp) if np.isfinite(corr_sp) else 0.0
    rc_pos = max(0.0, corr_rc) if np.isfinite(corr_rc) else 0.0
    total = sp_pos + rc_pos
    inconclusive = False
    if total <= 1e-9:
        proposed = (0.5, 0.5)
        inconclusive = True
        print("\nINCONCLUSIVE: both correlations <=0 or NaN; keeping 0.5/0.5 default.", flush=True)
    else:
        w_sp = sp_pos / total
        w_rc = rc_pos / total
        # Round to one decimal so the default is a clean published number; clamp away from 0.0/1.0
        # (a 0-weight component is a config bug, not a default).
        w_sp_r = round(w_sp * 10) / 10
        w_rc_r = round(w_rc * 10) / 10
        if w_sp_r < 0.1:
            w_sp_r, w_rc_r = 0.1, 0.9
        elif w_rc_r < 0.1:
            w_sp_r, w_rc_r = 0.9, 0.1
        # Renormalise the rounded pair (should already sum to 1.0 modulo rounding)
        s = w_sp_r + w_rc_r
        if s != 1.0:
            w_sp_r = w_sp_r / s
            w_rc_r = w_rc_r / s
        proposed = (w_sp_r, w_rc_r)
        print(f"\nraw weights (corr-proportional): ({w_sp:.3f}, {w_rc:.3f})", flush=True)
        print(f"PROPOSED default fidelity_weights = ({proposed[0]:.2f}, {proposed[1]:.2f})", flush=True)

    return dict(rows=rows, corr_spearman_recovery=corr_sp, corr_recall_recovery=corr_rc, proposed=proposed, inconclusive=inconclusive)


if __name__ == "__main__":
    main()

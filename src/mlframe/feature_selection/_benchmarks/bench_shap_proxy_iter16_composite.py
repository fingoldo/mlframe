r"""iter16 composite-fidelity bench: measure spearman + recall@k + proxy_fidelity_score across
the iter14/iter15 levers (uniform / F-stratified / Zipf alpha sweep) on the width=6000 regime so
we can decide whether to FLIP iter14's F-stratified default and/or iter15's Zipf alpha=0.25 default.

Run with the worktree on PYTHONPATH:
  $env:PYTHONPATH='<worktree>\src'
  python -m mlframe.feature_selection._benchmarks.bench_shap_proxy_iter16_composite
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard


def _heartbeat(msg: str) -> None:
    print(f"[iter16 bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_phi_and_unit_f(X: pd.DataFrame, y: np.ndarray, *, top_cols: int = 400):
    """Cheap stand-in for the production phi/unit_to_members/unit_f_scores. Picks the top ``top_cols``
    by |corr| with y (a fast univariate-F surrogate on the 6k cohort), then constructs a per-column
    SHAP-like attribution phi[:, j] = coef_j * (x_j - mean). Each surviving column is its own unit so
    unit_to_members is None. This keeps the bench reproducible and the trust-guard's anchor space
    matches what production sees post-prefilter (a few hundred surviving columns)."""
    Xn = X.to_numpy(dtype=np.float64)
    yn = y - y.mean()
    # Pearson |corr| as a cheap proxy for the production F-score ranking on this regime synthetic.
    num = (Xn - Xn.mean(0)).T @ yn
    den = np.linalg.norm(Xn - Xn.mean(0), axis=0) * np.linalg.norm(yn) + 1e-12
    fscore_all = np.abs(num / den)
    keep = np.argsort(-fscore_all)[:top_cols]
    keep.sort()
    Xk = Xn[:, keep]
    lr = LinearRegression().fit(Xk, y)
    phi = (Xk - Xk.mean(0)) * lr.coef_[None, :]
    base = np.full(len(y), float(y.mean()))
    return phi, base, keep, fscore_all[keep]


def _split(X: pd.DataFrame, y: np.ndarray, frac: float = 0.7):
    n = len(y)
    cut = int(n * frac)
    return (X.iloc[:cut].reset_index(drop=True), y[:cut], X.iloc[cut:].reset_index(drop=True), y[cut:])


def main(n_samples: int = 3000, n_informative: int = 12, n_features: int = 6000, n_anchors: int = 30):
    _heartbeat(f"building regime dataset n={n_samples} f={n_features} k_informative={n_informative}")
    n_noise = n_features - n_informative
    X, y, roles = make_regime_dataset(
        n_samples=n_samples, n_informative=n_informative, n_redundant=0, n_noise=n_noise,
        snr=4.0, task="regression", seed=0,
    )
    informative_names = [c for c, r in roles.items() if r == "informative"]
    _heartbeat(f"dataset built ({len(roles)} cols, {len(informative_names)} informative)")

    _heartbeat("prefiltering to top-400 cohort via |corr|")
    phi, base, keep_idx, unit_f = _build_phi_and_unit_f(X, y, top_cols=400)
    X_use = X.iloc[:, keep_idx].reset_index(drop=True)
    # Recovery: how many of the planted informatives survive the prefilter?
    informative_in_cohort = sum(1 for c in informative_names if c in set(X_use.columns))
    _heartbeat(f"cohort {len(keep_idx)} cols; recovery {informative_in_cohort}/{len(informative_names)}")

    Xs, ys, Xh, yh = _split(X_use, y, frac=0.7)
    phi_s = phi[: len(ys)]
    base_s = base[: len(ys)]

    template = RandomForestRegressor(n_estimators=40, max_depth=8, n_jobs=-1, random_state=0)
    common = dict(classification=False, metric="rmse", n_anchors=n_anchors, min_card=1, max_card=20, n_jobs=-1)

    def _run(label: str, **kwargs):
        _heartbeat(f"running variant: {label}")
        t0 = time.time()
        rng = np.random.default_rng(0)
        rep = proxy_trust_guard(phi_s, base_s, ys, template, Xs, Xh, yh, rng=rng, **common, **kwargs)
        elapsed = time.time() - t0
        sp = rep["spearman"]
        rc = rep["recall_at_k"]
        fid = rep["proxy_fidelity_score"]
        _heartbeat(f"  {label}: spearman={sp:.4f} recall@k={rc:.4f} composite={fid:.4f} ({elapsed:.1f}s)")
        return dict(variant=label, spearman=sp, recall_at_k=rc, composite=fid, trustworthy=rep["trustworthy"], seconds=elapsed)

    results = []
    # Baseline: uniform-uniform (legacy default).
    results.append(_run("uniform / uniform-k"))
    # F-stratified anchors, uniform-k (iter14).
    results.append(_run("F-stratified / uniform-k", unit_f_scores=unit_f))
    # Uniform anchors, Zipf-k at alpha sweep (iter15).
    for alpha in (0.25, 0.5, 1.0):
        results.append(_run(f"uniform / zipf alpha={alpha}", cardinality_dist="zipf", zipf_alpha=alpha))
    # Combined: F-stratified + Zipf alpha=0.25 (the iter15 sweet spot under composite).
    results.append(_run("F-stratified / zipf alpha=0.25", unit_f_scores=unit_f, cardinality_dist="zipf", zipf_alpha=0.25))

    print("\nresults table:", flush=True)
    print(f"{'variant':<36} {'spearman':>10} {'recall@k':>10} {'composite':>10} {'gate':>6}", flush=True)
    for r in results:
        print(f"{r['variant']:<36} {r['spearman']:>10.4f} {r['recall_at_k']:>10.4f} "
              f"{r['composite']:>10.4f} {('OK' if r['trustworthy'] else 'LOW'):>6}",
              flush=True)
    return results


if __name__ == "__main__":
    main()

"""Does the null computed on a row subsample, rescaled by n_sub/n, reproduce the full-row null?  n=100000, 3 data seeds, B=100 draws on both sides (so the 95 % quantile is
not dominated by 40-draw noise).  TARGET: tau(full) vs tau(sub 50000 rows, scaled).  FEATURE: median over columns of tau(full) vs tau(sub), and agreement of the accepted columns."""
import warnings
import numpy as np
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
import tt_b5_cost as C, tt_b5_cost_np as N

n, p = 100000, 24
for seed in range(3):
    rng = np.random.default_rng(900 + seed)
    X = rng.standard_normal((p, n)).astype(np.float32)
    X[3] = np.exp(0.8 * rng.standard_normal(n)); X[5] = rng.chisquare(3, n)
    yl = np.exp(0.8 * X[0] + 0.5 * X[1] + 0.3 * X[2] + 0.3 * rng.standard_normal(n))
    prov = C.ArrayProvider(X, yl, 5000)
    f = C.target_criterion(prov, K=9, B=100, seed=seed)
    s = C.target_criterion_subnull(prov, K=9, B=100, null_rows=2500, seed=seed)  # tau_sub is the unscaled threshold
    print(f"TARGET seed {seed}: observed gain {f['best_gain']:.3f} ({f['best']}); tau full {f['tau']:.4f} vs tau sub (50000 rows) unscaled {s['tau']:.4f}, rescaled by n_B,sub/n_B {s['tau_sub'] * 0.5:.4f}; accepted full {f['accepted']} sub {s['accepted']}", flush=True)
    yy = np.log(X[3]) + 0.7 * np.sqrt(X[5]) + 0.5 * X[7] + 0.3 * rng.standard_normal(n)
    D = np.hstack([np.ones((n, 1)), X.T.astype(np.float64)])
    coef = np.linalg.lstsq(D, yy, rcond=None)[0]
    r = yy - D @ coef
    pr2 = C.ArrayProvider(X, r, 5000)
    a = N.feature_criterion_np(pr2, K=9, B=100, seed=seed, beta=coef[1:])
    b = N.feature_criterion_np_sub(pr2, K=9, B=100, null_rows=2500, seed=seed, beta=coef[1:])
    ratio = b["tau"] / np.where(np.abs(a["tau"]) > 1e-12, a["tau"], np.nan)
    print(f"FEATURE seed {seed}: median tau full {np.median(a['tau']):.2e} sub {np.median(b['tau']):.2e} (median column ratio {np.nanmedian(ratio):.2f}); accepted full {np.flatnonzero(a['accepted']).tolist()} sub {np.flatnonzero(b['accepted']).tolist()}", flush=True)

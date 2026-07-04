"""Noise-floor cap (30k strided) vs full-n: does the REJECT decision hold + how much faster at 1M?

The cap changes the null the reject decision uses (mi_real vs null_p95 x 1.5). Selection-equivalence bar:
for a batch of engineered pairs at 1M, the accept/reject verdict (r is None) must match between the
full-n noise-floor (env cap disabled) and the 30k cap (default). Also time the noise-floor block."""
import os, time
import numpy as np

os.environ["MLFRAME_FE_SMART_POLYNOM_BACKEND"] = ""  # no forced backend

from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

rng = np.random.default_rng(0)
N = 1_000_000

def mk(kind):
    a = rng.standard_normal(N).astype(np.float64)
    b = (rng.standard_normal(N) + 0.3).astype(np.float64)
    if kind == "prod":     yc = a * b
    elif kind == "sq":     yc = a * a - 0.5 * b * b
    elif kind == "cubic":  yc = a**3 - 2*a*b
    elif kind == "noise":  yc = rng.standard_normal(N)          # should be REJECTED by the null
    elif kind == "weak":   yc = 0.03*a*b + rng.standard_normal(N)  # borderline
    else:                  yc = a*b
    yc = np.nan_to_num(yc) + rng.standard_normal(N) * 0.3
    y = np.digitize(yc, np.quantile(yc, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
    return np.nan_to_num(a), np.nan_to_num(b), y

kw = dict(n_trials=100, min_degree=3, max_degree=6, basis="chebyshev",
          mi_estimator="plugin", plugin_n_bins=20, optimizer="cma_batch",
          discrete_target=True, sweep_degrees=True, seed=42, noise_floor_n_perms=50)

# warm
xa, xb, y = mk("prod"); optimise_hermite_pair(xa, xb, y, **kw)

print(f"{'target':8} {'FULL rej':>9} {'CAP rej':>8} {'match':>6} {'FULL ms':>9} {'CAP ms':>8} {'speedup':>8}")
matches = 0; total = 0
for kind in ("prod", "sq", "cubic", "noise", "weak"):
    xa, xb, y = mk(kind)

    os.environ["MLFRAME_FE_NOISE_FLOOR_MAX_ROWS"] = "0"  # disable cap => full n
    t = time.perf_counter(); r_full = optimise_hermite_pair(xa, xb, y, **kw); t_full = (time.perf_counter()-t)*1e3
    rej_full = r_full is None

    os.environ["MLFRAME_FE_NOISE_FLOOR_MAX_ROWS"] = "30000"  # cap
    t = time.perf_counter(); r_cap = optimise_hermite_pair(xa, xb, y, **kw); t_cap = (time.perf_counter()-t)*1e3
    rej_cap = r_cap is None

    m = (rej_full == rej_cap); matches += m; total += 1
    print(f"{kind:8} {str(rej_full):>9} {str(rej_cap):>8} {str(m):>6} {t_full:9.0f} {t_cap:8.0f} {t_full/max(t_cap,1e-9):8.2f}x")

print(f"\nreject-decision match: {matches}/{total}")

# Wide-data scaling of the MRMR-FE synergy screen — benchmark results (2026-06-18)

Hardware: GTX 1050 Ti (cc 6.1, 4 GB VRAM), 4 physical CPU cores. Data: n=8000, 8 quantile bins,
binary target, multi-seed, fixed data per seed. Settles whether the O(p^2) all-pairs joint-MI synergy
sweep (capped at `fe_synergy_screen_max_features=250`) can scale to wide frames two ways.

Scripts (re-runnable; all write throwaway state, none modify the package):
- `h1_bench.py` — exhaustive C(p,2) joint-MI wall-time vs p, CPU njit-prange vs CUDA vs cupy.
- `h1_gpu_large.py` — the p=5000 / p=10000 CUDA points (long-running; captured via live monitor).
- `h2_bench.py` — interaction-propensity ranking recall of planted pure-pair operands.

## H1 — exhaustive C(p,2) joint-MI sweep, wall-time vs p

| p | pairs | CPU njit | CUDA | CUDA pairs/s |
|------|----------|----------|--------|--------------|
| 250 | 31,125 | 6.07 s | 0.74 s | 4.2e4 |
| 500 | 124,750 | (contended) | 2.51 s | 5.0e4 |
| 1000 | 499,500 | 167.3 s | 9.61 s | 5.2e4 |
| 2000 | 1,999,000 | 679.3 s | 37.97 s | 5.3e4 |
| 5000 | 12,497,500 | ~3390 s (extrap) | 241.2 s | 5.2e4 |
| 10000 | 49,995,000 | ~13,560 s (extrap) | **1004 s (~16.7 min)** | 5.0e4 |

Clean O(p^2): CUDA quadruples per p-doubling; measured p=5000/10000 within 6% of the extrapolation.
CUDA is a flat ~13.5x over CPU. cupy was measured and is **non-competitive** (per-pair Python bincount
loop: 45 s at p=250). Memory at p=10000: device ~1.5 GB (codes 320 MB + pair-index 800 MB + out 400 MB),
fits the 4 GB card; host pair-index (800 MB) is the larger pressure. Correctness: a planted XOR pair
ranked **#0 of 49,995,000** with MI = ln2 — the exhaustive sweep finds the needle exactly.

Wall-time crossovers (this GPU): <30 s to p~=1715, <120 s to p~=3430, <600 s to p~=7670.

**H1 verdict:** exhaustive p=10000 IS feasible on this GPU (~17 min, fits VRAM) but too slow to default.
A GPU-exhaustive path lifts the practical cap to ~2000–3500 at a 30–120 s budget; keep it force/opt-in for
moderate p when balanced-interaction correctness outranks wall-time.

## H2 — O(p) interaction-propensity ranking recall (top-m of planted operands)

n=8000, p=2000, K=6 planted pure-pair interactions (12 operands), 5 seeds. L = main-effect leakage.

| criterion | L0/250 | L.1/100 | L.1/250 | L.3/250 |
|-----------|--------|---------|---------|---------|
| marginal_MI (baseline) | 0.10 | 0.45 | 0.68 | 1.00 |
| **2nd_moment** | 0.22 | 0.70 | **0.88** | 1.00 |
| cond_resp_var | 0.10 | 0.45 | 0.68 | 1.00 |
| dcor (sampled) | 0.12 | 0.23 | 0.27 | 1.00 |
| **gbm_splits** | 0.35 | **0.92** | 0.93 | 1.00 |
| random_base | 0.12 | 0.05 | 0.12 | 0.12 |

Timing (O(p*n)): at p=10000 — 2nd_moment 5.2 s, marginal 12 s, dcor 20.7 s, gbm 87.7 s.

**H2 verdict:**
- **L=0.0 (perfectly balanced): irreducible** — every criterion sits at the random baseline. No O(p)
  per-variable score recovers a zero-higher-moment interaction; only the exhaustive sweep can.
- **L>=0.1:** `gbm_splits` is best but 18x costlier; `2nd_moment` is the cheap sweet spot (0.88 @ top-250,
  beats marginal MI 0.68 by a wide margin). `cond_resp_var` ties marginal MI (no lift); `dcor` underperforms.

## Recommendation (implemented + planned)

Two-stage funnel, not GPU-everything:
1. **SHIPPED** (`fe_synergy_prerank`, default ON; `_fe_interaction_prerank.py`): above the cap, pre-rank all
   p by `2nd_moment` and keep the top `fe_synergy_screen_max_features` for the existing O(cap^2) sweep,
   instead of skipping the bootstrap (the legacy "engineer nothing on a wide zero-marginal frame"). At
   top-250 the sweep is ~0.7 s GPU / 6 s CPU.
2. **PLANNED** (force/opt-in GPU-exhaustive): for moderate p (<=~3000 at a 2-min budget) when correctness on
   a balanced interaction outranks wall-time, run the full exhaustive CUDA sweep (the only path that
   recovers the irreducible L=0 case). Dispatch via `kernel_tuning_cache` on (n, p).

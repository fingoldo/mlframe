# Cross-cut audit: silent wrong-number substitution

**Scope:** `src/mlframe` (584k LOC). Read-only. ~95 files examined across the three named shapes
(catastrophic cancellation in moment formulas at any k>=2; additive epsilon that replaces rather than
guards; non-neutral `except` substitute logged at or below debug).

**Method:** pattern sweep, then numerical verification of every candidate against a stable two-pass
centred reference. Findings below are split into **CONFIRMED (measured)** and **LEADS (unmeasured)**.

---

## CONFIRMED (measured)

### XNUM-01 [P0] raw-sum Pearson in the FE-pair correlation gate
**File:** `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py:43`
(twin at `:78`)

```python
va = saa - sa * sa / n          # line 43
vy = syy - sy * sy / n          # line 44
if va <= 1e-24 * n or vy <= 1e-24 * n:
    return 0.0                  # "near-constant" -> no correlation
r = (say - sa * sy / n) / denom # line 51
```

**Summary:** This is the k=2 case of shape 1 that earlier skew/kurtosis-only sweeps missed. Both
`_abs_corr_finite_njit` and `_abs_corr_zerofill_njit` accumulate **uncentred** power sums
(`saa += av*av`, `say += av*yv`) and subtract `sa*sa/n` afterwards. On a column with a large offset
relative to its spread the subtraction cancels away every significant bit. Two distinct wrong numbers
come out, both indistinguishable from a real result:

1. a corrupted but finite `|r|` (can be *larger* than the truth, so it fabricates redundancy), and
2. **exactly `0.0`**, because the destroyed `va` trips the `<= 1e-24 * n` "near-constant" guard. `0.0`
   is not a neutral value here: in the redundancy / noise-wrap dedup gate it means "not redundant,
   keep the candidate", and at the y-correlation call sites it means "no signal, drop the candidate".

The function's own docstring asserts *"FP-equivalent to the numpy `abs(corrcoef(a[m], y[m])[0,1])`
to ~1e-15 - selection-safe"*. That claim is false outside near-centred data, and it is what makes the
substitution invisible.

Call sites are all selection decisions: `_pairs_emit.py:541` (div-corr veto), `_pairs_score.py:1152`
(winner-vs-operand dedup veto), `_ratio_delta_fe.py:261` (ratio/log-ratio redundancy gate, called with
`min_n=2`), `_group4.py:275` (mRMR monotone-twin survivor pick).

**Failure scenario:** any column whose |offset| / spread ratio exceeds ~1e6 -- which engineered
features produce routinely (this gate runs on *engineered* columns: `x**2` of a price, `add(x, c)`,
epoch timestamps, account balances, an absolute sensor reading).

Measured against `abs(np.corrcoef(...))`, n=20000-30000:

| column | offset/spread | true \|r\| | returned | absolute error |
|---|---|---|---|---|
| synthetic | 1e5 | 0.308429820807 | 0.308378379411 | 5.14e-05 |
| synthetic | 1e6 | 0.298996612196 | 0.295808614577 | 3.19e-03 |
| synthetic | 1e7 | 0.300081818793 | **0.767145514915** | **4.67e-01** |
| synthetic | 1e8 | 0.313591326180 | **0.000000000000** | **3.14e-01** |
| epoch-seconds, 1 min of ticks | 2.8e7 | 0.496590376 | **0.000000000** | **4.97e-01** |
| epoch-millis, 1 min of ticks | 2.8e7 | 0.501293495 | 0.244894434 | 2.56e-01 |
| engineered `x**2` of a 5e4 price | 5.0e7 | 0.496705254 | 0.087776343 | 4.09e-01 |
| account balance 1e9 +- 100 | 1.0e7 | 0.498629217 | 0.422125857 | 7.65e-02 |
| sensor 1013.25 hPa +- 1e-3 | 1.0e6 | 0.495629217 | 0.495822294 | 1.93e-04 |

At offset 1e7 the gate sees `|r| = 0.767` where the truth is `0.300` -- on either side of every
plausible redundancy threshold. For one minute of tick data keyed on epoch seconds a genuine
`|r| = 0.497` is reported as exactly `0.0`.

**Evidence:** `v1.py` / `v3.py`

```python
import numpy as np, os, sys
os.environ["NUMBA_DISABLE_JIT"]="1"; sys.path.insert(0,"src")
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import (
    _abs_corr_finite_njit, _abs_corr_zerofill_njit)
rng=np.random.default_rng(0); n=20000
for offset in [0.0, 1e3, 1e5, 1e6, 1e7, 1e8]:
    base=rng.standard_normal(n)
    a=offset+base*1.0
    y=offset+(0.3*base+0.95*rng.standard_normal(n))
    ref=abs(np.corrcoef(a,y)[0,1])
    got=_abs_corr_finite_njit(a,np.ascontiguousarray(y),np.ones(n,dtype=np.bool_),8)
    got2=_abs_corr_zerofill_njit(a,np.ascontiguousarray(y))
    print(f"offset={offset:9.0e} ref={ref:.12f} finite={got:.12f} err={abs(got-ref):.3e}  zerofill_err={abs(got2-ref):.3e}")
```

```
offset=    0e+00 ref=0.300903325418 finite=0.300903325418 err=4.441e-16  zerofill_err=4.441e-16
offset=    1e+03 ref=0.300198726054 finite=0.300198726394 err=3.398e-10  zerofill_err=3.398e-10
offset=    1e+05 ref=0.308429820807 finite=0.308378379411 err=5.144e-05  zerofill_err=5.144e-05
offset=    1e+06 ref=0.298996612196 finite=0.295808614577 err=3.188e-03  zerofill_err=3.188e-03
offset=    1e+07 ref=0.300081818793 finite=0.767145514915 err=4.671e-01  zerofill_err=4.671e-01
offset=    1e+08 ref=0.313591326180 finite=0.000000000000 err=3.136e-01  zerofill_err=3.136e-01

epoch-s, 1 min of ticks      ratio=2.8e+07 ref=0.496590376 got=0.000000000 abs_err=4.966e-01
epoch-ms, 1 h of ticks       ratio=4.7e+05 ref=0.501279034 got=0.500779597 abs_err=4.994e-04
epoch-ms, 1 min of ticks     ratio=2.8e+07 ref=0.501293495 got=0.244894434 abs_err=2.564e-01
sensor 1013.25 hPa +-0.001   ratio=1.0e+06 ref=0.495629217 got=0.495822294 abs_err=1.931e-04
engineered x**2 of price 5e4 ratio=5.0e+07 ref=0.496705254 got=0.087776343 abs_err=4.089e-01
acct balance 1e9 +- 100      ratio=1.0e+07 ref=0.498629217 got=0.422125857 abs_err=7.650e-02
```

**Suggested fix:** keep the one-pass structure but centre by a **shifted origin** -- the classic
shifted-data / Welford form. Cheapest change that preserves the single pass and the njit signature:
take `a[0]`, `y[0]` (or the first joint-finite row) as offsets `K_a`, `K_y` and accumulate
`(av - K_a)`, `(yv - K_y)`. Pearson is translation-invariant, so `r` is unchanged mathematically and
the cancellation floor drops from `offset^2 * eps` to `spread^2 * eps`. A true two-pass centred
computation (mean, then centred sums) is exact and is what the reference above uses. Either way the
`<= 1e-24 * n` degeneracy guard then means what it says, and `0.0` stops being reachable by accident.
The docstring's "FP-equivalent to ~1e-15" claim must be corrected or made true.

---

### XNUM-02 [P0] raw prefix-sum segment SSE in the streaming Chow change-point test
**File:** `src/mlframe/training/composite/streaming.py:133`

```python
def _seg_sse(m, sb, sy, sbb, sby, syy):
    m = m.astype(np.float64)
    sxx = sbb - (sb * sb) / m          # line 133
    sxy = sby - (sb * sy) / m          # line 134
    syy_c = syy - (sy * sy) / m        # line 135
    ...
    return np.maximum(sse, 0.0)        # line 143
```

**Summary:** The O(n) split scan builds `cumsum(base*base)`, `cumsum(base*y)`, `cumsum(y*y)` and
recovers each segment's centred sums by subtraction -- shape 1 at k=2, twice over (once in the
cumulative reduction, once in the `- (sb*sb)/m` de-centring). The in-code comment claims *"Numerically
equivalent to the per-segment `_ols_alpha_beta_sse` (same sufficient statistics) up to FP rounding"*,
which is exactly the assumption that hides it.

Two things make the wrong number indistinguishable downstream:

- `sse_full` is computed by the **centred** `_ols_alpha_beta_sse`, while `best_sse` comes from this
  **uncentred** scan. `f_stat = ((sse_full - best_sse)/q) / (best_sse/dof)` mixes the two, so the
  inconsistency lands directly in the reported statistic.
- `np.maximum(sse, 0.0)` clamps a cancellation-negative SSE to `0.0`; `best_sse <= 0.0` then takes the
  `no_break` return, so a genuine regime break is reported as **`found=False, cp_index=-1,
  f_stat=nan`** -- identical to a healthy stationary buffer.

`_detect_change_point`'s output is not diagnostic-only: `streaming_alpha_check_and_refit`
(`streaming.py:262-270`) slices the refit window to `y_clean[cp:]`. A missed break means the refit
blends the dead and live regimes and deploys wrong `(alpha, beta)`; a spurious/shifted `cp_index`
truncates the window at the wrong row. `f_stat` and `sse_split` are also returned to the caller as
reported numbers.

**Failure scenario:** any streaming buffer of prices, revenues, counts or timestamps -- i.e. `y` and
`base` not centred at 0. n=400, min_segment_n=50, a genuine slope break planted at row 250, compared
against a two-pass centred reference doing the identical scan:

| offset | true cp | returned cp | true SSE | returned SSE | true F | returned F | outcome |
|---|---|---|---|---|---|---|---|
| 0 | 251 | 251 | 9.578e+01 | 9.578e+01 | 192.170 | 192.170 | ok |
| 1e3 | 249 | 249 | 8.304e+01 | 8.304e+01 | 171.839 | 171.839 | ok |
| 1e5 | 250 | 250 | 9.910e+01 | 9.911e+01 | 166.581 | 166.574 | ok |
| **1e7** | 256 | 255 | 1.075e+02 | **9.042e+00** | 142.615 | **3851.601** | F inflated **27x**, SSE **12x** too small |
| **1.7e9** | 251 | **-1** | 9.379e+01 | **nan** | 156.960 | **nan** | break **missed entirely**, `found=False` |

**Evidence:** `v5.py`

```python
import numpy as np, sys; sys.path.insert(0,"src")
from mlframe.training.composite.streaming import _detect_change_point as dcp
rng=np.random.default_rng(3)
def ref_chow(y,base,min_seg):
    def sse(yy,bb):
        if yy.size<2: return 0.0
        bc=bb-bb.mean(); yc=yy-yy.mean(); sxx=float(bc@bc)
        if sxx<=1e-300: return float(yc@yc)
        return max(float(yc@yc)-float(bc@yc)**2/sxx,0.0)
    n=y.size; best=(np.inf,-1)
    for k in range(min_seg,n-min_seg+1):
        s=sse(y[:k],base[:k])+sse(y[k:],base[k:])
        if s<best[0]: best=(s,k)
    full=sse(y,base); q=2; dof=n-2*q
    return best[1],best[0],(((full-best[0])/q)/(best[0]/dof) if best[0]>0 else np.inf)
for name,off in [("centred (offset 0)",0.0),("price offset 1e3",1e3),("price offset 1e5",1e5),
                 ("revenue offset 1e7",1e7),("epoch-like offset 1.7e9",1.7e9)]:
    n=400; min_seg=50
    base=off+rng.standard_normal(n)*1.0
    y=off+0.8*(base-off)+rng.standard_normal(n)*0.5
    y[250:]+=0.9*(base[250:]-off)          # genuine slope break at 250
    r=dcp(y,base,min_segment_n=min_seg,f_threshold=5.0)
    rk,rs,rf=ref_chow(y,base,min_seg)
    print(f"{name:32s} {rk:7d} {r['cp_index']:7d} {rs:13.6e} {r['sse_split']:13.6e} {rf:10.3f} {r['f_stat']:10.3f}   found={r['found']}")
```

```
scenario                          ref_cp  got_cp       ref_sse       got_sse      ref_F      got_F
centred (offset 0)                   251     251  9.578416e+01  9.578416e+01    192.170    192.170   found=True
price offset 1e3                     249     249  8.303889e+01  8.303889e+01    171.839    171.839   found=True
price offset 1e5                     250     250  9.910441e+01  9.910645e+01    166.581    166.574   found=True
revenue offset 1e7                   256     255  1.074992e+02  9.041840e+00    142.615   3851.601   found=True
epoch-like offset 1.7e9              251      -1  9.378630e+01           nan    156.960    nan       found=False
```

**Suggested fix:** centre `y` and `base` **once** before building the prefix sums (subtract the global
means, or any fixed shift such as `base[0]`/`y[0]`). OLS SSE is invariant to a shift of either variable,
so the scan and its O(1)-per-split structure are untouched while the cancellation floor drops by
`(offset/spread)^2`. Then compute `sse_full` from the same centred statistics as `best_sse` so the F
ratio is internally consistent, and make the `np.maximum(sse, 0.0)` clamp raise or return NaN rather
than silently producing a `no_break` that is indistinguishable from a stationary buffer.

---

### XNUM-03 [P1] `+ 1e-12` on a power-law denominator in kNN local density
**File:** `src/mlframe/feature_engineering/spatial.py:515`

```python
local_density = float(k) / (dist_to_kth**d + 1e-12)
```

**Summary:** Shape 2. `d` is the **coordinate dimensionality**, so the denominator is `r^d`, which
shrinks geometrically in `d`. The pad reads as a divide-by-zero guard but for any moderately
high-dimensional, moderately dense point set it is the *same order as or larger than* the true
denominator and simply replaces the density. The output is a plain finite number in the emitted
`local_density` feature; nothing distinguishes a saturated `~1e12 * k` from a genuine one.

**Failure scenario:** unit-scaled coordinates (the normal case after a scaler) with `d >= 8` and a kNN
radius below ~0.1 -- i.e. dense data in 8-12 dimensions.

| d | r (dist to kth) | r^d | true density | returned | relative error |
|---|---|---|---|---|---|
| 8 | 0.10 | 1.000e-08 | k*1.0000e+08 | k*9.9990e+07 | 0.01% |
| 8 | 0.05 | 3.906e-11 | k*2.5600e+10 | k*2.4961e+10 | **2.50%** |
| 10 | 0.10 | 1.000e-10 | k*1.0000e+10 | k*9.9010e+09 | 0.99% |
| 10 | 0.05 | 9.766e-14 | k*1.0240e+13 | k*9.1103e+11 | **91.10%** |
| 12 | 0.10 | 1.000e-12 | k*1.0000e+12 | k*5.0000e+11 | **50.00%** |
| 12 | 0.05 | 2.441e-16 | k*4.0960e+15 | k*9.9976e+11 | **99.98%** |

At `d=12, r=0.05` the feature is off by a factor of 4096 and, worse, the pad *saturates*: every row
with a small enough radius returns ~`1e12 * k`, collapsing the density feature's dynamic range to a
constant exactly where it carries the most information (the densest rows).

**Evidence:** `v4.py`

```python
for d in [2,3,5,8,10,12]:
    for r in [0.5,0.2,0.1,0.05]:
        true=1.0/r**d; got=1.0/(r**d+1e-12); rel=abs(got-true)/true
        if rel>1e-6: print(f"  d={d:2d} r={r:.2f}  r^d={r**d:.3e}  true=k*{true:.4e} got=k*{got:.4e} rel_err={rel*100:.2f}%")
```

```
  d= 5 r=0.05  r^d=3.125e-07  true_density=k*3.2000e+06 got=k*3.2000e+06 rel_err=0.00%
  d= 8 r=0.10  r^d=1.000e-08  true_density=k*1.0000e+08 got=k*9.9990e+07 rel_err=0.01%
  d= 8 r=0.05  r^d=3.906e-11  true_density=k*2.5600e+10 got=k*2.4961e+10 rel_err=2.50%
  d=10 r=0.20  r^d=1.024e-07  true_density=k*9.7656e+06 got=k*9.7655e+06 rel_err=0.00%
  d=10 r=0.10  r^d=1.000e-10  true_density=k*1.0000e+10 got=k*9.9010e+09 rel_err=0.99%
  d=10 r=0.05  r^d=9.766e-14  true_density=k*1.0240e+13 got=k*9.1103e+11 rel_err=91.10%
  d=12 r=0.20  r^d=4.096e-09  true_density=k*2.4414e+08 got=k*2.4408e+08 rel_err=0.02%
  d=12 r=0.10  r^d=1.000e-12  true_density=k*1.0000e+12 got=k*5.0000e+11 rel_err=50.00%
  d=12 r=0.05  r^d=2.441e-16  true_density=k*4.0960e+15 got=k*9.9976e+11 rel_err=99.98%
```

**Suggested fix:** work in log space, which removes the guard entirely and is the natural scale for a
density anyway: `log_density = log(k) - d * log(dist_to_kth)`, emitting `log_density` (or
`exp(log_density)` where the linear value is required). If the linear form must be kept, make the guard
**relative and explicit**: `np.where(dist_to_kth > 0, k / dist_to_kth**d, np.inf)` -- the honest answer
for a coincident point is an infinite (or NaN, or sentinel) density, not `1e12 * k`.

---

### XNUM-04 [P2] `+ 1e-12` on `dist**power` in inverse-distance weighting
**File:** `src/mlframe/feature_engineering/spatial.py:576` (also `:577`, `:585` on the weight sums)

```python
weights = 1.0 / (compact_dist**power + 1e-12)
```

**Summary:** Same shape as XNUM-03. `power` defaults to `2.0`, which is safe for unit-scaled
coordinates, but the parameter is caller-settable and the guard does not scale with it. Once
`dist**power` approaches 1e-12 the pad flattens the weights toward uniform, which **changes the emitted
`idw` prediction and `idw_loo_residual`**, not just the internal weight vector -- normalisation does not
cancel it, because the pad is a different fraction of each neighbour's weight.

**Failure scenario:** `power >= 4` on unit coordinates, or `power = 2` on coordinates whose scale is
1e-6 or smaller. Measured on 5 neighbours with labels ~N(0,10):

| power | coord scale | mean d^p | true IDW | returned IDW | error |
|---|---|---|---|---|---|
| 2 | 1e-3 | 7.32e-07 | +6.007427 | +6.007422 | 4.34e-06 (0.00%) |
| 3 | 1e-2 | 9.64e-07 | -5.634030 | -5.634025 | 5.76e-06 (0.00%) |
| **6** | **1e-2** | 5.95e-13 | -1.374972 | **-3.519845** | **2.14 abs (155.99%)** |
| 2 | 1e-4 | 9.84e-09 | +2.446972 | +2.447016 | 4.38e-05 (0.00%) |

Weight-level error reaches 99.93% (`power=6`, `d=0.003`) and 50.00% (`power=6`, `d=0.01`).

**Evidence:** `v4.py`

```python
rng=np.random.default_rng(0)
for p,scale in [(2,1e-3),(3,1e-2),(6,1e-2),(2,1e-4)]:
    dist=np.sort(rng.uniform(0.5,1.5,(1,5)))*scale
    lab=rng.standard_normal((1,5))*10
    w_true=1.0/dist**p; w_got=1.0/(dist**p+1e-12)
    pt=(lab*(w_true/w_true.sum())).sum(); pg=(lab*(w_got/w_got.sum())).sum()
    print(f"  power={p} coord scale={scale:.0e} idw_true={pt:+.6f} idw_got={pg:+.6f} abs_err={abs(pg-pt):.4e} rel={abs(pg-pt)/abs(pt)*100:.2f}%")
```

```
  power=2 coord scale=1e-03 (d^p ~ 7.32e-07) idw_true=+6.007427 idw_got=+6.007422 abs_err=4.3405e-06 rel=0.00%
  power=3 coord scale=1e-02 (d^p ~ 9.64e-07) idw_true=-5.634030 idw_got=-5.634025 abs_err=5.7574e-06 rel=0.00%
  power=6 coord scale=1e-02 (d^p ~ 5.95e-13) idw_true=-1.374972 idw_got=-3.519845 abs_err=2.1449e+00 rel=155.99%
  power=2 coord scale=1e-04 (d^p ~ 9.84e-09) idw_true=+2.446972 idw_got=+2.447016 abs_err=4.3824e-05 rel=0.00%
```

**Suggested fix:** exact-zero-distance is the only case that actually needs handling, and it has a
correct answer: a query point coincident with a reference point should take that point's label
outright. Replace the pad with `w = np.where(compact_dist > 0, compact_dist**-power, np.inf)` and
normalise (the `inf` normalises to a one-hot on the coincident neighbour, which is the right IDW
limit). Failing that, make the guard relative to the row's own distance scale, e.g. add
`(eps * compact_dist.max(axis=1, keepdims=True))**power` rather than an absolute constant. Same
treatment for the `w_sum` / `loo_sum` pads at `:577` and `:585`.

---

### XNUM-05 [P2] raw-moment model-to-model SHAP variance, clipped to zero
**File:** `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_explain.py:752`

```python
mean = s / n_models
var = np.clip(sq / n_models - mean * mean, 0.0, None) if return_variance else None
```

**Summary:** Shape 1 at k=2 again, in the multi-model SHAP variance. `sq` accumulates `pf * pf` over
`n_models` fits of raw phi values. SHAP phi in margin/log-odds space for a dominant feature sits at
3-30 while the model-to-model spread can be orders of magnitude smaller, so `sq/n - mean*mean`
cancels. `np.clip(..., 0.0, None)` then converts the noise-signed result into `0.0`, and `0.0` is the
disabling direction: `phi_var` feeds `subset_uncertainty_many` (`_shap_proxy_objective.py:283`) which
feeds the `uncertainty_penalty` subtraction in `_shap_proxied_fit.py:709-712`. Zero variance means
"the models agree perfectly", so an unstable subset is scored as maximally stable and no penalty is
applied.

**Failure scenario:** near-deterministic boosters (`config_jitter=False`, cached fold fits, seed
affecting only tie-breaks) where per-model phi differ by 1e-7 or less against a phi magnitude of 10-30.
n_models=8, 6 features:

| phi magnitude | per-model sd | true penalty term | returned | penalty rel. error | max per-feature var rel. error | features clipped to 0 |
|---|---|---|---|---|---|---|
| 12.0 | 1e-3 | -- | -- | 1.83e-07 | -- | 0/4 |
| 12.0 | 1e-6 | 2.2986e-06 | 2.3116e-06 | 0.57% | **4.16%** | 0/6 |
| 12.0 | 1e-7 | 2.4406e-07 | 2.3842e-07 | 2.31% | **1822%** | **5/6** |
| 12.0 | 1e-8 | 2.4785e-08 | 2.9200e-07 | **1078%** | **21850%** | 4/6 |
| 30.0 | 1e-7 | 2.2894e-07 | 5.8400e-07 | **155%** | **2520%** | 5/6 |
| 30.0 | 1e-8 | 2.5631e-08 | 6.7435e-07 | **2531%** | **231900%** | 3/6 |

Rated P2 rather than P1 because the regime is narrow: it needs per-model spread below roughly 1e-7
relative to phi. At sd=1e-3 the error is 1.8e-07 and harmless. But the failure mode is one-sided
(clipping only ever *removes* uncertainty) and the fix is free.

**Evidence:** `v2.py` / `v3.py`

```python
def raw_var(P):
    s=np.zeros(P.shape[1]); sq=np.zeros(P.shape[1])
    for r in P: s+=r; sq+=r*r
    m=s/P.shape[0]; return np.clip(sq/P.shape[0]-m*m,0.0,None)
for phi_mean,sd in [(12.0,1e-6),(12.0,1e-7),(12.0,1e-8),(30.0,1e-7),(30.0,1e-8)]:
    P=phi_mean+rng.standard_normal((8,6))*sd
    got=raw_var(P); ref=P.var(axis=0)
    rel=np.abs(got-ref)/np.maximum(ref,1e-300)
    print(f"phi~{phi_mean:5.1f} sd={sd:.0e} ref_sd={np.sqrt(ref.sum()):.4e} got_sd={np.sqrt(got.sum()):.4e} "
          f"penalty_rel_err={abs(np.sqrt(got.sum())-np.sqrt(ref.sum()))/np.sqrt(ref.sum()):.3e} "
          f"max_var_rel={rel.max():.3e} clipped_to_0={(got==0).sum()}/6")
```

```
phi~ 12.0 sd=1e-06 ref_sd=2.2986e-06 got_sd=2.3116e-06 penalty_rel_err=5.657e-03 max_var_rel=4.155e-02 clipped_to_0=0/6
phi~ 12.0 sd=1e-07 ref_sd=2.4406e-07 got_sd=2.3842e-07 penalty_rel_err=2.310e-02 max_var_rel=1.822e+01 clipped_to_0=5/6
phi~ 12.0 sd=1e-08 ref_sd=2.4785e-08 got_sd=2.9200e-07 penalty_rel_err=1.078e+01 max_var_rel=2.185e+02 clipped_to_0=4/6
phi~ 30.0 sd=1e-07 ref_sd=2.2894e-07 got_sd=5.8400e-07 penalty_rel_err=1.551e+00 max_var_rel=2.520e+01 clipped_to_0=5/6
phi~ 30.0 sd=1e-08 ref_sd=2.5631e-08 got_sd=6.7435e-07 penalty_rel_err=2.531e+01 max_var_rel=2.319e+03 clipped_to_0=3/6
```

(`penalty_rel_err` is a fraction: 2.531e+01 = 2531%.)

**Suggested fix:** the loop already accumulates into `s`, so shift each model's contribution by the
first model's phi before squaring -- `sq += (pf - pf0) * (pf - pf0)`, `s0 += (pf - pf0)` -- and recover
`var = sq/n - (s0/n)**2`, which is exact to rounding because the shifted values are the actual spread.
Cheapest alternative given `n_models` is small: keep the per-model `pf` arrays in a list and call
`np.var(stack, axis=0)` (a genuine two-pass), at the cost of `n_models * n_rows * f` floats. Either
way, drop the `np.clip(..., 0.0, None)`; with a centred computation a negative variance is impossible,
so the clip is only ever laundering a numerical failure into a confident zero.

---

### XNUM-06 [P2] `sqrt(expected + 1e-12)` in the correspondence-analysis chi-square residual
**File:** `src/mlframe/feature_engineering/cat_cooccurrence_svd.py:115` (and `:117`)

```python
S = (P - expected) / np.sqrt(expected + 1e-12)
row_coords = (U[:, :n_eff] * s[:n_eff]) / np.sqrt(r + 1e-12)
```

**Summary:** Shape 2. `expected = r @ c` is a product of two marginal *probabilities*, so it is
quadratically small for rare-by-rare category pairs. When `expected` falls to 1e-12 the pad is the
same order as the true denominator and shrinks the standardised residual -- which is precisely the
entry a rare-pair association would show up in. The SVD then under-weights the rare-pair structure,
and the emitted `row_coords` are ordinary finite numbers with nothing marking the loss.

**Failure scenario:** two high-cardinality categoricals (user-id-like, hash-bucketed IDs) where a
category with relative frequency ~1e-6 co-occurs with another at ~1e-6, giving `expected ~ 1e-12`.
Testing a pair with a genuine 50% lift over expectation:

| expected | true residual S | returned S | shrinkage |
|---|---|---|---|
| 1e-06 | 5.0000e-04 | 5.0000e-04 | 0.00% |
| 1e-09 | 1.5811e-05 | 1.5803e-05 | 0.05% |
| 1e-11 | 1.5811e-06 | 1.5076e-06 | **4.65%** |
| 1e-12 | 5.0000e-07 | 3.5355e-07 | **29.29%** |
| 1e-13 | 1.5811e-07 | 4.7673e-08 | **69.85%** |

**Evidence:** `v4.py`

```python
for e in [1e-6,1e-9,1e-11,1e-12,1e-13]:
    resid=0.5*e                     # a 50% lift over expectation
    true=resid/np.sqrt(e); got=resid/np.sqrt(e+1e-12)
    print(f"  expected={e:.0e} true_S={true:.4e} got_S={got:.4e} shrink={100*(1-got/true):.2f}%")
```

```
  expected=1e-06 true_S=5.0000e-04 got_S=5.0000e-04 shrink=0.00%
  expected=1e-09 true_S=1.5811e-05 got_S=1.5803e-05 shrink=0.05%
  expected=1e-11 true_S=1.5811e-06 got_S=1.5076e-06 shrink=4.65%
  expected=1e-12 true_S=5.0000e-07 got_S=3.5355e-07 shrink=29.29%
  expected=1e-13 true_S=1.5811e-07 got_S=4.7673e-08 shrink=69.85%
```

**Suggested fix:** an `expected` of exactly 0 means the row or column marginal is 0, i.e. an unobserved
category whose whole row/column is structurally zero. Handle that structurally rather than with a pad:
`S = np.divide(P - expected, np.sqrt(expected), out=np.zeros_like(P), where=expected > 0)`, and
likewise mask the `sqrt(r)` division on `r > 0`. That is exact for every observed pair and correct
(0, not a shrunken value) for the structurally-empty ones.

---

## LEADS (unmeasured -- not findings)

Each of these matches a hunted shape by inspection but was **not** numerically verified, so no error
magnitude is claimed. Listed for completeness per the "report every finding" rule.

| ID | File:line | Shape | Why it is only a lead |
|---|---|---|---|
| L-01 | `feature_engineering/bayesian.py:220`, `:487`, `:645`, `:658` | 2 | `K = var_pred / (innovation_var + 1e-12)` -- Kalman gain. `innovation_var = var_pred + obs_noise` is normally much larger than 1e-12; would only bite on a heavily down-scaled target (sd <~ 1e-5). Not exercised. |
| L-02 | `feature_engineering/anchor.py:93`, `:562` | 2 | `slope = num / (den + 1e-12)` where `den = sum(dx*dx)`. `den` is a sum of squares, so it is quadratically small for a narrow anchor window. Note `anchor.py:206` already documents a *relative* guard for the sibling path, suggesting this pair was missed by that fix. |
| L-03 | `feature_engineering/hurst.py:255`, `:272`, `:317`, `:562`, `:622` | 2 | Same `num / (den + 1e-12)` least-squares slope shape, `den` a sum of squared log-scale deviations. Multiple sites, all feeding emitted Hurst-exponent features. |
| L-04 | `feature_engineering/spatial.py:286`, `:313` | 2 | `/ (w.sum(axis=1) + 1e-12)` where `w` is itself the padded IDW weight from XNUM-04; compounding, but the sum is large whenever any weight is, so likely benign. |
| L-05 | `feature_engineering/transformer/local_classifier.py:62` | 2 | `W = w_safe * p * (1-p) + 1e-9` -- IRLS weight. For a saturated logistic (`p` ~ 1e-6 or 1-1e-6) `p*(1-p)` ~ 1e-6, so the pad is 1e-3 relative; plausible but the downstream sensitivity was not traced. |
| L-06 | `feature_engineering/transformer/distributional_moments.py:70-76` | 2 | `iqr = (q75-q25) + 1e-9` then `skew_proxy = .../iqr`. A near-degenerate quantile spread on a small-scale target makes the pad dominant. Quantile-based, so no cancellation; magnitude untested. |

---

## Shape 3 (non-neutral `except` substitute): swept, no new finding

I traced every `except` handler in `src/mlframe` that returns a bare `True` / `False` / `0.0` / `+-inf`
(roughly 90 sites). The disabling-direction archetypes named in the brief are **already hardened** in
this tree, with the reasoning written into the code:

- `feature_selection/filters/_fe_gpu_vram.py:117-124` -- the VRAM cushion probe explicitly **fails
  closed** on a `memGetInfo` exception, with an in-code note that *"returning True is the value that
  ALLOWS the upload, so failing open removed the protection at precisely the moment the device is
  unhealthy"*. The one remaining permissive `return True` (`:110`) is on `ImportError` for cupy, i.e. a
  host with no GPU at all, where the value is genuinely neutral.
- `_feature_engineering_pairs/_pairs_setup.py:150` -- `return True` on a validation failure is a
  documented deliberate choice (`# validation failure -> fall back to accepting the warp`).
- `_mrmr_fit_impl/_assign_support.py:489` -- `return True` with the note `# estimator error -> do not
  silently drop a possibly-genuine operand`, i.e. the conservative direction.
- The `return 0.0` substitutes I checked (`_fe_additive_fusion.py:78`, `_orth_extra_basis_fe.py:894`
  and `:955`, `_group4.py:274`, `_mi_aggregator.py:217`) all substitute a score/lift/R2 where `0.0`
  means "no evidence", which **rejects** the candidate. Conservative, not disabling.
- `_mi_aggregator.py:205` and `:217` log at `WARNING`, above the debug threshold in the brief.
- The `-np.inf` substitutes (`_hinge_basis_fe.py:267`, `:734`, `_group2.py:231`/`:234`/`:492`,
  `_group3.py:107`) sit in **maximisation** contexts, where `-inf` guarantees rejection -- again the
  conservative direction, not the guaranteeing-acceptance one.

Two related sites that were *already fixed* before this audit, confirming the class is live in this
codebase and worth the k=2 re-sweep:

- `calibration/_independence_check.py:44` -- now centres each column before forming the sufficient
  statistics, with a comment recording the exact failure (a saturated member's `var_a` going
  noise-signed, clipping to 0.0, and reporting correlation 0.0 = "perfectly independent" for the most
  redundant member possible).
- `votenrank/adversarial_stochastic_blend.py:196` -- now centres against the running mean instead of
  `E[w^2] - E[w]^2`, with a comment recording a `stability_score` of 1.0 reported as a numerical
  artifact.

XNUM-01, XNUM-02 and XNUM-05 are the same defect in three places those two fixes did not reach.

---

## Summary table

| ID | Severity | File:line | Shape | Measured error |
|---|---|---|---|---|
| XNUM-01 | **P0** | `feature_selection/filters/_feature_engineering_pairs/_pairs_core.py:43` (+`:78`) | 1 (k=2) | \|r\| 0.300 -> 0.767 (abs 4.67e-01); true 0.497 -> **exactly 0.0** on epoch-second ticks |
| XNUM-02 | **P0** | `training/composite/streaming.py:133` | 1 (k=2) | Chow F 142.6 -> **3851.6** (27x); genuine break **missed** (`found=False`) at offset 1.7e9 |
| XNUM-03 | P1 | `feature_engineering/spatial.py:515` | 2 | **99.98%** relative error on `local_density` at d=12, r=0.05; 91.10% at d=10, r=0.05 |
| XNUM-04 | P2 | `feature_engineering/spatial.py:576` (+`:577`, `:585`) | 2 | **155.99%** error on the emitted `idw` value at power=6, coord scale 1e-2 |
| XNUM-05 | P2 | `feature_selection/shap_proxied_fs/_shap_proxy_explain.py:752` | 1 (k=2) | uncertainty penalty off by **2531%**; per-feature variance off by 2.3e5%; 5/6 features clipped to 0 |
| XNUM-06 | P2 | `feature_engineering/cat_cooccurrence_svd.py:115` (+`:117`) | 2 | chi-square residual shrunk **69.85%** at `expected=1e-13`, 29.29% at 1e-12 |
| L-01..L-06 | -- | see LEADS table | 2 | **not measured -- leads, not findings** |

**Totals:** 6 confirmed (2 P0, 1 P1, 3 P2), 6 leads, 0 shape-3 findings.

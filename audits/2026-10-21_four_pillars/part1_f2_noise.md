# Four pillars, part 1: F2 across distributions and pure noise (hostile review)

Scope: `src/mlframe/feature_selection/_benchmarks/bench_four_pillars.py`, `tests/feature_selection/mrmr/core/test_f2_single_compound_across_distributions.py`
(called `F2T` below), `tests/feature_selection/biz_val/test_pure_noise_selects_nothing.py` (called `NZT`), `audits/2026-10-16/_TRACKER.md`.
Nothing was fitted. Every generator below was only smoke-run for data generation (n <= 3000, small p).
Tracker state at review time: F2 5/5 (with_outliers fixed by the warp `bulk_gain` gate, row 16), pure noise 9/9 on the bench.
"Pass" below therefore means "passes a weak contract on one seed", which is the point of this review.

## 1. Findings

| # | Weakness | Evidence | Severity |
|---|----------|----------|----------|
| 1 | F2 contract is a name-token test. A single column of the WRONG functional form that mentions a,b,c,d (for example the known bad `min(rint(b),invqubed(oofwarp(a)))` combined with c,d) is classified `full` and passes. No check that the selected features reproduce the target, no held-out `transform` check, no correlation with the noise-free signal. | bench:49-54,67; F2T:193-223 | High |
| 2 | Name-token matching is blind in both directions. `_ID=[a-zA-Z_]\w*` takes whole identifiers, so a column called `a_dup`, `c_x1000` or `d_deg` yields no token `a`/`c`/`d` and escapes the classifier; a selected column with no a-e token at all (junk `gate_select__...`, `cell3`, row index, constants) lands in no bucket and is never counted. `len(names)==1` is only enforced in the slow scale tests. | F2T:24,188-205; bench:49-54,66-67 | High |
| 3 | One seed per case. F2 uses `seed=42` only for the 10k profile test; noise derives `seed=11+index` and reuses it for all 5 binning strategies (so those are the same data, not 5 trials). A pass has probability roughly 0.95 under a true 5% false-positive rate, so a single green carries almost no information; no power/FP interval is ever reported. | F2T:211; NZT:64; bench:62,77 | High |
| 4 | The bench runs 9 of the 90 noise cases (3 of 9 distributions, 3 of 5 strategies, binary target for 2 of 3). `student_t2`, `lognormal`, `exponential`, `binary`, `near_constant`, `uniform` and the `quantile`/`uniform` strategies are not in the fast loop. | bench:121-122 vs NZT:43,56,59-61 | Medium |
| 5 | Noise pass criterion is only the flag `no_significant_features_ is True`. It does not assert `get_feature_names_out()==[]` or that `transform(X)` has 0 columns, so a flag/selection inconsistency (flag True but features returned, or FE columns built then ignored) is invisible. The bench prints the names but does not gate on them. | NZT:72-74; bench:82-83 | Medium |
| 6 | Noise frames are homogeneous and tiny: P=8 columns all from ONE law (plus one normal column for two kinds), N=2000, no NaN, no duplicates/collinear columns, no string/`category` dtype (the "categorical" kind is 6-level floats), no ids beyond a permutation, no correlated noise columns (so no spurious pair structure for the FE pair search to chase), no p>>n, no small n. Targets: fair coin or N(0,1) only; no ties, rare class, multiclass, heavy tail. | NZT:16-17,22-52,49,67-70 | High |
| 7 | F2 regime is a single narrow slice: p=5 with ONE distractor (`e`), n=10000, regression target, additive noise `f/5` from the same family as the operands, no label noise, no NaN, no discrete/mixed dtype, no duplicate or monotone-transformed decoys, no scale/offset stress, no time order. "A single compound among distractors" is untested because a single independent column is almost never picked by chance. | F2T:170-185,217 | High |
| 8 | `with_outliers` only contaminates the operands a,b (3%, 15 IQR) in 10k rows. No outliers in the target, no leverage via divisor near 0 plus heavy-tailed target noise, no outliers in c,d. The outlier fix (tracker row 16) rests on this single contamination law. | F2T:73-75,179-184; tracker:11,22 | Medium |
| 9 | Bench F2 runs with hand-tuned `full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05` and only runs default `MRMR()` when `--f2-default` is passed. The shipped claim is about defaults; the headline bench never tests them. | bench:63,117-118 | Medium |
| 10 | Seeds inconsistent: F2 fit has no `random_state` (bench:63, F2T:214) while noise sets `random_state=0` (bench:81). F2 results may vary between runs for reasons other than data. | bench:63,81 | Medium |
| 11 | Harness: no per-case wall budget or timeout (a 333 s conjunction or 264 s xor3 case blocks the loop), no memory accounting, first case pays numba/JIT compile (tracker row 9, ~10 s) so per-case times depend on case order, exceptions become `ok=False, dt=nan` and silently drop out of `total`, JSON truncates names to 8 and omits seed/git hash/package versions/env flags, `--only` with a typo silently runs nothing, imports rely on `os.getcwd()` plus repeated `sys.path.insert`. | bench:43-45,59,74,88,102,133,139-157,162,168-169 | Medium |
| 12 | `scaled_1_5` builds its operands on a different code path (`rng.uniform`) than the other profiles (`sample_operands`) and uses `f~U(1,5)` (non-zero mean offset `f/5`); `d` is U(0,2pi) there but whatever `sample_operands` yields elsewhere. Profiles are not a clean factorial of (scale x tail x type). | F2T:172-180 | Low |
| 13 | No negative control paired with F2: the same frame with a permuted target must select nothing. That is the cheapest way to find leakage between "dependent columns" structure and the permutation null (case N9). | absent | Medium |
| 14 | `log(abs(c)+1e-9)` makes truth explode for tiny `c` in heavy-tailed profiles; the contract never asks how much Var(y) the log spike owns, so a profile may be dominated by a handful of rows. Not wrong, but undocumented regime. | F2T:184 | Low |

## 2. Shared statistical machinery

Pass/fail on a panel, never on a single seed. With k failures in m seeds the one-sided Clopper-Pearson bounds are
`cp_upper(k,m)` (false-positive rate or failure rate) and `cp_lower(k,m)` (power or pass rate).
Useful anchors: 0 failures in 59 seeds gives FP rate < 5.0% at 95% confidence; 0 in 20 gives < 13.9%; 0 in 10 gives < 25.9%;
20/20 passes give pass-rate > 86%; 10/10 give > 74%.

```python
import re
import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr
from scipy.special import ndtr

def cp_upper(k, m, conf=0.95):
    return 1.0 if k >= m else float(beta.ppf(conf, k + 1, m - k))

def cp_lower(k, m, conf=0.95):
    return 0.0 if k == 0 else float(beta.ppf(1 - conf, k, m - k + 1))

def refs(name, cols):
    """Exact column-name references inside an engineered name (no whole-identifier regex blind spot)."""
    return {c for c in cols if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", str(name))}

def _sig(a, b, c, d):
    return a ** 2 / b + np.log(np.abs(c) + 1e-9) * np.sin(d)

def _base(r, n):
    """Clean F2 operands: a,b ~ U(.5,2), c ~ U(.5,4), d ~ U(0,2pi), e noise."""
    return dict(a=r.uniform(.5, 2, n), b=r.uniform(.5, 2, n), c=r.uniform(.5, 4, n), d=r.uniform(0, 2 * np.pi, n), e=r.normal(size=n))
```

### Shared F2 contract (replaces the token test)

For a case generator `gen(seed, n) -> (X, y, truth)` (truth is the noise-free `_sig`), after fitting on `gen(seed, n)`:

1. Fresh draw `gen(seed + 10_000, n)` (same law, new rows) is passed through `sel.transform`; OLS (intercept + selected columns, fitted on the train transform) predicts it.
2. Spearman(pred, truth) >= 0.90 AND >= 0.95 * Spearman of the oracle column `truth` itself (so heavy-tailed regimes are not unfairly strict).
3. No selected name references a declared noise column (`refs(name, noise_cols)`), checked with exact column names, not tokens.
4. `len(selected) <= max_cols` (default 3). Anything beyond what the signal needs is junk; columns with no operand reference count too.
5. `sel.transform` of a frame that contains NaN does not raise.
6. Panel: 10 seeds regression gate needs 10/10 (pass-rate > 74%); a 20-seed nightly panel needs >= 19/20 on easy cases and reports `cp_lower` on hard cases (power cases are report-only unless noted).

```python
def f2_verdict(sel, gen, seed, noise_cols=("e",), n=3000, min_rho=0.90, max_cols=3):
    Xa, ya, ta = gen(seed + 10_000, n)
    Xt, yt, _ = gen(seed, n)
    names = [str(s) for s in sel.get_feature_names_out()]
    Ft = np.asarray(sel.transform(Xt), float); Fa = np.asarray(sel.transform(Xa), float)
    if Ft.ndim == 1: Ft, Fa = Ft[:, None], Fa[:, None]
    ok = np.isfinite(Ft).all(1)
    w = np.linalg.lstsq(np.c_[np.ones(ok.sum()), Ft[ok]], yt[ok], rcond=None)[0]
    oka = np.isfinite(Fa).all(1)
    pred = np.c_[np.ones(oka.sum()), Fa[oka]] @ w
    rho = spearmanr(pred, ta[oka]).correlation
    oracle = spearmanr(ya[oka], ta[oka]).correlation
    bad_noise = [m for m in names if refs(m, noise_cols)]
    return dict(ok=bool(rho >= min_rho and rho >= 0.95 * oracle and not bad_noise and len(names) <= max_cols),
                rho=float(rho), oracle=float(oracle), noise=bad_noise, n_sel=len(names))
```

### Shared noise contract

For `gen(seed) -> (X, y)` with y independent of X by construction: fit `MRMR(verbose=0, random_state=seed, n_jobs=1, nbins_strategy=s)`.
Fail if `sel.no_significant_features_ is not True` OR `len(sel.get_feature_names_out()) > 0` OR `sel.transform(X).shape[1] > 0`.
Panel: each (case, strategy) is run on m fresh seeds (data seed AND `random_state` both vary). Gate tiers:
* Smoke (every run): 10 seeds on the cheapest strategy (default), require 0 failures (FP < 25.9%, a tripwire, not a guarantee).
* Nightly: 59 seeds on 3 representative cases (N1, N2, N5), require 0 failures (FP < 5.0%); the rest 20 seeds, 0 failures (FP < 13.9%).
* "report" cases (N6, N7) record the FP rate with `cp_upper` and gate nothing until a dependence-aware null exists.

## 3. Proposed F2 cases (target: one compound `a**2/b + log(c)*sin(d)` among distractors)

All generators return `(X, y, truth)`. Preamble of section 2 is assumed.

### F1 `f2_monotone_decoys`
Hard because: columns `q0..q4` are monotone transforms of the signal operands. Rank-based MI sees `exp(a)` and `a` as identical; the selector may pick a decoy or both, or build the compound from a mixture, creating fragments and duplicates.
Expected: one compound (or at most 3 columns) with held-out rho >= 0.90; `e` not referenced; decoys may stand in for their operand but must not add fragments (max_cols=3).
```python
def f2_monotone_decoys(seed, n=3000):
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["q0"] = np.exp(o["a"]); X["q1"] = np.log(o["b"]); X["q2"] = o["c"] ** 3
    X["q3"] = np.sin(o["d"]) + 1e-3 * r.normal(size=n); X["q4"] = pd.Series(o["a"]).rank().to_numpy()
    return X, y, truth
```
Contract: `f2_verdict(..., noise_cols=("e",), max_cols=3)`, 10/10 seeds.

### F2 `f2_leverage_tiny_b`
Hard because: divisor `b` is lognormal (reaches 1e-2), so `a**2/b` has huge-leverage rows; target noise is Student-t(1.5) (infinite variance); `e` is t(2) so rank MI of `e` vs tail rows is spurious-prone. The rank-vs-linear disagreement path from tracker row 5/16 is exercised with leverage in the target, not only in the operands.
Expected: compound recovered (rank metric), `e` not referenced. Pearson is meaningless here, so the contract is rank-only.
```python
def f2_leverage_tiny_b(seed, n=3000):
    r = np.random.default_rng(seed)
    a = r.normal(1.0, 0.3, n); b = r.lognormal(-1.0, 1.0, n)
    c = r.uniform(.5, 5, n); d = r.uniform(0, 2 * np.pi, n); e = r.standard_t(2, n)
    truth = _sig(a, b, c, d); y = truth + 0.5 * r.standard_t(1.5, n)
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e)), y, truth
```
Contract: `f2_verdict`, `min_rho=0.80` (tail noise caps attainable rho; oracle clause still applies), 20 seeds, >= 18/20 (report `cp_lower`).

### F3 `f2_nan_mnar`
Hard because: NaN in `c` is MCAR (5%), NaN in `a` is MNAR on the target (large y => a missing, so the missingness indicator itself leaks y), NaN in `b` is MAR on the noise column `e`. A selector that imputes with the median can build a biased compound; one that drops NaN rows can lose the signal.
Expected: transform never raises; compound built (complete-case rho >= 0.90 on rows where a,b,c,d finite); `e` not referenced (a column that references only `isnan(a)` flags is allowed and is not noise).
```python
def f2_nan_mnar(seed, n=3000):
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X.loc[r.random(n) < 0.05, "c"] = np.nan
    X.loc[r.random(n) < 0.3 / (1 + np.exp(-2 * (y - np.median(y)))), "a"] = np.nan
    X.loc[o["e"] > 1.5, "b"] = np.nan
    return X, y, truth
```
Contract: `f2_verdict` (it already restricts to finite transformed rows) + `rho_complete_case >= 0.90`; also assert fraction of finite rows in the transform >= 0.70 so "NaN everywhere" is not a way out. 10/10.

### F4 `f2_scale_and_range`
Hard because: columns live on wildly different scales (`a` ~ 1e3, `b` ~ 1e6, `c` ~ 1e-4, `d` in float32 quantised radians) and `e` has a huge offset with a tiny range (`1e9 + 1e-3*N`). The truth is defined on the raw scaled columns, so `a**2/b` stays O(1) and `log(c)` is `-9.2 + log(c0)`, adding a `-9.2*sin(d)` term (a legitimate part of the signal).
Expected: compound recovered; `e` not referenced; no overflow/NaN from squaring 1e3 or log of 1e-4.
```python
def f2_scale_and_range(seed, n=3000):
    r = np.random.default_rng(seed); o = _base(r, n)
    a, b, c = 1e3 * o["a"], 1e6 * o["b"], 1e-4 * o["c"]
    d = o["d"].astype(np.float32).astype(np.float64)
    truth = _sig(a, b, c, d); y = truth + 0.2 * r.normal(size=n)
    e = 1e9 + 1e-3 * o["e"]
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e)), y, truth
```
Contract: `f2_verdict`, 10/10. Additionally `np.isfinite(sel.transform(X)).all()`.

### F5 `f2_dupes_collinear`
Hard because: exact duplicate (`a_dup`), near duplicate (`a_jit`, 1e-6 jitter), a 1000x rescaled copy (`c_x1000`), the same angle in degrees (`d_deg`), a half-signal proxy (`ab_ratio=a/b`, correlated but not equal to `a**2/b`) and a duplicate of the noise column (`e_dup`). Name tokens cannot see most of these (finding 2); `refs` can.
Expected: <= 3 selected columns, rho >= 0.90, neither `e` nor `e_dup` referenced, no column pair selected that is |corr| > 0.999.
```python
def f2_dupes_collinear(seed, n=3000):
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["a_dup"] = o["a"].copy(); X["a_jit"] = o["a"] + 1e-6 * r.normal(size=n)
    X["c_x1000"] = 1000 * o["c"]; X["d_deg"] = np.degrees(o["d"])
    X["ab_ratio"] = o["a"] / o["b"]; X["e_dup"] = o["e"].copy()
    return X, y, truth
```
Contract: `f2_verdict(noise_cols=("e","e_dup"))`, 10/10; extra assert on pairwise |corr| of the transform < 0.999.

### F6 `f2_wide_small_n` (power case)
Hard because: n=600 with 150 noise columns generated from 5 latent factors (correlated noise, so spurious pair structure exists) and a randomly permuted column order. Power drops with n, FP pressure rises with p.
Expected: honest power measurement; this is a REPORT case. Gate: pass-rate `cp_lower` over 20 seeds reported; hard gate only on false-discovery side: mean number of referenced noise columns per fit <= 0.2 (any `n*` column referenced counts).
```python
def f2_wide_small_n(seed, n=600, p=150):
    r = np.random.default_rng(seed); o = _base(r, n); o.pop("e")
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    Z = r.normal(size=(n, 5)) @ r.normal(size=(5, p)) * 0.7 + r.normal(size=(n, p))
    X = pd.concat([pd.DataFrame(o), pd.DataFrame(Z, columns=[f"n{j}" for j in range(p)])], axis=1)
    return X[r.permutation(X.columns)], y, truth
```
Contract: `f2_verdict(noise_cols=tuple(f"n{j}" for j in range(150)), max_cols=4)`; success if verdict ok; track `ok` rate and mean `len(noise)`. 20 seeds; expectation recorded as baseline (target >= 70%), noise FDR <= 0.2 per fit gated.

### F7 `f2_discrete_mixed`
Hard because: `a` is an ordinal integer 1..8, `b` gamma, `c` an integer code 1..6, `d` integer degrees (360 levels) in radians, and `e` a 5-level categorical code. Discretisation makes `a**2/b` and `log(c)` piecewise and creates ties; binning strategies disagree on integer columns.
Expected: compound recovered with rho >= 0.85 (quantised operands carry less information; the oracle clause scales the bar).
```python
def f2_discrete_mixed(seed, n=3000):
    r = np.random.default_rng(seed)
    a = r.integers(1, 9, n).astype(float); b = r.gamma(3.0, 0.5, n) + 0.1
    c = r.integers(1, 7, n).astype(float); d = np.round(r.uniform(0, 360, n)) * np.pi / 180
    e = r.integers(0, 5, n).astype(float)
    truth = _sig(a, b, c, d); y = truth + 0.3 * r.normal(size=n)
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e)), y, truth
```
Contract: `f2_verdict(min_rho=0.85)`, 10/10.

### F8 `f2_class_targets` (binary, multiclass, label noise)
Hard because: F2 is only ever regressed. Here the class is a thresholded noisy latent `truth + logistic noise`, k=2 (50/50), k=2 rare (5% positive via `rare=True`), or k=4 quantile classes, with 10% of labels flipped. The compound must survive information loss through thresholding and label noise.
Expected: selection references both a/b and c/d halves (a column set containing at least one of {a,b} and one of {c,d}), `e` not referenced, held-out AUC (binary) or macro one-vs-rest AUC (k=4) of a logistic model on the selected columns >= oracle AUC - 0.03 where the oracle is the same model on the truth column. Returned `truth` is the latent so the contract can compute the oracle.
```python
def f2_class_targets(seed, n=3000, k=2, flip=0.10, rare=False):
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); lat = truth + r.logistic(size=n)
    qs = [0.95] if rare else list(np.linspace(0, 1, k + 1)[1:-1] if k > 2 else [0.5])
    y = np.searchsorted(np.quantile(lat, qs), lat)
    m = r.random(n) < flip; y[m] = r.integers(0, k, m.sum())
    return pd.DataFrame(o), y.astype(np.int64), truth
```
Contract: 10/10 seeds for k=2, k=4; the `rare=True` variant needs n >= 5000 (see memory rule on rare classes) and is a 20-seed report case.

### F9 `f2_time_autocorr_walk`
Hard because: all operands are AR(1) (rho=0.98, uniform marginals via the normal CDF), noise `e` is a random walk (non-stationary, so it correlates spuriously with any trending y), target noise is AR(1) rho=0.9. IID-permutation MI tests are anti-conservative here and a time-agnostic CV leaks.
Expected: judged on a chronological hold-out (train = first 70%, evaluate = last 30%): rho >= 0.85, `e` not referenced.
```python
def _ar1(r, n, rho):
    x = np.empty(n); x[0] = r.normal(); eps = r.normal(size=n) * np.sqrt(1 - rho ** 2)
    for i in range(1, n): x[i] = rho * x[i - 1] + eps[i]
    return x

def f2_time_autocorr_walk(seed, n=3000):
    r = np.random.default_rng(seed); u = lambda: ndtr(_ar1(r, n, 0.98))
    a, b, c, d = .5 + 1.5 * u(), .5 + 1.5 * u(), .5 + 3.5 * u(), 2 * np.pi * u()
    truth = _sig(a, b, c, d); y = truth + 0.3 * _ar1(r, n, 0.9)
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=np.cumsum(r.normal(size=n)))), y, truth
```
Contract: fit on rows[:2100], verdict on rows[2100:] (use `f2_verdict` with the held-out passed explicitly; the fresh-draw rule is replaced by the time split), 20 seeds, REPORT with `cp_lower`; hard gate only that `e` is not referenced in >= 18/20 fits.

### F10 `f2_id_columns`
Hard because: a monotone row index `rid` (time order, correlates with nothing but is perfectly rank-informative about itself), a unique permuted id `uid`, a 300-level group `grp` (10 rows per level) and a constant column. High-cardinality columns blow up binned MI plug-in estimates (finding for any id-like column), and `uid` can win by chance at small n.
Expected: compound recovered; none of `rid,uid,grp,const` referenced.
```python
def f2_id_columns(seed, n=3000):
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["rid"] = np.arange(n); X["uid"] = r.permutation(n)
    X["grp"] = r.integers(0, 300, n); X["const"] = 1.0
    return X, y, truth
```
Contract: `f2_verdict(noise_cols=("e","rid","uid","grp","const"))`, 10/10.

## 4. Proposed pure-noise cases (y independent of X by construction)

All generators return `(X, y)`. Contract is the shared noise contract of section 2 unless stated.

### N1 `noise_factor_blocks`
Hard because: 30 columns built from 3 latent factors (within-block corr ~0.9) plus two explicit interaction columns of noise columns, so the pair-search / FE sees large, perfectly real, pair-level structure that is still independent of y. A selector that tests columns independently will report many "hits" when one block fluctuates.
Expected: nothing selected, no engineered features.
```python
def noise_factor_blocks(seed, n=2000, p=30, k=3, rho=0.9, binary=True):
    r = np.random.default_rng(seed); g = r.normal(size=(n, k)); blk = np.arange(p) % k
    X = pd.DataFrame(np.sqrt(rho) * g[:, blk] + np.sqrt(1 - rho) * r.normal(size=(n, p)),
                     columns=[f"x{j}" for j in range(p)])
    X["x_prod01"] = X.x0 * X.x1; X["x_ratio23"] = X.x2 / (np.abs(X.x3) + 0.1)
    y = (r.random(n) > 0.5).astype(np.int64) if binary else r.normal(size=n)
    return X, y
```
Contract: nightly panel 59 seeds, 0 failures (FP < 5.0%), both target types, strategy `None`; 20 seeds for `mdlp`, `mdlp_validated`.

### N2 `noise_wide_pn`
Hard because: n=300, p=600 of mixed laws (normal, lognormal, binary, 5-level int by `j%4`): maxT/family-wise control must hold with 600 candidate columns and 300 rows; this is where the Westfall-Young style correction either holds or collapses.
Expected: nothing selected.
```python
def noise_wide_pn(seed, n=300, p=600):
    r = np.random.default_rng(seed); cols = {}
    for j in range(p):
        t = j % 4
        cols[f"x{j}"] = (r.normal(size=n) if t == 0 else r.lognormal(0, 1.5, n) if t == 1
                         else r.integers(0, 2, n).astype(float) if t == 2 else r.integers(0, 5, n).astype(float))
    return pd.DataFrame(cols), (r.random(n) > 0.5).astype(np.int64)
```
Contract: nightly 59 seeds, 0 failures.

### N3 `noise_tied_targets`
Hard because: the target is almost all ties: `zero_infl` = 97% exact zeros else lognormal, `ordinal` = 3 levels 70/25/5, `rare` = ~0.6% positives at n=5000 (about 30 positives). Binning the target yields a degenerate or 2-bin y; MI plug-in bias is largest here.
Expected: nothing selected, no exception on near-degenerate y.
```python
def noise_tied_targets(seed, kind="zero_infl", n=None):
    r = np.random.default_rng(seed); n = n or (5000 if kind == "rare" else 2000)
    X = pd.DataFrame({f"x{j}": (r.normal(size=n) if j % 2 else r.integers(0, 6, n).astype(float)) for j in range(10)})
    if kind == "zero_infl": y = np.where(r.random(n) < 0.03, r.lognormal(0, 1, n), 0.0)
    elif kind == "ordinal": y = np.searchsorted([0.70, 0.95], r.random(n))
    else: y = (r.random(n) < 0.006).astype(np.int64)
    return X, y
```
Contract: 20 seeds per kind, 0 failures; for `rare`, seeds where the realised positive count < 10 are resampled.

### N4 `noise_highcard_strings`
Hard because: real `category`/string dtype columns with 1000 and 200 levels at n=2000 (about 2 and 10 rows per level), plus a 50-level column and a numeric id. The existing "categorical" kind is 6-level float and never exercises string handling, target encoding or the cat-aware MI path. Target-encoding style leakage shows up here.
Expected: nothing selected; no engineered target-encoded column survives.
```python
def noise_highcard_strings(seed, n=2000):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"s{L}": pd.Categorical(r.integers(0, L, n).astype(str)) for L in (1000, 200, 50)})
    X["id"] = r.permutation(n); X["num"] = r.normal(size=n)
    return X, (r.random(n) > 0.5).astype(np.int64)
```
Contract: nightly 59 seeds, 0 failures; additionally assert fit does not raise.

### N5 `noise_missing_dupes_constants`
Hard because: duplicated column, negated column, 1e-9 jittered column, a constant, an all-NaN column, a 0.2% non-zero near-constant, MCAR NaN (30%), MNAR NaN where missingness depends on the column's own value, and NaN-ness correlated across columns. The missingness indicator is independent of y, so nothing is a valid signal.
Expected: nothing selected; the all-NaN and constant columns are neither selected nor crash.
```python
def noise_missing_dupes_constants(seed, n=2000):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 8)), columns=[f"x{j}" for j in range(8)])
    X["dup"] = X.x0; X["neg"] = -X.x1; X["jit"] = X.x2 + 1e-9 * r.normal(size=n)
    X["const"] = 7.0; X["allnan"] = np.nan; X["near_const"] = (r.random(n) < 0.002).astype(float)
    X.loc[r.random(n) < 0.3, "x3"] = np.nan
    X.loc[X.x4 > 0.5, "x4"] = np.nan
    X.loc[X.x5.isna() | (X.x6 > 1.0), "x7"] = np.nan
    return X, (r.random(n) > 0.5).astype(np.int64)
```
Contract: nightly 59 seeds, 0 failures.

### N6 `noise_ar1_stationary` (report only)
Hard because: columns and target are independent stationary AR(1) series (rho=0.9, effective sample size about n(1-rho)/(1+rho) ~ 105 of 2000). An IID permutation null is anti-conservative; a nonzero FP rate here is a methodological finding (block / circular-shift permutation is the remedy), not necessarily a bug.
Expected: nothing selected is the pillar goal; it is NOT guaranteed by the current null. Record FP rate and `cp_upper`.
```python
def noise_ar1_stationary(seed, n=2000, p=8, rho=0.9):
    r = np.random.default_rng(seed)
    def ar(m):
        x = np.empty(n); x[0] = r.normal(); e = r.normal(size=n) * np.sqrt(1 - rho ** 2)
        for i in range(1, n): x[i] = rho * x[i - 1] + e[i]
        return x
    return pd.DataFrame({f"x{j}": ar(0) for j in range(p)}), ar(0)
```
Contract: report only: 30 seeds, print `k/m` and `cp_upper`. Gate becomes "FP < 5%" only after a time-aware null is implemented.

### N7 `noise_random_walks` (report only, spurious regression)
Hard because: cumulative sums of independent noise (non-stationary): the classical spurious-regression setting where independent walks look strongly correlated (|corr| often > 0.3 at n=2000).
Expected: ideally nothing; documented as a known violation of the exchangeability assumption. Report FP rate; a selector that is silent here is doing de-trending or block nulls.
```python
def noise_random_walks(seed, n=2000, p=8):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(np.cumsum(r.normal(size=(n, p)), axis=0), columns=[f"w{j}" for j in range(p)])
    return X, np.cumsum(r.normal(size=n))
```
Contract: report only, 30 seeds; also report median |corr(X_j, y)| to show how strong the spurious signal was.

### N8 `noise_heavy_tail_ties`
Hard because: columns are lognormal(sigma=3) and Cauchy (infinite variance), the target is Cauchy rounded to integers (ties plus extreme leverage). Equal-frequency and uniform bins collapse into one huge bin plus a few outlier bins; MDLP boundaries land on single extreme rows.
Expected: nothing selected, finite everything.
```python
def noise_heavy_tail_ties(seed, n=2000, p=8):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{j}": (r.lognormal(0, 3, n) if j % 2 else r.standard_cauchy(n)) for j in range(p)})
    return X, np.round(r.standard_cauchy(n))
```
Contract: 20 seeds across all 5 binning strategies (strategy varies fastest, data seed per trial), 0 failures; strategies share data only within a trial and are counted as separate Bernoulli trials only across seeds.

### N9 `noise_permuted_f2` (paired negative control)
Hard because: it reuses the real F2 frame (heavy-tailed operands, duplicated and monotone decoys, real structure among columns) and shuffles the target. It is the exact null twin of F1/F5: any selection is a false positive created by the feature structure alone, not by the marginal laws.
Expected: nothing selected.
```python
def noise_permuted_f2(seed, n=2000):
    r = np.random.default_rng(seed); o = _base(r, n)
    y = _sig(o["a"], o["b"], o["c"], o["d"]) + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["q0"] = np.exp(o["a"]); X["q1"] = np.log(o["b"]); X["a_dup"] = o["a"].copy()
    X["ab_ratio"] = o["a"] / o["b"]; X["sqa_over_b"] = o["a"] ** 2 / o["b"]
    return X, r.permutation(y)
```
Contract: nightly 59 seeds, 0 failures. Note: `sqa_over_b` is the exact signal half; permuting y destroys its link, so it is a decoy here.

### N10 `noise_float32_offset_quantised`
Hard because: columns `1e9 + N(0,1)` cast to float32 (spacing 64 at 1e9 so only 1-3 distinct values survive), an int64 nanosecond timestamp column, a boolean, and a float16 column. Different dtypes and effective cardinality 1-3 make the binner and the permutation null degenerate.
Expected: nothing selected, no exception.
```python
def noise_float32_offset_quantised(seed, n=2000):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"big32": (1e9 + r.normal(size=n)).astype(np.float32),
                      "ts_ns": (1_700_000_000_000_000_000 + np.sort(r.integers(0, 10 ** 12, n))).astype(np.int64),
                      "flag": r.random(n) > 0.5, "half": r.normal(size=n).astype(np.float16),
                      "ok": r.normal(size=n)})
    return X, r.normal(size=n)
```
Contract: 20 seeds, 0 failures.

## 5. Benchmark harness improvements

1. Panel runner: `--seeds M` (default 1 for the dev loop, 10 smoke, 59 nightly) with data seed and `random_state` both varied; emit `k/m`, `cp_lower`/`cp_upper` per case and gate on the bound, never on one lucky seed (finding 3).
2. Contract module: move `f2_verdict` and the noise contract into a shared importable module (for example `mlframe.feature_selection._benchmarks._contracts`) used by BOTH pytest and the bench; the bench currently re-implements the F2 predicate (bench:49-54,67) so the two can drift. Include the held-out `transform` check (finding 1, 5).
3. Per-case wall budget: run each case in a worker subprocess (`multiprocessing` spawn) with `--case-timeout` (default 300 s) and record `status in {pass, fail, timeout, error}`; exceptions must not turn into `dt=nan` that vanishes from totals (bench:154,162).
4. Memory: record peak RSS per case (`psutil` in the worker, `resource` not available on Windows), the commit charge at start (very low-memory hosts), and abort the case with status `oom_risk` if free commit < configurable floor instead of paging the host. Free the frame and `sel` explicitly between cases.
5. Determinism: pass the same `random_state` in every case (bench:63 vs :81), set `PYTHONHASHSEED`, record numba thread count, `MLFRAME_*` env flags, git hash, library versions and the full selected-name list (not `[:8]`, bench:157) in the JSON. Add a `--repeat 2` mode that refits the first seed twice and fails if the selected names differ.
6. Warm-up: run one throwaway tiny fit before timing (numba compile, tracker row 9) and report `import_s` / `warmup_s` separately so per-case seconds do not depend on case order (bench:43-45).
7. Report both tuned and default parameters for F2 in the same run (flip `--f2-default` to default ON and keep the tuned run as the opt-out), since defaults are what ships (bench:63,117).
8. Argument hygiene: reject unknown `--only` tokens, add `--list`, `--case NAME` and `--exclude`; drop `os.getcwd()`/`sys.path.insert` by importing the generators from an installed `tests`-independent module (the generators in this file can live in `_benchmarks/_pillar_cases.py`).
9. Report format: a table per pillar (case x strategy) with pass-rate, CI bound, median/p95 seconds, peak MB, and the first failing seed's selected names; a `--compare prev.json` mode that flags a regression in pass-rate bound or a >20% median wall change, so optimisation work is checked against correctness in one command.
10. Cost control: put the 59-seed FP panels behind `--tier nightly`; the dev loop keeps the current one-seed-per-case behaviour so profiling stays fast.

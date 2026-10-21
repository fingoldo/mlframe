# Four pillars review, part 2: 3-way XOR among noise and conjunction with decoys

Scope: `src/mlframe/feature_selection/_benchmarks/bench_four_pillars.py` (pillars 3 and 4), `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_xor3_noise.py`, `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_conjunction_decoys.py`, and `audits/2026-10-16/_TRACKER.md` (rows 1, 4, 8, 15 and the pillar-state line: xor3 37 s binary / 264 s continuous, conjunction 333 s). Nothing was fitted; every generator below was smoke-run for shape and target balance only. The "n needed" figures are derived, not measured. Gate thresholds are proposals and have not been calibrated against MRMR runs.

## 1. Findings

| # | Weakness | Evidence | Severity |
|---|---|---|---|
| 1 | One seed, one fit per case; a pass says nothing about power. The tracker records that two identical default conjunction fits gave different selections (row 15), so a single PASS has unknown reliability. | bench:125 (`_xor3_case(yk, 0)` only), bench:127, bench:105 (`seed=0`); tracker row 15 | High |
| 2 | Conjunction bench visits only the native column order; the pytest file checks perm 1 and 2, the bench never does. The data seed is fixed at 0 in both; only the column order varies in the test. | bench:127 `_conj_case(None)`; test_conj:79, :116 | High |
| 3 | xor3/conj fits pass no `random_state` / `n_jobs`, while the noise pillar passes `random_state=0, n_jobs=1`. Run-to-run variation and thread nondeterminism are uncontrolled and unrecorded. | bench:94 `_fit(X, y)` vs bench:81 | High |
| 4 | Fixed regime: xor3 n=4000, 100 Gaussian noise columns, balanced binary / U(-1,1) operands; conj n=50000, 50 Gaussian proxies per operand, W=3, prevalence 0.5. The test comment admits conj recovery fails at n=4000, so the n between 4000 and 50000 where it starts working is unmapped, and the 50k case costs 333 s so nobody maps it. | test_xor3:28-30; test_conj:19-23 | High |
| 5 | The contract hard-codes the engineered name `x1*x2*x3__T1_T1_T1`. A correct selection under another transform name fails; a name match certifies nothing about information content. Renaming a recipe breaks the pillar with no selection regression. | test_xor3:48-62 | Medium |
| 6 | The same expected name is used for the binary and the continuous variant. For 0/1 operands the product x1*x2*x3 is AND, not parity, so the docstring claim that the composite fully explains the target (line 9) needs T1 to be a centering/sign map. Worth verifying; if T1 is not, the binary pass is carried by a different mechanism than documented. | test_xor3:9, :48, :70 | Medium (verify) |
| 7 | The docstring says the continuous y has zero marginal signal, but Var(y given x1) = x1^2/27 (y = product of three U(-1,1)), so the dependence sits in the variance and a binned-y MI has marginal signal on every operand. The case is easier than described; a sign-parity regression (R1) is the real no-marginal version. | test_xor3:3-7, :73 | Medium |
| 8 | Only a boolean per case; "composite missing" (power loss) and "extra selection" (false positive / redundancy) are not distinguished, and only the first 8 names are stored. Power and FP rates cannot be separated. | bench:96, :109, :157 | High |
| 9 | No null control with the same frame shape: permuted y on a 103 or 253 column frame must give `no_significant_features_`. The pure-noise pillar uses other shapes. | bench:81 only | Medium |
| 10 | Conj decoys are only Gaussian-noisy copies (rho 0.7) of single operands. No exact functional copies (affine/monotone), no partial AND proxies, no proxy of the composite, no categorical decoys. Exact ties and partial AND proxies are the realistic decoy classes. | test_conj:21-24, :37-40 | High |
| 11 | Operands are always binary (conj) or balanced binary / uniform (xor3). No NaNs, no few-level categoricals, no unequal balance, no label noise, no heavy tails, no k other than 3, no regression conjunction. | test_xor3:68-73; test_conj:31-35 | High |
| 12 | Noise columns are all N(0,1). Discrete, ID-like, heavy-tailed, constant or NaN-laden noise (where binning and null calibration break) never sit next to the hidden interaction. | test_xor3:74; test_conj:41-42 | Medium |
| 13 | Any exception becomes `ok=False, dt=nan`, dropped from the total and counted as one failure; OOM, timeout and assertion look alike. No per-case wall budget (a 330 s case can hide a 5x regression until the end) and no memory peak although the tracker documents host-memory exhaustion (row 1; conj peak 2.2 GB). | bench:152-154, :162-164 | Medium |
| 14 | The contract is exact-set equality. That is a correct redundancy gate but it fuses recall of the interaction and parsimony into one boolean; one flaky extra column zeroes the power estimate. Report both separately. | test_xor3:9; test_conj:7 | Medium |
| 15 | CPU vs GPU: the bench prints the mode but never compares selections between modes on the same data; xor3 continuous time is dominated by gray-zone families (tracker row 4), so divergence in null calibration would go unnoticed. | bench:140, :158 | Low |
| 16 | Parity order k=3 only. Search size and cell count grow as C(p+k,k) and 2^k, and the heuristic that finds a triple need not find a 4-way or 5-way tuple; nothing exercises k=4,5, p_noise=200, or any n near the detectability floor (the current n is about 25x above the practical floor). | test_xor3:28-29 | High |

## 2. Detectability and the statistics used

Plug-in MI of the true k-tuple (balanced parity, no label noise) is ln2 = 0.693 nats. A random k-tuple has null MI of about chi2 with 2^k - 1 degrees of freedom, divided by 2n. Searching M = C(p+k, k) tuples, the Bonferroni threshold is the chi2 quantile at 1 - 0.05/M. Parity with flip rate f has MI = ln2 - H(f): 0.368 nats at f=0.1, 0.193 at f=0.2, 0.082 at f=0.3. So the information-theoretic floor is n_min = chi2_{1-0.05/M, 2^k-1} / (2 MI).

| k | p_noise | chi2 threshold | n_min f=0 | f=0.1 | f=0.2 | f=0.3 |
|---|---|---|---|---|---|---|
| 3 | 50 | 38.8 | 28 | 53 | 101 | 236 |
| 3 | 100 | 43.4 | 31 | 59 | 113 | 264 |
| 3 | 200 | 48.0 | 35 | 65 | 124 | 291 |
| 4 | 100 | 67.8 | 49 | 92 | 176 | 412 |
| 5 | 100 | 105.2 | 76 | 143 | 273 | 640 |
| 5 | 200 | 114.4 | 83 | 155 | 297 | 695 |

These are floors for an oracle that tests the right tuple. A plug-in estimate with 2^k cells also needs about 10 to 20 observations per cell, so the practical floor is roughly `n >= 20 * 2^k / (1 - 2f)^2`: k=3: 160, k=4: 320, k=5: 640 at f=0 (the (1-2f)^2 factor is the signal attenuation of the flipped parity). With B quantile bins per continuous operand the cell count is B^k; a cellwise estimator with B=10, k=3 needs n >= 20000, which is why a sign-aware composite recipe exists, and why the number of bins MRMR selects is a hidden power parameter. The current xor3 setting (n=4000, p=100) is 25x above the practical k=3 floor, so it is far from the power cliff. Cases marked CLIFF put n at 1 to 4 times the floor.

Panel statistics used by every contract below:

* Power: R seeds, S successes (required composite present and nothing in the forbidden set). "Power >= pi0" is claimed only if the one-sided 95% Clopper-Pearson lower bound `beta.ppf(0.05, S, R-S+1)` >= pi0. Lower bounds for all-pass: R=20 gives 0.861, R=30 gives 0.905, R=59 gives 0.950. For R=30: 28 of 30 gives 0.805, 27 gives 0.761. The present tests (R=3) give 0.368 even at 3 of 3.
* False positive: F = runs with a forbidden column (noise, proxy, partial AND decoy). "FP rate <= alpha0" is claimed if the upper bound `beta.ppf(0.95, F+1, R-F)` <= alpha0: 0 of 30 gives 0.095, 0 of 59 gives 0.049.
* Gates written as "power >= 0.90 at R=30" therefore mean 30 of 30 passes; "power >= 0.80 at R=30" means at least 28 of 30.
* Cost: at 30 to 330 s per fit, R=30 is feasible at the cheap tier (n at 2 to 16x the practical floor, p_noise <= 100, a few to 40 s per fit, untested) or nightly. Each case below is the generator plus a Tier-S / Tier-L reading in the contract.

## 3. Cases

Conventions: every generator returns `(X: DataFrame, y: Series)`, is deterministic in `seed`, uses only numpy / pandas, and shares this helper (included once):

```python
def _frame(cols, p_noise, rng, n):
    d = dict(cols)
    for i in range(p_noise):
        d[f"noise_{i}"] = rng.standard_normal(n)
    return pd.DataFrame(d)
```

"Forbidden" means a column named `noise_*` or a decoy named in the case. Supports are compared on name tokens (split on non-alphanumerics, as the existing tests do), so "a composite covering x1,x2,x3" means one selected feature whose tokens include x1, x2, x3. Cases P2 and P8 reuse the generator of P1 / P7 with different arguments.

### P1. Parity k=3,4,5 at floor n, p_noise 50 to 200 (CLIFF power curve)

**Why hard.** The existing case sits 25x above the floor, so it shows nothing about the cliff. k and p_noise set the search size M = C(p+k,k) (k=5, p=200 gives 3.4e9 tuples), which no exhaustive scan can enumerate, so the heuristic that finds the tuple must be shown to still work. Floors: oracle table in section 2; practical k=3 n>=160, k=4 n>=320, k=5 n>=640.

```python
def g_parity_k(seed, k=3, n=3000, p_noise=100, flip=0.0, continuous=False):
    rng = np.random.default_rng(seed)
    if continuous:
        Z = rng.uniform(-1, 1, (n, k)); par = (Z > 0).sum(1) % 2
    else:
        Z = rng.integers(0, 2, (n, k)).astype(float); par = Z.sum(1).astype(int) % 2
    y = np.where(rng.random(n) < flip, 1 - par, par)
    X = _frame({f"x{i+1}": Z[:, i] for i in range(k)}, p_noise, rng, n)
    return X, pd.Series(y)
```

**Contract.** Grid k in {3,4,5}, p_noise in {50,200}, n in {2x, 4x, 16x practical floor}. Expected: exactly one feature covering x1..xk, nothing else. Tier-S: R=20; gate power >= 0.86 (20 of 20) at 16x floor for every k; at 2x and 4x floor only report power with its CP interval (this is the curve). FP: no noise token in any run (upper bound 0.14 at R=20). Tier-L: R=59 at 4x floor, gate power >= 0.95 for k=3 and record k=4,5 as a table. Continuous variant (`continuous=True`, sign parity) at the same grid; CLIFF cells are report-only.

### P2. Label-noise parity (flip rate 0.1, 0.2, 0.3)

**Why hard.** Labels are noisy in practice; the MI of the true tuple falls to ln2 - H(f) (0.368, 0.193, 0.082 nats) and the oracle n_min at k=3, p=100 rises 31, 59, 113, 264 for f = 0, 0.1, 0.2, 0.3. A selector with a fixed significance margin collapses somewhere on this axis; the current contract cannot see it.

```python
def g_parity_k(seed, k=3, n=3000, p_noise=100, flip=0.0, continuous=False):
    rng = np.random.default_rng(seed)
    if continuous:
        Z = rng.uniform(-1, 1, (n, k)); par = (Z > 0).sum(1) % 2
    else:
        Z = rng.integers(0, 2, (n, k)).astype(float); par = Z.sum(1).astype(int) % 2
    y = np.where(rng.random(n) < flip, 1 - par, par)
    X = _frame({f"x{i+1}": Z[:, i] for i in range(k)}, p_noise, rng, n)
    return X, pd.Series(y)
```

**Contract.** Call with `flip` in {0.1,0.2,0.3}, k=3, p_noise=100, n=3000 (11x the f=0.3 oracle floor, so all three should be recoverable by a good selector). Expected: one feature covering x1,x2,x3, nothing else. R=30: f=0.1 gate power >= 0.90; f=0.2 gate power >= 0.80; f=0.3 report only. FP gate in all three: 0 noise tokens in 30 runs (upper bound 0.095); an over-eager selector grows noise picks as f rises.

### P3. Parity among continuous, binary and few-level categorical operands

**Why hard.** Operands are mixed in practice: a sign-parity of a Gaussian, a binary and a 4-level categorical (parity of the level index). A composite recipe that works for x1*x2*x3 on U(-1,1) may not exist for a categorical operand, which must go through a cell cross. Effective cells: 2*2*4 = 16 (parity of the level index folds 4 levels to 2 but the cross still needs all 4), practical floor about 20*16 = 320.

```python
def g_xor_mixed_types(seed, n=3000, p_noise=50):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n); b = rng.integers(0, 2, n); c = rng.integers(0, 4, n)
    y = ((a > 0).astype(int) + b + (c % 2)) % 2
    cols = {"x1": a, "x2": b.astype(float), "x3": pd.Categorical(c.astype(str))}
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: one feature (composite or cell cross) covering x1,x2,x3, nothing else. R=30, n=3000: gate power >= 0.90; 0 noise tokens (upper bound 0.095). The fit must not raise on the categorical dtype.

### P4. Unequal balance: XOR of Bernoulli(q) operands

**Why hard.** With q=0.15 each operand has a strong marginal (P(y=1 given x1=1) is about 0.74 against 0.26, measured at n=20000), so this is not a no-marginal case: the raw operands are individually attractive and a greedy forward selector takes x1 first. The correct support is still the composite of x1,x2,x3, since parity determines y; the raw operand is redundant once the composite is in. y prevalence is 0.34, so MI estimates are also noisier. Tests the redundancy gate (tracker row 8) against an individually informative operand.

```python
def g_xor_unbalanced(seed, n=3000, q=0.15, p_noise=50):
    rng = np.random.default_rng(seed)
    Z = (rng.random((n, 3)) < q).astype(float)
    y = Z.sum(1).astype(int) % 2
    return _frame({f"x{i+1}": Z[:, i] for i in range(3)}, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: exactly the composite covering x1,x2,x3. R=30, n=3000, q in {0.5, 0.3, 0.15} (call with q=...): gate power >= 0.90 for each q. Any raw x_i beside the composite is a failure but is also tallied as `redundant_operand_rate` with upper bound <= 0.15 (R=30 allows 1 of 30 at most: 0.157, so require 0 of 30 for the gate).

### P5. Operands with NaNs (MCAR), 10 percent and 30 percent

**Why hard.** y is computed on complete values and the operands are masked afterwards, so the informative rows are those with all three operands present: n*(1-r)^3, i.e. 2187 rows at r=0.1 and 1029 at r=0.3 of n=3000 (floor 160). Rows with a missing operand carry no parity information (the missing operand makes it a coin flip). A composite built by arithmetic on NaN returns NaN there, a cell cross treats NaN as a level; the case separates NaN-aware from NaN-propagating composites.

```python
def g_xor_nan(seed, n=3000, nan_rate=0.1, p_noise=50):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 3)).astype(float)
    y = Z.sum(1).astype(int) % 2
    Zm = np.where(rng.random((n, 3)) < nan_rate, np.nan, Z)
    return _frame({f"x{i+1}": Zm[:, i] for i in range(3)}, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: one feature covering x1,x2,x3. R=30: nan_rate=0.1 gate power >= 0.90; nan_rate=0.3 gate power >= 0.80. No noise token in any run; the fit must not raise.

### P6. XOR with one operand carrying a weak marginal effect

**Why hard.** y = Bernoulli(sigmoid(4*(parity - 0.5) + beta*(x1 - 0.5))). The additive x1 term makes x1 significant on its own at modest n, so the greedy first step takes x1 and the 3-way structure is visible only conditionally. The expected support is the composite alone: the composite already contains x1, so the raw x1 has no conditional gain even though its marginal is real.

```python
def g_xor_weak_marginal(seed, n=3000, beta=0.5, p_noise=50):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 3)).astype(float)
    par = Z.sum(1) % 2
    logit = 4.0 * (par - 0.5) + beta * (Z[:, 0] - 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return _frame({f"x{i+1}": Z[:, i] for i in range(3)}, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: exactly the composite covering x1,x2,x3. R=30, n=3000, beta in {0.5, 1.5}: gate power >= 0.90; raw-x1-beside-composite must be 0 of 30; 0 noise.

### P7. Two independent XOR triples at once (hidden-selector mixture)

**Why hard.** y takes the parity of triple 1 (x1,x2,x3) on a random half of the rows and the parity of triple 2 (x4,x5,x6) on the other half; the selector is hidden. Each triple then explains only part of y: P(y = t1) = 0.75, so MI per triple is ln2 - H(0.25) = 0.131 nats (the full 6-column cell cross has the whole ln2 minus the mixing term). The second triple has to be found after the first is explained. Per-triple oracle floor at k=3, p=50: 38.8/(2*0.131) = 148.

```python
def g_two_xor_triples(seed, n=3000, p_noise=50, overlap=False):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 6)).astype(float)
    t1 = Z[:, :3].sum(1) % 2
    t2 = (Z[:, [0, 3, 4]] if overlap else Z[:, 3:]).sum(1) % 2
    sel = rng.random(n) < 0.5
    y = np.where(sel, t1, t2).astype(int)
    cols = {f"x{i+1}": Z[:, i] for i in range(6)}
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: two features, one covering {x1,x2,x3} and one covering {x4,x5,x6}, no feature mixing triples, no noise. R=30, n=3000: gate that at least one triple is found at power >= 0.90, and both triples at power >= 0.80; the recall (1 or 2 of 2) is reported separately.

### P8. Overlapping triples sharing one operand

**Why hard.** P7 with `overlap=True`: triples {x1,x2,x3} and {x1,x4,x5} share x1. A selector that deduplicates by raw-column coverage drops the second composite because x1 is already used; the compaction that builds a cell3 from exactly three raw columns (tracker row 15) can create this failure. The union has five columns, so a merged 5-way cross is a distinct, legitimate outcome.

```python
def g_two_xor_triples(seed, n=3000, p_noise=50, overlap=False):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 6)).astype(float)
    t1 = Z[:, :3].sum(1) % 2
    t2 = (Z[:, [0, 3, 4]] if overlap else Z[:, 3:]).sum(1) % 2
    sel = rng.random(n) < 0.5
    y = np.where(sel, t1, t2).astype(int)
    cols = {f"x{i+1}": Z[:, i] for i in range(6)}
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Call with `overlap=True`, n=4000. Expected: two features with tokens {x1,x2,x3} and {x1,x4,x5}, no noise. Outcome `merged5` (one feature covering x1..x5) is counted separately, not as failure and not as success. R=30: gate that at least one triple composite or merged5 appears at power >= 0.90; report both-found and merged5 rates; 0 noise tokens.

### P9. Gated (conditional) parity

**Why hard.** y equals the parity of x1..x3 only when g=1 and is a fair coin otherwise. g has zero marginal effect, and gating halves both the effective sample and the signal: MI of (g,x1,x2,x3) is 0.5*ln2 = 0.347 nats, so the oracle floor doubles (about 2*28 at p=50, k=4 search: 61.2/(2*0.347) = 88). The minimal explanation is the composite of x1,x2,x3; g is informative only jointly.

```python
def g_gated_parity(seed, n=4000, p_noise=50):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 3)); g = rng.integers(0, 2, n)
    y = np.where(g == 1, Z.sum(1) % 2, rng.integers(0, 2, n))
    cols = {"g": g.astype(float), **{f"x{i+1}": Z[:, i].astype(float) for i in range(3)}}
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: one feature covering x1,x2,x3; a feature that also includes g is acceptable and tallied as `with_gate`. R=30, n=4000: gate power >= 0.90; 0 noise tokens; report with_gate rate.

### P10. Parity among monotone-transform twins

**Why hard.** Each operand appears three times (x, exp(2x), x^3). All three triples carry identical information, because the monotone images bin identically. The right answer is ONE composite using one representative per operand. Failure modes: two twin composites selected (pairwise MI cannot separate them, rank correlation is 1), or a composite mixing twins of the same operand (redundant). Stresses the tie-handling the tracker lists as open for engineered exact copies.

```python
def g_parity_monotone_twins(seed, n=3000, p_noise=50):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, (n, 3))
    y = ((Z > 0).sum(1) % 2).astype(int)
    cols = {f"x{i+1}": Z[:, i] for i in range(3)}
    cols.update({f"x{i+1}_exp": np.exp(2 * Z[:, i]) for i in range(3)})
    cols.update({f"x{i+1}_cube": Z[:, i] ** 3 for i in range(3)})
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: exactly one feature whose tokens, after dropping the suffixes `exp` and `cube`, cover x1,x2,x3. R=30, n=3000: gate power >= 0.90 and the rate of runs with more than one selected feature must be 0 of 30.

### C1. Conjunction with exact and partial decoys

**Why hard.** Exact decoys: A_copy (affine) and A_exp (monotone) are functions of A, so any one of the class {A, A_copy, A_exp} is acceptable but two is not. Partial AND decoys: AB, AC, BC each carry a large share of the conjunction signal (AB is on in 25% of rows and is on in every row of the A&B&C event, so it predicts the event with precision 0.5). The minimal complete explanation is one member of the A class plus one composite covering A,B,C. The existing decoys are noisy copies and produce no ties and no partial proxies. Detectability: the AND cell holds n/8 rows; the logit lift W=3 over the A-only level gives a clean separation, so the floor is about 20 rows per cell, n >= 160 for the oracle, and several times that for the 5 decoys and 50 noise.

```python
def g_conj_exact_partial_decoys(seed, n=3000, p_noise=50, prev=0.5):
    rng = np.random.default_rng(seed)
    A, B, C = (rng.random((3, n)) < prev).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 2.5 * A + 3.0 * A * B * C)))).astype(int)
    cols = {"A": A, "B": B, "C": C, "A_copy": 10.0 * A + 3, "A_exp": np.exp(A), "AB": A & B, "AC": A & C, "BC": B & C}
    return _frame({k: np.asarray(v, float) for k, v in cols.items()}, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: exactly one A-class column plus exactly one feature covering the A class, B and C; not AB/AC/BC; no noise. R=30, n=6000 (the pytest case needs 50000 only because of 150 proxies plus 100 noise; here there are 5 decoys and 50 noise): gate power >= 0.90; partial-decoy selection must be 0 of 30; noise 0 of 30. Run half the replicates under a column permutation and require equal selected token sets between each replicate and its permuted twin.

### C2. Rare conjunctions of 3 and 4 operands (varying prevalence)

**Why hard.** y = AND of k Bernoulli(prev) operands with a small flip rate. Positive rate is prev^k: 1.6% at (k=3, prev=0.25), 0.4% at (k=4, prev=0.25), 0.1% at (k=3, prev=0.1). The usable requirement is about 30 true positives with true positives outnumbering flip-created positives: n >= 30/prev^k, giving 1920, 7680 and 30000 respectively. This is where plug-in MI has its largest bias and the null calibration is least accurate, and where an 'all-sample' significance gate ignores a feature that is informative in 0.4% of the rows.

```python
def g_rare_conj(seed, k=3, prev=0.25, n=3000, flip=0.02, p_noise=50):
    rng = np.random.default_rng(seed)
    Z = (rng.random((n, k)) < prev).astype(float)
    y = Z.prod(1).astype(int)
    y = np.where(rng.random(n) < flip, 1 - y, y)
    return _frame({f"x{i+1}": Z[:, i] for i in range(k)}, p_noise, rng, n), pd.Series(y)
```

**Contract.** Grid (k, prev, n, flip): (3, 0.25, 6000, 0.002), (4, 0.25, 16000, 0.0005), (3, 0.1, 30000, 0.0002). Expected: one feature covering x1..xk (raw x_i have a small marginal and are redundant given the composite), nothing else. R=30: the first two settings gate power >= 0.90, the third is CLIFF report-only (Tier-L); 0 noise tokens in all (upper bound 0.095).

### C3. AND-OR mixture: (A and B) or (C and D)

**Why hard.** Two conjunctions joined by OR; every raw column has a marginal. No single 3-way composite explains y compactly; two pair composites or one 16-cell cross do (practical floor about 20*16 = 320). Tests that the selector does not stop after the first conjunction and that the second pair is still found after the first is explained. 3% label flips.

```python
def g_and_or_mixture(seed, n=3000, p_noise=50, flip=0.03):
    rng = np.random.default_rng(seed)
    A, B, C, D = rng.integers(0, 2, (4, n))
    y = ((A & B) | (C & D)).astype(int)
    y = np.where(rng.random(n) < flip, 1 - y, y)
    cols = {k: v.astype(float) for k, v in zip("ABCD", (A, B, C, D))}
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: selected features whose tokens cover exactly {A,B,C,D} (any decomposition: two pair composites, one 4-way, or raw plus composite), at most 3 features, no noise. R=30, n=3000: gate coverage power >= 0.90; noise 0 of 30; more-than-3 rate 0 of 30.

### C4. Interaction through a many-level categorical

**Why hard.** cat has L=20 levels, half of them in a hidden set S; y = [cat in S] XOR b. cat alone and b alone have zero marginal signal; the pair determines y. The pair has 2*20 = 40 cells, practical floor 20*40 = 800; with S unknown a selector must either learn the level partition or accept the 40-cell cross. A numeric pair-composite recipe does not apply to a categorical, and per-level shrinkage matters at ~75 rows per level.

```python
def g_many_level_cat_xor(seed, n=3000, L=20, p_noise=50):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, L, n); b = rng.integers(0, 2, n)
    S = np.random.default_rng(99).permutation(L)[: L // 2]
    y = (np.isin(cat, S).astype(int) + b) % 2
    cols = {"cat": pd.Categorical([f"c{v}" for v in cat]), "b": b.astype(float)}
    return _frame(cols, p_noise, rng, n), pd.Series(y)
```

**Contract.** Expected: one feature covering {cat, b}, nothing else. R=30, L=20, n=3000: gate power >= 0.90, 0 noise. CLIFF report: L=50, n=8000 (floor 2000).

### R1. Regression: heavy-tailed sign-parity times capped magnitude, plus a conjunction offset

**Why hard.** Heavy-tailed (t3) operands, y = sign(x1 x2 x3) * min(|x1 x2 x3|, 5) + 0.5 eps + 2*(A and C). The sign term has no conditional-mean signal on any proper subset of the operands. Unlike the present continuous case, whose variance depends on each operand, the cap here bounds the magnitude and t3 tails put most mass in a few equal-width bins, stressing the binning. The offset 2*A*C adds an independent conjunction with marginal means of 0.5 on each of A and C.

```python
def g_regression_xor_conj(seed, n=3000, p_noise=50, heavy=True):
    rng = np.random.default_rng(seed)
    Z = rng.standard_t(3, (n, 3)) if heavy else rng.normal(size=(n, 3))
    B = rng.integers(0, 2, (2, n))
    y = np.sign(Z).prod(1) * np.minimum(np.abs(Z).prod(1), 5) + 0.5 * rng.normal(size=n)
    cols = {"x1": Z[:, 0], "x2": Z[:, 1], "x3": Z[:, 2], "A": B[0].astype(float), "C": B[1].astype(float)}
    return _frame(cols, p_noise, rng, n), pd.Series(y + 2.0 * B[0] * B[1])
```

**Contract.** Expected: one feature covering {x1,x2,x3} and features covering {A,C} (raw A, C allowed), no noise. R=20 (Tier-S), n=3000: gate coverage 20 of 20 (lower bound 0.86) and 0 noise tokens; Tier-L: R=59 with the 0.95 gate. Report the held-out R2 of a model on the selection against the analytic noise ceiling.

### N1. Null control with the same frame shape

**Why hard.** Permuted y on the P1 frame (103 columns): the correct selection is empty (`no_significant_features_` True). Without this, every power number above can be bought with a lax significance gate. It is also the cheapest per-commit check.

```python
def g_null_control(seed, n=3000, p_noise=50):
    X, y = g_parity_k(seed, 3, n, p_noise)
    return X, pd.Series(np.random.default_rng(seed + 7).permutation(y.to_numpy()))
```

**Contract.** R=59 fits at n=3000, p_noise=100: 0 selected columns in every run, giving FP upper bound 0.049. If any run selects a column, print its token and the family-wise rate.

## 4. Harness improvements

1. **Seed panel instead of one fit.** `--seeds R` (default 1 for the per-commit loop, 30 nightly); seed and column permutation vary per replicate. For each case print successes S/R with the CP lower bound, FP count F/R with the CP upper bound, median and max wall. Panel code:

```python
from scipy.stats import beta

def cp_lower(s, r, a=0.05):
    return 0.0 if s == 0 else float(beta.ppf(a, s, r - s + 1))

def cp_upper(f, r, a=0.05):
    return 1.0 if f == r else float(beta.ppf(1 - a, f + 1, r - f))

def panel(rows, power_gate=0.90, fp_gate=0.10):
    s = sum(r["ok"] for r in rows); f = sum(r["n_forbidden"] > 0 for r in rows); n = len(rows)
    lo, hi = cp_lower(s, n), cp_upper(f, n)
    verdict = "PASS" if lo >= power_gate and hi <= fp_gate else ("INCONCLUSIVE" if s / n >= power_gate else "FAIL")
    return f"power {s}/{n} CP95 lo {lo:.3f} | FP {f}/{n} CP95 hi {hi:.3f} | {verdict}"
```

   The three-valued verdict matters: a small panel can prove PASS only when the lower bound clears the gate; otherwise it must report INCONCLUSIVE, not FAIL.
2. **Failure taxonomy per run** instead of a boolean: `missing_composite`, `noise_selected`, `decoy_selected`, `redundant_operand`, `duplicate_composite`, `merged_cross`, `exception`, `timeout`. `_support_problems` already produces this information as strings (bench:96, :109); return it structured and write it to the json rows (bench:157).
3. **Per-case wall budget.** Run each fit in a subprocess with `--case-timeout` (default 2x the median of a stored baseline json) and report `TIMEOUT` as its own verdict; flag a wall regression above 25 percent as soon as the case finishes, not at the end (bench:150 has no bound). A subprocess also isolates the 2.2 GB peak of the conjunction fit (tracker row 1).
4. **Determinism.** Pass `random_state=0, n_jobs=1` in the contract run (as bench:81 does for noise) and add a double-fit check on one seed: two identical default fits must give identical selections (tracker row 15 shows they once did not). Record a `deterministic` column.
5. **Memory.** Record peak RSS of the child and the free commit charge before each case; emit `SKIPPED_LOW_MEMORY` instead of crashing the host. Cast frames to float32 before the fit (float64 copies contributed to the exhaustion in row 1); the generators here keep float64 for clarity.
6. **Tiering.** Tier-S (per commit, each under about 40 s, untested): N1 at R=20, P1 k=3 at 16x floor, C1 at n=6000. Tier-L (nightly): CLIFF grids and R=59. Keep the existing 4000-row xor3 and 50k conj as a `legacy` tier so history stays comparable.
7. **Power table at the end.** After the per-case lines print `case | n | p_noise | R | power (CP lo) | FP (CP hi) | median s | verdict`, plus a power-curve block for P1 and P2 (rows = n, columns = k or flip) so the cliff location is visible and diffable between commits. Write the same table into `--json`.
8. **CPU/GPU parity.** When both modes ran the same seed, diff the selected sets and print `MODE_DIVERGENCE` on any difference (bench:140).
9. **Contract by semantics, not name.** Score the support by token coverage and by a held-out refit of a small model on a fresh draw from the generator (AUC or R2 within a CI of the analytic Bayes rate, computable for every generator here), keeping the name check as a secondary assertion. This removes findings 5 and 6.
10. **Order invariance as a panel.** Half the replicates use a column permutation and the selected token sets must equal those of the unpermuted twin with the same data seed; the test currently compares native with one permutation, the bench compares nothing.

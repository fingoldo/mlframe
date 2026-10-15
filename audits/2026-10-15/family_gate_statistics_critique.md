# Critique: statistical validity of the family-wise permutation gate

Source: read-only critique agent, 2026-10-15. Scope: `src/mlframe/feature_selection/filters/`. Claims marked "measured" were not reproduced in this report; numbers quoted from the brief are marked as such.

## Findings

### F01 - Quantile threshold does not give the stated level
- **Disposition**: COMPLETED (exact p-value `(1 + R) / (B + 1) <= alpha` in `_family_maxt_gate.py` and in the fallback; checked on a toy null: p=1/1000 passes at alpha 0.001, p=2/1000 does not)
- The gate takes `np.quantile(null, 1 - alpha, method="higher")` and passes when `observed > threshold` (`_family_maxt_gate.py`, the threshold lines in `family_pass_batches`, `family_maxt_pass_codes`, `family_maxt_threshold`). With B = 999 permutations the index is `ceil((B-1)(1-alpha))`.
- Probability of passing under H0 is `(k+1)/(B+1)` where k is the number of null maxima at or above the observed one. Consequences by calculation (not measured here): alpha = 0.0025 gives an effective level near 3/1000; alpha = 1e-4 gives the maximum of the null, effective level 1/1000 (10x the stated value).
- Fix direction: pass when `(1 + R)/(B + 1) <= alpha`, where R counts null maxima at or above the observed value; require `B + 1 >= 1/alpha`.

### F02 - 1e-4 is not reachable with 999 permutations
- **Disposition**: OPEN
- A p-value test at alpha = 1e-4 needs at least 10^4 permutations per family. Permutation-only certification of strong signals is therefore not affordable inside every fit.
- Fix direction: sequential Monte Carlo for the decision that matters (F03), and an analytic tail (chi-square / G-test with Bonferroni over K, or a GPD fit to the null maxima) for certifying acceptance. Mark the analytic route as an approximation wherever it is used.

### F03 - Sequential stopping (Besag-Clifford style)
- **Disposition**: OPEN
- Continue permuting until h exceedances are seen (h around 10) or N_max is reached; report `p = (R + 1)/(N + 1)`. This is a valid p-value for rejection/acceptance decisions and is stable under early stopping.
- Expected cost under H0 about h * ln(1/alpha) (about 100 permutations for alpha near 1e-4). Expected cost under a clear signal is near 1/alpha, which is why F02 needs the analytic route.
- Determinism: seed derived from a hash of the pair, check stopping only at block boundaries (block 256).

### F04 - Early rejection changes the level only if its premise fails
- **Disposition**: OPEN
- Rule (`_family_maxt_gate.py`, early block): reject when the observed maximum is at or below the median of the first 256 null maxima. This does not inflate the level by itself. The comment "the decision is unchanged" holds only if the premise holds in every case; it is not proven.
- Fix direction: state the premise explicitly and test it on simulated null families; or drop the claim from the comment.

### F05 - Table floor is a nearest-neighbour lookup, not a test
- **Disposition**: OPEN
- `_null_mi_threshold.py:37-52` returns the entry nearest on (log n, bins, entropy, log K). Its level is not guaranteed and it is used as a cheap pre-filter (`_pairs_emit.py:106-110`).
- Fix direction: keep it only as a pre-filter whose rejections are cheap and whose level is documented; never count it as part of the family-wise guarantee.

### F06 - Possible leakage through configuration construction
- **Disposition**: OPEN
- The validity of the permutation null requires that no configuration is fitted to y (prewarp, weight-of-evidence, target encoding). Not checked in this report.
- Fix direction: audit every transformation that reads y before the family test.

### F07 - Overall level is a union over gates
- **Disposition**: OPEN
- Probability of any noise selection is bounded by the sum of levels of every gate that can admit a feature: pair family (per pair, Bonferroni over prospective pairs), table pre-filter, raw screen floor (`_screen_predictors_prescreen.py`), fallback family (`_fallback_maxt.py`, FAMILY_ALPHA = 0.01), modular family (`_periodic_fe.py:297`, default alpha 0.01). Each gate's level must be listed; the guarantee is their sum, not a single alpha.
- The honest statement today: the pair gate is near 0.003 at a stated 0.0025 and near 0.001 at a stated 1e-4; the whole-run guarantee is not established until F01 and the other gates are fixed.

### F08 - Regression test for the level
- **Disposition**: OPEN
- Add a test: under H0 (permuted labels, known null), the pass frequency of the family gate at alpha must not exceed alpha by more than the Monte Carlo error at the chosen repetition count. Start with alpha = 0.05 (cheap) and extend to smaller alpha as the stopping rule lands.

## Recommendation (from the critique)
1. Replace the quantile rule with the exact p-value (F01), and require `B + 1 >= 1/alpha` (F02).
2. Add sequential stopping for the decision (F03) and the analytic tail only for certifying strong signals, marked as approximate (F02).
3. Audit y-dependence of every configuration (F06).
4. Document and sum the gate levels (F07), and treat the table floor as a pre-filter only (F05).
5. Add the level regression test (F08).

# MRMR core selection algorithm — correctness critique (READ-ONLY)

Repo: `master_wt`. Scope: relevance/redundancy trade-off, null-mean debiasing,
significance gating, redundancy aggregation, stopping/fallback.

## Findings (own deep-dive: significance/debiasing)

### F1 [P1] GPU relevance null uses 2 perms, CPU uses 32 → hardware-dependent selection
- `gpu.py:974-1006` (loop `for _i in range(npermutations)`), called from `evaluation.py:530-550`.
- The CPU `mi_direct(return_null_mean=True)` path internally bumps the null budget to
  `_null_nperms = max(npermutations, _NULL_MEAN_MIN_PERMS=32)` (`permutation.py:642`), so the
  significance p-value has resolution ~1/33 and a genuine feature reads p≈0 < alpha (0.05),
  KEEPING its full MI. The GPU `mi_direct_gpu` path does NOT bump: it loops exactly
  `npermutations = _bnp = max(2, baseline_npermutations)` = **2** shuffles and returns
  `_p_value = _perm_pvalue(nfailed, _nchecked<=2, full_budget=2)` whose MINIMUM attainable value is
  `(1+0)/(2+1) = 0.333`.
- Consequence: on GPU hosts `p_value >= alpha` is ALWAYS true, so `direct_gain = max(0, direct_gain - null_mean)`
  is applied to EVERY feature including strong genuine signal, AND `null_mean` itself is a noisy 2-sample
  estimate. This is exactly the CPU/GPU divergence the comment at `evaluation.py:546-548` claims to have fixed
  but did not — GPU-present hosts select differently from CPU-only hosts, and strong signals are wrongly demoted.
- Failure scenario: dataset with one strong low-cardinality signal + high-card noise; CPU keeps signal MI intact
  (p≈0), GPU subtracts its null mean, so the signal's relevance drops and, near the `min_relevance_gain`/relative
  floor, it can be dropped or reordered below noise. Selection is non-reproducible across machines.
- Fix: in the `return_null_mean` branch of `mi_direct_gpu` compute the null over
  `max(npermutations, _NULL_MEAN_MIN_PERMS)` shuffles (mirror `permutation.py:642`), and pass that same
  budget as `full_budget` to `_perm_pvalue`. Add a CPU-vs-GPU parity regression test pinning identical
  `null_mean`/`p_value`/selection on a fixed fixture.

### Verified-correct (no defect) during the pass
- Add-one MC p-value `(1+nfailed)/(denom+1)` with budget-consistent `denom` — correct (`permutation.py:62-74`).
- Significance sense is consistent everywhere: significant = `p < alpha` (keep full MI); `p >= alpha` → subtract
  null mean. Raw-rescue gates (`_fit_impl_core.py:7255,8065-8075,9045`) all use `< alpha` for "keep". Consistent.
- MI kernels and the `H(y)` relevance floor both use natural log (nats) — units match (`_batch_kernels.py`,
  `_fit_impl_core.py:6475`). No base mismatch.
- CPU null bump to 32 perms is correct and the `min_nonzero_confidence=0.0` rate-floor preserves the
  Wave-9.1 unanimous-rejection semantics (`permutation.py:630-665`).

## Redundancy-aggregation findings (verified)

### F2 [P1 when jmim active] JMIM confirmation uses the wrong (CMIM) statistic
- `fleuret.py:257-278`: `evaluate_gain(...)` is called with `use_su=use_su` but NO `use_jmim=`, so it
  defaults False. Under `redundancy_aggregator='jmim'/'auto'`, candidates are SCORED by JMIM joint-MI but
  CONFIRMED against the CMIM conditional-MI null → statistic mismatch; JMIM synergy picks can be spuriously
  rejected/admitted. Verified: `get_fleuret_criteria_confidence` has no `use_jmim` param at all.
- Fix: thread `use_jmim` (+ jmim cache/counter) through
  `get_fleuret_criteria_confidence_parallel → parallel_fleuret → get_fleuret_criteria_confidence → evaluate_gain`,
  reading `use_jmim_aggregator()` at the Python boundary (mirror how `use_su` is threaded).

### F3 [P2, jmim opt-in] Fleuret `**(nexisting+1)` exponent wrongly applied to JMIM joint-MI
- `evaluation.py:363-364`: `additional_knowledge = additional_knowledge ** (nexisting+1)` on the JMIM branch.
  Joint MI `I({X,Z};Y)` routinely exceeds 1 bit, so the exponent AMPLIFIES it (Fleuret intent is to shrink
  values <1 as a redundancy penalty). Bennasar-2015 JMIM is a pure `min` of joint MI with no exponent; the
  amplification is nonlinear across candidates with different `nexisting`, distorting the min-ranking.
- Fix: skip the exponent on the `use_jmim` branch; keep it only on CMI/SU branches.

### F4 [P2, off by default] positive_mode/extra_knowledge breaks partial_gains monotonicity
- `evaluation.py:411-431` + `618-628`: with `extra_knowledge_multipler>0` the objective flips to MAX-over-Z
  (non-monotone as selected_vars grows), but the early-exit (`_stored_gain<=best_gain → return`) and the
  partial_gains resume assume monotone-decreasing scores, and `positive_mode` is reset each entry (`:255`).
  A candidate whose best synergy is with a newly-added var can be pruned or silently switched max→min across
  the step boundary. Default −1.0 (off) → P2.
- Fix: when `extra_knowledge_multipler>0` disable the partial_gains early-exit/resume (re-evaluate from
  scratch) and persist `positive_mode` alongside the partial gain.

### F5 [Low] redundancy_aggregator typos silently degrade to Fleuret
- `_mrmr_class.py:~3355`: `_jmim_on = (agg == "jmim")`; any other string (`'JMIM'`, `'jimm'`) silently means
  plain Fleuret with no error. Fix: validate `redundancy_aggregator in {None,'jmim','auto'}` and raise.

Verified NON-defects: greedy `min_Z` aggregation and relevance-minus-redundancy sign are correct; order≥2
stale-resume is correctly guarded; JMIM cache key uses `"|"` separator (no multiset collision).

Improvements (redundancy): (a) biz_value test pinning JMIM-confidence uses the JMIM statistic; (b) share
`cached_jmim_MIs` across greedy rounds (currently self-provisioned per-call → within-call memo only);
(c) materialise X/y/selected columns once per candidate instead of re-`_materialize_var` in each of the
4 post-Fleuret bonus branches (BUR/RelaxMRMR/PID/CMI-stop).

## Stopping / fallback findings

(pending agent)

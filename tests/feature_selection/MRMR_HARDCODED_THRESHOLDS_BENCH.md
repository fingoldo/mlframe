# MRMR hardcoded-threshold CONVERSION benchmark (results)

Companion to `MRMR_HARDCODED_THRESHOLDS_AUDIT.md`. For each audited threshold we measure whether a
data-derived value would beat the hardcoded default, via the sensitivity harness
`_bench_hardcoded_thresholds.py`: sweep the value across a plausible band on a grid of
magnitude-carrying synthetic archetypes (F2 heavy-tail ratio+trig, bilinear `a*b`, additive-only)
and read the downstream LINEAR test MAE of the MRMR-selected/transformed feature space.

**Verdict rule.** *FLAT* = test MAE insensitive to the value across the band on every archetype AND
the per-archetype argmin value does not move -> converting to a data-derived value buys nothing; KEEP
the hardcoded default. *DATA-DEPENDENT* = the optimal value shifts with the archetype (a wrong fixed
value costs MAE) -> a real conversion candidate; implement the audit's permutation-null / CV
replacement and bench the derived value vs the fixed default.

> Reliability caveat: the harness fits a full MRMR+FE pipeline per cell, so it runs at small n
> (n=2500, single seed) for tractability. The F2 archetype's heavy-tail `a**2/b` term needs larger n
> for a stable linear fit, so the F2 cells are NOISY; bilinear / additive cells are stable. A
> DATA-DEPENDENT verdict driven ONLY by an F2 swing is treated as UNCONFIRMED pending a larger-n
> multi-seed re-run before any conversion is implemented.

## HIGH-priority (measured 2026-06-13)

| Threshold | Default | Verdict | Evidence (linear test MAE across the swept band) | Disposition |
|-----------|---------|---------|--------------------------------------------------|-------------|
| `fe_min_pair_mi_prevalence` | 1.05 | DATA-DEPENDENT | bilinear `a*b`: 1.0->0.207, 1.05->0.207, **1.2->0.092** (a higher gate drops noise pairs); F2: 1.05->0.917, 1.2->1.079 (higher admits cross-mix, hurts). Optimal moves 1.05<->1.2 with archetype. | CONVERT candidate -- permutation-null debiased floor (shuffle y, 95th-pct null prevalence ratio). Confirm at larger n first. |
| `fe_synergy_min_prevalence` | 1.5 | DATA-DEPENDENT | bilinear: 1.2->0.207, 1.5->0.207, **2.0->0.051** (the genuine `a*b` synergy needs a HIGHER gate to clear noise here); F2/additive flat. Default 1.5 is sub-optimal on bilinear. | CONVERT candidate -- CV permutation null on the synergy prevalence ratio. Confirm at larger n first. |
| `fe_escalation_pairness_margin` | 1.15 | DATA-DEPENDENT (max spread 5.7%) | F2: 1.05->0.917, 1.15->0.917, **1.3->0.970** -- an over-tight margin costs 5.7% MAE. 1.15 is optimal HERE, but a 5.7% mis-tuning penalty means the gate genuinely binds, so on data whose optimum sits away from 1.15 the fixed value would pay that 5.7%. | CONVERT candidate -- a fold-adaptive null margin (95th-pct pair/single ratio under shuffled y) that tracks the per-data optimum. 5.7% is a worthwhile win, NOT noise to dismiss. |
| `fe_escalation_underdelivery_self_ratio` | 3.0 | **FLAT** (0.000% spread) | identical MAE at 2.0 / 3.0 / 4.5 on all three archetypes. | KEEP -- documented no-win; the rescue-escalation gate does not bind on these targets. |

(`_FE_MARGINAL_UPLIFT_MIN_RATIO=1.30`, the 5th HIGH item, is a module constant not a constructor
param; benched separately via monkeypatch -- see below.)

## MEDIUM / LOW priority (constructor-param items, measured 2026-06-13, n=2500)

| Threshold | Default | Verdict | Evidence / note | Disposition |
|-----------|---------|---------|-----------------|-------------|
| `fe_stability_vote_k` | 5 | DATA-DEPENDENT (83.9%) | F2: **k=3->1.985**, k=5->0.917, k=7->0.917 -- LOWERING k drops the good feature; k>=5 is fine. The audit's `min(5, max(2, n//100))` keeps 5 for n>=500, so it ONLY reduces k for tiny n -- a naive reduce-k would DEGRADE, the guarded formula is safe. | CONVERT (guarded) -- adaptive K with an n-floor; degenerates to 5 on normal n. |
| `fe_engineered_cmi_retain_frac` | 0.15 | FLAT (0%) | identical MAE at 0.08/0.15/0.25 on all archetypes. | KEEP (or guarded-hybrid on principle -- cannot degrade). |
| `fe_sufficient_summary_maxt_quantile` | 0.95 | FLAT (0%) | identical across 0.90/0.95/0.99. | KEEP. |
| `fe_sufficient_summary_residual_frac` | 0.25 | FLAT (0%) | identical across 0.15/0.25/0.40. | KEEP. |
| `fe_rung_rel_floor` | 0.40 | FLAT (0%) | identical across 0.25/0.40/0.55. | KEEP (budget knob, not admission). |
| `fe_stability_vote_quorum` | 0.6 | FLAT (0%) | identical across 0.5/0.6/0.75. | KEEP. |
| `fe_escalation_min_val_corr` | 0.15 | FLAT (0%) | identical across 0.08/0.15/0.25. | KEEP. |
| `min_relevance_gain_relative_to_first` | 0.05 | FLAT (0.008%) | argmin tiebreak between equal MAEs (0.008% spread = noise); not a real dependence. | KEEP. |
| `min_relevance_gain_frac` | 0.001 | FLAT (0%) | identical; audit notes it is a safety bound, not an active gate. | KEEP. |
| `fe_confirm_undersample_rows_per_cell` | 5.0 | FLAT (0%) | identical across 3.0/5.0/8.0. | KEEP. |
| `fe_pair_perm_null_excess_frac` | 0.05 | FLAT (0%) | identical across 0.02/0.05/0.10 (re-run, no OOM). | KEEP. |
| `fe_min_nonzero_confidence` | 0.99 | FLAT (0%) | identical across 0.95/0.99/0.999; audit rates "keep as-is". | KEEP. |
| `fe_min_pair_mi` | 0.001 | FLAT (0%) | identical across 0.0005/0.001/0.005. | KEEP. |
| `fe_good_to_best_feature_mi_threshold` | 0.98 | FLAT (0%) | identical across 0.90/0.98/0.999. | KEEP. |
| `fe_adaptive_relax_factor` | 0.9 | FLAT (0%) | identical across 0.8/0.9/0.95. | KEEP. |

**All 20 constructor-param thresholds now measured.** Summary: 4 conversion candidates
(`fe_min_pair_mi_prevalence`, `fe_synergy_min_prevalence`, `fe_escalation_pairness_margin`,
`fe_stability_vote_k` -- the last SHIPPED as guarded `"auto"`), `fe_escalation_underdelivery_self_ratio`
+ 14 others FLAT (KEEP / migrate as no-op guarded hybrids on principle). The strong FLAT majority says
most hardcoded MRMR constants are well-placed (or do not bind on typical data) -- converting them would
add estimation variance with no bias to remove, so on principle they migrate to guarded hybrids that
provably degenerate to the constant, never as naive replacements.

## Module-constant items (need monkeypatch, not ctor-param sweep) -- pending

`_FE_MARGINAL_UPLIFT_MIN_RATIO` (1.30), `_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO` (0.82),
`_FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO` (0.84) in `_pairs_gates.py`; `RAW_SELF_RETAIN_FRAC` (0.05)
in `_fe_raw_redundancy_drop.py`; `_HINGE_MIN_HELDOUT_R2_UPLIFT` (0.02), `_HINGE_CAND_Q_LO/HI`
(0.10/0.90), `_HINGE_PRECHECK_MIN_SSE_DROP` (0.005) in `_hinge_basis_fe.py`.

## Design principle for conversions (the bias-variance guard)

A hardcoded threshold has ZERO variance but bias; a permutation-null adaptive value has low bias but
ADDS estimation variance (finite sample + finite shuffles). On in-distribution data the fixed value's
zero variance can BEAT a noisy adaptive estimate (pure downside), and on under-sampled nulls the
adaptive value is unreliable -- so a naive "replace constant with raw null quantile" WILL degrade some
cases (the FLAT gates especially: no bias to remove, only variance to add; and `fe_stability_vote_k`
shows lowering the gate degrades F2). Therefore every conversion ships as a GUARDED HYBRID:
1. compute the adaptive value, but FALL BACK to the proven constant when n / rows-per-cell is below a
   reliability floor (mirror `fe_confirm_undersample_rows_per_cell`);
2. CLAMP / blend the adaptive value within a band anchored on the constant so a noisy estimate cannot
   stray far.
This makes the conversion degenerate to today's behaviour on the typical case (cannot degrade) and
only bite when the data clearly warrants it -- which is also how the FLAT-but-"convert-on-principle"
gates can be migrated safely. Each conversion is benched BOTH directions (does it hurt in-distribution
+ help out-of-distribution) before landing.

## Conversion candidates (to implement as guarded hybrids)

1. `fe_min_pair_mi_prevalence` (HIGH) -- permutation-null debiased prevalence floor.
2. `fe_synergy_min_prevalence` (HIGH) -- CV permutation null on the synergy ratio.
3. `fe_escalation_pairness_margin` (HIGH, 5.7%) -- fold-adaptive null margin.
4. `fe_stability_vote_k` (MED) -- DONE: `resolve_adaptive_vote_k` (`_fe_stability_vote.py`) accepts
   `"auto"` = n-floored guarded K (== 5 for n>=500, downward only for tiny n; explicit int incl. the
   default 5 honoured verbatim -> byte-identical). Opt-in until a tiny-n bench confirms default-on.
   Test `test_adaptive_vote_k.py` (31 cells: byte-identity of explicit int + the guarded "auto" band).

All other measured gates are FLAT -> migrate as no-op guarded hybrids on principle (cannot degrade)
or keep the constant.

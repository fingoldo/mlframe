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
| `fe_escalation_pairness_margin` | 1.15 | weakly sensitive (max spread 5.7%) | F2 only: 1.05->0.917, 1.15->0.917, 1.3->0.970 (default == optimal; only an over-tight 1.3 mildly hurts). bilinear/additive flat. | KEEP -- default 1.15 already at the per-archetype optimum; conversion would not beat it. |
| `fe_escalation_underdelivery_self_ratio` | 3.0 | **FLAT** (0.000% spread) | identical MAE at 2.0 / 3.0 / 4.5 on all three archetypes. | KEEP -- documented no-win; the rescue-escalation gate does not bind on these targets. |

(`_FE_MARGINAL_UPLIFT_MIN_RATIO=1.30`, the 5th HIGH item, is a module constant not a constructor
param; benched separately via monkeypatch -- see below.)

## MEDIUM / LOW priority

Pending -- swept by the same harness (constructor-param thresholds) + monkeypatch for module
constants. Recorded here as each completes.

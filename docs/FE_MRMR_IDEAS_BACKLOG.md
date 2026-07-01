# MRMR feature-engineering — status note (shipped / rejected / open)

This was a 22-idea ideation backlog from a 4-agent FE sweep. It is now ~86%
resolved: every shipped item carries an inline module reference and every
rejected item carries a `# bench-attempt-rejected` note at its call site, so the
full rationale lives in the source and in git history. This file is trimmed to
the residual — the three ideas that are neither shipped nor rejected — plus a
short ledger of what landed, so nobody re-treads resolved ground.

> **Guiding lesson (still applies):** across the rejected ideas the binding
> constraint was always the **admission / selection machinery**, not the richness
> of feature construction. The highest-confidence wins let a *genuinely-good
> feature pass a gate it currently fails*. Anything that lowers a bar must keep
> the **order-2 maxT permutation-null floor** as the outer guard and must debias
> **both** sides of any ratio consistently.

## Open items (NOT started)

- **#5 — Permutation-null-calibrated prevalence bar.** Replace the fixed
  engineered-MI-prevalence threshold with a per-run permutation-null-calibrated
  bar so the admission cutoff adapts to the data's null MI level. No module yet.
- **#10 — CMI complementarity Apriori lattice.** Bottom-up Apriori-style lattice
  over candidate interactions gated by conditional-MI complementarity. Distinct
  from the shipped greedy-CMI path (`_mi_greedy_cmi_fe.py`,
  `_orthogonal_cmim_fe.py`); the lattice enumeration is not built.
- **#19 — KSG continuous-MI gate tie-breaker.** Use the KSG estimator
  (`_ksg.py`, already present as a general estimator) as a tie-breaker inside the
  FE gate's 0.90–0.97 retained-MI band. The general estimator exists; the gate
  wiring does not.

## Shipped this session (default-on or opt-in)

Order-2 Westfall-Young maxT permutation-null floor (`_permutation_null.py`),
per-gate rejection ledger (`_fe_rejection_ledger.py`), occupied-K / Miller-Madow
bias correction (`_feature_engineering_pairs/`), interaction-information ranking +
routing (`_interaction_information.py`, opt-in), hinge / change-point basis
(`_hinge_basis_fe.py`, opt-in), conditional-dispersion features
(`_extra_fe_families_dispersion.py`), Haar wavelet basis (`_wavelet_basis_fe.py`,
`_wavelet_basis_fe_batched.py`), cross-fold recipe stability voting
(`_fe_stability_vote.py`), successive-halving rung schedule
(`_fe_rung_schedule.py`), robust heavy-tail 1-D prewarp (`hermite_fe/`),
gradient-interaction seeder (`_gradient_interaction_seeder.py`, opt-in),
sufficient-summary early-stop (`_fe_sufficient_summary.py`).

## Benchmarked + rejected (rationale in-code)

Trimmed-abs-floor, surrogate-GBM co-occurrence seeder
(`_surrogate_interaction_seeder.py`), order-3 maxT floor (blocked on the seeder),
RFF interaction pre-screen, isotonic reshaping (subsumed by spline),
confidence-weighted MI under imbalance (`_orthogonal_univariate_fe/`), cross-fit
warm-start prior. Each carries a `bench-attempt-rejected` note at its site so the
negative result isn't re-run.

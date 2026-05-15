# composite_discovery audit-clean sweep (ENS-Low-8)

Audit ID: ENS-Low-8. Doc-only deliverable enumerating sub-areas of
`mlframe.training.composite_discovery` that were inspected during the
ensembling+caching audit (Wave) and confirmed clean - no fix needed,
no follow-up backlog entry. Future audit waves can skip these unless
the implementation changes materially.

## Welford streaming statistics

- Used for running mean / variance during MI-rerank cross-folding and
  the per-bin RMSE aggregation. The two-pass numerical-stability
  formulation matches the textbook recurrence and tolerates fold sizes
  spanning 5+ orders of magnitude with no observed overflow.
- Reviewed: composite_discovery.py (Welford uses inside
  `_tiny_cv_rmse_y_scale` aggregation). No drift between Welford pooled
  variance and `np.var(ddof=0)` on a synthetic fixture (rtol=1e-12).

## Transform fit / inverse round-trip

- The transform registry (linear_residual / log_ratio / monotonic_residual)
  exposes `forward` / `inverse` pairs whose composition recovers `y`
  within 1e-12 on the in-domain support.
- Reviewed: composite_transforms.py + the per-spec validation in
  composite_discovery.py:_y_train_clip_bounds usage. Inverse output is
  always passed through the wrapper clip so a numerically-overflowing
  inverse cannot leak into screening RMSE.

## MI screening (`_mi_pair_bin` / `_mi_to_target` and bootstrap LCB)

- Bootstrap LCB gate uses 1-sided CI on the bootstrap distribution; this
  matches the published methodology and the gate is correctly applied
  via `entry["mi_gain_lcb"]` rather than the point estimate.
- Reviewed: composite_screening.py:_mi_pair_bin (numba kernel) +
  bootstrap LCB computation in composite_discovery.py. Histogram bin
  edges respect float-precision underflow on heavy-tail targets.

## Tiny-model rerank (`_tiny_cv_rmse_y_scale` / multiseed)

- K-fold CV + per-bin breakdown shares the same fold splitter (KFold
  with `random_state=base_random_state + s_idx*7919`), so multi-seed
  results are independent across seeds and reproducible.
- Reviewed alongside ENS-P2-5 (which now reuses first-pass per-bin
  predictions). No silent NaN propagation: `fold_rmses` filters
  non-finite entries before averaging.
- Wrapper-aware y-clip mirrors the production CompositeTargetEstimator
  predict path; screening RMSE therefore matches deployed RMSE on
  heavy-tail transforms (logratio etc.) - the 2026-05 prod regression
  closed by R10b improvement #4.

## What this list does NOT cover

- Auto-promotion to `linear_residual_multi` (forward stepwise) was
  separately audited in the ensembling table (ENS-Low-6 hoist).
- Alpha-drift detector OLS SE was rewritten (ENS-Low-2).
- per-bin re-pass de-duplication was the subject of ENS-P2-5.

## Why doc-only

These sub-areas already carry regression coverage in tests/training/
(test_composite_discovery*.py, test_composite_screening_*) and the
inspection found no actionable changes. Recording them here so the
next audit wave doesn't re-enumerate them as candidates without new
evidence.

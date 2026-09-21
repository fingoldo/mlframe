# training-log audit 2026-09-20 -- master tracker

Findings derived from a full production training run of the suite (`jobsdetails_shuffled`, 576_646x127, 14 targets,
CatBoost-only on GPU, 12:56 -> 14:05). The log is the evidence: every row below cites the log line that exposes it.
Reports: [sensors_and_gates.md](sensors_and_gates.md) (drift / temporal / collapse / ceiling sensors and the gates that
never gate), [training_loop.md](training_loop.md) (early stopping, loss recommendation, prediction path),
[discovery_economics.md](discovery_economics.md) (composite discovery cost vs delivered value),
[waste_and_reporting.md](waste_and_reporting.md) (recomputation, misleading log lines, metric instability).

Statuses: **RESOLVED** (fixed in code; the note names the test that pins it), **PARTIAL** (part fixed; the note says what
remains), **TODO** (open), **REJECTED** (measured and declined; evidence in the note), **NOT A DEFECT** (the claimed
behaviour does not occur, or is outside mlframe; the note says where it lives).

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT |
|---|---|---|---|---|---|---|
| `sensors_and_gates.md` | 12 | 12 | 0 | 0 | 0 | 0 |
| `training_loop.md` | 7 | 7 | 0 | 0 | 0 | 0 |
| `discovery_economics.md` | 6 | 6 | 0 | 0 | 0 | 0 |
| `waste_and_reporting.md` | 8 | 5 | 2 | 0 | 1 | 0 |
| **Total** | **33** | **30** | **2** | **0** | **1** | **0** |

The two PARTIAL rows are partial for the same reason: each bundled two claims, one real and one that did not survive
verification. `WST-01`'s PSI half is fixed and its adversarial half was never a defect (already cached on master --
the log shows 4 dataset builds in total, not 14x4); `WST-07`'s stale-poll half is fixed and its ETA half was never a
defect (the number printed is CatBoost's own ETA from its progress file, not an mlframe extrapolation). Both
corrections are written into the finding bodies rather than quietly dropped.

Regression tests added: `tests/training/test_target_maturity_and_temporal_scale.py` (12),
`tests/training/test_drift_report_heavy_tail_regression.py` (6),
`tests/training/test_saturated_early_stopping_refit.py` (13),
`tests/training/test_per_group_baseline_shrinkage.py` (7),
`tests/training/composite/test_discovery_gates_audit_2026_09_20.py` (13), `tests/training/test_threshold_optimizer_silent_noop.py` (17), 68 in total. Final sweep over the touched
areas (the five new files plus `tests/training/baselines`, `tests/training/targets`, `tests/reporting/test_drift_report.py`,
`tests/training/test_metrics_registry_classification.py`, `tests/metrics/classification`): 657 passed, 1 skipped, and one
unrelated failure -- `test_ks_fused_gate_perf_sentinel`, a 2%-margin microbenchmark over `_classification_extras.py`,
which is byte-identical to HEAD; it measured 0.98x on a host with 3.8/15.9 GB RAM free, in the same run where the
conftest's own contention detector skipped a sibling perf test.

### `sensors_and_gates.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P0 | `SEN-01` | The temporal audit compared an ABSOLUTE segment-rate spread against a RELATIVE threshold (`0.10`), so the "target rate is NOT stable" WARN fired for every target whose scale exceeded 0.1 | probability-valued rates keep the percentage-point rule, unbounded ones compare the spread against the segment level, and the message names which scale it used (`_target_temporal_audit_from_agg._spread_exceeds_threshold`; `test_target_maturity_and_temporal_scale.py::test_temporal_audit_does_not_warn_on_small_relative_spread_at_large_scale`) |
| **RESOLVED** | P0 | `SEN-02` | Regression label drift was tested only as a mean shift in TRAIN-SIGMA units, so a heavy-tailed target could lose 59% of its mean and report "no drift warnings" | added scale-free level / dispersion / upper-tail ratio checks beside the sigma test, each gated on the train reference being resolvable above its own standard error (`drift_report._regression_shape_warnings`; `test_drift_report_heavy_tail_regression.py`) |
| **RESOLVED** | P1 | `SEN-03` | No sensor detected target RIGHT-CENSORING, the mechanism behind SEN-02's numbers and the negative test R2 | new `targets/target_maturity_audit.py` separates censoring from regime change by where the per-bin statistic extrapolates at a zero observation window, and runs free on the bins the temporal audit already computed (`test_target_maturity_and_temporal_scale.py::test_maturity_audit_*`) |
| **RESOLVED** | P1 | `SEN-04` | The temporal audit dropped thin bins before segmenting, and on this run the dropped bins were the whole test window, so the stability verdict described a window excluding the evaluated period | the dropped-bin note names the bins, states the window the audit actually covers, and flags trailing drops explicitly (`test_target_maturity_and_temporal_scale.py::test_dropped_bin_warning_names_the_bins_and_the_covered_window`) |
| **RESOLVED** | P1 | `SEN-05` | The achievable-ceiling precheck conflated "no base candidate was evaluated" with "every evaluated candidate collapsed"; both left `best_composite_rmse=inf` and took the same `proceed` branch | evaluated / collapsed counts are carried into the verdict and the log line, and an all-collapsed measurement now reaches the existing strong-floor rule instead of being reported as an absence of measurement (`_achievable_ceiling.measure_achievable_ceiling`) |
| **RESOLVED** | P1 | `SEN-06` | `honest_rmse_gate` admitted specs up to `tol=1.05` WORSE than raw, then reported "all N spec(s) passed", which read as an endorsement | the summary splits survivors into beat-raw, beat-raw-significantly (paired SE) and within-tolerance-but-worse (`_honest_rmse_gate.apply_honest_rmse_gate`) |
| **RESOLVED** | P2 | `SEN-07` | The collapse sensor and the envelope clip both fired on a model that was then saved and reported with no aggregated verdict | new `_reporting_regression/_sensor_ledger.py` records every trip; `save_mlframe_model` warns at the point of persistence naming each one; cleared at suite entry |
| **RESOLVED** | P2 | `SEN-08` | `per_group_prior` / `per_group_mean` used an UNSMOOTHED group mean, so a singleton group emitted p=0 or p=1 and any unbounded proper scoring rule exploded on it | size-weighted shrinkage toward the global mean with an empirical-Bayes strength (`sigma_within^2 / sigma_between^2`), so weak-signal groups shrink hard and strong-signal ones keep their estimate (`_dummy_baseline_compute._empirical_bayes_pseudocounts`; `test_per_group_baseline_shrinkage.py`) |
| **RESOLVED** | P2 | `SEN-09` | The suite reported calibration metrics while allocating no calibration split, and never recalibrated after a detected prior shift | the prior-shift warning now names `TrainingSplitConfig.calib_size` as the remedy at the point of detection (`drift_report.compute_label_distribution_drift`) |
| **RESOLVED** | P2 | `SEN-10` | Decision-threshold optimisation is ON by default but fits only on a calib slice, so with `calib: 0` it found nothing and returned in total silence -- the run's only well-separated model kept the 0.5 cut at recall 0.39/0.48 under a flagged +6.6pp prior shift | reported by the aggregated calib-skip line (see `SEN-11`); the docstring claiming "default OFF" is corrected to match the `True` default (`_phase_finalize_calibration._optimize_decision_threshold_on_calib_slice`; `test_threshold_optimizer_silent_noop.py`) |
| **RESOLVED** | P2 | `SEN-11` | SEN-10 was one instance of a class: every calib-slice finalize step (probability calibration, isotonic risk check, threshold optimisation, conformal sets) reports only on success, so with no slice they all fell silent together; four comments/docstrings also called ON-by-default steps "Default OFF" | new `core/_phase_finalize_calib_skipped.py` emits ONE warning naming every enabled step the missing slice cost, mirroring each step's exact gate so a silence with a different cause is never blamed on the slice; per-step copy of the SEN-10 warning removed; stale default claims corrected (`test_threshold_optimizer_silent_noop.py`) |
| **RESOLVED** | P2 | `SEN-12` | `apply_confidence_shrinkage` (and the leaderboard diversity recommendations) are ON by default but need OOF predictions, which the default `oof_n_splits=0` never produces -- at defaults they can never run, and say nothing | an INFO line (not a WARN: the codebase deliberately treats `oof_n_splits` as the caller's cost choice, and that decision is kept) names the inert steps and the knob; the shrinkage docstring now states the dependency (`_phase_finalize_calib_skipped.report_oof_dependent_steps_inert`; `test_threshold_optimizer_silent_noop.py`) |

### `training_loop.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P0 | `TRN-01` | Aligning `eval_metric` to a ROBUST objective made early stopping blind to tail blow-up: ES tracked Huber for all 1000 iterations while val R2 reached -4.61 | the aligned surface is kept (it fixes the opposite bug) and the missing half is added -- see `TRN-02` |
| **RESOLVED** | P0 | `TRN-02` | There was a refit safety net for `best_iter` far too LOW but none for `best_iter` saturated at the cap, the other half of the same failure class | `_maybe_refit_on_saturated_best_iter` refits on the RMSE family when a robust loss ran to the budget AND lost to a constant predictor on the early-stopping rows (`_training_loop_refit.py`; `test_saturated_early_stopping_refit.py`, 13 tests covering every gate) |
| **RESOLVED** | P1 | `TRN-03` | Every CatBoost model was pre-flagged `_mlframe_polars_fastpath_broken = True` unconditionally, so working builds paid a polars->pandas conversion on every predict | the flag is driven by `catboost_polars_fastpath_broken()`, which asks the installed build the question the pessimisation assumed the answer to; on this machine it answers "accepts polars", removing all 52 conversions |
| **RESOLVED** | P1 | `TRN-04` | The sticky-fastpath log line asserted a history ("flagged as having missed the fastpath earlier") that never happened when the flag came from the blanket pre-set | the two provenances are tracked separately (`_mlframe_polars_fastpath_miss_observed`) and the message says which applies |
| **RESOLVED** | P1 | `TRN-05` | `_apply_loss_recommendation_in_place` received knob overrides for backends absent from the run and none for the backend that was present | the heavy-tail branch emits `cb_kwargs` alongside the other three (`_target_distribution_analyzer_target_fn`; `test_target_distribution_analyzer.py::test_heavy_tail_recommends_huber`) |
| **RESOLVED** | P2 | `TRN-06` | `residual_audit` recommended Huber on targets whose kurtosis `loss_recommendation` uses to REJECT Huber | one ladder across all three components: past `EXCESS_KURT_HUBER_FAILS` the audit's advice switches to RMSE and the distribution analyzer emits no robust-loss override at all (`test_target_distribution_analyzer.py::test_extreme_kurtosis_recommends_no_robust_loss`) |
| **RESOLVED** | P2 | `TRN-07` | `emit_per_model_composite_y_scale_test` swallowed CatBoost's polars `TypeError` instead of taking the pandas retry `_cb_pool` already implements, losing the y-scale emit for six composites | `_composite_predict` routes the wrapper's predict through the same recovery (`_phase_composite_wrapping.py`) |

### `discovery_economics.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P1 | `DSC-01` | `min_honest_gain_to_train=0.001` was an absolute constant with no relation to the sampling noise of the gain it filtered | the floor is `max(constant, z * se)` with the spec's own paired standard error and a new `min_honest_gain_z=2.0` (`_phase_composite_discovery`; `test_discovery_gates_audit_2026_09_20.py`) |
| **RESOLVED** | P1 | `DSC-02` | The honest-RMSE gate discarded the raw baseline's prediction vector, so no paired significance test was possible downstream | the vector is retained and `_paired_rmse_gain_se` stamps `honest_holdout_rmse_gain_se` on each spec, O(n) with no extra fits |
| **RESOLVED** | P1 | `DSC-03` | Specs were selected on differences smaller than the GPU non-determinism the suite warned about in the same breath | closed by DSC-01 + DSC-02: the selection now has a measured noise scale |
| **RESOLVED** | P2 | `DSC-04` | Zero-inflated targets were routed to curved y-compressors whose inverse cannot reproduce the spread, caught only AFTER a full fit | new `discovery/_point_mass_gate.py` skips the curved family above a dominant point mass, mirroring the existing left-skew gate; clipping transforms (which kept a piecewise-linear inverse and were the one production composite that beat raw) are untouched |
| **RESOLVED** | P2 | `DSC-05` | Composite discovery ran BEFORE `BaselineDiagnostics` reported `composite_recommendation`, so its own "unlikely_to_help" verdict could not gate it | the verdict is read at the decision point from the same precompute the hint comes from; `unlikely_to_help` ("no dominant features") narrows the search to the base-free unary family it does not bear on, 24 transforms -> 5 (`_maybe_narrow_to_unary_transforms`) |
| **RESOLVED** | P2 | `DSC-06` | `_run_interaction_bases` did not pass `train_mask` although every row it supplies is already train-only, producing a leak WARNING for a leak that cannot occur | an explicit all-train mask makes the invariant checkable instead of assumed (`_opt_in_steps._run_interaction_bases`) |

### `waste_and_reporting.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **PARTIAL** | P1 | `WST-01` | Target-independent drift work repeated once per target | PSI: **RESOLVED** -- `_psi_cache_key` + `_PSI_CACHE` mirror the adversarial panel's existing cache (`diagnostics_dispatch`). Adversarial validation: **NOT A DEFECT** -- `_ADVERSARIAL_CACHE` already handles it on master, and the log confirms it (4 dataset builds in total, all for the first target, not 4 per target). The first-pass claim of 14 adversarial fits was wrong |
| **RESOLVED** | P2 | `WST-02` | `_maybe_auto_drop_after_feature_analyzer` KEPT 9 NaN-heavy columns as "signal" that the suite-wide pre-screen dropped as constant 12 minutes later | the keep decision now applies the pre-screen's own null-fraction bar, so a >99%-null column is dropped at the first opportunity rather than after discovery has screened it four times (`_main_train_suite_target_distribution`) |
| **REJECTED** | P2 | `WST-03` | `feature_distribution_analyzer` reports redundant pairs that nothing applies; `_auto_base` re-derives them per target | the two are not redundant: the suite-level analyzer says WHICH pairs are correlated, while `_auto_base` additionally needs the target's own MI ranking to choose which member of a pair survives, and that is target-dependent by construction. The only duplicated work is the per-target correlation recompute, ~0.2s x 4 targets on this run. Consuming the suite-level list would change which feature survives per target -- a selection change with no measured benefit, so it is not made. The differing `|corr|` values across targets (0.975 / 0.979 / 0.985) are subsample estimates of one quantity, not a contradiction |
| **RESOLVED** | P2 | `WST-04` | The split report printed two different train/val boundaries for the same split | the pre-augmentation line is labelled as the implied layout BEFORE val augmentation and says the realised split is reported separately (`splitting.make_train_test_split`) |
| **RESOLVED** | P2 | `WST-05` | A val split augmented with random in-period rows silently disabled every time-series baseline, reported 13 times as an incidental per-target skip | throttled to one WARNING per process naming the suite-level consequence and the split knob responsible (`_dummy_baseline_regression`) |
| **RESOLVED** | P2 | `WST-06` | `quadratic_weighted_kappa` / `weighted_kappa` were reported for BINARY targets, where both degenerate to plain Cohen's kappa -- one number printed twice under two ordinal names | the ordinal pair is registered for multiclass only; binary keeps the unweighted statistic already reported as `Cohen_kappa` (`metrics_registry`; `test_metrics_registry_classification.py` reframed) |
| **PARTIAL** | P3 | `WST-07` | The CatBoost GPU monitor extrapolated its ETA from the first sample and kept polling after the fit returned | stale poll: **RESOLVED** -- `_safe_poll` re-checks the stop flag after its interval wait. ETA: **NOT A DEFECT** -- the printed `cb-ETA` is CatBoost's own estimate read from its progress file (`read_time_left_tail`), not an mlframe extrapolation; the first-sample rate is used only for the collapse check |
| **RESOLVED** | P3 | `WST-08` | The classification report printed an empty `RICEs:` section | the header is emitted only when it has contents (`_reporting_probabilistic`) |

## Out of scope (not mlframe)

| Item | Where it lives |
|---|---|
| 14-minute unlogged gap between the GPU probe (12:58:57) and suite start (13:12:52) | the caller's `new_modelling` driver script, before `train_mlframe_models_suite` is entered |
| `private commit 52.7GB` at suite entry with `RSS 0.7GB` (leaked commit from earlier work in a reused interpreter) | the caller's process lifecycle; mlframe already reports it via `COMMIT_PRESSURE` |
| Val split augmented with `+25949Rnd` in-period rows | the caller's `TrainingSplitConfig`; mlframe's consequence is tracked as `WST-05` |
| The run being killed at 14:05 with no exit line | external; mlframe's crash diagnostics correctly flagged the absence |

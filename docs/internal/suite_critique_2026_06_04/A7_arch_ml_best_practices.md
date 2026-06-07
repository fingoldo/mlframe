# A7 — `train_mlframe_models_suite` ML-best-practices / statistical-rigor audit

Date: 2026-06-04
Lens: methodology & statistical correctness ONLY (train/val/test/OOF discipline, CV nesting, calibration placement, metric honesty, seeding, imbalance/threshold, selection-bias). SW-engineering concerns out of scope (separate agent).
Method: traced actual data flow across the suite entry, split phase, per-target training body, trainer eval, calibration modules, ensemble chooser, FS config, predict path. Each finding verified at file:line by static reading.

Overall verdict: **The suite is methodologically strong and unusually self-aware about leakage.** The honest-holdout discipline is genuinely good — test/OOS is touched exactly once for metric reporting and is explicitly guarded out of every decision surface (FS, calibration, ensemble selection). The main real defects are (1) a half-wired `calib_size` knob that promises a calibration carve it never performs, (2) a time-/group-unaware OOF K-fold used for ensemble selection on time-series suites, and (3) a fixed 0.5 decision threshold that is never tuned (quality gap, not leakage). Most other items are positive confirmations or low-severity caveats.

---

## Findings

### A7-01 — `calib_size` config knob is declared, validated, documented, but never carves a calibration slice
- Severity: **P1**
- file:line: `src/mlframe/training/_preprocessing_configs.py:125` (declaration), `:183-190` (sum validator), `src/mlframe/training/core/_phase_helpers_fit_split.py:327` (comment "calib_size -- downstream post-train carve"), `src/mlframe/training/evaluation.py:169,198` (docstrings claiming it is a "reserved slice from train ... opt-in via TrainingSplitConfig.calib_size").
- What's the issue: `calib_size` is filtered OUT of the splitter kwargs (the splitter signature `make_train_test_split` at `splitting.py:79` has no `calib_size` parameter, and `_phase_helpers_fit_split.py:332-337` only passes signature-matching fields), and a repo-wide grep finds NO code path that slices a `calib_idx`/calibration subset from train using `calib_size`. The only consumers are the standalone `post_calibrate_model`/`train_postcalibrators` (evaluation.py / calibration/post.py), which are themselves never called from `core/` (confirmed: no `post_calibrate_model`/`train_postcalibrators` reference anywhere under `training/core/`). So a user who sets `calib_size=0.1` gets the sum-validator enforced (train shrinks conceptually) but receives no calibration slice and no auto-calibration — the knob is inert.
- Why it matters: a user reasonably expects `calib_size>0` to reserve a disjoint, honest calibration split and have the suite fit post-hoc calibrators on it. Silent no-op means either (a) they think their probs are calibrated when they are not, or (b) they discover post-hoc calibration must be wired by hand, contradicting the docstrings. This is a methodology trap: the config invites a correct workflow then doesn't deliver it.
- Recommendation: either (a) wire `calib_size` end-to-end — carve `calib_idx` from train in the split phase, thread it to an auto-invoked `post_calibrate_model`/`train_postcalibrators` at finalize, and assert disjointness from test (the assert machinery already exists in `post.py:480-503`); or (b) if auto-calibration is deliberately out of scope, demote `calib_size` to clearly "reserved for manual use" and stop the docstrings asserting the splitter carves it. Option (a) is the fuller fix and matches the "enable corrective mechanisms by default" project norm.
- Confidence: **High** (verified the knob has no carve site and the calibrators are not auto-invoked).

### A7-02 — OOF K-fold (`_compute_oof_preds`) is time-/group-shuffle-aware only partially: shuffled KFold used even when the main split is temporal
- Severity: **P1**
- file:line: `src/mlframe/training/trainer.py:205-275` (`_compute_oof_preds`), specifically `:247` `KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)`. Call site `_trainer_train_and_evaluate.py:696-725` passes `group_ids` but NOT `timestamps`.
- What's the issue: `_compute_oof_preds` accepts `group_ids` (→ `GroupKFold`, good) but has no `timestamps` parameter and no `TimeSeriesSplit` branch. When the suite split was temporal (`timestamps` present → `make_train_test_split` orders by time, and RFECV correctly uses `TimeSeriesSplit` per `_helpers_training_configs.py:767-768`), the OOF pass nonetheless does a **shuffled** KFold over train rows. Those OOF probs feed (a) the ensemble winner selection (`_ensemble_chooser.py:32-39` ranks on `oof.integral_error`/`oof.rmse` first) and (b) OOF-preferred calibration. On an autocorrelated / non-stationary target, shuffled-OOF leaks future-into-past within train, producing optimistic OOF estimates that bias which ensemble flavour is chosen.
- Why it matters: the suite is otherwise careful to keep temporal ordering (split, RFECV). The OOF surface used for a real decision (ensemble selection) breaks that ordering, so the "honest CV test-analog" is not actually honest under temporal drift — the exact regime where ensemble choice matters most.
- Recommendation: thread `timestamps` (or a `has_time` flag) into `_compute_oof_preds`; when temporal, use `TimeSeriesSplit` (mirroring the RFECV path). Keep shuffled KFold only for the genuinely i.i.d. case. Pin a regression test that a temporal target yields ordered OOF folds.
- Confidence: **High** (verified the KFold call, the missing timestamps param, and the OOF→ensemble-selection consumption).

### A7-03 — OOF and CalibratedClassifierCV use a hardcoded seed (42 / fixed folds) rather than the threaded `split_config.random_seed`
- Severity: **P2**
- file:line: `_trainer_train_and_evaluate.py:708` (`random_seed=42` passed into `_compute_oof_preds`), `trainer.py:247` (`random_state=random_seed`→42), `models.py:412` (`CalibratedClassifierCV(model, cv=DEFAULT_CALIBRATION_CV_FOLDS, method="isotonic")` — internal KFold seed left to sklearn default).
- What's the issue: the suite threads `split_config.random_seed` (default `DEFAULT_RANDOM_SEED`, `_preprocessing_configs.py:136`) into the split and the CatBoostEncoder (`_main_train_suite.py:488`), but the OOF fold seed is hardcoded to 42 and `CalibratedClassifierCV`'s internal CV gets no `random_state`. A run is therefore reproducible for the split but the OOF/calibration randomness is fixed-but-independent of the user's seed — two users with different `random_seed` get identical OOF folds, and the calibration CV is not seed-pinned at all.
- Why it matters: reproducibility is per-decision-surface here. Hardcoded 42 means the OOF estimate (which drives ensemble selection) cannot be varied with the run seed for variance/stability analysis, and an unseeded CalibratedClassifierCV CV makes the calibrated probabilities mildly non-reproducible across runs. Neither is a leak; both undercut full reproducibility.
- Recommendation: plumb `split_config.random_seed` into `_compute_oof_preds(random_seed=...)` and into `CalibratedClassifierCV(..., random_state=seed)` (sklearn accepts it via the inner CV splitter). Document the single suite seed as the master.
- Confidence: **High**.

### A7-04 — Decision threshold is a fixed 0.5 everywhere; never tuned, but also never leaked
- Severity: **P2** (quality gap, NOT a leak)
- file:line: `core/utils.py:4` / `core/_setup_helpers.py:74` `DEFAULT_PROBABILITY_THRESHOLD = 0.5`; applied at `core/_predict_main_suite.py:368,379,446,456,475,484` and `core/_predict_main_from_models.py:499,508,571,580,597,605`; `evaluation.py:505` uses `DEFAULT_BINARY_THRESHOLD`.
- What's the done-right part: because the threshold is a constant and is never fit on any split, there is **no threshold-on-test leakage** — a common failure this suite avoids by construction. This is the safe choice for the leakage lens.
- What's the issue (quality): for imbalanced / cost-asymmetric targets a fixed 0.5 produces poor `preds` (the probabilities can still be good). The suite computes rich calibration metrics (ICE/Brier/ECE) on probs, so the constant threshold mostly affects the hard-label `preds` and any downstream confusion-matrix-style metric.
- Recommendation: optionally tune the threshold on **val or OOF** (never test) per target — e.g. maximise F-beta / minimise expected cost on OOF — and stamp the chosen threshold into metadata so predict reuses it. Gate behind a config flag; keep 0.5 as the leak-safe default. If left as-is, document explicitly that hard-label metrics assume 0.5.
- Confidence: **High** that it is fixed-0.5 and unleaked; **Medium** that per-target tuning would materially help (target-dependent).

### A7-05 — Tree-model "calibration" under `prefer_calibrated_classifiers=True` is an in-training eval-metric swap, not held-set post-hoc calibration; the post-hoc hook is a no-op
- Severity: **P2** (semantic/labeling, not a leak)
- file:line: `_trainer_configure.py:525,528,558,728,738,757` (CB/XGB/LGB pick `*_CALIB_CLASSIF` params or set `eval_metric=integral_calibration_error`); `_calibration_models.py:258-274` (`_maybe_apply_posthoc_calibration` returns `model` unchanged — explicit no-op); `_training_loop.py:745-752` (calls the no-op hook).
- What's the issue: `prefer_calibrated_classifiers=True` (default, `_model_configs.py:470`) for **tree** models only changes the early-stopping/eval metric toward calibration — it does NOT produce a held-set-calibrated probability. The historical post-hoc isotonic wrapper was deliberately disabled (no-op at `_calibration_models.py:271-274`, with the stated rationale "avoids the val-set overfitting problem"). Only **linear** models get a true `CalibratedClassifierCV(cv=3)` wrap (`models.py:410-412`), which is correctly internal-CV (honest, no test touch).
- Why it matters: the flag name implies calibrated probabilities for all classifiers; in practice tree probs are "calibration-optimized during fit" but not post-hoc calibrated on a disjoint set. This is defensible (avoids val overfitting) and is NOT a leak, but it is easy to misread. A reviewer expecting calibrated tree probs by default would be surprised.
- Alternative reading: the design choice is sound — fitting post-hoc isotonic on val would burn the ES budget twice / overfit val, exactly the trap the no-op avoids. The honest path is the standalone `post_calibrate_model` on a disjoint calib slice (which the suite doesn't auto-wire — see A7-01).
- Recommendation: rename/clarify in the docstring that for trees the flag tunes the training objective, not a post-hoc calibrator; surface in metadata whether each model's probs are post-hoc calibrated or merely calibration-trained. Pair with A7-01 to give an honest auto-calibration path.
- Confidence: **High**.

### A7-06 — Standalone calibration tools enforce disjoint-from-test discipline rigorously (POSITIVE)
- Severity: **OK-positive**
- file:line: `calibration/post.py:423-503` (`train_postcalibrators` refuses to run without explicit `calib_probs_per_model`+`calib_target`, and raises if `calib_target` matches any model's `.test_target` via `np.array_equal` at `:492-498`); `evaluation.py:166-198` (`post_calibrate_model` documents the two honest sources — OOF-train probs or a `calib_idx` disjoint from test, asserts the intersection empty); `calibration/post.py:494-497` (comment confirming "the calibrator never touches test, so every test row is an honest holdout").
- What's done right: these are textbook-correct. They actively forbid the "fit calibrator on the honest holdout" anti-pattern with a value-equality guard, not just a docstring. `pick_best_calibrator` (`calibration/policy.py:280-437`) selects the calibrator on **OOF only** (`:298-303` "selection is OOF-only to keep test honest"; optional `probs,y` are diagnostic-only).
- Why it matters: confirms the calibration *machinery* is methodologically sound; the only gap is that the suite doesn't auto-invoke it (A7-01).
- Confidence: **High**.

### A7-07 — Ensemble winner selection ranks on OOF first, val second, and only falls back to test with a loud one-time WARN (POSITIVE, with residual caveat)
- Severity: **OK-positive** (caveat tie-in to A7-02)
- file:line: `core/_ensemble_chooser.py:32-39` (`_ENSEMBLE_RANK_METRIC_CANDIDATES` ordered oof → val → test), `:109-118` (test-path emits a WARN that "Using test for selection converts it into a model-selection surface and biases downstream test-set metrics"). Consumed at `_phase_train_one_target_ensembling.py:138-145`.
- What's done right: the selection surface priority is exactly correct — OOF (honest CV analog) preferred, val (biased ES detector) next, test only as a last-resort unit-fixture fallback with an explicit operator warning. The terminology in the comments (oof = honest, val = biased ES detector, test = honest estimate not to be selected on) matches project ML-terminology norms precisely.
- Caveat: the "honest" OOF this relies on is the shuffled-KFold OOF flagged in A7-02; on temporal data the OOF rank is optimistic. The selection *structure* is right; the OOF *generation* is the weak link.
- Confidence: **High**.

### A7-08 — Composite-target discovery operates strictly on train indices — no val/test leakage in discovery (POSITIVE)
- Severity: **OK-positive**
- file:line: `_composite_discovery_fit.py:186-296` — `y_train = y_full[train_idx]` (`:220`), feature filtering on `y_train`/`train_idx` (`:267`), MI screening on a sub-sample of `train_idx` (`train_idx_screen = train_idx[sample_idx]`, `:296`), base-column extraction `[train_idx]` (`:351,793,881`). The discovery `random_state` is config-driven (`:291`).
- What's done right: target engineering (a notorious leakage vector) is computed only from train rows; val/test never inform which composite target is discovered. The 50-row guard (`:195-199`) prevents discovery on degenerate train sizes.
- Confidence: **High** (multiple independent `[train_idx]` confirmations).

### A7-09 — Split phase is leakage-careful: auto-stratification, group-awareness, time-ordering, and group-spanning-cutoff resolution all correct (POSITIVE)
- Severity: **OK-positive**
- file:line: `splitting.py:606-632` (timestamp path argsorts with `kind="stable"` for reproducible temporal split), `:820-906` (group-spanning-cutoff: a group straddling a temporal cutoff is assigned wholesale to the LATER split with partition-invariant `raise` checks at `:865-876`, so train never sees rows from a query leaked into val/test), `_phase_helpers_fit_split.py:559-642` (pl.Enum domain built from train+val UNION only, **test excluded** to avoid label-time leakage — explicit at `:215,575`). Fairness subgroups computed from full df BEFORE split (`_phase_helpers_fit_split.py:373-374`) — but these are reporting-only strata, not features, so no leak.
- What's done right: the split honors non-IID structure (groups stay whole), temporal ordering (stable sort, later-split assignment for spanning groups), and the categorical-encoding domain is built without test. Regression bucket-stratify (`:247-282`) prevents heavy-tail concentration in val/test. These are the right defenses.
- Minor caveat (not a leak): the Enum domain includes **val** categories (train+val union, `:575-589`), deliberately so val=ES detector doesn't silently null-cast val-only categories. Documented as intentional (Wave 72). Defensible — val is allowed to inform encoding domain because it is part of the model-development budget; only test must be excluded, and it is.
- Confidence: **High**.

### A7-10 — RFECV feature selection is fit on train-only with time-/group-aware internal CV (POSITIVE)
- Severity: **OK-positive**
- file:line: `_helpers_training_configs.py:764-789` — `TimeSeriesSplit` when `has_time` (`:767-768`), `cv_shuffle=not has_time` (`:787`) so folds never shuffle across the time boundary; `KFold(cv_n_splits)` otherwise. RFECV runs inside the trainer's `_apply_pre_pipeline_transforms` on `train_df`+`train_target` only (`_trainer_train_and_evaluate.py:390-404`), with `group_ids` sliced to the train range (`:358-367`). FS scoring uses `integral_calibration_error` (classification) / regression scorer, not a test-derived metric.
- What's done right: FS is properly nested — selection happens inside an inner CV on train, never sees val or test, and respects time/group structure. This is the correct nesting for the model-selection-inside-honest-outer-loop principle (the outer "honest loop" here is the held-out test reported once).
- Confidence: **High**.

### A7-11 — Honest holdout (test) is consumed exactly once, for metric reporting, and is structurally barred from every decision (POSITIVE, headline)
- Severity: **OK-positive**
- file:line: `_trainer_train_and_evaluate.py:806-924` (train/val/test metrics computed; test via `_compute_split_metrics` once, `:899-913`); test never feeds FS (A7-10), calibration (A7-06), ensemble selection (A7-07), composite discovery (A7-08), or threshold (A7-04). The only places test appears in a "decision-ish" context are the standalone calibration tools, which explicitly refuse to fit on test (A7-06), and the ensemble chooser's last-resort fallback, which WARNs loudly (A7-07).
- What's done right: this is the core best-practice and the suite gets it right end-to-end. Early stopping uses **val** (`_trainer_train_and_evaluate.py:451-534`, `_setup_eval_set`), correctly treating val as the biased ES detector, not test.
- Confidence: **High**.

### A7-12 — Selection-bias / prior-shift handling matches the SELECTION_BIAS.md doc, but the diagnostic is wired while the remediation is opt-in standalone (mostly POSITIVE; minor doc-vs-code gap)
- Severity: **Low**
- file:line: `docs/SELECTION_BIAS.md:30-44` claims `compute_label_distribution_drift` "is wired into train_mlframe_models_suite automatically ... right after the split and BEFORE training." Verified: the drift report machinery exists (`drift_report.py`) and is referenced; `PULearningWrapper` (`pu_learning.py`) is a standalone remediation, correctly documented as user-invoked (`SELECTION_BIAS.md:73-162`). Saerens-Latinne-Decaestecker correction is referenced correctly.
- What's done right: the doc's *diagnostic auto-wiring* claim is consistent with the suite computing/log­ging label-distribution drift around the split, and the remediation tools are honestly described as opt-in (not silently auto-applied — correct, since prior-shift correction needs a user-supplied `true_prior`). The math citations (Saerens 2002, Elkan-Noto 2008, Lipton 2018) are accurate.
- Minor gap: confirm at runtime that `compute_label_distribution_drift` is invoked from the split phase for the standard (non-precomputed) path; I traced the doc + module presence but did not pin the exact suite call site in this pass (the drift snapshot log helper `_log_cardinality_and_drift_snapshot` is called at `_main_train_suite.py:454-461`, which is consistent). If the auto-wiring is gated only behind `verbose`, that would weaken the doc's "automatically" claim.
- Recommendation: verify (and pin a test on) the unconditional drift-report call independent of `verbose`; if it's verbose-gated, either ungate it or soften the doc.
- Confidence: **Medium** (doc claim plausible and module present; exact unconditional call site not pinned in this pass — flagged honestly per "no hand-wave" rule).

### A7-13 — Multilabel calibration is correctly refused-or-downgraded, not silently mis-applied (POSITIVE)
- Severity: **OK-positive**
- file:line: `_trainer_configure.py:327-343` — `prefer_calibrated_classifiers=True` + `MULTILABEL_CLASSIFICATION` raises `NotImplementedError` (CalibratedClassifierCV is single-output) unless `allow_uncalibrated_multi=True`, in which case it WARNs and drops calibration. `_calibration_models.py:99-178` provides a proper per-class isotonic calibrator (`_PerClassIsotonicCalibrator.fit`) for the multiclass/multilabel post-hoc path with constant-label guards (`:169-173`).
- What's done right: the suite does not pretend to calibrate multilabel via a single-output calibrator; it fails loud or downgrades with a warning. The per-class isotonic path, when used, fits each class independently on its own calibration probs — correct.
- Confidence: **High**.

### A7-14 — Bootstrap CIs and honest-diagnostics report on the honest holdout with stratified resampling and seeded reproducibility (POSITIVE)
- Severity: **OK-positive**
- file:line: `evaluation/bootstrap.py:40-90` (`bootstrap_metric` requires `random_state` for reproducible diagnostics, supports `stratify=` for class-balance-sensitive metrics like AUC/Brier on rare classes); `honest_diagnostics.py:66-120` (bootstrap block computes binary top-line CIs from test probs; uses numba-port metric kernels asserted numerically identical to sklearn). PSI drift across train/val/test surfaced (`honest_diagnostics.py:10-12`).
- What's done right: imbalance is handled in the CI machinery via stratified resampling; metrics carry uncertainty bands rather than point estimates; the report is built from the honest holdout. Seeding is mandatory for the diagnostics path.
- Caveat (overlaps A7-03): the bootstrap seed defaults are per-target but fixed (`rng_seed = 0` at `honest_diagnostics.py:76`), not derived from the suite seed — minor reproducibility coupling gap, same class as A7-03.
- Confidence: **High**.

---

## Disposition rollup
- P1: 2 (A7-01 calib_size dead-wire, A7-02 shuffled-OOF on temporal data)
- P2: 3 (A7-03 hardcoded OOF/calib seed, A7-04 fixed 0.5 threshold [quality, not leak], A7-05 tree "calibration" is eval-metric not post-hoc)
- Low: 1 (A7-12 drift auto-wiring doc-vs-code, unconfirmed call site)
- OK-positive: 8 (A7-06 calibration disjointness guards, A7-07 OOF-first ensemble selection, A7-08 composite discovery train-only, A7-09 split leakage hygiene, A7-10 RFECV nested+time-aware, A7-11 single honest-test consumption, A7-13 multilabel calib refusal, A7-14 bootstrap CIs)

No P0 found: there is no path where the honest holdout silently informs a model/feature/calibration/ensemble decision. The two P1s are a missing-feature trap (calib_size) and a temporal-OOF correctness gap, both fixable without architectural change.

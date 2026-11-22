# MRMR module audit tracker (2026-11-23)

Five read-only audit waves (Haiku agents, code reading only, nothing executed) plus two more on uncovered aspects. Each finding carries a status (CONFIRMED-CODE: read in the code path; HYPOTHESIS: needs a run) and a disposition (OPEN, FIX, REJECTED with reason, FUTURE with trigger). Nothing here is implemented yet. A finding becomes a defect only after a reproducing run.

Status values follow the other trackers: RESOLVED, PARTIAL, TODO, REJECTED, FUTURE.

| Status | ID | Area | Finding | Evidence (file:line) | Disposition |
|---|---|---|---|---|---|
| TODO | AUD23-01 | state / cache | Fingerprint memo keyed on id(X) samples about 10 cells; an in-place edit of an unsampled cell then refit returns the old support_ from the fit cache | _mrmr_fingerprints.py:602-611, _fit_cache.py:111,135 | OPEN; reproduce first with a single-cell edit test |
| TODO | AUD23-02 | GPU | Circuit breaker trips for the whole process on any exception, including cupy OutOfMemoryError | _permutation_null_pair_resident.py:223-241,275 (compare _fe_gpu_vram.py:57-92 cooldown) | OPEN; split: context fault trips, OOM retries with a smaller block |
| TODO | AUD23-03 | state | `signature` not restored by __setstate__ defaults; old pickles raise on next fit | _mrmr_class.py:3218, _mrmr_setstate_defaults.py, _fit_cache.py:135 | OPEN; add default and a legacy-pickle test |
| TODO | AUD23-04 | concurrency | Merge of stage state loses an attribute created by two stages in different groups (last setattr wins) | _fe_stage_concurrency.py:132-133, _hybrid_orth_family_variants/_concurrent.py:104-127 | OPEN; reproduce with two groups sharing a new name |
| TODO | AUD23-05 | concurrency | Rebuilt list discards earlier appends from another stage | _fe_stage_concurrency.py:120-124 | OPEN; same reproduction as AUD23-04 |
| TODO | AUD23-06 | concurrency | A failed kept stage drops the successful siblings' state (sequential run keeps it) | _fe_stage_concurrency.py:233-237 | OPEN; decide semantics with owner |
| TODO | AUD23-07 | transform | Cat-FE replay maps key by str(), so int 1 and string "1" collide | engineered_recipes/_recipe_extract.py:108 | OPEN; reproduce on [1, "1", 2] |
| TODO | AUD23-08 | transform | Cat-FE replay map built before the min_support fold; the fold shifts codes at transform | _recipe_extract.py:62-124 vs discretization/_categorical_encoding.py:89-99 | OPEN; reproduce with a thin level |
| TODO | AUD23-09 | transform | Non-integer categorical source without cat_code_map becomes constant code 0, no error | _recipe_extract.py:236-237 | OPEN; reproduce with legacy recipe |
| TODO | AUD23-10 | transform | ndarray input is not checked for column order | _mrmr_validate_transform.py:611-614, 844-846 | FIX (docstring) plus test; no behaviour change |
| TODO | AUD23-11 | transform | Output container type depends on usability_aware_lists for ndarray input | _mrmr_class.py:3644-3645, _mrmr_validate_transform.py:204-209,223 | OPEN; decide contract with owner |
| TODO | AUD23-12 | discretization | Knuth quantile edges not de-duplicated; column collapses to one bin | _discretization_edges.py:145-146, _adaptive_nbins.py:288 | OPEN; reproduce with knuth_edge_type="quantile" (non-default knob) |
| TODO | AUD23-13 | discretization | Uniform path gives NaN edges for an all-NaN column; quantile path gives a single bin | _discretization_edges.py:131-146, 395-408, core/arrays.py:33-34 | OPEN; low; MI unchanged |
| TODO | AUD23-14 | significance | Serial confirm path uses count/n with no +1; p=0, confidence 1.0 at budget 3 | _confirm_predictor_helpers.py:119 | OPEN; deliberate per code comment; owner decision (add-one vs starvation) |
| TODO | AUD23-15 | significance | All candidates share one permutation seed (correlated exceedances) | _family_maxt_gate.py:690-911, _honest_mdlp_null.py:404-405,497-498 | OPEN; measure effect on FDR accounting |
| TODO | AUD23-16 | significance | Quantile convention: np.quantile linear in maxT/FDR floor vs method="higher" in honest path; stale docstring in _fallback_maxt.py | _permutation_null.py:455,735,822; _honest_mdlp_null.py:365,430; _fallback_maxt.py:3-6 | FIX (docstring); decide convention |
| TODO | AUD23-17 | significance | Analytic null applicability guard before one direct call not verified | _feature_engineering_pairs/_pairs_analytic_gpu.py:359 | OPEN; trace the caller |
| TODO | AUD23-18 | GPU | Permutation-null block allocates (n, nperm) with no free-memory check; about 28 bytes per cell of temporaries | _fe_cmi_perm_null_gpu.py:456,484-486 | OPEN; trace nperm gating at n=1M |
| TODO | AUD23-19 | GPU | Batch pair-MI precision: cupy path float32 by default, CUDA and njit float64 | batch_pair_mi_gpu.py:125-163, _batch_pair_mi_cuda_kernels.py:395-401 | OPEN; measure selection change at ties |
| TODO | AUD23-20 | GPU | Honest-null float32 pass-1 error may exceed the 0.5-nat window at large n | _honest_null_device.py:168-171,208 | HYPOTHESIS; measure error vs n |
| TODO | AUD23-21 | GPU | Pair-subchunk re-uploads factors_data every subchunk (not via resident_operand); H2D grows with subchunk count | _batch_pair_mi_cuda_kernels.py:733-742 | OPEN; measure H2D at many subchunks |
| TODO | AUD23-22 | IT | Chao-Shen MI: per-term coverage in two kernels; the shared-coverage fix is not on the production path | _entropy_kernels.py:390-393, _class_mi_kernels.py:105-109, _chao_shen.py:149-171, _class_mi_kernels.py:297-298 | OPEN; compare values on one table |
| TODO | AUD23-23 | IT | group_blocked_mi hard-codes use_mm=True, ignores the Miller-Madow toggle | info_theory/_group_mi.py:238,264,276 | FIX unless owner wants MM always on for groups |
| TODO | AUD23-24 | IT | Possible double Miller-Madow correction in FE pair gate | _feature_engineering_pairs/_pairs_gates.py:163-164, _pairs_score_helpers.py:328-330 | HYPOTHESIS; trace the producer of pair_mi |
| TODO | AUD23-25 | IT | entropy(min_occupancy) drops bins without renormalising (0.3065 vs 0.6730 nats for [0.6, 0.4]) | _entropy_kernels.py:17-18,139-140 | OPEN; check callers pass non-zero min_occupancy |
| TODO | AUD23-26 | IT | Naive (1-p)^n in entropy_chao_shen | _entropy_kernels.py:338 | REJECTED for now: unreachable on real counts (n*p bounded below) |
| TODO | AUD23-27 | IT | Possible int overflow in GPU cell index (int in C kernel) | info_theory/_cmi_cuda.py:837,455,476; _cmim_min_gpu.py:150,154 | HYPOTHESIS; read kernel bodies |
| TODO | AUD23-28 | API | Docstring cv default None, code default 3 | mrmr/_mrmr_class.py:208 vs :1198, fs_params/mrmr.py:185 | FIX (docstring) |
| TODO | AUD23-29 | API | Comment says fe_fast_search default ON; code default False | mrmr/_mrmr_class.py:552 vs :576,580 | FIX (comment) |
| TODO | AUD23-30 | API | pld_max / pld_mean accepted but only warn; not wired to a path | _mrmr_param_constants.py:20, fs_params/mrmr.py:80, _mrmr_validate_strings.py:19-22,127-135 | OPEN; trace redundancy path; remove or implement (owner) |
| TODO | AUD23-31 | API | categorical_fold_strategy typed as plain str in fs_params; allow-list only at fit | fs_params/mrmr.py:49, _mrmr_validate_strings.py:150 | FIX (type as Literal) |
| TODO | AUD23-32 | API | Unvalidated fe_mi_estimator and fe_acceptance | _step_core_parts.py:678,146 | HYPOTHESIS; trace |
| TODO | AUD23-33 | API tests | fs_params drift test ignores annotations and compares a render, not the live signature | tests/training/test_fs_params_in_sync.py:27,36-50 | FIX (test) |
| TODO | AUD23-34 | API tests | Fold-strategy test checks an argument, not behaviour (passes with a no-op fold) | tests/feature_selection/mrmr/test_categorical_fold_param.py:32-54 | FIX (test) |
| TODO | AUD23-35 | API tests | pytest.raises(ValueError) without match | same file :28-29 | FIX (test) |
| TODO | AUD23-36 | API tests | Docs-token test instead of behaviour | tests/test_meta/test_default_scorer_choice_is_deliberate.py:42-43 | FIX (test) |
| TODO | AUD23-37 | selection | Operand re-attach uses a cols index against a raw-index set; swallowed at DEBUG | _mrmr_fit_impl/_assign_support.py:746,756-757,773,786-787 | OPEN; reproduce with redundancy_policy="drop" |
| TODO | AUD23-38 | FE | Prewarp replay falls back to a var-keyed spec that may be another chunk's warp | _mrmr_fe_step/_step_score_parts2.py:222-223, _feature_engineering_pairs/_pairs_gates.py:80-83 | OPEN; reproduce with two chunks |
| TODO | AUD23-39 | FE | Prewarp recipe without spec: fit succeeds, replay raises KeyError | _step_score_parts2.py:222-223, engineered_recipes/_recipe_unary_binary.py:133-136,338-341 | OPEN; reproduce |
| TODO | AUD23-40 | FE ledger | Counts dedupe recipes by name; emitted and selected can disagree | _fe_group_counts.py:32-36,56 | OPEN; reproduce duplicate names |
| TODO | AUD23-41 | FE | Name-based skip of a distinct recipe with the same name | _fe_pure_form_retention.py:802-806, _fe_stage_cascade_mid_a.py:278 | OPEN; reproduce |
| TODO | AUD23-42 | FE | Bare failure in dtype introspection marks every column numeric (debug log only) | _mrmr_fe_step/_helpers.py:39-43 | FIX (log at warning) |
| TODO | AUD23-43 | FE | Probe exception passes the candidate (documented fail-open) | _fe_accuracy_gate.py:235-237,386,176-178 | OPEN; owner decision: fail-open vs fail-closed |
| TODO | AUD23-44 | categorical | Bool columns not classified as categorical in the target_stat null; codes not refolded under permuted target | _mrmr_fit_impl/_fit_impl_stages/_binning.py:70,74,292 | OPEN; measure null bias on a bool source |
| TODO | AUD23-45 | categorical | target_stat fold built from all rows (in-sample by design, partly bounded by BIC) | _binning.py:485-492, _categorical_target_fold.py:8,137 | REJECTED as defect: documented; honest null is the correction |
| TODO | AUD23-46 | categorical | NaN block shift taken from the transform frame, may differ from fit | _discretization_dataset.py:701-713, _recipe_extract.py:73-92 | HYPOTHESIS; trace |
| TODO | AUD23-47 | concurrency | Pools keyed (device,i) can nest on the same device; set_limit read-modify-write outside lock | _fe_device_scheduler.py:113-140 | HYPOTHESIS; trace call graph |
| TODO | AUD23-48 | concurrency | Lap timers accumulate per thread; laps from edge workers leak into the next gate | _fe_family_timing.py:111 | HYPOTHESIS; trace gate calls inside map_on_edge_pool |
| TODO | AUD23-49 | state | Unbounded module cache _MRMR_IDENTITY_FP_CACHE; identity key omits sample_weight | _mrmr_fingerprints.py:85,545-560, _mrmr_class.py:3512,3520 | OPEN; bound the cache; reproduce weight-sensitive identity |
| TODO | AUD23-50 | state | Single-slot last-X hash cache and unlocked background-warmup flag under concurrency | _mrmr_fingerprints.py:512, _numba_warmup.py:83-85 | FIX (lock); performance only |

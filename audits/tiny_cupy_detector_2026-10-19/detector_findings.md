# Detector findings — tiny-cupy loop detector

Written for: maintainers of the mlframe meta-tests. Every finding from the critique agent's report is listed; none is merged.

Legend: verdict CONFIRMED / PLAUSIBLE / REJECTED; disposition FIX / DEFER / DISMISS.

## A. Detector false negatives (sites the detector cannot see)

| ID | Finding | Evidence | Verdict | Disposition |
|----|---------|----------|---------|-------------|
| A1 | Method calls `self._step(x)` are not followed; a loop calling a method that issues cupy calls is not flagged. | Old detector returns no hits on the snippet; regression `test_method_call_through_self_is_flagged`. | CONFIRMED | FIX (done: `self`/`cls` resolve to same-module methods) |
| A2 | Two-level helper chains are not followed (loop -> helper -> helper with the cupy calls). | Old detector returns no hits; regression `test_call_chain_two_levels_deep_is_flagged`; depth limit pinned as a cost bound by `test_call_chain_beyond_the_depth_limit_is_not_followed`. | CONFIRMED | FIX (done: `_CALL_DEPTH = 2`) |
| A3 | cupy operator launches (`a[p] * b[p]`, `x + y` on device arrays) are not counted as launches. | Operators are not `ast.Call`. | PLAUSIBLE | DEFER: needs type inference for which names are device arrays; no reliable AST signal without it. Target: next wave. |
| A4 | A single cupy call per iteration over a large range is not flagged. | Threshold is >= 3 calls (or 2 with a heavy routine). | PLAUSIBLE | DISMISS: a single launch per iteration over a large range is the batched shape of an unavoidable loop; the threshold is deliberate. Revisit if a site appears. |
| A5 | Loop bodies that call a helper whose cupy calls live in a branch the loop never takes are counted in full. | This is the source of the false positives in B3. | CONFIRMED | Covered by B3 |

## B. Detector false positives (sites the detector flags but should not)

| ID | Finding | Evidence | Verdict | Disposition |
|----|---------|----------|---------|-------------|
| B1 | Chunk and tile loops (3-arg `range`, bound named *chunk/*tile/*block) are real loops but run a handful of iterations. | `_permutation_null_pair_tiles.py::_pooled_pair_null_perm_tiles` | CONFIRMED | FIX already in 3e00704d8 (severity low). Keep. |
| B2 | A once-guarded branch inside the loop body still counts. | Critique self-test. | CONFIRMED | FIX (count only statements not under a branch that tests a loop-invariant condition) — DEFER to next wave if it breaks existing hits; see note. |
| B3 | Callee bodies count in full even when their cupy calls sit in a GPU branch the CPU loop does not take. | Probable false positives in section C, rows marked "likely false positive". | PLAUSIBLE | DEFER: needs per-site review; the detector cannot know which branch is taken. Handled by triage in section C. |

## C. The 49 new baseline entries from the call-following rule

Judged by the critique agent from loop headers, first body lines and callee call counts. NOT read in full. Each row needs a per-site check before its disposition is final.

| Group | Sites (file::function or file:line) | Agent verdict | Disposition |
|-------|-------------------------------------|---------------|-------------|
| Real per-iteration GPU launches | `_gpu_resident_extval` 77/79; `_gpu_resident_shared` 685, 700-702; `_wavelet_basis_fe_batched` 131-132; `_usability_pool_resident` 244; `_hinge_detect_gpu_resident` 277; `_usability_greedy_gpu_resident` 269, 410; `_usability_greedy_clf_gpu_resident` 446; `_extra_basis_resident` 142; `_gpu_resident_basis` 757, 883; `_mi_greedy_cmi_fe_steps` 277; `_conditional_gate_fe` 577; `_cupy_polynom_optimizer` 240; `_hermite_prewarp_gpu_resident` and `_als_fused_gpu` 209; `_pairwise_modular_resident` 140 | launch-bound | DEFER: real worklist; each needs its own batching change with an A/B. |
| Real, chunk-level (low) | `_gpu_resident_fe` 636; `_gpu_resident_k_chunk_ktc` 94; `_gpu_resident_pair_mi_parts` 65; `_step_batched_marginals` 45; `_streamed_gpu_mi` 195; `_usability_pool_resident_pairs` 31; `_family_maxt_gate` 245 | chunk loops | DEFER: low severity, no action. |
| Probable, not confirmed | `_fe_additive_fusion_gpu_resident` 195, 384, 390, 429; `_fourier_batch_gpu` 337; `_streamed_gpu_mi` 147; `_binned_numeric_agg_resident` 435; `_hinge_detect_gpu_resident_batch` 145; `_fe_cmi_redundancy_gate` 479, 740 | not read | DEFER until per-site read (next step of this wave). |
| Likely false positives (GPU branch not taken on CPU) | `_fe_additive_fusion` 197; `_fe_auto_escalation` 190, 219, 283, 399, 428; `_fe_cmi_redundancy_null` 418; `_mi_greedy_cmi_fe` 618; `_mi_greedy_cmi_fe_steps` 843; `_unified_fe_gate` 475; `_usability_aware_selection` 484; `_mrmr_fe_step` `_step_score` 56, 59 and `_step_score_parts2` 461, 464; `hermite_fe/__init__` 418; `_hermite_prewarp` 318; `_gpu_resident_pair_mi` 666 | host loops with GPU calls in a branch | DEFER: verify the branch condition per site; if the call is guarded, add the guard to the detector as an accepted pattern, not a baseline pin. |

## D. Test-quality findings

| ID | Finding | Evidence | Verdict | Disposition |
|----|---------|----------|---------|-------------|
| D1 | `test_known_blind_spots_are_pinned` asserted the current false negatives as correct behaviour. | Test body in 3e00704d8 parent. | CONFIRMED | FIX: removed (commit `3e00704d8`). The pin was the wrong direction; the blind spots are now A1, A2, A3 above. |
| D2 | Self-test coverage of the helper case and the batched form. | Existing tests cover a per-pair loop and its batched form. | CONFIRMED | FIX: add a helper-hidden regression test for A1 and A2 when they are fixed. |

## E. Open-ended proposals from the critique

| ID | Proposal | Verdict | Disposition |
|----|----------|---------|-------------|
| E1 | Run the detector over `_benchmarks` as well (currently excluded because those loops time launches on purpose). | PLAUSIBLE | DISMISS: exclusion is deliberate; benchmark loops measure launch cost. |
| E2 | Parse once per run and share the summary cache across `test_perm_loop_invariants_hoisted.py` and this test. | PLAUSIBLE | DEFER: a speed-only change; scan is already about 11 s. |

## Summary of dispositions

Note: the 35 sites added to the baseline by the depth-2 and `self` rules are NOT yet reviewed per site. They are a worklist, same status as section C.

- FIX in this wave: A1, A2, B1 (done earlier), D1 (done), D2 (with A1 and A2).
- DEFER with a target: A3, B2, B3, the 49 entries in section C, E2.
- DISMISS with reason: A4, E1.

## F. Triage of the 84 new baseline entries

Full per-site verdicts: `triage_a.md` (keys 1-42 of `/d/Temp/triage_a.txt`) and `triage_b.md` (keys 43-84). Verdicts were read from source, with callee bodies read where cited.

Totals: REAL 25 (A 9, B 16), CHUNK 24 (A 13, B 11), FALSE_POSITIVE 27 (A 17, B 10), UNCLEAR 8 (A 3, B 5).

Caveat for the FALSE_POSITIVE group: most are gated off by default (`_cmi_gpu_enabled`, `_gate_cands_resident()`, `fe_gpu_strict_resident_enabled()`, `MLFRAME_EXTVAL_FUSED`). The detector is right that the default host run does not launch them, but these gates are opt-in switches, not kernel-tuning-cache decisions. Per the project rule (dispatch by hardware and size via kernel_tuning_cache, not opt-in flags), those gates are a dispatch gap to fix, not a reason to dismiss. Status: OPEN, not dismissed.

Example: `hermite_fe/__init__.py` `_polyeval_dispatch` (key 37 in the triage list). The CUDA crossover `_cuda_threshold` from kernel_tuning_cache is computed in `_lookup_polyeval_thresholds` but the default path ignores it and reaches CUDA only when `MLFRAME_POLYEVAL_BACKEND=cuda` is set. Disposition: FIX through KTC, not DISMISS.

Shapes of FALSE_POSITIVE, to be recognised by the detector or fixed at source:
1. Host function calling a helper whose cupy branch sits behind a config gate (about 13 sites).
2. `xp` or `cp` parameter always bound to numpy by callers (keys 34, 36 in A).
3. Env flag that returns early before the cupy loop (key 31 in A).
4. Prewarp/spec branch in a callee that skips the cupy quantiser (key 19 in A).

UNCLEAR, needs a further read: A 27, 28, 38; B 5 keys (see `triage_b.md`).

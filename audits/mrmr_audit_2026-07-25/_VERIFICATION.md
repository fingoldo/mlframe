# mrmr_audit_2026-07-25 — independent verification pass + dispositions

An independent read-only agent re-checked every tracker `DONE` against real source, and a second agent mined
all 16 audit docs for meta-testable bug classes. Both were told to be adversarial. They found genuine gaps —
including damage done by this wave's own comment sweep. This file is the honest record; where the first round
overstated a `DONE`, that is said plainly here rather than quietly left in the tracker.

## Gaps the verification pass found, and what was done

| Gap | Verdict | Action |
|---|---|---|
| `DISCRETIZATION-1` guard missing at 3 out-of-package call sites (`_mrmr_fe_step/_step_score.py:577,897,1005`) | REAL | FIXED — `reserve_nan_slot=(quantization_method == "uniform")` passed at all three; pinned by `test_step_score_safe_code_dtype_reserves_uniform_nan_slot` |
| `INFO_THEORY-10` RelaxMRMR cap placed AFTER its first dense alloc, and the `n_S == 0` early return bypassed it entirely | REAL | FIXED — all caps hoisted above the first allocation; pinned by `test_relax_mrmr_caps_cardinality_before_any_dense_alloc` |
| `INFO_THEORY-10` JMIM empty-selected-set fallback allocates `(K_x, K_y)` un-capped | REAL | FIXED — cap added on that path; pinned by `test_jmim_caps_cardinality_on_empty_selected_set` |
| `INFO_THEORY-6` had no kernel-side clamp (only the 3 dispatch call sites were widened) | REAL | FIXED — both `batch_mi_with_noise_gate` kernels now raise when `classes_dtype` cannot hold `max(nbins)-1`; verified to compile and fire under numba; pinned by `test_batch_mi_noise_gate_kernel_rejects_overflowing_classes_dtype` |
| `INFO_THEORY-6` 4th unguarded site `_gpu_resident_basis.py:1320` hardcoded `np.int16` | REAL | FIXED — routed through `_fe_classes_dtype` |
| `FE_PAIRS-3` memo key still omitted `MLFRAME_FE_GPU_DISCRETIZE` / `MLFRAME_FE_GPU_BINNING`, which the gates read live | REAL | FIXED — both folded into the key; pinned by `test_fe_gpu_gate_key_includes_the_per_gate_kill_switches` |
| `CORE_CLASS-2` comment claimed a same-call flat kwarg beats the config; the code does the opposite | REAL (comment) | FIXED — comment corrected to state the actual precedence, which matches `__init__` ("the nested config IS the state, once provided") |
| CPU/GPU parity hole not in any audit doc: `_pairwise_modular_resident.py:77,127,207` kept the bare `y.astype(int64)` its CPU twin was fixed for | REAL | FIXED — routed through `encode_y_for_classif_mi`; pinned by `test_pairwise_modular_resident_encodes_y_like_its_cpu_twin` |
| Perf regression risk: `encode_y_for_classif_mi` runs `np.unique` per call inside per-pair x per-modulus x 12-perm loops | REAL | FIXED — O(n) bincount fast path for already-dense codes; equivalence pinned by `test_encode_y_dense_fast_path_matches_np_unique` |
| `test_audit_hermite_fixes.py` subprocess did not propagate `PYTHONPATH` | REAL | FIXED — explicit `env=` |
| This wave's own `bench_fe_batch_free_blocks.py` printed a finding ID | REAL | FIXED |
| `INFO_THEORY-5`: triple kernels also omit `n_classes_y` | **REJECTED** | Their working histogram is `(n_dense, n_classes_y)` with `n_dense <= n`, so it is bounded by the data, not by cardinality; the `remap` table (sized by the raw product) is the real OOM hazard and is already capped. Not the same defect as the pair kernels. |
| `_resident_candidate_mi_ktc.py:97` still uses `equiv_rtol/atol=5e-2` — "the sibling of GPU_RESIDENT-4 was missed" | **REJECTED** | Not the same situation: that sweep compares rank binning vs percentile-edge binning, two genuinely different schemes that disagree at ties, documented as the approved FE-PAIR selection-equivalence trade. `GPU_RESIDENT-4`'s pair were FP-reorder twins of one algorithm. Recorded as a deferred entry (with the reason) in the new tolerance meta-test rather than retightened blind. |

## Damage done by this wave's own comment sweep, and its repair

The metadata sweep had a real bug: when a marker sat mid-sentence on a **continuation line** of a multi-line
comment, it stripped the marker and then capitalized the now-leading word, splicing a fake sentence boundary
into the middle of a sentence.

- 11 mid-sentence capitalizations reverted (the 215 other capitalizations are correct — those markers sat at
  the start of the comment, so capitalizing the next word is right).
- 247 tracked files under `_benchmarks/` and `profiling/` restored to `HEAD`: the sweep should never have run
  there (CROSS-2 excluded that tree) and the dates it stripped were measurement records, not process narration.
- Individually repaired: the orphaned `)` in `_grouped_recipes.py`; the lost `None`-preserves-legacy rationale
  in `_confirm_predictor_context.py`; the "regression fix" qualifier at two `_fit_impl_core.py` guards; the
  carve-ordering sentence in `_orth_auto_scorer_fe.py`; and the `_confirm_predictor.py` header, which said
  "RETIRED" while the code below it had become an env-gated opt-in.
- `bench-attempt-rejected` records: **199 in HEAD, 199 after** — the REJECTED-is-not-DELETED rule held.

## Honest status of the sweep findings

`CROSS-2` and `CROSS-4` are **partially** done, not complete, and the tracker should be read that way:

- `CROSS-2`: the unambiguous marker forms are gone from `src/mlframe/feature_selection` (finding-IDs down to
  zero real ones; `Wave N` / `loop iter N` / pure date parentheticals stripped; all 185 stale
  `suppressed in <file>:<line>` citations removed **repo-wide**). What remains is ~840 date mentions that are
  part of prose sentences, where blind deletion destroys meaning. Those need human reading, not a regex.
- `CROSS-4`: ~80% of the prose `--` in the audited tree is now ` - `; the remainder sits in contexts the
  space-delimited rule deliberately does not match.
- `FE_STEP-7`: the FE-step helpers that had no direct coverage now have it, but 4 of the 5 blocks the finding
  names still have no direct test. Recorded as remaining work rather than closed.

## A regression this wave introduced and fixed, and one pre-existing failure it did NOT cause

**Introduced and fixed here.** The `FE_STEP-1` rewrite replaced `list(combinations(...))` with
`np.asarray(numeric_vars_to_consider, dtype=np.int64)` - but that caller passes a **set**, which `np.asarray`
cannot convert. The order-2 maxT permutation-null floor therefore raised on *every* fit and was silently
swallowed ("maxT permutation-null floor failed; continuing without it"), so the FE step ran with no noise
floor at all. Fixed with `np.fromiter(..., count=len(...))`, which also preserves the exact iteration order
`combinations()` walked, so the per-pair MM bias vector stays index-aligned. Two regression tests now pin it
(pair-order equivalence, and a real fit asserting the fallback warning is absent); both fail pre-fix.
The bench missed this because it fed a `list` while production feeds a `set` - the bench now uses a set.

**Pre-existing, NOW FIXED (see the follow-up section at the bottom).** `biz_val/
test_biz_value_mrmr_fe_hybrid_orth/test_autowired_hybrid_fe.py` failed 11 / 38. Verified identical on a clean `origin/master` worktree:
**11 failed, 27 passed on both**. All 11 share a single root cause - the hybrid-orth stage emits nothing
(`hybrid_orth_features_=[]`), so the XOR `He_1*He_1` term, the `x1__He2` quadratic detector and the downstream
LogReg AUC lift all fail together; in one seed a pure-noise column (`binagg_skew(noise_b|qbin(noise_a))`) is
the only thing appended. That is a genuine product bug and the obvious P0 for the next wave, but it is out of
this audit's scope and predates it. (For reference, the same file now runs in 58 s here vs 249 s on master,
from the maxT index-generation speedup.)

## Meta-tests added (the "catch this class next time" deliverable)

| Test | Class it catches | Status on the current tree |
|---|---|---|
| `test_no_stale_source_line_citations.py` | a log message citing its own `<file>:<line>`, which drifts on every edit above it | GREEN (self-verifying: a correct citation passes by construction, so no allowlist is needed) |
| `test_config_dataclass_defaults_match_ctor.py` | a nested-config default drifting from the flat ctor default it unconditionally overwrites | GREEN — and it immediately found 4 more instances in `FastSearchConfig`, correctly classified as an intentional override profile and recorded with the reason |
| `test_ktc_sweep_tolerances.py` | a backend-equivalence sweep whose tolerance is orders looser than its own numerics claim | GREEN with 6 pre-existing sweeps explicitly deferred (each with a reason), so it fires on anything new |

Proposals deliberately NOT written, with reasons: a `--`-in-prose gate (a regex cannot separate prose from
CLI flags, reST rules and `i--` without an exception list longer than the rule, and it prevents zero
behavioural bugs); an implicit-`Optional` gate (mypy already reports it — widen the beachhead instead of
adding a weaker duplicate); a truncate-before-filter gate (a semantic question about intent, not a pattern);
a dead-parameter gate (fires on every signature-parity stub and `**kwargs` shim); an overloaded-sentinel gate
(`return 0.0` is statically indistinguishable from `return 0.0`).

## Follow-up: the hybrid-orth biz-value suite, 49 failures -> 0

The whole `test_biz_value_mrmr_fe_hybrid_orth/` directory was failing **49 / 385** on `origin/master`. Every
failure traced to three causes, none of which was a defect in the hybrid-orth algorithm itself (verified
standalone: it produces exactly the expected `x2*x1__He1_He1` and `x1__He2` terms).

**1. The test presets disabled the very stage under test.** The orth-basis families are FE stages, and the
`fe_max_steps=0` "no FE at all" contract - correctly - suppresses them. Both `make_fast_mrmr` and this
directory's own `_mrmr_kw` pinned `fe_max_steps=0` for speed, so every hybrid fit came back raw-only and each
assertion reduced to "the roster is empty". Fixed by giving the FE budget to callers that ask for the family
(`make_fast_mrmr` now sets `fe_max_steps=1` when `fe_hybrid_orth_enable` is passed; `_mrmr_kw` defaults to 1).
Scenario A's control has to stay genuinely raw - it asserts the target is linear-UNSOLVABLE - so `_fit_mrmr`
grew an explicit `fe_steps` knob and A's baseline passes `fe_steps=0`; with a budget the control engineered
its own cross term and scored AUC 0.91 instead of ~0.5.

**2. A real product bug: an MI floor above the information-theoretic ceiling.** `hybrid_orth_mi_fe` gates
candidates on `engineered_mi >= max(legacy_floor, median+k*MAD(raw), median+k*MAD(engineered))`. The MAD terms
assume the candidate sample IS a noise band. On a small candidate set (`cols=["x1"], degrees=(2,3)` -> two
candidates: one signal, one dud) the MAD measures the signal-to-noise SEPARATION instead, and the ~5*1.4826
multiplier pushed the floor to **2.21 nats against H(y) = 0.62** - unsatisfiable by construction, silently
disabling the family with no error and no log. Since `MI(X; y) <= H(y)` is a hard bound, a MAD floor at or
above `H(y)` can only mean "reject everything"; it is now dropped as degenerate (with a debug log). Verified
both directions: the quadratic detector is recovered on every seed, and the all-noise control still engineers
nothing, so the Layer-27 false-positive protection is intact.

**3. Structural assertions outlived the representation they were written for.** With FE enabled the step
often delivers a signal FUSED into a composite rather than as a standalone basis column - on the mixed-bag
scenario it emitted `add(add(sqr(a),b__He2),mul(cbrt(c1),cbrt(c2)))`, literally the target's generating
function. Name-shaped assertions (`startswith("a__")`, `"*" in name`) missed it even though the model was
strictly better: holdout AUC **0.86 raw-only -> 0.98 with FE**. Those assertions were re-framed to match the
real contract (an engineered column built from the signal's sources, standalone or fused), and the AUC bar
was raised 0.80 -> 0.90 accordingly - above the raw-only ceiling, so it now pins a genuine FE lift.

**Observability added.** `hybrid_orth_features_` is intersected with `support_`, so when a sibling FE family
emits an equivalent column and wins the greedy the roster goes empty - indistinguishable from the stage never
having run. That ambiguity is what made this take a manual bisect to diagnose. `MRMR.hybrid_orth_candidates_`
now records everything the family produced, whether or not it survived selection.

Result: **395 passed, 0 failed** in that directory (was 49 failed / 336 passed).

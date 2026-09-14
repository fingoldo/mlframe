# mrmr_audit_2026-09-14 — master tracker

9-agent parallel read-only audit (see `_BRIEF.md`) against git `57f649fb6`. Surface: 77 MRMR modules,
~31 440 LOC. Six agents took file clusters; three took cross-cutting bug classes (numerical cancellation +
silent fallbacks, performance, test gaps).

**Totals: 201 findings — P0 ×5, P1 ×25, P2 ×76, P3 ×67 (+41 proposed tests).**

Status legend: TODO / DONE (fixed+tested/benched) / DOC / REJECTED (with reason).

## Spot-checks performed by the orchestrator

Seven claims were re-verified against source before being relayed. Six held exactly; two carried a
correction, recorded here so the next wave does not inherit an overstatement:

| Claim | Verdict |
|---|---|
| `transform()` identity fast-path bypasses the column-name check | **HELD** — order confirmed at `_mrmr_validate_transform.py:347 / 358 / 381` |
| `_orth_dedup` variance via raw power sums (k=2) | **HELD** — `:98-107`, and `nan` → "not a duplicate" per its own comment |
| relaxMRMR-3D CMI argument slots transposed | **HELD** — helper signatures at `_relaxmrmr_3d.py:44, 81` confirm the mismatch is structural |
| `fit_kwargs` (`sample_weight`/`groups`) swallowed by `_stability_outer_fit` | **HELD** — the name occurs only in the signature, `_mrmr_class_fit_helpers.py:218` |
| `_pairwise_modular_fe` batching is cupy-STRICT-only | **HELD** — host fallback loop at `:294-297`; the sibling batches on host at `_integer_lattice_fe.py:208-220` |
| gate/cross-group prune regexes inert on real column names | **HELD** — `([a-z](?:[a-z]?\d+)?)` matches `a`/`x12`, not `price`/`revenue` |
| `maxT` floor `0.0` on exception is undiagnosable | **CORRECTED** — the finding stands (`0.0` is the documented gate-off sentinel), but it logs at `logger.warning` with `exc_info=True`, not `debug` |
| `nbins_method` `marx`/`sci`/`mah_sci` collapse to `mah` is a silent no-op | **CORRECTED → P3** — MAH/SCI is one method (Marx 2021); the aliasing is intentional. Residual real gap: no test pins the alias map, so a typo in it would pass unnoticed |

## P0 — silent wrong results (fix first)

| ID | File:Line | Summary | Status |
|----|-----------|---------|--------|
| PERIPHERY-1 | `_mrmr_validate_transform.py:352-362` | `transform()` identity fast-path compares only a column COUNT; the width check is deliberately skipped for named frames so the name-drift `RuntimeError` at `:381` can fire, but the fast-path returns before it — a reordered or entirely different same-width frame is returned unchanged, every column mis-mapped downstream | RESOLVED |
| NUM-1 | `_orth_dedup.py:92-183` | FE collinearity-dedup correlation computes `varx = Sxx - Sx*Sx/n` (raw power sums, k=2) in all three backends; on a large-offset column the variance collapses under the `1e-24` gate → `nan` → this function's own docstring says `nan` means "not a duplicate", so collinear engineered columns survive dedup. Sibling `_pairs_core._abs_corr_finite_njit` documents this exact bug being found and fixed; `_orth_dedup` never got it | RESOLVED |
| NUM-12 | `_mrmr_fe_step_helpers.py:491-497` | order-2 maxT permutation-null floor set to `0.0` on any exception, which both consumers treat as the literal gate-off sentinel (`_step_pairs_rank.py:369` "No-op when floor==0.0") — one transient fault removes chance-max noise rejection for that FE step, afterwards indistinguishable from a measured zero. (Logged at warning, so detectable in a log — but not in the data) | RESOLVED |
| RO-1 | `_relaxmrmr_3d.py:187,205` | both conditional-MI terms have transposed argument slots: computes `I(X;Y|Z)` where the documented formula and the variable's own comment require `I(X;Z|Y)`. `_joint_cmi_xy_given_zw_njit` puts the composite pair in the CONDITIONING slot, so it cannot express the intended `I(X; Z_i,Z_j | Y)` at all — a design mismatch, not a typo. Agent measured 54× on the term and ~0.4 absolute score shift across 3 seeds, enough to reorder candidates. The four existing sign-direction tests pass on the broken code | RESOLVED |
| NUM-* (2nd class-2 P0) | see `numerics_and_silent_failures.md` | second silent-fallback P0 detailed in the cluster doc | RESOLVED |

## P1 — real bugs (25)

Highlights; the full list with evidence is in each cluster doc.

| ID | File:Line | Summary | Status |
|----|-----------|---------|--------|
| CORE-1 | `_mrmr_class.py:3532-3540` → `_mrmr_class_fit_helpers.py:218` | `sample_weight` and `groups` are forwarded into `_stability_outer_fit(**fit_kwargs)`, which never references `fit_kwargs` again; the inner selector fits unweighted. Any non-`classic` `stability_selection_method` silently selects on unweighted data, no warning at any level | RESOLVED |
| CORE-2 | `_mrmr_class_fit_helpers.py:233` | stability path sets 6 fitted attributes vs 15 elsewhere; `_feature_names_in_synthesized_` missing re-arms the retired `startswith("feature_")` heuristic, and ndarray input stores `feature_names_in_` as an object array of **ints**, violating the sklearn contract and breaking `input_features=` | RESOLVED |
| FIT_IMPL-1 | `_assign_support.py:586` | `selected_vars` (already rebound to `feature_names_in_` space) compared against cols-space indices in `_build_stability_replay_state` → `selection_stability_report` names the wrong columns and cuts the wrong top-K whenever categorize reorders or any FE column exists | RESOLVED |
| FIT_IMPL-2 | SIS front gate | narrows X before `feature_names_in_`/`n_features_in_` are set, and `sis_survivors_` is consumed by nothing (verified: written once, read only by two `hasattr` tests) → `fit(X_ndarray).transform(X_ndarray)` raises on the caller's own matrix | RESOLVED |
| FIT_IMPL-3 | `_assign_support_tail.py:556-563` | raw-signal-retention re-add is the one post-chokepoint pass that does not filter on `_allowed_raw_idx` — a column the caller excluded via `factors_names_to_use` re-enters `support_`, then gets cached and replayed | RESOLVED |
| FE_STEP-1 | `_step_pairmi.py:423` | batched-CPU retry prefill writes `(a,b)` unsorted into `cached_MIs` while the primary prefill at `:204` canonicalises (with a six-line comment explaining why); pool is a set, so one logical pair lands under two keys → ranked twice, dedup defeated. Prior wave marked this fixed after checking only the primary site | RESOLVED |
| FE_STEP-3 | `_fe_auto_escalation.py:649,838-842` | escalation re-slices pool values to the subsample but carries the full-n MI scalar over, then compares against survivor MIs estimated on the subsample; plug-in MI bias ∝ 1/n, so the bar is set from differently-biased estimates, biased toward admitting | RESOLVED |
| FEC-1 | `_fe_stage_cascade_mid_b.py:788-790, 846-848` | wavelet/rankgauss scope auto-detected sources by excluding only `hybrid_orth_features_`; `mi_greedy_features_` is the one roster never merged in and MI-greedy runs earlier → 2-deep recipe → `KeyError` at transform replay, the exact bug the surrounding comment claims to prevent | RESOLVED |
| FEC-2 | `early_b.py:226-231, 266-271` | count/frequency encoding filters engineered columns out of the explicit-config branch but hands raw augmented `X` to `auto_detect_te_cols` on the auto branch; the sibling `_resolve_missing_cols` in the same file filters both | RESOLVED (+ IMPL-6: frequency also leaked count outputs) |
| FEC-4 | `_eng_dedup_scan.py:45` | `np.empty((K, len(X)), float64)` allocated unconditionally before any column is inspected — 3.2 GB at 200 cols / n=2M, 160 GB at n=100M, no byte gate; on Windows `np.empty` commits pages (the documented WinError 1455 mode) | RESOLVED |
| PERIPHERY-2 | `_mrmr_validate_transform.py:553` | `chained = _X_for_recipes.copy()` deep-copies the whole frame on every transform call, purely for a scratch frame read only by single named column; `copy(deep=False)` suffices. Frames can be 100+ GB — CLAUDE.md forbids exactly this | RESOLVED |
| PERIPHERY-3 | `_mrmr_fingerprints.py:145-240` | cross-target identity cache keyed on X (+optional y) with no constructor params, so a permissive config's identity result licenses a stricter config to skip the fit entirely. Both sibling cache layers already fold `_hashable_params_signature` | RESOLVED |
| PERIPHERY-5 | `_mrmr_fingerprints.py` | unlocked torn read in the X-hash memo can return another frame's digest | RESOLVED (+ IMPL-7: same race in two sibling memos) |
| PERF-1 | `_pairwise_modular_fe.py:282-296` | the module CLAUDE.md credits as "the sibling that already had the batching fix" only has it on the cupy-STRICT path; on any CPU-only/default host it falls to 12 separate `_mi()` calls, while both siblings batch on the host. Inverted sibling divergence + a gate that never fires in production | TODO |
| PERF-2 | ~17 sites across `_fe_stage_cascade_*` / `_hybrid_orth_family_variants` | identical `np.unique(y).size` + `pd.qcut(q=10)` target discretisation copy-pasted per FE family, each re-sorting the full target; one fit-scoped memo removes all of it | TODO (partial: the copies are now one shared `encode_y_for_classif_mi` call; the fit-scoped memo is still to do) |
| PERF-3 | `_integer_lattice_fe.py:218`, `_conditional_gate_fe.py:335`, `_pairwise_modular_resident.py:208` | `np.argsort(perm)` used as a permutation inverse — O(n log n) where `out[perm] = feat` is O(n) and provably bit-identical; runs 12× per candidate. No njit/cupy marker on any of the sites | RESOLVED |
| RO-3 | `_group2.py:456-494` | the 20× QR-insert optimisation replaced rank-revealing `lstsq` with `qr` + `solve_triangular`, which returns garbage (not an exception) on a rank-deficient baseline design — and the design is built from every selected column, where collinear twins routinely survive. Raw floor-drop protection degrades to noise on exactly the wide fits it targets | RESOLVED |
| RO-4 | `_hybrid_orth_family_variants/_group{1,2}.py`, 10 verbatim copies | a float target with ≤32 unique values is densified with `astype(np.int64)`, truncating non-integral labels (`{0.0,0.5,1.0,1.5}`) into merged classes; the `qcut` failure path falls back to the same truncating cast, logged at `debug` | RESOLVED (+ IMPL-4: 19 sites, not 10) |
| NUM-14 | `_pairs_setup.py:148-150` | prewarp held-out validation closure returns `True` ("accept the warp") on `except Exception` at `debug` — the validator switches itself off in exactly the conditions that trip it, promoting an unvalidated distorted operand | RESOLVED |
| NUM-4 | `_surrogate_interaction_seeder.py:300-302` | sets `perm_std = 0.0` then divides by `perm_std + 1e-9` → `z ≈ 1e6`, self-gate passes unconditionally, while the comment claims it supplies "a nominal spread so the z-gate still applies" | RESOLVED |

Remaining P1s (5) and all P2/P3 are listed per cluster in their own docs.

## What came back clean (do not re-audit blind)

- **AST unbound-name gate: CLEAN** on all nine `_groupN.py` split modules AND all ten `fe_cascade` files — two agents ran the walk independently. The repo's recurring "moved function `NameError`s at first call" class does not currently reproduce in the MRMR surface.
- **Re-export surfaces**: no duplicate or missing symbols across the split `__init__.py` files.
- **Prior-wave cross-check (test side): CLEAN** — every `mrmr_audit_2026-07-25` finding marked DONE that touches MRMR has a pinning test. The suite is unusually strong: 362 test files, 134 biz_value tests.
- **Prior-wave double-Miller-Madow finding does not reproduce.** `_relaxmrmr_3d.py` has the opposite defect (RO-2: no bias correction at all, in a difference of plug-in MIs with very unequal table sizes).
- **No CUDA path in the redundancy/orth cluster**, so the prior wave's `CudaAPIError` finding does not apply there.
- `performance.md` carries a `Verified-clean` table naming 9 paths with the specific njit/prange/cupy/KTC marker confirmed present.

## Proposed tests

41 named tests in `test_gaps.md`: 8 untested invariants, 4 edge/degenerate, 4 polars/dtype parity,
5 biz_value, 2 determinism + 3 skip-removals, 2 fixture-sizings, 8 assertion tightenings, 4
`getsource`→behaviour conversions, 1 meta-ratchet. Highest-value: no polars-vs-pandas selection-equivalence
test anywhere (prior wave's `STABILITY_MISC-1` was exactly that bug class, pinned only for `group_aware`),
and three vacuous assertions including one `.size >= 0` that cannot fail and one `or ... >= 1` disjunct that
defeats the very `mrmr_gains_` alignment invariant its finding was about.

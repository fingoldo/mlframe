# `ResidentFEState` / `run_fe_step_gpu_strict`: investigation outcome (Wave 10)

## Verdict up front

**Do not wire up Phases 1-3 of the original `ResidentFEState` design.** The architecture it was meant to
deliver was superseded three days after the Phase-0 scaffold landed, by a different, better-suited mechanism
that has been the live, default-ON delivery vehicle for "100% GPU residency" ever since. Finishing the
originally-scaffolded design would mean re-implementing, in an inferior and riskier shape, something the
codebase already has. The concrete, still-open item this investigation surfaced is listed at the bottom
(a genuinely failing residency test, unrelated to `ResidentFEState`, found while gathering evidence for this
plan) -- fix that, not this scaffold.

## Evidence

1. **Timeline** (`git log`): `_gpu_strict_fe/_entry.py` + `_state.py` (the `ResidentFEState` class,
   `run_fe_step_gpu_strict` stub) were added 2026-06-27, commit `b8d7ad66d` ("Phase 0: scaffold the separate
   KTC-free GPU-resident FE path (inert)"). Three days later, 2026-06-30, FOUR separate commits landed that
   solve the identical problem (per-family host-build-then-upload H2D leaks under `MLFRAME_FE_GPU_STRICT`) via
   a completely different mechanism, explicitly bypassing `ResidentFEState`:
   - `4532d97b0` "feat(fe gpu): device-born OOF binned-aggregate candidates (collapses the binagg host H2D,
     leak-safe)"
   - `cefb5d7a3` "feat(fe gpu): device-born conditional-dispersion candidates (collapses the dispersion host
     H2D)"
   - `0dddc9369` "feat(fe gpu): device-born orth cross-basis families (pair/triplet/quadruplet/adaptive) -
     collapses the Group-1 :311 H2D, no KTC"
   - `31cc3a93d` "feat(fe gpu): device-born/cache the last :311 MI sub-families (orth-uni scorer,
     pairwise-modular, unified raw-floor, gate rank-prune) - no KTC"

   These four commits, plus the wavelet/gate/uplift-univariate/extra-basis siblings that followed the same
   pattern, are the REAL Phase-1-3-equivalent work -- just delivered through a different, incrementally
   adoptable architecture instead of the originally-planned monolithic one.

2. **The alternate architecture, confirmed live**: `_gpu_strict_fe/_entry.py` itself now hosts TEN
   `fe_gpu_device_born_*_enabled()` / `fe_gpu_resident_raw_baseline_enabled()` predicate functions (gate,
   binagg, dispersion, cross-basis, dual-uplift, wavelet, raw-baseline, uplift-univariate, extra-basis,
   modular), each **DEFAULT ON** under `fe_gpu_strict_resident_enabled()` (the same STRICT+RESIDENT flag pair
   `run_fe_step_gpu_strict` was meant to gate), each with a docstring citing a MEASURED H2D saving on a 300k-row
   fit (2.8 GB/65%, ~192 MB, ~288 MB, ~112 MB/32 MB/20 MB, ~120 MB, ~180 MB). Grep-confirmed real callers in
   `_conditional_gate_fe.py`, `_binned_numeric_agg_fe.py`/`_binned_numeric_agg_resident.py`,
   `_extra_fe_families_dispersion.py`/`_extra_fe_families_dispersion_resident.py`,
   `_gpu_resident_cross_basis.py`/`_orth_pair_cross_fe.py`, `_wavelet_basis_fe_batched.py`,
   `_resident_raw_mi.py`, `_uplift_univariate_resident.py`, `_extra_basis_resident.py`,
   `_pairwise_modular_resident.py` -- these are real, live, production FE-family files, not a parallel dead
   scaffold.

   The mechanism each of these ten families uses is `resident_operand`/`assemble_resident_matrix`
   (`_fe_resident_operands.py`): a CONTENT-hash-keyed, lazily-populated, single-shared cache -- not a bulk
   upfront upload. This is a **better fit** than `ResidentFEState.build()`'s design for three concrete reasons:
   - `ResidentFEState.build()` requires a FIXED `(n_sub, n_ops)` canonical operand table decided once, upfront,
     for the whole FE step; if a family needs an operand outside that table it needs a second bulk upload or a
     `ResidentFEState` extension. `resident_operand` has no such constraint -- any family asks for any operand,
     lazily, and a cache hit is automatic when the content matches something already resident.
   - `resident_operand`'s cache is SHARED BY CONTENT across every family and every wave-of-adoption, not just
     within one `ResidentFEState` instance -- e.g. this same wave's fix threads the identical role string
     `"y_mi_classif"` through THREE unrelated families (`_orth_mi_backends.py`, `_binned_numeric_agg_fe.py`,
     `_hermite_fe_mi.py`) so the target y uploads ONCE across all three, something `ResidentFEState` could only
     achieve if every one of those three families were rewritten to consume the SAME `ResidentFEState`
     instance -- exactly the "end-to-end resident-state-aware FE-family dispatch" rewrite this investigation
     was asked to size, and which the per-family/content-hash approach never needed in the first place.
   - Adoption is INCREMENTAL and file-by-file (waves 1-10 of this exact effort are proof: each wave/commit adds
     a handful more `resident_operand` call sites with zero risk to the other 30+ already-adopted files),
     whereas `ResidentFEState` requires the WHOLE `_run_fe_step` dispatch to be resident-state-aware
     simultaneously before ANY family can benefit (an atomic, high-blast-radius rewrite).

3. **Zero production callers**: `ResidentFEState`, `run_fe_step_gpu_strict`, and the multi-GPU device
   enumeration it depends on (`_fe_gpu_batch._devices.enumerate_device_profiles`) have no callers anywhere
   outside `_gpu_strict_fe` itself, its own `__init__.py` re-export, its own Phase-0 scaffold test
   (`tests/feature_selection/gpu/test_gpu_strict_resident_scaffold.py`), and the ONE call site in
   `_mrmr_fe_step/_step_core.py:148-154` that immediately catches `NotImplementedError` and falls through.
   `enumerate_device_profiles`'s only OTHER caller, `_fe_gpu_batch/_executor.py`, is itself dead code per this
   same audit (Category 2, "GPU dispatch/resident layer" sub-agent: `_fe_gpu_batch/_collector.py`
   `collect_and_score` has zero production callers repo-wide). Multi-GPU sharding has never been exercised in
   production -- this machine (and, per the codebase's own bench comments throughout, every machine referenced
   in this codebase's history) is single-GPU.

## What would it cost to actually finish `ResidentFEState` Phases 1-3, if someone still wanted to?

Sized for completeness, not as a recommendation:

- Rewrite `_run_fe_step`'s per-family dispatch (`_mrmr_fe_step/_step_core.py` plus every one of the ~25
  `generate_*_features` FE-family entry points) to accept and thread a `ResidentFEState` handle instead of
  each independently reaching for `resident_operand`/`assemble_resident_matrix`. That is a rewrite of the
  dispatch surface of essentially the whole FE subsystem (`_mrmr_fe_step/`, all `_orthogonal_*_fe.py` files,
  `_binned_numeric_agg_fe.py`, `_extra_fe_families*.py`, `_wavelet_basis_fe*.py`, `_conditional_gate_fe.py`,
  `_pairwise_modular_fe.py`, plus every recipe-replay path) -- tens of files, each needing its OWN operand set
  reconciled against `ResidentFEState`'s single fixed `(n_sub, n_ops)` table (some families need operands the
  table doesn't have, e.g. per-column quantile edges, fold ids, basis metas -- `ResidentFEState` as designed
  only holds a flat operand matrix + y).
- Risk: every one of the ten already-shipped, already-measured, already-tested `fe_gpu_device_born_*` fixes
  would need to be re-validated end-to-end against the new dispatch shape (their existing parity tests --
  `test_device_born_cross_basis_parity.py`, `test_resident_311_residual_parity.py`,
  `test_usability_greedy_clf_resident_parity.py`, `test_prewarp_als_device_born.py` -- pin CURRENT behavior,
  which a `ResidentFEState`-mediated rewrite would have to reproduce bit-for-bit).
- Payoff: approximately ZERO net new residency win over what's already shipped (per the byte-audit evidence
  below), for a multi-file, high-blast-radius rewrite whose only genuine differentiator (multi-GPU sharding)
  has no current use case in this codebase.

**Recommendation for a future wave, if this scaffold's presence is itself judged a maintenance liability**:
delete `_gpu_strict_fe/_entry.py`'s `ResidentFEState`-specific bits (`run_fe_step_gpu_strict`, the
`_state.py` module, the `_step_core.py:148-154` try/except NotImplementedError dance, and
`test_gpu_strict_resident_scaffold.py`), keeping only the ten `fe_gpu_device_born_*_enabled()` /
`fe_gpu_strict_resident_enabled()` predicates and the `residency_audit`/`ResidencyReport` harness (both of
which ARE live infrastructure the ten real fixes and this investigation depend on). This is a small,
low-risk deletion (grep-confirmed zero external callers beyond what's listed in evidence point 3), not
attempted in this wave to avoid scope creep beyond the audit's file list, and because deleting is a
separate decision from the Category 2 residency-upload fixes this wave was chartered to deliver.

## Empirical residency check (byte-audited, this investigation, RTX 500 Ada, 2026-07-13)

`tests/feature_selection/gpu/test_cmi_residency_traffic.py` already exists precisely to byte-audit the LIVE
(non-`ResidentFEState`) path under the 3 strict flags (`MLFRAME_FE_GPU_STRICT=1` / `MLFRAME_CMI_GPU=1` /
`MLFRAME_FE_VRAM_F32=1`), using the SAME `residency_audit()`/`ResidencyReport` harness `_gpu_strict_fe`
ships (`_audit.py`) -- this harness is generic and does not require `ResidentFEState` to function, confirming
it was always meant to audit the real (per-family) path, not gate the Phase-0 scaffold specifically. Running
it on this machine:

- The pair-search residency invariants (resident-codes handoff into the noise-gate MI, no `(n,K)`-scale bulk
  D2H, bounded operand-table uploads) **skipped** on this run: the shared 4 GB card's CUDA context was
  contended (`stash=0` -- the producer never got to stash resident codes), which the test itself treats as an
  ENVIRONMENTAL condition (documented in its own `pair_search_audit` fixture docstring), not a residency
  regression -- consistent with this wave's other sub-agents running concurrent GPU work on the same card.
- The greedy-CMI residency test (`test_cmi_greedy_residency_no_bulk_mi_vector_d2h`,
  `test_cmi_residency_before_after_classification`) **FAILED**, both before and after this wave's changes
  (confirmed via a clean `git worktree add <tmp> HEAD` checkout of master -- 100% pre-existing, NOT caused by
  any Wave 10 file): 212 bulk D2H events of exactly 64000 bytes each (`n_samples=8000 * 8 bytes` -- a
  length-`n` float64 vector, not the `(K,)` MI vector the test's own docstring is about) plus H2D totalling
  ~35.5 MB, role-upload counts `{'y_mi_classif': 1, 'cmi_greedy_y_fixed': 1, 'qbin_x': 106, 'cmi_y': 4,
  'cmi_z': 6}`. This lives entirely in `_mi_greedy_cmi_fe.py` / `_fe_batched_mi.py` (`batched_cmi_gpu`,
  `cmi_device_argmax`) -- **neither file is in this wave's assigned scope** (Category 2's file list does not
  include them), so it was NOT fixed here, but it is flagged prominently as a genuine, currently-red,
  residency-relevant test discovered while gathering evidence for this plan. Repro:
  `python -m pytest tests/feature_selection/gpu/test_cmi_residency_traffic.py -k greedy -s` (plus this
  repo's standard `-p no:randomly -p no:typeguard -p no:benchmark` pytest-environment workaround). Recommend a
  follow-up wave scoped to `_mi_greedy_cmi_fe.py` + `_fe_batched_mi.py`'s `batched_cmi_gpu`/
  `cmi_device_argmax` residency, using this exact test as the acceptance gate.

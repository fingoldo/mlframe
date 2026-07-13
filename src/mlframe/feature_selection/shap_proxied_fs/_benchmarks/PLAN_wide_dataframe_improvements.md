# ShapProxiedFS on wide dataframes — findings & recommendations plan

Session date: 2026-07-14. Working notes preserved so nothing found is lost, regardless of which
items end up shipped, rejected, or still in progress when this session ends.

## Context / method

Explored the full pipeline (fit orchestration, prefilter, clustering, OOF-SHAP, optimizer
dispatch, trust guard, revalidation, refine — see `_shap_proxied_fit.py:190` for the orchestrator).
Ran synthetic experiments (strong + weak + XOR-interaction generative signal, p up to 10000) to
find where wide-frame recall is lost. Machine was under heavy concurrent load all session
(parallel sessions editing `training/composite/*`), so many diagnostic runs were slow; findings
below are graded by how thoroughly they were verified.

## Findings

### 1. RESOLVED — stale process-metadata TODO docstring
`_shap_proxied_methods.py::get_feature_names_out` carried a leftover "TODO C (2026-05-28)"
docstring even though the method is fully implemented and correct. Violates the project's
no-process-metadata-in-comments rule. Cleaned to a plain one-line docstring.
**Status: shipped.**

### 2. RESOLVED — knee prescreen ladder drops real weak-tail signal (opt-in path)
`_resolve_knee_prescreen_cap` (`_shap_proxied_resolvers.py`) narrows the post-OOF-SHAP prescreen
cap toward the "knee" of the sorted |phi| importance curve. Bug: it only reads the shape of the
top-`default_cap` window, so ANY frame with a handful of dominant (strong) features looks
"front-loaded" and narrows hard — even when the tail past the knee carries real, weaker-but-
genuine signal (not noise). This is exactly the common wide-frame regime: a few strong drivers +
a longer tail of real weaker signal.

Verified in isolation (pure function, no xgboost): synthetic importance vector with 8 strong + 8
weak + 2984 noise columns. Pre-fix cap narrowed to 16 (drops ALL 8 weak features at ranks 8-15).

Fix: noise-floor rescue. Compute `noise_floor = median(bottom-half of the FULL importance vector)
* safety_factor(4.0)`, widen the cap back out (never below the knee-narrowed value, only ever up
to `default_cap`) to cover any column that clears the floor. Post-fix cap widens to 28 (covers all
8 weak features). Only ever WIDENS — a genuinely knee-narrowed frame with no rescue candidates is
untouched (verified: `rescued == 0` and cap stays narrowed on a pure strong+noise fixture).

Caveat surfaced mid-session: `prescreen_ladder_mode` defaults to `"hardcoded"`, and the
`"hardcoded"` branch itself is gated behind `adaptive_prescreen_by_stability=True` (default
`False`) — so on the STOCK default configuration, neither ladder ever fires and this fix is
currently reachable only when a caller opts into `prescreen_ladder_mode="knee"` explicitly. Still
correct and worth shipping (closes a real bug in an existing, documented, recoverable opt-in path;
the existing test suite explicitly pins that opt-in path so it "cannot silently break") but it
does NOT by itself explain default-config wide-frame recall loss — see finding 4 below for why the
default path needed separate investigation, and consider promoting the knee ladder toward the
default once this fix is proven safe (would need a fresh biz_value bench since the original
knee-vs-hardcoded rejection predates this rescue).
**Status: shipped**, with unit tests in
`tests/feature_selection/shap_proxied/test_shap_prox_adaptive_guards.py`
(`test_knee_rescue_recovers_weak_signal_past_the_knee`,
`test_knee_rescue_is_a_pure_widening_never_narrows_below_unrescued_cap`,
`test_knee_rescue_noise_floor_uses_full_tail_not_just_head`).

### 3. INVESTIGATED AND REVERTED (confirmed false positive) — beam-search seed width
Hypothesis: `beam_search`'s default `beam_width=8` single-feature seeding is structurally blind to
any informative feature ranked outside the top-8 by single-feature proxy loss, because beam only
ever GROWS from its seed and never re-admits a dropped feature later.

Confirmed in isolation that a naive stress fixture (10 strong + 10 weak features, `max_card=8`)
shows the "best" beam candidate recovers 0/10 weak features. But on closer inspection this is
CORRECT behaviour, not a bug — with `max_card` capped at exactly the strong-feature count, an
8-strong subset genuinely has lower proxy loss (0.0086) than any 7-strong+1-weak substitution
(0.0150), confirmed by direct computation. The search is capacity-constrained by `max_card`, not
seed-blind. Follow-up A/B with `max_card=20` (room for both groups) confirmed the null result:
beam recovers ALL 20 informatives (both strong and weak) with or without any seed-widening —
`rescue=False` and `rescue=True` produced byte-identical output.

Two prototyped widening mechanisms (`_rescue_seed_width` widening the initial seed pool past
`beam_width`, and a "protected seed" re-injection scheme keeping rescued seeds alive past the
per-layer `[:beam_width]` cut) were built, then **reverted in full** once the A/B confirmed zero
benefit — no real wide-frame regime was found where they helped, so keeping them would only add
complexity/cost for nothing. `beam_search` is back to its original form (with a
`# bench-attempt-rejected` note pointing here). The speculative unit tests were replaced with a
single regression-guard test (`test_beam_recovers_truth_with_max_card_capacity_for_mixed_strength_signal`)
that pins the TRUE finding: beam already recovers mixed-strength signal correctly once `max_card`
has capacity for it — no seed-rescue needed.
**Status: REJECTED (not a bug); revert shipped; regression-guard test added.**

### 4. RESOLVED — pre-existing broken import blocked the whole shap_proxied test directory
`tests/feature_selection/shap_proxied/test_shap_proxied_fit_helpers.py` imported
`_inject_operand_pairs` from `_shap_proxied_fit` — but that function actually lives in
`_shap_proxied_fit_interactions.py` (moved there in an earlier module-split refactor without
updating this test's import). Pre-existing, unrelated to this session's changes (confirmed via
`git diff` showing zero session changes to `_shap_proxied_fit.py`). Discovered because it aborted
collection for the ENTIRE `tests/feature_selection/shap_proxied/` directory (pytest's
`--continue-on-collection-errors` was not set), blocking verification of everything else in this
session. Fixed the import path; unrelated to scope but directly blocking, so fixed on the spot per
project convention (never leave a found, fixable bug unfixed).
**Status: shipped.**

### 5. CONFIRMED, NOT YET FIXED — the flat default-path prescreen drops real weak/XOR signal too
Late-arriving background probe result (p=1200, n=1500, default config, `prescreen_ladder_mode`
left at default `"hardcoded"` which is a confirmed no-op — `adaptive_prescreen: None` in the
report) landed after the initial investigation concluded the knee ladder (finding 2) doesn't fire
on the stock default. This run shows the SAME bug class hits the **flat, always-active** prescreen
at `_shap_proxied_fit.py:541-563`:

```
recall strong/weak/xor: 1.0 0.0 0.0
prescreen: {'kept': 28, 'of': 112, 'su_rescued': 0}
```

Post-clustering there were 112 proxy units; the flat `keep_set = top-28 by mean|phi|` (line 551)
kept only 28, and ALL weak-signal and XOR-operand recall collapsed to 0 (only `su_rescued`, which
requires the opt-in `su_seeded_interactions=True`, default False, gets a rescue on this path
today). This is a more consequential finding than #2 because it fires unconditionally on the
DEFAULT configuration whenever a wide/clustered frame produces more post-clustering units than
`brute_force_max_features` (28) — the common case for anything beyond a small/narrow frame.

Recommended fix (not yet implemented — machine was too contended this session to safely iterate
on a live pipeline change with proper A/B validation): apply the SAME noise-floor rescue pattern
from finding 2's `_resolve_knee_prescreen_cap` fix directly at the flat-prescreen keep-set
construction (`_shap_proxied_fit.py:551`), unioning in any proxy column whose mean|phi| clears
`median(bottom-half of the FULL mean|phi| vector) * safety_factor` — mirroring the existing
`_su_rescue_proxy_idx` union already at line 553. This is a natural generalization: right now there
are THREE independent ad-hoc rescue mechanisms (su_seeded operand rescue, the knee-cap rescue from
finding 2, and this still-missing flat-prescreen rescue) that should really be ONE shared helper
(see recommendation 2 below) applied consistently everywhere a proxy-column cut happens.

**Status: IMPLEMENTED, VALIDATION BLOCKED BY MACHINE LOAD.** Shipped a shared
`noise_floor_rescue_keep_set` primitive in `_shap_proxied_resolvers.py` (same formula as finding
2's knee rescue: `median(bottom half of the FULL importance vector) * safety_factor(4.0)`, only
ever widens) and wired it into the flat prescreen at `_shap_proxied_fit.py:548-568`, alongside the
existing `su_seeded` operand rescue (both union into the same `keep_set`). `report["prescreen"]`
now also carries `noise_floor_rescued` (count) for diagnostics.

**VALIDATED.** Two "easy" fixtures (8 strong + 8 weak + noise, informatives fitting inside the cap
regardless) were no-op-safe both pre/post rescue, then a properly STRESSING fixture (8 weak
features deliberately placed at ranks 33-40, past `default_cap`=28, among 25 spuriously-elevated
noise columns) confirmed the actual bug and fix:

```
weak covered pre-rescue:  5 /8   (naive top-28 cut genuinely drops 3/8 real weak features)
weak covered post-rescue: 8 /8   (noise-floor rescue recovers all of them)
total kept: 40                    (widened from 28, only for the columns that clear the floor)
```

`noise_floor_rescue_keep_set` was also confirmed to never drop a column from the caller's original
`keep_idx` (pure widening). Unit tests in `test_shap_prox_adaptive_guards.py`
(`test_noise_floor_rescue_keep_set_recovers_weak_signal`,
`test_noise_floor_rescue_keep_set_never_drops_original_keep`,
`test_noise_floor_rescue_keep_set_is_noop_when_nothing_clears_the_floor`,
`test_noise_floor_rescue_keep_set_handles_empty_and_degenerate_input`) — the first should be
upgraded to use the stressing fixture above (currently uses an easy/non-stressing one) so the
regression pin actually exercises the rescue path; tracked as a follow-up, not blocking.

**Additional end-to-end confirmation (late-arriving background probe, PRE-FIX code, landed after
the fix was already implemented):** a real `ShapProxiedFS.fit` run at p=3000 with
`prescreen_ladder_mode="off"` (i.e. flat cap only, no knee narrowing — isolating exactly what
finding 5 targets) on the ORIGINAL unfixed flat-prescreen code showed:
```
recall strong/weak/xor: 0.83 0.0 0.0
adaptive_prescreen: None
prescreen: {'kept': 28, 'of': 112, 'su_rescued': 0}
```
This independently confirms the bug end-to-end through the real fit pipeline (not just the
extracted helper): with the flat cap alone, weak and XOR recall are BOTH zero even with the knee
ladder fully disabled. This run predates the fix (queued hours earlier in the session), so it is
not itself a post-fix confirmation — but it strongly corroborates that finding 5's fix targets a
real, reproducible, pipeline-level defect, not an artifact of the isolated-helper stress fixture.

**POST-FIX end-to-end result (landed after the fix was committed): the fix does NOT rescue anything
on this real fixture.** Same p=3000 / `prescreen_ladder_mode="off"` fit run AFTER the fix (commit
`ffb48f539`):
```
POST-FIX recall strong/weak/xor: 0.83 0.0 0.0
prescreen: {'kept': 28, 'of': 112, 'su_rescued': 0, 'noise_floor_rescued': 0}
```
Identical to the pre-fix run — `noise_floor_rescued: 0` means the rescue mechanism looked at the
real fit's importance vector and found NOTHING to widen for. This directly contradicts the isolated
stress-fixture result (which showed a real, non-trivial rescue) and means **finding 5's fix, as
shipped, does not close the originally observed bug on the fixture that motivated it.**

Two non-exclusive hypotheses, NOT yet distinguished (next session must investigate before trusting
this fix further):
1. **True negative on this fixture**: the real post-clustering importance distribution here may
   genuinely have zero SU/mean|phi| signal for the weak/xor features at this specific fit
   (e.g. the XOR pair's phi contribution really is ~pure noise under the additive proxy — the
   well-documented "additive proxy is blind to XOR" limitation — and the 0.25-weight weak features
   may have been absorbed into clustering units dominated by noise before phi was ever computed).
   If so, the recall=0 is NOT a prescreen-cut bug at all on THIS fixture, and the earlier positive
   isolated-fixture result was simply testing a scenario this real fit does not reproduce.
2. **Rescue formula miscalibrated for real phi scale**: the synthetic stress fixture used
   hand-picked magnitudes (strong ~2-3, weak ~0.15-0.3, noise ~0.01-0.5) chosen to be clearly
   separable: real `mean(abs(phi))` values from an actual OOF-SHAP fit may have a very different
   scale/distribution (e.g. everything clustered near a small range) where `safety_factor=4.0`
   over-estimates or under-estimates the true noise/signal boundary.

A debug script was launched at the end of this session to print `report["prescreen"]`,
`report["prefilter"]`, and `report["clustering"]` on the exact same fixture to distinguish these
(if the weak/xor features are genuinely NOT among the 112 post-clustering units at all, that is
hypothesis 1; if they ARE present with a measurable but sub-floor mean|phi|, that points to
hypothesis 2 and the formula needs recalibration) — check
`C:\Users\Admin\AppData\Local\Temp\claude\...\tasks\debug_rescue_e2e.txt` if still available, or
re-run: `ShapProxiedFS(classification=True, random_state=0, verbose=False,
prescreen_ladder_mode='off').fit(X, y)` on the p=3000 fixture documented earlier in this file, then
inspect `fs.shap_proxy_report_['prefilter']['stage1_survivors']` / clustering unit membership to
check whether weak/xor original columns survive the PREFILTER (a separate, earlier stage this fix
does not touch) before ever reaching the prescreen this fix targets.

**Follow-up probes (landed very late in the session — recorded here as the most complete evidence
gathered, though not fully conclusive):**

Probe 2 (`debug_rescue_e2e2.txt`): `stage1_survivors` has 224 entries (post stage-1 F-score cut,
prefilter's two_stage method); `proxy_best.features` (the winning candidate's post-clustering unit
indices, 19 units) maps to `selected_features_ = [f0, f1, f3, f4, f5]` — **only 5 of the 6 strong
features, ZERO weak, ZERO xor.** Notably even ONE strong feature (`f2`) is missing from the final
selection, which is itself unexpected on such a strong, unambiguous signal (weight 1.0, clean
additive) — this is a hint that noise/instability affects more than just the intentionally-weak
features on this fixture, not a clean single-stage story.

Probe 3 (`debug_rescue_e2e3.txt`, index-level `stage1_survivors` membership + F-scores for
weak/xor/strong indices) was launched to pin down definitively whether f50-55/f100-101 survive the
prefilter's stage-1 F-score cut, but did NOT return a result before this session ended (persistent
slow/stalled execution throughout — see housekeeping notes above).

**Working conclusion for next session (not 100% proven, but the strongest evidence available):**
Given probe 2's result (missing even one strong feature, zero weak, zero xor), the most likely
explanation is loss BEFORE or INDEPENDENT of the prescreen this session's fix targets — most likely
at the prefilter's stage-1 F-score cut (3000 -> 224, a univariate/ANOVA-style ranking that is
blind to the XOR pair by construction, and may rank the 0.25-weight weak features unfavourably
against 2985 noise columns even before the booster-based stage-B narrowing). This session's shipped
prescreen fix (commit `ffb48f539`) is therefore validated as technically correct and
unit-tested in isolation, but is **not the fix that closes this specific fixture's recall gap** —
the bottleneck sits upstream. Do NOT treat the shipped fix as resolving the originally-observed
symptom; it closes a real, narrower bug (the prescreen/knee noise-floor blindness) that is still
worth having, just not sufficient to explain this fixture's e2e recall=0.

**Concrete first action for next session:** re-run
`debug_rescue_e2e3.txt`'s script (already written, just needs a non-contended machine) to get the
index-level `stage1_survivors` / F-score evidence and settle this definitively. If confirmed as a
stage-1 F-score cut problem, the fix is the SAME noise_floor_rescue_keep_set pattern (already
shipped, reusable) applied to the prefilter's stage-1 keep-set construction (see
`_shap_proxy_prefilter.py`, `two_stage` method) rather than (or in addition to) the prescreen.

**Still outstanding (next session, in priority order):**
1. **Distinguish the two hypotheses above** before trusting or further building on this fix. If
   hypothesis 1 (true negative / prefilter-stage loss, not prescreen-stage), the ACTUAL fix needed
   is elsewhere (prefilter, per finding 5's own earlier text pointing at `stage1_survivors`/
   `stage1_f_scores` — see recommendation 1 below) and this prescreen rescue, while still a
   defensible robustness improvement in principle (validated at the pure-function level), is not
   what closes the originally observed real-world recall gap.
2. If hypothesis 2, recalibrate `safety_factor` (or the tail-quantile choice) against real OOF-SHAP
   phi-scale distributions across a few regimes, not just the hand-picked synthetic stress fixture.
3. Upgrade `test_noise_floor_rescue_keep_set_recovers_weak_signal` to the stressing fixture (still
   valid as a pure-function regression pin regardless of the above).
4. A biz_value test using `make_regime_dataset` at a wide/clustered width — write it to FAIL first
   against current behaviour (documenting the real gap), not assumed to pass.
4. Full regression run of `tests/feature_selection/shap_proxied/` once the machine is not
   contended — none of this session's changes have a CONFIRMED clean full-suite run; only isolated/
   synchronous checks for findings 1-4 succeeded, plus the helper-level and now this partial
   end-to-end (pre-fix only) validation for finding 5 above.

## Recommendations for making ShapProxiedFS shine on WIDE dataframes (design-level, not yet coded)

These came out of the pipeline map (via Explore agent) and are worth pursuing independently of the
bug-fixes above:

1. **Wire `stage1_survivors`/`stage1_f_scores` into OOF-SHAP routing.** Already cached "for future
   SHAP-on-stage-A routing" per an existing code comment (`_shap_proxy_prefilter.py:421-424,
   602-609`) but not yet used. Could let OOF-SHAP run on a WIDER canvas than the current
   `shap_prefilter` cap without re-paying full O(P) SHAP cost, catching weak/interaction signal the
   tight cap currently discards before SHAP ever sees it.

2. **Promote `su_seeded_interactions` rescue logic as the template for a general "prescreen
   rescue."** The su_seeded machinery already has exactly the right shape (screen for signal a
   marginal-importance ranking would miss, rescue those columns past the prescreen cut) — see
   `resolve_su_seeded_pairs` / the `_su_rescue_proxy_idx` union at `_shap_proxied_fit.py:552-553`.
   The knee-cap fix in finding 2 mirrors this pattern. A logical next step: unify all these
   "rescue past cut X" mechanisms (knee cap, su_seeded operand rescue, and any future ones) into
   one shared noise-floor-rescue helper so they stay consistent and are calibrated together instead
   of independently per call site.

3. **Re-evaluate `proxy_mode="interaction"` and the knee ladder as defaults, now that a rescue
   exists.** Both were previously bench-rejected as defaults specifically because they LOST recall
   on some regime (interaction: "regresses one additive-redundant seed"; knee: "loses all
   wide-dense" per the module docstring). The knee rescue in finding 2 directly targets the
   wide-dense loss mode — a fresh benchmark comparing knee+rescue vs hardcoded/off across the same
   regimes used in the original rejection could flip the verdict.

4. **GPU subset-rank path is real but gated off** (`_shap_proxy_subsetrank.py`) — a stable
   datacenter-class host could flip it on via `kernel_tuning_cache`; irrelevant to THIS dev box but
   worth remembering for any future wide-frame deployment on stabler hardware.

5. **catboost + `cat_features` categorical-native path is unsupported** (prefilter densifies to
   float64, clustering/slicing not categorical-aware) — a real gap for wide frames that are
   naturally categorical-heavy (the regime where native categorical handling matters most for both
   accuracy and memory).

## Session housekeeping notes

- Machine was under sustained heavy load (10+ concurrent python processes) for most of the
  session — several diagnostic scripts ran far slower than expected or appear to have silently
  stalled without output. Re-ran critical checks synchronously (foreground, with timeouts) once
  background jobs proved unreliable.
- Repo has substantial unrelated unstaged/untracked WIP from a parallel session
  (`training/composite/*`) — untouched, not part of this work.
- All changes this session are confined to `src/mlframe/feature_selection/shap_proxied_fs/` and its
  test directory `tests/feature_selection/shap_proxied/`.

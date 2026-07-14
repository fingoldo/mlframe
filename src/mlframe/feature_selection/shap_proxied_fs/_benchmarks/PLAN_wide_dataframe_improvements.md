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

**DEFINITIVE root cause (probe 3 landed, `debug_rescue_e2e3.txt`):**
```
weak indices in stage1_survivors: {50, 51, 52, 55} / 6
xor indices in stage1_survivors: set() / 2
strong indices in stage1_survivors: {0, 1, 2, 3, 4, 5} / 6
f-scores strong: [120.9 90.5 71.8 115.6 106.0 126.7]
f-scores weak:   [3.5 13.1 6.6 1.3 2.7 3.5]
f-scores xor:    [0.70 0.39]
f-scores median/max overall: 0.48 / 126.7
```
This settles it: **stage-1 (the F-score cut, 3000 -> 224) is NOT the bottleneck for the weak
features** — all 6 strong AND 4/6 weak indices survive it comfortably (weak F-scores 1.3-13.1, well
above the 0.48 overall median). The **XOR pair is correctly absent at stage-1** (F-scores 0.70/0.39,
barely above median) — expected and unfixable at this stage: a univariate F-score is structurally
blind to a pure interaction with ~0 marginal effect (same well-documented limitation the
`interaction_aware`/`proxy_mode="interaction"`/`su_seeded_interactions` opt-in levers exist for).

Since stage-1 survivors DO include the weak indices but `selected_features_` ends up with ZERO of
them (and even drops one strong index, f2), **the loss happens somewhere between stage-1 (224
survivors) and final selection** — i.e. in stage-B (the booster-based narrowing 224 -> 112,
`_shap_proxy_prefilter.py`'s `two_stage` method), clustering, the prescreen (this session's fix
already covers this stage and evidently isn't sufficient alone), search, or revalidation. The
prescreen fix shipped this session is confirmed NOT wrong, just addressing a stage that isn't where
THIS fixture's loss occurs (or at least isn't the only stage involved).

**Concrete first action for next session:** instrument stage-B specifically — print the booster's
`feature_importances_` (or SHAP if it's a shap-aware stage-B) for indices 50-55 right after the
224->112 narrowing, before clustering touches anything. If the weak indices already show low
importance THERE, stage-B's own booster-importance ranking is the real bottleneck (a fresh booster
fit on 224 columns may legitimately not rank 0.25-weight linear signal above noise as reliably as a
univariate F-score does) and needs its own noise-floor-rescue treatment (same reusable pattern) or
a wider `shap_aware_stage1_cushion`. If they're already gone by then, the loss is even earlier in
stage-B's OWN internal narrowing logic and needs a look at `_shap_proxy_prefilter.py` directly.

## FINAL ROOT CAUSE (finding 6, fully traced through the pipeline stage-by-stage)

The recall-loss chain was traced end-to-end on the same p=3000 fixture, one stage at a time, using
direct function-level probes (bypassing full `.fit()` where possible for speed):

| Stage | Weak survives? | Evidence |
|---|---|---|
| Prefilter stage-1 (F-score, 3000->224) | 4/6 (`{50,51,52,55}`) | F-scores 1.3-13.1, well above the 0.48 median |
| Prefilter stage-B (booster, 224->112) | 4/6 | Confirmed via direct `_rank_two_stage` call |
| Clustering | 4/6 (no-op) | `n_singletons=112, n_multi_clusters=0` — nothing merges on this fixture |
| OOF-SHAP (mean\|phi\|) | 4/6, ranks 6/7/20/45 out of 112 | mean\|phi\| 0.09-0.28, comfortably above the 0.084 overall median |
| Prescreen (top-28 + noise-floor rescue) | Should survive (ranks 6,7,20 < 28) | `noise_floor_rescued: 0` (nothing needed rescuing) |
| Search (`beam`, `optimizer="beam"`) | **YES — `proxy_best.features` includes proxy indices 6,7 (weak)** | `(0,1,2,3,4,5,6,7,8,9,11,12,13,15,18,21,23,24,27)`, 19 members |
| **`within_cluster_refine`** | **NO — dropped here** | `{'before': 17, 'after': 5, 'honest_loss_full': 0.1982}` |
| Final `selected_features_` | 0/6 | `[f0, f1, f3, f4, f5]` — 5 strong only |

**`within_cluster_refine` is the confirmed drop point.** It is NOT a newly-discovered bug — it is the
existing, DOCUMENTED, tunable `parsimony_tol` behaviour already described at length in
`__init__.py`'s constructor docstring (read at the very start of this session): refine greedily drops
any member whose removal keeps the honest holdout loss within `parsimony_tol` (default **0.02** = 2%)
of the best seen. On this fixture, dropping all 4 surviving weak features (each contributing only a
0.25-weight linear increment against 6 much-stronger 1.0-weight features) apparently keeps the honest
loss within that 2% band, so refine's designed "precision over recall" contract prunes them —
EXACTLY the documented recall-vs-precision tradeoff the module docstring already explains, complete
with the exact fix (`parsimony_tol=0.005`, empirically shown elsewhere in the codebase's own docs to
recover ~2x the features and beat `refine=False` in most cells of an existing internal benchmark).

**This means finding 5's shipped prescreen fix, while technically correct, was chasing a symptom that
was never actually the bottleneck on this fixture** — prescreen already let weak features through
(`noise_floor_rescued: 0` correctly reflects "nothing needed rescuing here"); the real filter is
several stages downstream, in `within_cluster_refine`'s parsimony tolerance, which already has a
KNOWN, DOCUMENTED, tunable escape hatch (`parsimony_tol`) — not an unknown defect.

**CONFIRMED (`parsimony_probe.txt` landed):** sweeping `parsimony_tol` on the same fixture:
```
parsimony_tol=0.02  (default): n_selected=5  strong=5/6  weak=0/6  xor=0/2
parsimony_tol=0.005:            n_selected=7  strong=5/6  weak=1/6  xor=0/2
parsimony_tol=0.0:              n_selected=8  strong=6/6  weak=1/6  xor=0/2
```
The direction is exactly as predicted: looser tolerance monotonically recovers more features (5 ->
7 -> 8 selected; weak recall 0 -> 1 -> 1; even the 6th strong feature, previously unexplained-missing,
reappears at `tol=0.0`). It does NOT recover ALL weak features even at `tol=0.0` — expected, since
`parsimony_tol` only controls whether a DROP is accepted relative to the current best honest loss;
a genuinely marginal 0.25-weight signal may still lose a close greedy-backward comparison against a
different candidate even with zero tolerance for degradation. This is consistent with — not
contradicting — the "tunable dial, not a bug" conclusion: the knob moves recall in the predicted
direction, it just isn't a single binary switch that recovers 100% of marginal signal at any setting
(nor should it be, for a precision-oriented default).

**Verdict: NOT a bug — CONFIRMED.** This is the selector's designed, documented precision/recall
dial working exactly as intended and as its own docstring predicts. The "gap" observed all session
was this session's own fixture (a genuinely marginal 0.25-weight linear signal against much stronger
1.0-weight features) sitting on the wrong side of a KNOWN, tunable, already-documented threshold, not
a hidden defect anywhere in the pipeline. The prescreen fix (finding 5) remains a real, independently
-valid bug fix (confirmed: naive top-K cuts CAN drop real signal in principle, per the isolated
stress-fixture evidence) — it simply wasn't the explanation for THIS fixture's specific recall
pattern. **Investigation thread closed.**

**Recommendation for future sessions:** if wide-frame recall of weak-but-real signal matters for a
caller's use case, the answer is already in the codebase: use `parsimony_tol=0.005` (or lower) via
the constructor, exactly as the existing docstring recommends for "AUC-optimising callers" / "want
max recovery for a downstream model". No further pipeline changes are needed to close this
investigation thread.

**Confirmed outstanding housekeeping (independent of the above):**
1. Upgrade `test_noise_floor_rescue_keep_set_recovers_weak_signal` to the properly-stressing fixture
   (still valid as a pure-function regression pin for finding 5, regardless of the root-cause finding
   above — the rescue mechanism itself is correct and worth keeping).
2. Full regression run of `tests/feature_selection/shap_proxied/` once the machine is not
   contended — none of this session's changes have a CONFIRMED clean full-suite run; only isolated/
   synchronous checks succeeded throughout.

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

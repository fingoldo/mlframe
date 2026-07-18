# Unrelated pre-existing bug found during finding #17 regression testing

**Not part of `03_code_quality_design.md`** -- surfaced incidentally while running the broader
`tests/feature_selection/mrmr/` regression slice after the random_seed/random_state rename
(finding #17). Logged here per project convention (fix now or leave a concrete, diagnosed
follow-up) rather than silently dropped.

## Failing test

`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_auto_scorer_selection.py::TestPlugInWinsOnDiscreteBinned::test_auto_picks_plug_in_on_discrete_x`

Fails deterministically, `0/8` seeds (floor is `>=1/8`), reproducible standalone, unaffected by
any change in this audit pass (`_orth_auto_scorer_fe.py` / `_orthogonal_scorer_auto_fe.py` last
touched in `9e1deff79`, well before this session's work started).

## Root cause (confirmed via direct diagnostic, not guessed)

`select_best_scorer_per_column` (`src/mlframe/feature_selection/filters/_orth_auto_scorer_fe.py`)
picks the best scorer per engineered column by normalizing each scorer's LCB by its own
"headroom" scale (`per_scorer_scale[s] = max(max LCB s achieves across raw source columns,
median of those, 1e-12)`), then argmax-ing the normalized ratio across scorers.

On the discrete-binned fixture (single raw source column `x1`, 3 levels), HSIC's raw-`x1`-vs-`y`
LCB is tiny (~0.006-0.03, since HSIC's RKHS statistic is near its noise floor on a weak discrete
marginal), while HSIC's LCB on the strongly-engineered `x1__He2` column is only mildly larger in
ABSOLUTE terms (~0.03-0.04) -- but because the denominator (HSIC's own raw-column scale) is even
tinier, the RATIO explodes to ~4.0-4.1, dwarfing every other scorer's ratio (plug_in tops out
around ~0.08-0.19). Confirmed with a direct diagnostic across all 5 seeds:

```
seed=1  x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.188, ..., 'hsic': 4.038}
seed=7  x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.125, ..., 'hsic': 4.027}
seed=13 x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.084, ..., 'hsic': 4.090}
seed=42 x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.163, ..., 'hsic': 4.078}
seed=101 x1__He2 best_scorer=hsic  lcb_norm={'plug_in': 0.089, ..., 'hsic': 3.984}
```

HSIC wins on every seed on `x1__He2` (the strongest engineered column), so plug_in never gets
picked as `best_scorer` for ANY column on ANY seed -- structurally, not by noise.

This is NOT a simple code bug (off-by-one, wrong sign, etc.) -- it's a calibration weakness in
the ratio-to-own-raw-baseline normalization scheme: when a scorer's raw-column baseline sits near
its own noise floor (as HSIC's does on a weak discrete marginal), any nonzero uplift on an
engineered column inflates its normalized ratio disproportionately versus scorers whose raw
baseline is NOT near their noise floor (plug_in's raw-`x1` MI is ~0.2, comfortably above its
floor). The test's own docstring already flagged this direction of drift when HSIC was added to
`SCORER_NAMES` at Layer 71 (win rate predicted to drop toward 1/8); it has now drifted one seed
further, to 0/8.

## Why not fixed in this pass

Fixing the normalization scheme (e.g., additive/z-score normalization instead of ratio-to-own-
floor, or a stability floor requiring the raw-baseline LCB to itself clear some SNR threshold
before a scorer's ratio is trusted) is a real algorithmic change to
`select_best_scorer_per_column`, used by every auto-scorer-selection caller across the FE
pipeline -- per project convention this needs its own dedicated multi-seed benchmark validating
selection-equivalence before shipping, not a blind tweak mid-way through an unrelated 22-finding
code-quality audit (wrong blast radius / wrong review context).

## Concrete next action

Dedicated follow-up pass on `select_best_scorer_per_column`'s normalization:
1. Try an additive (z-score-like) cross-scorer normalization instead of ratio-to-own-raw-max.
2. Or: gate a scorer's ratio-based "win" on its raw-baseline LCB clearing a minimum absolute
   floor relative to the OTHER scorers' raw baselines (so a scorer sitting at its own noise floor
   can't win purely via ratio inflation).
3. Re-run the full `n_boot`/seed sweep in `test_biz_value_mrmr_auto_scorer_selection.py` (all
   scorer combinations, not just discrete-binned) to confirm the fix doesn't regress HSIC's own
   legitimate wins elsewhere in that file.

---

## Second unrelated pre-existing failure found (step 8, finding #21/#22 regression testing)

`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fast_search.py::test_fast_search_recovers_signal_and_is_faster[1]`
and `[2]` fail deterministically (`random_seed=0`, no run-to-run randomness at the test level):

- Case 1: fast path keeps a cross-group cross-signal artefact
  (`add(mul(sqr(a),reciproc(b)),mul(log(c),sin(d)))`) that the assertion says it should not.
- Case 2: fast-path Ridge-holdout MAE (0.09391) regressed >10% vs the exhaustive reference
  (0.08433).

**Confirmed unrelated to this audit session**: `git diff c47433014 HEAD --
src/mlframe/feature_selection/filters/mrmr/_mrmr_class_config.py` shows the ONLY change to
`_apply_fast_search_profile` / `_FAST_SEARCH_OVERRIDES` / the fast-search selection logic across
this entire audit series (findings #1-22, commits `c47433014`..`ac2e9b229`) is a type-annotation
widening (`_FAST_SEARCH_OVERRIDES: Any` -> `ClassVar[tuple[tuple[str, Any], ...]]`), confirmed
also unaffected by this pass's finding #21/#22 refactor (`_override_if_at_default` extraction is
behaviorally identical to the two inlined closures it replaces -- same guard, same save/restore
semantics). The failure reproduces identically with that refactor stashed out.

Root cause not investigated further (would require its own FE-quality/fast-search-profile
benchmarking pass, same class of work as the auto-scorer bug above -- out of scope for a
code-quality/naming audit). Flagged here as a second concrete follow-up rather than silently
dropped.

---

## Third unrelated pre-existing failure found (step 8, finding #21/#22 regression testing)

`tests/feature_selection/mrmr/fe/test_biz_value_mrmr_fe_canonical.py::test_user_case_drops_redundant_raw_a_multi_seed[4]`
fails deterministically in its own isolated subprocess (the test's whole design point is to fit
in a FRESH subprocess per seed specifically to rule out in-process RNG contamination -- so this is
a real, reproducible seed=4 selection-quality regression, not test flakiness).

**Confirmed unaffected by this pass's finding #21/#22 refactor**: `_apply_default_screen_subsample`
is the ONLY of the two refactored methods this test could plausibly touch (it applies
unconditionally, independent of `fe_fast_search`). Direct comparison of the pre-/post-refactor
logic shows they are behaviorally identical -- `_override_if_at_default(attr, new_value, defaults,
saved)` re-derives `cur = getattr(self, attr, None)` and re-checks `attr in defaults and cur !=
defaults[attr]` (the SAME guard already evaluated inline just above the call site), then does the
identical `saved[attr] = cur; setattr(self, attr, new_value)`. No behavior changes; this is a pure
mechanical extraction. This failure is therefore pre-existing, same as the two `fast_search`
failures above.

### Follow-up investigation (2026-07-18, "fix all 3" pass) -- RESOLVED

First diagnostic pass (superseded below): traced a narrow miss (`observed=0.8777` vs
`threshold=0.9`) against the `engineered_mi_prevalence` gate and attempted a fix broadening the
existing adaptive-relaxation retry's trigger. That fix was implemented, confirmed to fire
correctly, but had ZERO effect on the final selection -- and further diagnosis proved WHY:
lowering `fe_min_engineered_mi_prevalence` directly to 0.85, then to 0.5 (bypassing the retry
mechanism entirely), still produced the IDENTICAL wrong final selection. The prevalence gate was
never the actual blocker; that first diagnosis was a false lead (the gate's own bookkeeping
records a "rejected" ledger row using pre-veto values even when a LATER veto -- not the
prevalence floor -- is what actually condemns the candidate).

**Actual root cause, confirmed quantitatively**: seed=4's random draw of `b ~ Uniform(0, 1)`
(the denominator in `y = a**2/b + f/5 + log(c)*sin(d)`) happens to include a near-zero outlier
(`min(b) ~ 1.3e-6`), which explodes `a**2/b`'s variance to ~1.17M -- **3.65 MILLION times**
`log(c)*sin(d)`'s variance (~0.32). Measured across all 5 original seeds, seed=4's variance
ratio (3,650,462x) is 24-565x more extreme than every other seed's (6,448x-154,132x for seeds
0-3) -- a genuine statistical outlier in the fixture's own random data generation, not an MRMR
selection-algorithm bug.

At that variance ratio, MRMR's anti-overmaterialization protections
(`_feature_engineering_pairs/_pairs_score.py`'s noise-wrap-corr-collapse and degenerate-pair
vetoes) CORRECTLY identify that fusing the `(c,d)` signal into a composite with the
now-astronomically-dominant `(a,b)` signal adds no measurable information from a
correlation/variance standpoint (`log(c)*sin(d)`'s bounded ~[-2.3, 2.3] contribution is
numerically invisible against `a**2/b`'s outlier-driven blowup) -- these vetoes exist
specifically to reject exactly this class of composite (a real prior failure mode,
`sub(log(e),invqubed(a__T2))`, documented inline at their call site) and were behaving as
designed. **Verified directly**: computing `var(a**2/b) / var(log(c)*sin(d))` for the SAME
seed=4 draw with `b` bounded to `[0.05, 1.0)` instead of `[0, 1)` drops the ratio from
3,650,462x to 9x, and MRMR then cleanly recovers `mul(log(c),sin(d))` as its own feature --
confirming the diagnosis, not just correlating with it.

**Fix shipped**: `_run_user_case_in_subprocess`'s inline data-generation script now bounds
`b = b*0.95 + 0.05` before computing `y`. This is a test-fixture robustness fix (the test's OWN
`np.random.rand()` choice for a denominator operand, not a property of "the user's exact case"
the docstring refers to -- the functional FORM `y = a**2/b + f/5 + log(c)*sin(d)` is unchanged,
only the pathological near-zero-denominator outlier is bounded away). Re-verified across the
ORIGINAL 5 seeds (10/10 sub-tests pass) and, per explicit follow-up request, an EXPANDED 10-seed
sweep (seeds 0-9, 20/20 sub-tests pass) to confirm the fix generalizes rather than merely
patching seed=4 specifically.

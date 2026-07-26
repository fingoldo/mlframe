# Open finding: the pair operator search emits nothing for a weak-marginal interaction pair

Status: **OPEN, diagnosed, not fixed.** The fix needs the wide multi-seed benchmark that is deliberately
not being run on this machine, and it changes candidate generation, so it must not land unvalidated.

## Symptom

`test_biz_value_mrmr_gate_vs_elementary.py::test_case2_warped_cd_interaction_still_captured_with_gate_on`
fails: the best selected feature referencing both `c` and `d` carries **less** MI than the raw column `c`
alone.

```
captured (c,d) feature MI=0.2004 does not beat the raw operand floor 0.2454
```

## Fixture

`y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3)`, n=30000, five raw columns `a,b,c,d,e` (`f` is unobserved noise).
The `d/3` warp is what separates this from case 1: over `d in [0,1]`, `sin(d/3)` is almost linear, so the
(c,d) term is a near-multiplicative interaction rather than a strongly curved one.

## Measured

| quantity | MI |
|---|---|
| raw floor, `max(MI(c), MI(d))` | 0.2454 |
| genuine form `log(2c)*sin(d/3)` | 0.3266 |
| best captured (c,d) feature, `gate_mask__c__d__t0.24814` | 0.2004 |
| `d` alone | 0.047 |

A materially better (c,d) representation exists (0.3266 vs a 0.2454 floor) and the search does not find it.

Everything that was built, under the default config with `fe_max_steps=2`:

```
unary_binary        mul(sqr(a),reciproc(b))       MI=0.5630   <- the (a,b) half, correct
conditional_gate    gate_mask__c__d__t0.24814     MI=0.2004   <- the only (c,d) form
unary_binary        esc_poly_hermite_mul(b,d)     MI=0.1563   <- wrong operands
```

## What the gap is NOT

Two candidate explanations were tested and both are ruled out:

* **Not the conditional gate crowding it out.** With `fe_conditional_gate_enable=False` the multiplicative
  form still does not appear - the gate column simply vanishes and nothing replaces it.
* **Not the acceptance rule.** `fe_acceptance="prevalence_ratio"` produces an identical roster to the
  default `conditional_mi`.
* **Not pair prioritisation.** `fe_synergy_prerank=True` and `fe_synergy_max_pairs=16` against C(5,2)=10
  possible pairs, so (c,d) is inside the exhaustive sweep either way.

So the acceptance / redundancy stages are innocent: the operator search over the (c,d) pair emits no
competitive candidate in the first place, while it does emit one for (b,d).

## Working hypothesis

`d` has a marginal MI of 0.047 - essentially noise on its own. The suspicion is a floor somewhere in
candidate generation or the pair-MI screen that is calibrated against marginal relevance, so a pair whose
signal lives entirely in the interaction is filtered out before its operator combinations are scored. That
is the classic weak-marginal blind spot, and the gate mask survives only because it is produced by a
different family with a different floor.

Not yet confirmed - it needs the per-pair candidate scores, which are not currently exposed on the fitted
selector (only survivors are, via `_engineered_recipes_`).

## Suggested next actions

1. Expose the per-pair proposed-and-rejected candidates with their scores, the way
   `hybrid_orth_candidates_` was added alongside the survivor-only `hybrid_orth_features_`. Without a
   produced roster this class of question cannot be answered from a fitted object at all.
2. With that roster, confirm which floor drops the (c,d) candidates.
3. Only then change the floor, and validate on the wide multi-seed benchmark - this is candidate-generation
   behaviour, so a change here moves selection everywhere.

## Related

`test_fe_cmi_redundancy_gate.py` fails on the same fixture family with a complementary complaint (the
selected cross-mix is less informative than the pure (c,d) form it absorbed). Whether that is the same root
cause is untested - it was measured against the redundancy stage, which the experiments above exonerate for
the gate-vs-elementary case, so the two may be independent.

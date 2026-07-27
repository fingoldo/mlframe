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

---

# Open finding: `mi_direct_gpu(return_null_mean=True)` reads D2H once per permutation

Status: **OPEN, located, not fixed.** The fix is a kernel-shape change with a real design trade, so it needs
the wide benchmark that is being run elsewhere.

## Symptom

`test_cmi_lru_and_null_mean_residency_fixes.py::TestMiDirectGpuNullMeanBatchedReadback::test_return_null_mean_d2h_ops_do_not_scale_with_npermutations`

```
D2H op count grew with npermutations (8->64): 33 -> 64; expected roughly constant under the batched-readback fix
```

Not environmental: the fixture is `(n, 2) int32`, so VRAM plays no part (unlike the pair-search residency
tests in the same suite, which are correctly VRAM-gated).

## Location

`feature_selection/filters/gpu.py`, inside the permutation loop:

```python
for _i in range(npermutations):
    ...                                   # shuffle y, joint hist, MI -> totals (device)
    mi = totals.get()[0]                  # <- one D2H per permutation
    _null_sum += float(mi)
    if mi >= original_mi:
        nfailed += 1
        if nfailed >= max_failed:
            original_mi = 0.0
            break
```

## Why it is not a one-line fix

The read is load-bearing: `mi` drives the early-stop (`nfailed >= max_failed`), which is a genuine
optimisation, not decoration. Accumulating `_null_sum` and `nfailed` on device and reading once at the end
removes every intermediate D2H but also removes the early stop, and batching the check every K iterations
still leaves D2H scaling as `nperm/K` - the test's tolerance is `+3` ops, so any Python-level loop that must
observe `mi` to decide whether to break will fail it.

Satisfying both means moving the permutation loop itself into the kernel, with the early-stop condition
evaluated on device. That is the shape the test's name ("batched-readback fix") implies was intended; the
CPU twin's contract is mirrored here only in the returned values, not in the traffic profile.

## Suggested next actions

1. Decide the contract deliberately: is the early stop worth more than the D2H traffic at realistic
   `npermutations`? Measure both on the real fit, not in isolation.
2. If traffic wins, fuse the permutation loop into one kernel launch with a device-side `nfailed` and an
   in-kernel bail; if the early stop wins, re-frame the test to assert what the path actually guarantees.
3. Either way the current state is wrong: the test asserts a contract the code does not implement.

---

# Resolved by measurement: rankgauss "fails" on the regression axis because a better sibling wins

Status: **NOT a product defect.** Recorded so the next reader does not re-open it.

`test_fe_mechanisms_task_axis.py::test_fe_mech_fits_transforms_and_lifts[rankgauss-regression]` reports
`0/4 seeds produced a surviving 'rankgauss' column`. The family is not broken: calling
`hybrid_rankgauss_fe` directly on the same fixture returns `appended=['rankgauss__x1']` on every seed. The
column is produced and then loses selection to `add(cbrt(x1), z2)` - the general pair search finds the cube
root of the SAME heavy-tailed column and fuses it with z2.

Losing is the correct outcome. Ridge R^2 under 4-fold CV on the fixture:

| seed | raw | + cbrt | + rankgauss |
|---|---|---|---|
| 0 | 0.6217 | **0.9270** | 0.9182 |
| 1 | 0.6051 | **0.9250** | 0.9175 |
| 2 | 0.7559 | 0.9293 | **0.9350** |
| 3 | 0.5716 | **0.9306** | 0.9161 |

`cbrt` wins on three seeds of four and both transforms lift the raw baseline enormously (0.57-0.76 ->
~0.93). The selector is picking the better variance-stabilising transform of the same column.

What is stale is the assertion, not the mechanism. `rankgauss_features_` is a SURVIVOR roster, so requiring
it to be populated on a majority of seeds demands that this specific mechanism beat every sibling rather
than that it work. The same shape was re-framed in `test_recipe_fe_families.py` earlier, where the pair
search folded the rankgauss column into a composite.

The `kfold-te-regression` (1/4 seeds) and `rankgauss-multiclass` (median lift 0.0192 against a 0.03 floor)
legs of the same parametrisation are the same question and are NOT yet measured this way - the fix there is
either the same re-frame or a re-derived floor, and it needs the multi-seed benchmark, not a single run.

---

# Third instance of the same shape: the escalation never sees the pair that needs it

Status: **OPEN, localised.** Same neighbourhood as the first finding above, and probably the same root.

`test_fe_auto_escalation.py::test_biz_val_escalation_recovers_inner_frequency_pair_and_replays` fails with
`recipes=[]`. The fixture is `y = sin(3.7*x0)*x1` - an inner frequency the library unaries cannot express,
which is exactly what the demodulated adaptive-Fourier escalation exists to recover.

## The proposer is not the problem

Called directly on the (x0, x1) columns it returns **2 proposals**, and does so regardless of what it is
fed:

| target passed | proposals |
|---|---|
| raw y | 2 |
| rank-transformed y (what the fit actually uses) | 2 |
| z-scored y | 2 |
| `max_freqs` swept 1, 2, 3, 4, 6 | 2 each |

`_propose_poly` returns None on the same pair, which is expected - a degree-4 polynomial cannot track ~3.5
cycles, and that is the documented reason the Fourier leg exists.

## What actually happens

`fe_escalation_info_` after a default fit reports `proposed: 0`, `rejected: {}` - nothing was proposed, so
nothing could be rejected - against 8 eligible pairs:

```
('x1','x2') ('x1','x3') ('x1','x5') ('x1','x4') ('x0','x4') ('x0','x3') ('x0','x2') ('x0','x5')
```

Every pair is x0 or x1 crossed with a noise column. **(x0, x1) - the only pair carrying signal - is absent
from the eligible set entirely.** The escalation is never offered the pair it was built for, so a working
proposer produces nothing. The final support is the raw `['x1', 'x0']` with no engineered column at all.

Not caused by the fe_max_steps default change: `esc=[]` at both `fe_max_steps=1` and `fe_max_steps=2`.

## Why this is filed with the first finding

Both are "the right pair never reaches candidate generation", measured from opposite ends: there the (c,d)
operator search emitted nothing competitive for a weak-marginal pair, here (x0, x1) does not even enter the
eligible list. `x1` alone has essentially no marginal signal in the first fixture and the same is true of
the pure-interaction fixture here. The suggested first step is unchanged and would answer both: expose the
per-pair proposed-and-rejected candidates with their scores, so the stage that drops the pair is visible
from a fitted object instead of being inferred.

---

# Measured regression: the rankgauss mechanism's downstream contribution has fallen since calibration

Status: **OPEN, measured. Do NOT relax the floor** - the floor is correct and the mechanism regressed.

`test_fe_mechanisms_task_axis.py` fails on both rankgauss legs. The obvious move is to lower the lift
floor, and the file's own calibration header is what rules that out. It records the medians the floors were
derived from on 2026-06-10, "floors 60-85% below per CLAUDE.md":

| mechanism | reg R2 lift median (calibrated) | mc acc lift median (calibrated) |
|---|---|---|
| rankgauss | 0.225 | 0.073 |

Measured today on the same fixture and the same `_downstream_lift`, over 14 seeds instead of the test's 4:

```
seeds 0..13, multiclass: median 0.0269  mean 0.0274  min 0.0128  max 0.0462  p25 0.0199
                         2 of 14 seeds: mechanism absent from the support entirely
                         only 42% of seeds clear the 0.03 floor
```

So the 0.03 floor sits ~59% below the 0.073 it was derived from - exactly the documented margin - while the
mechanism now delivers 0.0269. The contribution has fallen ~2.7x. Lowering the floor to make the suite green
would erase the only signal that this happened.

The regression axis is worse: the calibrated median was 0.225, and today the rankgauss column does not reach
the support at all. `add(cbrt(x1), z2)` wins instead, and it deserves to - Ridge R^2 under 4-fold CV puts
cbrt ahead on three seeds of four (see the section above). That is a legitimate sibling win and explains the
regression leg on its own. It does NOT explain the multiclass leg, where the mechanism IS selected on 12 of
14 seeds and still only lifts 0.0269.

## What to look at

Something between 2026-06-10 and now changed either what rankgauss produces or what the downstream sees.
The two legs may share a cause with the sibling-transform story or may not - the multiclass leg's column is
selected, so its lower lift is about the VALUE of the column, not about losing selection. Bisecting the
downstream lift on this fixture across the intervening commits would name the change directly; that is
cheap to run and was not attempted here.

---

# The tail-subsume knob that is exposed is not the one that decides

Status: **test fixed; a product gap is noted below and left open deliberately.**

`test_raw_tail_subsume_survivor_gate.py` carried a sensor that lowered `fe_raw_tail_subsume_min_corr` to
0.6 and asserted raw `b` was then dropped, to prove the shipped 0.85 gate is load-bearing. It cannot prove
that. Instrumenting the leg on its own fixture gives:

| quantity | value |
|---|---|
| raw `b` linear \|corr(y)\| | 0.7564 |
| raw `b` RANK association | 0.8834 |
| subsuming survivor \|corr(y)\| | 0.7889 |

The drop requires all three of `survivor >= min_corr`, `raw_linear < survivor`, and
`raw_rank <= 0.7 * raw_linear`. The first two hold at either gate setting; the third is
`0.8834 <= 0.5295`, false by a wide margin. `b`'s rank association is HIGHER than its linear one - the
opposite of the tail-concentration signature - so it carries genuine monotone signal and is correctly kept.
The survivor-strength gate never gets to decide, which is why varying it changed nothing and the sensor
failed for a reason unrelated to its name. It now asserts the real guard instead: a rank-healthy raw column
survives at both settings.

## The gap

`tail_subsume_rank_frac` - the 0.7 that actually decides this case - is **not a constructor parameter**.
Only `fe_raw_tail_subsume_min_corr` is plumbed through to `MRMR.__init__`. So the knob a caller can reach is
the one that does not decide, and the deciding constant is reachable only by editing the function default.
Exposing it would also make a genuine sensor possible: at `rank_frac >= 1.17` the condition on this fixture
flips and the leg fires, which is the demonstration the old sensor was reaching for.

Not done here because adding a public ctor parameter is an API change with its own review, and the test no
longer depends on it.

# MRMR usability-aware (multi-list) feature selection -- validated design (2026-06-13)

A prototyped + measured design for letting MRMR serve a LINEAR downstream as well as it
serves trees, by producing MORE THAN ONE feature selection from one FE candidate pool.

## The problem (measured)

On the F2 target `y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3)` (heavy-tailed; `f` hidden):

- The achievable floor is **MAE ~0.05** (the irreducible `f/5` noise: `0.2*E|U-0.5| = 0.05`).
- A closed-form LINEAR model needs the engineered interaction `mul(log(c),sin(d))` to reach it:
  `[a**2/b, c]` -> MAE 0.092; `+ mul(log(c),sin(d))` -> **0.050**. A tree gets nothing from it.
- MRMR's Fleuret objective is `MI-relevance - MI-redundancy` -- RANK-based and MODEL-AGNOSTIC.
  MI ranks a tree-friendly monotone warp `sub(exp(c),cbrt(d))` (MI 0.288, linear MAE 0.093 = no
  help) ABOVE the linearly-aligned `mul(log(c),sin(d))` (MI 0.264, linear MAE 0.063). So pure-MI
  selection cannot surface the feature a linear model needs -- not a tractability issue, a
  metric-alignment one (handing MRMR MORE candidates made it WORSE: it picked high-MI cross-mix
  `esc_poly(a,c)` and fused them, MAE 0.092 -> 1.08).

## The validated mechanism

Re-run the greedy with a **usability-aware relevance**:

```
relevance(f | S) = (1-w) * (MI(f;y) - red * MI_redundancy(f,S))
                 +   w   * |held-out corr(f, RESIDUAL_after_S)|
```

The CRITICAL detail: the usability term is `|held-out partial corr of f with the RESIDUAL after
the already-selected features|`, NOT an R^2 / incremental-R^2. On a heavy-tailed target R^2 is
dominated by the `a**2/b` tail (0.998 -> 0.9999, the (c,d) increment drowns in estimation noise),
so an R^2-based usability term is blind to the weak interaction. Removing the selected features
(esp. the dominant `a**2/b`) leaves a BOUNDED residual (`log*sin + f/5`) whose linear correlation
with the genuine interaction feature is large and tail-free.

### Measured (prototype `_bench_usability_aware_selection.py`, n=60k, 2758-candidate pool)

| relevance weight | linear test MAE | (c,d) features selected |
| --- | --- | --- |
| `w=0.0` (pure MI -- the current selection) | **0.134** | none (a**2/b + cross-mix only) |
| `w=0.5` (blend) | **0.063** | `mul(log(c),exp(d))` |
| `w=0.8` (mostly usability) | **0.052** | `mul(log(c),exp(d))`, `mul(sqr(c),log(d))`, `mul(log(c),log(d))`, `mul(id(c),sqr(d))` |

`w=0.8` reaches **0.052 ~ the irreducible floor** -- the goal -- selecting genuine `(c,d)`
interaction forms that pure-MI never surfaces.

## The deployment: THREE selection lists from one pool

MRMR already exposes `metadata["selected_features_per_model"]`. The clean, non-disruptive
deployment is to produce THREE selections from the SAME FE candidate pool and route each to the
model family that wants it:

1. **List 1 -- pure MI (`w=0`)**: the current selection. For trees / nonlinear models (they build
   interactions internally; MI relevance + redundancy is the right model-agnostic objective).
2. **List 2 -- strict linear (`w->1`)**: usability-only relevance. For linear / additive models
   that cannot build interactions.
3. **List 3 -- blend (`w in [0.5, 0.8]`)**: `MI + lambda*usability`. A universal/linear-leaning
   list.

The suite trains linear/ridge from list 2 or 3, trees/mlp from list 1. List 1 stays byte-identical
to today's behaviour (no regression for existing tree pipelines).

## Open implementation considerations (for the core build)

- **Pool richness.** The usability greedy can only pick `mul(log(c),sin(d))` if it is IN the
  candidate pool. The current FE admission (prevalence gate) + one-best-per-pair PRUNE it out.
  So List 2/3 need a richer pool: the `fe_multi_emit_max_per_pair` knob (already shipped,
  default-off) emits diverse forms per pair, and the `(c,d)` pair must be admitted. For SMALL p
  the full all-pairs-all-forms pool is tractable (CASE2 p=5 -> 2758 candidates); for LARGE p the
  admission pruning is a real tractability constraint and List 2/3 must scope the pool (e.g. run
  the usability greedy only over the already-admitted + multi-emitted pool, accepting it may miss
  a `(c,d)` the prevalence gate dropped).
- **Residual cost.** One linear refit of the selected set per greedy step (cheap: k features x n).
- **Usability target.** Use the CONTINUOUS y residual (`_fe_prewarp_y_continuous_`) so the term is
  genuinely linear; fall back to the rank target only if continuous y is unavailable.
- **No-regression.** List 1 (`w=0`) must reproduce the current selection exactly; List 2/3 are
  additive metadata the suite consumes per-model.

## Remaining RED tests this design must turn green (status 2026-06-17)

After the small-n de-dup / FE-leak / polars fixes landed (commit `c7689a4b`), the FS suite's
residual failures collapse to ONE class -- the cross-mix-subsumes-the-pure-form problem this design
targets. These four are the acceptance gate for wiring `_usability_aware_selection` into the fit:

- `test_fe_cmi_redundancy_gate::test_user_f2_e2e...[20000]` -- the headline case, and NOT small-n:
  every engineered `(a,b)` feature also drags in `c`/`d` (e.g.
  `div(sqr(add(invcbrt(b),prewarp(d))),invsqrt(mul(exp(a),sqr(c))))`), so `_covers("a","b",
  exclude=("c","d"))` finds NO pure `a**2/b` form. The cross-mix wins on MI (F2's y is the additive
  sum of the two terms, so a single a&c feature informs about BOTH), but a linear model needs the
  pure form. List 2/3 (usability-blended) is what recovers the pure `(a,b)` term.
- `test_mrmr_create_keep_drop::test_create_keep_drop_nsweep[1000-MS_three_tier_strength]` --
  `y=5*(a*b)+sqrt(c)+...`; at n=1000 the greedy picks raw `a,b,c` and never forms the `mul(a,b)`
  interaction `_covers({a,b})` demands (a single feature spanning both operands).
- `test_mrmr_endtoend_invariants::test_I5...[ratio_plus_trig-uniform-regression-s101-fe2]` -- the
  fe_max_steps=2 structure-uplift invariant, same cross-mix-over-pure-form subsumption.
- `test_mrmr_weak_f2_seed_stability::test_weak_f2_stability_summary` -- the genuine `(c,d)` joint is
  recovered far less often than the `(a,b)` ratio; the asymmetry is the same pure-form instability.

These are intentionally NOT xfailed: they fail loudly until the usability-aware list (or the P3
joint-synergy screen feeding a richer pool) lands. The pure-MI List 1 (`w=0`) must stay byte-identical
so no tree pipeline or existing green test regresses.

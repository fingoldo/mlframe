# An atlas of when feature selection pays, and which method to reach for

This benchmark was designed and run by the author of one of the methods it judges (MRMR). The bed list is
hand-picked and is not a sample from any population of real problems. Read every count below as a property
of these beds, and the qualitative separations as the part that might transfer.

Three legs, one protocol, one roster of sixteen arms, `all-features` as the null hypothesis in every cell:

| leg | beds | cells | what it can answer |
|---|---|---|---|
| real (`phase0_confirm`) | 7 OpenML beds | 2 240 | does selection pay on data nobody generated |
| adversarial (`phase0_synth_control`) | 9 hand-written beds | 3 029 | is the real verdict about the data or the model |
| SCM (`scm_beds`) | 8 generated beds | 2 560 | what a method RECOVERS when the answer key comes from the graph |

## The first-order finding: the downstream model decides more than the selector does

| downstream | beds where some arm beats the null | |
|---|---|---|
| logistic | 15 of 17 synthetic beds | fails only on the two parity beds a linear model cannot use anyway |
| lightgbm | 5 of 17 synthetic beds | and on the real beds, none of 7 at any K |

On the eight SCM beds the split is starker still: with logistic, some arm beats the null on 7 of 8 at the
tightest cut and 8 of 8 at the widest; with lightgbm, on 1 of 8 and then none at all. A linear model gains
from selection almost everywhere. A gradient-boosted tree gains only where the bed is hard in a specific
way, and on clean generated beds essentially never.

If you are choosing where to spend effort, the downstream model is the bigger lever than the selector.

## Where a gradient-boosted tree does gain

Sorting the bed instances by whether lightgbm gained separates them by *what makes the bed hard*, not by
width:

| bed | p | truth | lgbm gained | why, as far as these beds show |
|---|---|---|---|---|
| `probe_flood_p1000` | 1000 | 8 | yes (3/3 K) | overwhelming noise: 992 probes against 8 signals |
| `fdr_under_budget` | 200 | 20 | yes (3/3) | many weak signals under a budget |
| `compensable_pair` | 33 | 2 | yes (3/3) | the pair compensates, so neither column is individually attractive |
| `group_additive` | 50 | 10 | yes (3/3) | a group that only sums to something |
| `linear_k5_p50` | 50 | 5 | no (0/3) | a tree already ignores 45 clean probes |
| `mb_spouse_collider` | 33 | 3 | no (0/3) | same |
| `mediator_chain_with_proxy` | 33 | 1 | no (0/3) | same |
| `redundant_exact_k5` | 35 | 1 | no (0/3) | a tree pays nothing for keeping four redundant copies |
| `xor3`, `xor3_plus_decoy` | 33-34 | 3-4 | no (0-1/3) | nothing hands it the operands: see the next section |

The rule these beds support: **a tree gains when noise columns outnumber signal ones by two orders of
magnitude, or when the signal is hard to see one column at a time. It does not gain by having clean probes
removed from a narrow bed** -- it was already ignoring them.

## Nobody in this roster solves parity

On the three-way parity bed every arm recovers the answer key at chance: 0.05 to 0.22 against a base rate of
0.09, with the unsupervised `variance-sort` control among the highest by luck. Not one of the thirteen
scored arms does better than the arm that never looks at the target.

That is a negative result, and it is the honest one. An earlier version of this document reported the
opposite -- a wrapper recovering the parity operands perfectly on all twenty seeds -- and that claim was an
artefact of this benchmark's own harness: the arm tied every column, the cut resolved ties with a stable
sort, a stable sort preserves input order, and the beds declared their informative columns first. Ties are
now randomised per cell and the beds shuffle their columns; the recovery fell from 1.000 to 0.050, which is
chance. **An arm's apparent success is worth auditing hardest when it is the result you were hoping for.**

## Three separations clean enough to act on

**Marginal filters cannot reach a Markov blanket through a collider.** On `mb_spouse_collider` the
multivariate arms -- `ace`, `boruta`, `boruta-shap`, `lars-order`, `sfm-lgbm` -- recover the blanket
perfectly (1.000). Every marginal filter stalls at two of the three members: `univariate-mi` 0.700, `skb-f`
and `select-fdr` 0.683, `skb-mi` and `knockoffs` 0.667. The one they miss is always the spouse, which is
independent of the target until the collider is conditioned on. Reach for a multivariate method when a
variable's relevance is conditional.

`rfecv` is the cautionary row in that table at 0.567 -- below every marginal filter, on a bed where being
multivariate is supposed to be the advantage.

**Binned mutual information fails on compensable structure.** On `compensable_pair`, `skb-mi` recovers
0.150 of a two-column truth and costs the downstream model 0.277 AUC -- worse than the tripwire control. It
does not merely fail to find the columns; it actively selects worse ones.

**MRMR buys purity with coverage, consistently.** `mb_spouse_collider`: precision 0.983 against recall
0.667. `fdr_under_budget`: 0.898 against 0.670, where `lars-order` takes 0.857 on both. That is the trade
its design makes, visible only because recovery is reported as two numbers rather than one.

## The pooled picture, and why "wins" overstates it

SCM leg, logistic downstream, hierarchical fit over the eight beds on normalized skill (so one region of
practical equivalence means the same thing on every bed):

| arm | pooled advantage | P(> 0) | P(inside the 1%-of-skill ROPE) | between-bed spread |
|---|---|---|---|---|
| `boruta-shap` | +0.0091 | 1.000 | 0.867 | 0.0017 |
| `ace` | +0.0090 | 1.000 | 0.892 | 0.0015 |
| `lars-order` | +0.0090 | 1.000 | 0.904 | 0.0015 |
| `sfm-lgbm` | +0.0089 | 1.000 | 0.774 | 0.0037 |
| `boruta` | +0.0036 | 0.798 | 0.916 | 0.0113 |
| `univariate-mi` | -0.0125 | 0.280 | 0.290 | 0.0607 |
| `select-fdr`, `skb-f` | -0.0147 | 0.267 | 0.262 | 0.0663 |
| `mrmr` | -0.0166 | 0.255 | 0.231 | 0.0715 |
| `knockoffs` | -0.0321 | 0.129 | 0.137 | 0.0786 |
| `skb-mi` | -0.0364 | 0.081 | 0.103 | 0.0698 |
| `rfecv` | -0.0977 | 0.003 | 0.004 | 0.0777 |
| `variance-sort` | -0.1923 | 0.003 | 0.001 | 0.1496 |

Two readings, and the second matters more. The top four are positive with posterior probability
indistinguishable from one -- and each also sits INSIDE the pre-registered region of practical equivalence
with probability 0.77 to 0.92, which says the gain is **real and small**. Meanwhile the negative arms carry
a between-bed spread five to fifty times the winners': they are not uniformly bad, they are bed-dependent,
which is what an atlas exists to map.

`rfecv` at -0.098 with P(> 0) = 0.003 is the corrected version of what this document once called its best
result.

## The pre-registration, scored

Each bed declares before the run which arms it expects to defeat, and those predictions are now checked
against what happened:

| leg | predictions held |
|---|---|
| adversarial | 18 of 29 (62%) |
| SCM | 16 of 31 (52%) |

Roughly half, which is the useful answer. A suite whose predictions always held would be tuned to its
author's beliefs rather than measuring anything; one whose predictions never held would not be measuring
what it thinks it is. The misses are named in the report: `mrmr` paid on beds built to defeat it, and
`boruta` gained 0.127 AUC on the thousand-column probe flood that named it.

## Cost

Per cell, SCM leg, model fits spent by the arm itself -- the downstream panel is paid identically by every
arm and is excluded -- with wall-clock as advisory:

| arm | fits by the arm | wall (s) |
|---|---|---|
| `skb-f`, `lars-order`, `select-fdr`, `mrmr`, `skb-mi` | 0 (no model fit at all) | 0.005-2.1 |
| `ace` | ~9 | 7.7 |
| `boruta` | ~20 | 8.8 |
| `rfecv` | ~38 | 52 |
| `boruta-shap` | ~46 | 9.6 |
| `shap-proxied` | ~76 | 93 |

The four arms with a real pooled advantage span the whole cost range: `lars-order` fits no model at all and
`ace` costs nine, while `rfecv` at thirty-eight fits and fifty seconds is the worst arm in the pooled table.
On these beds, cost buys nothing.

## What this does NOT establish

* **That selection does not pay on real data.** The real leg's minimum detectable effect at 20 seeds is
  0.013 AUC on the median contrast and 0.031 on the p90 one ([`BENCHMARK_POWER.md`](BENCHMARK_POWER.md)).
  Most gains measured on synthetic beds are smaller than that, so the real leg could not have seen them.
* **That these counts generalise.** Seventeen hand-picked bed instances, mostly at one sample size, one
  downstream panel of two models. The separations are structural and are the part most likely to transfer;
  the counts are not.
* **That the arms are configured optimally.** Each is driven bare with an explicit budget. A tuned RFECV or
  a differently parameterised MRMR would land elsewhere, and the pre-registration says so.
* **Anything about tail dependence.** The two copula beds were registered after this leg ran and are
  computing now; nothing here speaks to them.

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

> **Two corrections in flight (2026-09-08), both affecting the SCM rows only.**
>
> **The parity result was an artefact and is withdrawn.** This document claimed `rfecv` recovered the
> three-way parity answer key perfectly on all twenty seeds. An ablation run to explain that result found
> it selected ALL 33 columns, so its recovery at the matched cut came from somewhere else: its ranking
> ties every survivor at rank one, the harness broke ties with a stable sort, and a stable sort falls back
> to column order -- which on a generated bed whose informative columns are declared first IS the answer
> key. Ties are now permuted with a per-cell seed, and the SCM beds shuffle their column order per seed,
> so position carries no information. Any claim below resting on RFECV's parity recovery is void until
> the re-run lands.
>
> **Bed sizes were overridden.** The adapter replaced each bed's declared row count with a uniform 4 000,
> so `linear_gaussian_lowdim_n200` -- the bed that exists to be 200 rows -- ran at twenty times its design
> size. Fixed; the size now travels with the bed and the registry hash could not have caught it, because
> the spec was never edited.

The real and adversarial legs are unaffected by both.

## The first-order finding: the downstream model decides more than the selector does

Across the two synthetic legs -- seventeen bed instances, three K settings each -- the pattern is not about
which selector you choose:

| downstream | beds where some arm beats the null | |
|---|---|---|
| logistic | 16 of 17 | fails only on the two parity beds a linear model cannot use anyway |
| lightgbm | 7 of 17 | and on the real beds, none of 7 at any K |

A linear model gains from selection almost everywhere. A gradient-boosted tree gains on some beds and not
on others, and on real data it gained nowhere the design could see. If you are choosing where to spend
effort, the downstream model is the bigger lever.

## Where a gradient-boosted tree does gain

Sorting the seventeen bed instances by whether lightgbm gained separates them cleanly by *what makes the
bed hard*, not by width alone:

| bed | p | truth | lgbm gained | why, as far as these beds show |
|---|---|---|---|---|
| `probe_flood_p1000` | 1000 | 8 | yes (3/3 K) | overwhelming noise: 992 probes drown 8 signals |
| `fdr_under_budget` | 200 | 20 | yes (3/3) | many weak signals under a budget |
| `compensable_pair` | 33 | 2 | yes (3/3) | the pair compensates, so neither column is individually attractive |
| `group_additive` | 50 | 10 | yes (3/3) | a group that only sums to something |
| `xor3`, `xor3_plus_decoy` (SCM) | 33-34 | 3-4 | yes (3/3) | parity becomes learnable once the probes are removed |
| `linear_k5_p50` | 50 | 5 | no (0/3) | a tree already ignores 45 clean probes at n=4000 |
| `mb_spouse_collider` | 33 | 3 | no (0/3) | same |
| `mediator_chain_with_proxy` | 33 | 1 | no (0/3) | same |
| `latent_replicates_private_delta` | 26-33 | 3-6 | no (0/3) | collapsing the cluster destroys what drives the target |
| `redundant_exact_k5` | 35 | 1 | mostly no (1/3) | a tree costs nothing for keeping four redundant copies |

The rule these beds support: **a tree gains when the signal is hard to see one column at a time, or when
noise columns outnumber signal ones by two orders of magnitude. It does not gain by removing clean probes
from a narrow bed** -- it was already ignoring them.

## Four separations clean enough to act on

**~~A subset wrapper finds parity; nothing else does.~~ WITHDRAWN.** This read as the sharpest separation
in the benchmark: `rfecv` recovering the parity answer key perfectly on all twenty seeds while every other
arm sat at chance. An ablation designed to explain it -- swap the inner estimator, hold everything else --
found the cause was neither the search nor the estimator. RFECV selected all 33 columns and tied them at
rank one; the harness broke that tie with a stable sort, and a stable sort inherits column order, which on
these beds is the answer key. The perfect recall and the perfect stability were the same artefact seen
twice. Ties are now randomised per cell and the beds shuffle their columns; what the arms do on parity
will be restated from the re-run, and the honest current answer is that this benchmark has not yet shown
any method solving three-way parity.

**Marginal filters cannot reach a Markov blanket through a collider.** On `mb_spouse_collider`, every
marginal filter recovers exactly two of the three blanket members: `univariate-mi` 0.700, `skb-f` and
`select-fdr` 0.683, `skb-mi` and `knockoffs` 0.667. The missing one is always the spouse, which is
independent of the target until the collider is conditioned on. `ace`, `boruta`, `boruta-shap`,
`lars-order`, `rfecv` and `sfm-lgbm` all reach 1.000. Reach for a multivariate method when a variable's
relevance is conditional.

**Binned mutual information fails on compensable structure.** On `compensable_pair`, `skb-mi` recovers
0.150 of a two-column truth and costs the downstream model 0.277 AUC -- worse than the variance-sort
tripwire on the same bed. It is not a case of failing to find the columns; it actively selects worse ones.

**MRMR buys purity with coverage, consistently.** `fdr_under_budget`: precision 0.898, recall 0.670, where
`lars-order` takes 0.857 on both. `mb_spouse_collider`: precision 0.983, recall 0.667. That is the trade
its design makes, and it is visible only because recovery is reported as two numbers rather than one.

## The pooled picture, and why "wins" overstates it

On the SCM leg with logistic downstream, the hierarchical fit over the eight beds (normalized skill, so a
fixed region of practical equivalence means the same thing on every bed):

| arm | pooled advantage | P(> 0) | P(inside the 1%-of-skill ROPE) | between-bed spread |
|---|---|---|---|---|
| `rfecv` | +0.0099 | 1.000 | 0.513 | 0.0031 |
| `ace` | +0.0091 | 1.000 | 0.883 | 0.0015 |
| `boruta-shap` | +0.0091 | 1.000 | 0.910 | 0.0013 |
| `lars-order` | +0.0090 | 1.000 | 0.904 | 0.0015 |
| `sfm-lgbm` | +0.0085 | 0.999 | 0.815 | 0.0040 |
| `univariate-mi` | -0.0125 | 0.280 | 0.290 | 0.0607 |
| `select-fdr`, `skb-f` | -0.0147 | 0.267 | 0.262 | 0.0663 |
| `mrmr` | -0.0166 | 0.255 | 0.231 | 0.0715 |
| `knockoffs` | -0.0328 | 0.128 | 0.129 | 0.0798 |
| `skb-mi` | -0.0364 | 0.081 | 0.103 | 0.0698 |
| `variance-sort` | -0.1923 | 0.003 | 0.001 | 0.1496 |

Two things to read here, and the second is the more important. The top five are positive with a posterior
probability of essentially one -- but four of them also sit inside the pre-registered region of practical
equivalence with probability 0.8 or better, which says the gain is **real and small**. And the arms with
negative pooled effects have a between-bed spread five to fifty times larger than the winners': they are
not uniformly bad, they are bed-dependent, which is exactly what this atlas exists to map.

## Cost

Per cell, on the SCM leg, in model fits (the deterministic axis) with wall-clock as advisory:

| arm | fits | wall (s) |
|---|---|---|
| `skb-f`, `lars-order`, `select-fdr`, `mrmr`, `skb-mi` | 8 (i.e. the panel alone; the selector itself is free) | 0.005-2.1 |
| `ace` | 17 | 7.7 |
| `boruta` | 28 | 8.8 |
| `rfecv` | 46 | 52 |
| `boruta-shap` | 54 | 9.6 |
| `shap-proxied` | 84 | 93 |
| `knockoffs` | 7 | 86 |

`rfecv` is the only arm that solved parity, and it costs roughly six times the fits and three thousand
times the wall-clock of a univariate filter. On a bed where a filter suffices, that is a poor trade; on a
bed with interaction structure, nothing cheaper worked at all.

## What this does NOT establish

* **That selection does not pay on real data.** The real leg's minimum detectable effect at 20 seeds is
  0.013 AUC on the median contrast and 0.031 on the p90 one ([`BENCHMARK_POWER.md`](BENCHMARK_POWER.md)).
  Most gains measured on synthetic beds are smaller than that, so the real leg could not have seen them.
* **That these counts generalise.** Seventeen hand-picked bed instances, mostly at one sample size, with
  one downstream panel of two models. The separations in the section above are structural and are the part
  most likely to transfer; the counts are not.
* **That the arms are configured optimally.** Each is driven bare with an explicit budget. A tuned RFECV or
  a differently-parameterised MRMR would land elsewhere, and the pre-registration says so.

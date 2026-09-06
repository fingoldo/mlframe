# The synthetic control point: is the Phase 0 verdict about the data or about the model?

Phase 0's kill criterion fired. With lightgbm downstream, no arm beat the `all-features` null on four to
seven of the seven eligible real beds, at every K from k5 to k200. That result has two readings and they
call for opposite next steps:

* selection does not pay on **real** data at these widths, because real beds carry dense, genuinely
  predictive correlation structure a strong model exploits by itself; or
* selection does not pay for a **strong model**, full stop, in which case the criterion says nothing about
  real data and the leg was mis-designed.

Only a bed with known truth separates them. This run is that control: the same sixteen arms, the same
protocol, the same downstream panel, on nine adversarial beds whose informative columns are constructed and
therefore known.

**3 029 cells, 20 dataset seeds per (arm, bed), 2 OOM failures.** Development seeds; this run decides no
pre-registered hypothesis, it interprets one.

## The answer

Selection pays on the synthetic beds, with the same model that could not use it on the real ones.

| downstream | K | beds where some arm beats the null | beds where nothing does |
|---|---|---|---|
| lightgbm | 1k | 5 / 9 | 4 |
| lightgbm | 2k | 5 / 9 | 4 |
| lightgbm | 5k | 7 / 9 | 2 |
| logistic | 1k | 8 / 9 | 1 |
| logistic | 2k | 7 / 9 | 2 |
| logistic | 5k | 7 / 9 | 2 |

So the Phase 0 verdict is **about the data, not about the model**. A gradient-boosted tree does benefit from
feature selection when the bed has a small, genuinely sparse truth; the real beds simply do not look like
that, or their signal is spread across too many columns for a subset to help.

## The beds where nothing pays are the beds built so nothing would

With lightgbm, the no-pay beds are `xor2`, `xor3`, `xor3_plus_marginal_decoy` and
`latent_replicates_private_delta` -- and each of those declared, before the run, exactly the arms it expects
to defeat. Parity operands carry zero marginal association by construction, so a marginally greedy ranking
cannot see them; the private-delta cluster is jointly necessary, so any selector that collapses it destroys
signal it cannot recover. These are predicted failures, not null results, which is the difference between a
benchmark that learns something from a negative and one that merely records it.

`null_p1000` also shows no-pay at every absolute K, which is correct and uninteresting: nothing is relevant
there, so nothing can be gained, and that bed is scored on false discovery instead.

## The magnitudes matter more than the count, and they cut the other way

The gains here are small:

| bed | best arm | delta (lightgbm, 1k) | t |
|---|---|---|---|
| `compensable_pair` | ace / boruta-shap / lars-order / sfm-lgbm (tied) | +0.0171 | 11.4 |
| `fdr_under_budget` | lars-order | +0.0073 | 4.7 |
| `probe_flood_p1000` | ace and six others (tied) | +0.0066 | 8.5 |
| `group_additive` | ace and seven others (tied) | +0.0032 | 2.2 |

Read those against the power analysis of the real leg
([`BENCHMARK_POWER.md`](BENCHMARK_POWER.md)): at R = 20 the real beds resolved 0.013 AUC on the median
contrast and 0.031 on the p90 one. **Every gain measured here except `compensable_pair` sits inside that
blind spot.** The two legs are therefore consistent with "real gains of this size exist and the design could
not see them" just as much as with "there are no real gains". The control point rules out one explanation --
that a strong model cannot benefit from selection at all -- and it does not establish the opposite.

The tight standard errors here are not a contradiction: the synthetic beds are quieter per seed (tau of a
few thousandths) than the real ones (median 0.0205), which is exactly what one expects when the generating
process is fixed and the only thing redrawn is a sample from it.

## What the arms did, beyond the leaderboard

**The tripwire behaved.** `variance-sort` never looks at the target and recovers **0.000** of the truth on
`compensable_pair`, losing 0.28 AUC there and 0.31 on `probe_flood_p1000`. It is last or near-last almost
everywhere. Had it won anywhere, the bed would have been leaking variance ordering into relevance and the
result would have been an artefact of the generator.

**`skb-mi` collapses on `compensable_pair`**: recovery 0.150, delta -0.277. A compensable pair is precisely
the structure an equal-mass binned mutual-information score cannot see, and it takes the loss to the
downstream model rather than merely failing to find the columns.

**`rfecv` is the worst arm on `probe_flood_p1000`** at -0.3152, below even the variance-sort control. On a
thousand-column bed with eight informative ones, its stopping rule keeps a set that costs a third of the
achievable AUC.

**`mrmr` on `fdr_under_budget` buys purity with coverage**: precision 0.898 against recall 0.670, where
`lars-order` takes 0.857 on both. That is the trade its design makes, visible because support recovery is
reported as two numbers rather than one.

## Reproducing

```
python -m mlframe.feature_selection._benchmarks.fs_hybrid.run_synth_control --seeds 20
FS_HYBRID_RESULTS=src/mlframe/feature_selection/_benchmarks/_results/phase0_synth_control.jsonl \
    python -m mlframe.feature_selection._benchmarks.fs_hybrid.analyze
```

The run is resumable; cells already present are skipped.

## Declared limitations

The beds are hand-picked by the author of one of the arms, so the count of beds where selection pays is a
property of this bed list and not an estimate of anything. What transfers is the qualitative finding -- a
strong model CAN benefit from selection -- and the negative controls, which are structural rather than
distributional.

The run carries a manifest, and the report's declaration block flags two things about it, both true:

* **`null_p1000` is undeclared.** Twelve of its cells were written by a duplicate copy of an earlier run
  that survived a stop signal and kept appending to this file. They are gate cells, excluded from every
  aggregate here, and the mechanism catching them is the mechanism working.
* **The pre-registration changed after the run started.** Sections 6a and 2b were added afterwards, so the
  document as it stands today did not bind this run. Recorded rather than papered over.

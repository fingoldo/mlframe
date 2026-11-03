# _pair_admission_batch: resident codes and block analytic null

100k x 100, all families, `nbins_strategy="optimal_joint"` (warm fit 114.0 s): the pair admission had 5010 asymmetric-relax pairs and 955
prevalence-failing candidates in 9 anchor groups (mdlp: 3567 and 418). One captured `_score_groups` call (955 candidates, int8 codes 100k x 194):

| part | before |
|---|---|
| host column gather + int64 widen (27 blocks) | 1.41 s |
| `batched_cmi_gpu` | 0.50 s |
| per-pair `_conditional_perm_null` (analytic branch, 1.7 ms each: column and target copies) | 1.65 s |
| total | 3.52 s |

Now: the code matrix is uploaded once in its int8 dtype and the blocks gather on the device; the analytic null of a block is one vectorised
`chi2.ppf` over the cardinalities the batched CMI returns (the per-pair null still runs where the analytic branch would not apply).
Same call: 0.168 s, 955/955 verdicts identical. `batched_asymmetric_relax_verdicts` takes the same route.

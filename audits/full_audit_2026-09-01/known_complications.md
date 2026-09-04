# Known complications found during the 2026-09-01 audit implementation

Issues discovered while implementing audit findings that are NOT themselves findings, and that are not fully
resolved. Each records what was measured, what was ruled out, and what a fix would cost.

---

## MLPRanker: the FIRST fit in a process differs from every later one, from identical initial weights

**Found:** 2026-09-03, while fixing `XCUT_TEST_QUALITY-18` (an unrelated phantom monkeypatch in the same file).
`tests/training/neural/test_training_neural_fixes.py::test_f10_mlp_ranker_same_seed_is_reproducible` was
failing, and it fails identically on unmodified `HEAD`, so it is pre-existing and not caused by this audit round.

**Measured** (16-thread host, CUDA present; same data, same `seed=7`, four fits in one process):

| comparison | result |
|---|---|
| initial weights, fit 1 vs fit 2 | **bit-identical**, maxdiff exactly `0.0` |
| predictions, fit 1 vs fit 2 | differ, maxdiff `0.00631118` |
| predictions, fit 2 vs fit 3 | **bit-identical** |
| predictions, fit 3 vs fit 4 | **bit-identical** |

The magnitude `0.00631118` reproduces exactly across independent runs, so this is a deterministic difference
between "cold process" and "warm process", not random noise.

**Therefore:** the estimator's seeding is CORRECT -- `torch.manual_seed(self.random_state)` covers weight init
and dropout, and `GroupBatchSampler` takes its own seed, which the bit-identical initial weights confirm
directly. The divergence arises during TRAINING.

**Ruled out, each by direct measurement:**

* *CPU intra-op thread count.* An early probe appeared to show `torch.set_num_threads(1)` fixing it; that probe
  was wrong -- the single-threaded pair happened to run after a warm-up fit. Under pytest, with the
  single-threaded pair running first, it fails. Thread count is not the variable.
* *cuDNN autotuning.* `torch.backends.cudnn.benchmark` is already `False`, and additionally setting
  `cudnn.deterministic = True` changes nothing.
* *CUDA context / RNG initialisation order.* Calling `torch.cuda.init()` and allocating on the device before any
  fit changes nothing.

**Most likely remaining cause:** workspace-dependent cuBLAS algorithm selection. cuBLAS picks GEMM algorithms
partly from available workspace, which differs on the first call in a process; the documented controls are
`torch.use_deterministic_algorithms(True)` together with the `CUBLAS_WORKSPACE_CONFIG` environment variable.
Not confirmed.

**Why not fixed here:** both controls are process-wide and carry a real throughput cost on every model in the
suite, not just this one. Imposing them from inside an estimator would be exactly the kind of silent global
mutation `ranker.py`'s own Wave 49 comment removed. Whether to trade throughput for bitwise determinism is the
caller's decision.

**What was done instead:** the test was rewritten to assert what the estimator genuinely controls and to fail
loudly if any of it regresses -- bit-identical initial weights from the same seed, different initial weights
from a different seed, and same-seed predictions closer than different-seed predictions. It no longer asserts
bitwise prediction identity, which the code cannot deliver on a cold CUDA context, and the docstring records
the measurements above so the next reader does not repeat the three ruled-out hypotheses.

**Next action if picked up:** confirm the cuBLAS hypothesis by running the same four-fit comparison with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)` set before the first fit. If
that makes fit 1 match fits 2+, document the pair as the supported way to obtain bitwise-reproducible ranker
fits and add an opt-in `deterministic=True` constructor flag that sets them.

---

## MRMR loses downstream AUC on the 5-signal/15-noise ranking benchmark

**Found:** 2026-09-03, while tightening a neighbouring assertion in the same file (XCUT_NONDISCRIMINATING_ASSERTS-12/13).
`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_quality_metrics.py::TestRankingQuality::test_top_k_precision_5_signals_15_noise`
fails, and it fails identically on unmodified `HEAD`, so it is pre-existing and not caused by this audit round.

**Measured:** selection downstream AUC `0.8969` against an all-signal baseline of `0.9648` -- a gap of `0.068`
where the test allows `0.03`. The selected set is two features:
`['sig4', 'sub(add(neg(sig0),neg(sig1)),add(prewarp(sig2),prewarp(sig3)))']`.

**Reading of it:** MRMR is not selecting noise -- both survivors are signal-derived, and the second is an
engineered combination of four of the five planted signals. The loss is that five independent signals have been
compressed into two columns, one of them a lossy algebraic mixture, so the downstream model can no longer
separate their individual contributions. That is a redundancy-gate calibration question (the FE combination is
being scored as subsuming its own operands), not a correctness bug in the sense of a wrong number.

**Why not fixed here:** it is a selection-quality tuning question on a benchmark fixture, and the audit round
this session is implementing contains no finding about it. Changing the redundancy gate to keep the operands
would alter selection behaviour across the whole suite, which needs its own before/after measurement rather
than a change made in passing.

### Root-caused 2026-09-04: `fe_drop_redundant_raw_operands`

The next action below was carried out, and it isolates the cause to a single default. All four arms on the same
fixture (seed 200), downstream 5-fold ROC-AUC against the five-raw-signal baseline of `0.9648`:

| Arm | AUC | Gap | Selected |
|---|---|---|---|
| baseline (5 raw signals) | 0.9648 | -- | 5 |
| default (full mode, FE on) | 0.8969 | +0.0679 | 2 |
| `use_simple_mode=True` (FE off) | 0.9639 | +0.0009 | 25 |
| `fe_drop_redundant_raw_operands=False` (FE ON) | 0.9649 | -0.0001 | 5 |
| `fe_raw_retention_max_n=25` | 0.8969 | +0.0679 | 2 |

Disabling FE closes the gap, which confirms the compression reading. But the fourth arm is the informative one:
with FE still fully ON and only the operand-drop rule disabled, the selection lands on exactly five features and
recovers the baseline AUC outright. So the loss is not caused by engineering the composites -- it is caused by
DROPPING their raw operands afterwards. The subsumption rule treats a composite as covering the columns it was
built from, but an algebraic mixture like `add(add(neg(sig0),neg(sig1)),add(neg(sig2),neg(sig4)))` is lossy: it
preserves the sum and destroys the individual contributions, which is precisely what a downstream linear model
needs when the five signals carry different coefficients (`0.8 - 0.1k` here). Raising the retention cap does not
help (fifth arm, unchanged at 0.8969), confirming it is the drop rule and not a cap.

**Still not changed here, deliberately.** `fe_drop_redundant_raw_operands` defaults to True package-wide, so
flipping it changes selection for every MRMR user and every test that pins a selected-feature set. That is the
suite-wide before/after this note already said the change needs, and it is a separate piece of work from the
audit round. What has changed is the status: this is no longer a vague "redundancy-gate calibration question"
but a located one-parameter defect with a measured fix.

**Next action if picked up:** flip the default in a scratch branch, run the full suite, and count what moves --
the expectation is that selected-feature-set pins shift while quality assertions improve. If the blast radius is
tolerable, the better fix is narrower than a flag flip: keep an operand whenever the composite is lossy with
respect to it (the composite cannot reconstruct the operand), rather than dropping on name-containment alone.

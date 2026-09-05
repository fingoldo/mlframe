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

### 2026-09-04: the cuBLAS hypothesis is REFUTED, and four more with it

The next action below was carried out. It does not work, and neither do four further candidates. Each arm is
run in its own cold process (`benchmarks/mlp_ranker_cold_fit_probe.py`), because anything that warms the
process first destroys the effect -- the same trap that invalidated the original thread-count probe.

| Arm | fit 1 vs fit 2 | fits 2-4 |
|---|---|---|
| control (shipped defaults) | maxdiff `0.00631118` | bit-identical |
| `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | maxdiff `0.00631118` | bit-identical |
| device generator seeded and drawn before fit 1 | maxdiff `0.00631118` | bit-identical |
| `torch.cuda.empty_cache()` before EVERY fit | maxdiff `0.00631118` | bit-identical |
| **CPU only (`CUDA_VISIBLE_DEVICES=""`)** | **bit-identical** | **bit-identical** |

Two further direct measurements, both negative: the CUDA generator state is BYTE-IDENTICAL immediately before
all four fits (so the divergence is not an RNG-state difference), and no global precision switch
(`matmul.allow_tf32`, `cudnn.allow_tf32`, `float32_matmul_precision`, `cudnn.benchmark`, `cudnn.deterministic`)
is mutated by the first fit, ruling out a first-Trainer-construction side effect on a global.

**The actionable part:** do NOT build the opt-in `deterministic=True` constructor flag proposed below. It would
set exactly the pair that was just measured to change nothing, so it would cost throughput and deliver no
reproducibility. The magnitude is unchanged to all eight digits with both controls active.

**What is now established:** the effect is CUDA-specific (CPU is fully deterministic across all four fits), it
is not RNG, not allocator state, not workspace/algorithm selection as exposed by torch's determinism switches,
and not a mutated global. It survives everything reachable from outside the training loop.

### ROOT CAUSE: the GPU dropout mask draw on a cold CUDA generator

The in-training bisect was run and it lands the cause exactly.

**Dropout is necessary and sufficient.** With `dropout=0.0`, all four fits are bit-identical including the
first. With the shipped default `dropout=0.1`, only fit 1 diverges, by the usual `0.00631118`. Nothing else
about the estimator changes between those two runs.

**The divergence is in the CUDA generator, not the data and not the CPU.** Instrumenting the first training
batch:

| measured at first training batch | fit 1 vs fit 2 |
|---|---|
| batch CONTENT (hash of the tensors handed to the step) | byte-identical |
| CPU RNG state | identical |
| **CUDA RNG state** | **DIFFERENT** |
| module device | same (`cuda:0`) |
| batch 0 loss | `0.7212307453` vs `0.7187753320` |

So the very first backward pass already differs, on identical weights and byte-identical input, purely
because the GPU dropout masks differ. That also explains the magnitude: `2.5e-3` on a `0.72` loss is far too
large for float32 kernel-selection noise, and always was a hint that the earlier hardware-level hypotheses
were looking in the wrong place.

**Partial fix, measured but NOT shipped.** Re-seeding the CUDA generator from a Lightning `on_train_start`
callback shrinks the gap 16x, from `0.00631118` to `0.00038409`, but does not close it -- so device
randomness is still consumed differently between train start and the first dropout draw. It is left
unshipped for the same reason the cuBLAS pair was: it mutates global CUDA RNG, which is precisely the
process-wide mutation `ranker.py`'s own note removed, and a partial fix does not buy reproducibility anyway.
Eagerly seeding CUDA at `torch.manual_seed` time (rather than letting it happen lazily) changes nothing,
which rules out lazy CUDA seeding as the mechanism.

**The obvious fix is not available.** A per-fit `torch.Generator` seeded from `random_state` would be exact
and would carry no global-mutation objection -- but neither `nn.Dropout` (`p`, `inplace`) nor
`F.dropout` (`input`, `p`, `training`, `inplace`) accepts a `generator` argument, so there is nothing to pass
it to. Delivering it means a custom dropout module (`torch.rand(shape, generator=g) > p`, rescaled) in
`training/neural/flat.py`, which is the trunk every flat MLP in the package builds on. That changes the mask
SEQUENCE for every such model, so every pinned prediction/metric in the suite moves. It is an architecture
change with a package-wide blast radius, not a local fix, and it should be agreed before it is written.

### CLOSED: documented, not fixed

Decided by the maintainer: the custom dropout module is not worth its cost, and this is documented as a known
limitation instead.

**The limitation, stated plainly.** With `dropout > 0` on CUDA, the FIRST `MLPRanker` fit in a process is not
bitwise reproducible against later fits in that same process. Fits after the first are bitwise reproducible,
and with `dropout=0.0` every fit including the first is. This is not a correctness defect: both fits are valid
trainings that differ only in which dropout masks they drew.

**Why it was not fixed.** Neither `nn.Dropout` nor `F.dropout` accepts a `generator`, so pinning the mask
stream means a custom dropout module in `training/neural/flat.py` -- the trunk every flat MLP in the package
builds on. That changes the mask sequence for every such model and moves every pinned prediction and metric in
the suite. The benefit is confined to one scenario (reproducing a single cold fit bit-for-bit), which nothing
in the pipeline depends on.

**What holds the line.** The F10 test pins what the estimator genuinely delivers -- bit-identical initial
weights from an identical seed, different weights from a different seed, and same-seed predictions far closer
than two different initialisations -- and its docstring carries the measured cause so the next reader does not
re-derive it. `benchmarks/mlp_ranker_cold_fit_probe.py` and `benchmarks/mlp_ranker_batch_bisect.py` reproduce
the finding on demand.

**Do not reopen by probing process-global switches.** Seven were eliminated by measurement (thread count,
`cudnn.benchmark`/`deterministic`, CUDA context init, `use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG`,
a pre-seeded device generator, `empty_cache()` per fit, and any mutated precision flag), and the cause is known
to be the GPU dropout mask draw. Reopening is only worthwhile if the decision above changes.

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

### CLOSED: fixed narrowly, not by flipping the coarse flag

The narrow fix turned out to need no new machinery. `_fe_raw_redundancy_drop` already carried exactly the right
instrument -- `raw_retains_linear_signal_given_children`, a permutation-floored partial rank-correlation that
keeps a raw retaining private linear signal given its children, so a pure-noise raw still drops. It was gated
to `use_simple_mode` only, on the argument (in its own comment) that a subsumed monotone operand and a genuine
linear term are statistically indistinguishable per-raw. That argument holds per-raw and still cost real
accuracy in aggregate, on the very shape the comment's own example describes.

The leg now runs in full mode as well, under `fe_keep_linearly_usable_raw_operands` (default True).

| Arm | AUC | Selected |
|---|---|---|
| five raw signals (baseline) | 0.9648 | 5 |
| default, after the fix | 0.9649 | 5 |
| `fe_keep_linearly_usable_raw_operands=False` (previous behaviour) | 0.8969 | 2 |
| `fe_drop_redundant_raw_operands=False` (coarse off-switch) | 0.9649 | 25 |

The last row is why the narrow fix is the better one: the coarse flag reaches the same accuracy by keeping
five times as many features, because it disables the sweep for genuinely-subsumed operands too. Pinned by
`tests/feature_selection/mrmr/test_linearly_usable_raw_operands_kept.py`, which asserts the outcome (recovered
accuracy, still-compact selection) rather than the mechanism, and pins the opt-out so the fixture is shown to
discriminate rather than passing whatever the code happens to do.

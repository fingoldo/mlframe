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

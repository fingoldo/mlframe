# Audit report: training_composite_discovery

**Scope**: `C:/Users/Admin/Machine learning/mlframe/src/mlframe/training/composite/discovery/` (composite-model target-discovery subsystem: MI-based base ranking, transform screening/registry consumption, tiny-model rerank, honest-holdout / y-scale / structural-fragility gates, stability selection, forward-stepwise multi-base, auto-chain/interaction-base/region-adaptive opt-in steps, and their numba-kernel dispatchers).

**Files reviewed**: 50 of 65 files in scope (all 47 non-benchmark production files read in full, line-by-line; 3 of the 34 `_benchmarks/` scripts read in full as representative samples, the remaining 31 benchmark scripts spot-checked via targeted grep for common anti-patterns — mutable defaults, bare/broad excepts, unseeded RNG — given they are standalone `__main__` profiling/A-B harnesses never imported by any production code path, i.e. out of the blast radius that matters for "corrupt a trained model or its predictions").

**LOC reviewed**: ~15,778 LOC read in full (100% of the non-benchmark production code, `wc -l` confirmed) + ~163 LOC of benchmark scripts read in full + grep-level coverage of the remaining ~3,549 benchmark LOC. Total in-scope LOC per the assignment: 19,490 (`wc -l` over all `*.py` under the directory including `_benchmarks/`).

Out of scope per the assignment brief (not reviewed): `feature_selection/filters/**` (MRMR engine) and `feature_selection/shap_proxied_fs/**` — both already closed out in a dedicated 2026-07-25 audit cycle.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| TRAINING_COMPOSITE_DISCOVERY-1 | P2 | `_eval.py:649-651` | Bootstrap MI-gain CI crashes (`IndexError`) instead of degrading gracefully when `mi_gain_bootstrap_n=1` and that lone replicate fails | Guard with `boot_finite.size > 0 and boot_finite.size >= bootstrap_n // 2` before calling `np.percentile` | Fuzz any `size_guard = n // k` pattern feeding `np.percentile`/`np.median`/`np.quantile` with `n∈{0,1,2}` and an empty-array input; flag guards that admit `size==0` |
| TRAINING_COMPOSITE_DISCOVERY-2 | P2 | `_collinear_numba.py:326-331` vs `354-358` | Module-level `_KEEP_MASK_CACHE` (`OrderedDict`) is read (`.get()` + `.move_to_end()`) WITHOUT the lock that guards every write/evict path (`popitem`+`__setitem__`), risking a `KeyError` (crash) or corrupted LRU order under concurrent `CompositeTargetDiscovery.fit()` calls in the same process | Take `_KEEP_MASK_CACHE_LOCK` around the read-and-touch sequence too (or switch to a lock-free structure) | Grep for module-level mutable caches where a `Lock`/`RLock` name is referenced on some access sites but not others in the same file; flag any `.get(`/`[...]` read on a cache with a sibling locked write |
| TRAINING_COMPOSITE_DISCOVERY-3 | P2 | `_calibration_gate.py` (whole file, 202 LOC) | A fully-implemented, unit-tested "calibration-aware ranking signal" (OOF bias/variance-miscalibration penalty) is never imported or called by any discovery gate (`_tiny_rerank.py`, `_filter_and_gate.py`, etc.) — confirmed via repo-wide grep, zero call sites outside its own definition and its dedicated test file | Either wire it into `_tiny_model_rerank`'s scoring (the module was clearly designed for that purpose) or delete it and its test, and note the rejection rationale if it was tried and found not to help | Reachability scanner: for every non-`__init__`, non-test module in a package, flag it if every cross-reference to its public symbols resolves only to `__init__.py` re-exports and `tests/` |
| TRAINING_COMPOSITE_DISCOVERY-4 | P3 | `bayesian.py` (whole file, 332 LOC) | `bayesian_alpha_fit` / `bayesian_alpha_fit_bootstrap` (conjugate/bootstrap posteriors for `linear_residual`'s alpha/beta) are never called from any production discovery path — only unit-tested standalone and re-exported via `__init__.py`. Unlike `forward_stepwise.py` (whose docstring explicitly says "Not auto-integrated into Discovery.fit()... standalone helper ships now"), this module carries no such "intentionally standalone" framing, so it reads as an orphaned feature rather than a documented public utility | Add an explicit "standalone utility, not wired into `fit()`" note to the module docstring (matching `forward_stepwise.py`'s convention) if intentional, or wire it into a diagnostic/report field if it was meant to be consumed | Same reachability scanner as TRAINING_COMPOSITE_DISCOVERY-3; additionally require every "public standalone utility" module to carry an explicit "not auto-integrated" docstring marker so the reachability scanner doesn't need to re-adjudicate intent |
| TRAINING_COMPOSITE_DISCOVERY-5 | P3 | `_screening_tiny_perbin.py:42-43` | Dead, exact-duplicate module-level `_KFOLD_SPLIT_CACHE` / `_KFOLD_SPLIT_CACHE_MAX` declarations left over from the `_screening_tiny.py` module split; this file actually imports and uses the live cache from `_screening_tiny.py` (`_cached_kfold_splits`), so these two lines are never read or written | Delete the two dead lines | Grep for identically-named module-level globals declared in 2+ sibling files of a carved-out module family (`_foo.py` + `_foo_bar.py`); flag any declaration whose name is never referenced again in the same file (an AST unused-module-global check) |

## Counts

- P0: 0
- P1: 0
- P2: 3
- P3: 2

## Narrative

### TRAINING_COMPOSITE_DISCOVERY-1 (P2): bootstrap MI-gain CI can crash on `mi_gain_bootstrap_n=1`

In `eval_one_transform`'s bootstrap block (`_eval.py`), when `bootstrap_n > 0` each replicate's MI-gain is computed
inside a `try/except`; a failure sets `boot_gains[b] = float("nan")`. After the loop:

```python
boot_finite = boot_gains[np.isfinite(boot_gains)]
if boot_finite.size >= bootstrap_n // 2:
    mi_gain_lcb = float(np.percentile(boot_finite, 2.5))
```

With `bootstrap_n=1`, `bootstrap_n // 2 == 0`, so the guard `boot_finite.size >= 0` is trivially true even when
`boot_finite` is empty (the single replicate failed). `np.percentile` on a zero-length array raises `IndexError`
(verified live: `np.percentile(np.array([]), 2.5)` → `IndexError: index -1 is out of bounds for axis 0 with size
0`). This exception is NOT caught by the per-replicate `try/except` (that block only wraps the replicate loop body,
not the post-loop percentile call), so it propagates out of `eval_one_transform`, out of the (possibly joblib
`threading`-backed) work-item dispatch in `_fit.py`, and crashes the whole `fit()` call for that target — not just
a single spec being dropped. `mi_gain_bootstrap_n` is a plain `int` config field (`_composite_target_discovery_config_base.py:762`, default `0`) with no floor validator, so a caller setting `1` (a
plausible "just enable the bootstrap gate cheaply" mistake, or a hand-tuned low-cost config) is one degenerate
replicate away from a hard crash. The fix is a one-line guard change; a regression test should pin a mock/patch
that forces the sole bootstrap replicate's inner MI call to raise and assert `eval_one_transform` still returns a
reject-or-accept dict rather than propagating `IndexError`.

### TRAINING_COMPOSITE_DISCOVERY-2 (P2): `_KEEP_MASK_CACHE` read path is unlocked while the write path is locked

`_collinear_numba.py`'s `near_collinear_keep_mask_fast` maintains a bounded, module-level (process-wide, shared
across every `CompositeTargetDiscovery` instance) `OrderedDict` cache keyed by a content hash of the feature
matrix. The docstring explicitly motivates the cache by CROSS-TARGET reuse ("discovery runs once per target...
when two targets share the byte-identical matrix for a base... the O(B^2*n) walk is recomputed"). The write path
is correctly guarded:

```python
with _KEEP_MASK_CACHE_LOCK:
    if len(_KEEP_MASK_CACHE) >= _KEEP_MASK_CACHE_MAX_ENTRIES:
        _KEEP_MASK_CACHE.popitem(last=False)
    _KEEP_MASK_CACHE[_ck] = keep.copy()
```

but the read/touch path is NOT:

```python
_hit = _KEEP_MASK_CACHE.get(_ck)
if _hit is not None:
    _KEEP_MASK_CACHE.move_to_end(_ck)
    return np.asarray(_hit.copy())
```

If a second thread's `popitem(last=False)` evicts the very entry a first thread just found via `.get()`, the first
thread's subsequent `.move_to_end(_ck)` raises `KeyError` (an `OrderedDict.move_to_end` on a since-deleted key is
not a race CPython's GIL makes safe — it is two separate C-API calls with a window in between). Within a single
`fit()` call this cannot fire (`near_collinear_keep_mask` is only invoked from the strictly serial per-base setup
loop in `_fit.py`, before the parallel `eval_one_transform` dispatch), so I could not find an in-repo call site
that currently drives two `fit()` calls concurrently in the same process (the stability-selection and per-group
paths both run their replicate/group loop of `fit()` calls serially). The finding is downgraded from P1 to P2
because of that — but the risk is real and specifically invited by the module's own cross-target-reuse rationale;
any caller that fits multiple targets' `CompositeTargetDiscovery` instances from a `joblib.Parallel(backend=
"threading")` pool (a natural pattern this same codebase uses pervasively for per-spec/per-fold work one layer
down) would hit it. Fix: extend the lock to cover the read-and-touch sequence, or switch to a data structure with
lock-free reads (e.g. a plain `dict` + a separate insertion-order counter, or `functools.lru_cache`-style
thread-safety).

### TRAINING_COMPOSITE_DISCOVERY-3 (P2): `_calibration_gate.py` is fully built, fully tested, and never wired in

`_calibration_gate.py` implements `calibration_adjusted_score` / `calibration_penalty` — a bias+variance-
miscalibration penalty meant to dock a spec's ranking score when its OOF residuals are biased or mis-scaled
relative to its in-fold residuals, exactly the kind of "lucky-but-overfit" spec the tiny-rerank and stability-
selection machinery elsewhere in this package are built to catch. The module docstring frames it as "a PURE,
OPTIONAL ranking signal the rerank caller MAY consult." A repo-wide grep for `calibration_adjusted_score` /
`_calibration_gate` / `CALIBRATION_GATE_DEFAULT_ENABLED` outside the module's own file turns up exactly one other
hit: its own dedicated test file (`tests/training/composite/screening/test_calibration_gate.py`) plus one
reference in `tests/training/composite/discovery/test_training_composite_discovery_fixes.py`. Nothing in
`_tiny_rerank.py`, `_filter_and_gate.py`, `_honest_rmse_gate.py`, or `_yscale_holdout_gate.py` calls it. This is
202 lines of maintained, tested, dead production code — either a half-finished feature that should be completed
(wired into `_tiny_model_rerank`'s per-spec scoring, which already computes per-fold OOF/in-fold residuals via
`_tiny_cv_rmse_y_scale`'s fold loop and would need only to thread them through) or a rejected experiment that
should be removed along with its test, with the rejection reasoning recorded per the project's
REJECTED-≠-DELETED convention (which this module currently does not follow — it presents as live infrastructure,
not a documented rejected prototype like `_region_adaptive.py` does).

### TRAINING_COMPOSITE_DISCOVERY-4 (P3): `bayesian.py`'s posterior fitters are never called in production

`bayesian_alpha_fit` (conjugate Normal-Inverse-Gamma posterior) and `bayesian_alpha_fit_bootstrap` (empirical
bootstrap posterior) compute credible intervals for a `linear_residual` spec's `(alpha, beta)`. A repo-wide grep
for `bayesian_alpha_fit(` (call syntax, not just the def line) finds only the function's own definition — no
caller anywhere in `src/`. Contrast this with `forward_stepwise.py`, whose module docstring explicitly states
"Not auto-integrated into Discovery.fit()... Standalone helper ships now; auto-integration is opt-in" — a
deliberate, documented design choice for an unwired-but-intentional utility. `bayesian.py` carries no equivalent
framing, so a reader cannot tell whether this is (a) an intentional standalone diagnostic API for users who import
it directly (plausible — it is re-exported through `composite/__init__.py`), or (b) a feature that was meant to
feed the alpha-drift gate (`_eval_stats.py::apply_alpha_drift_gate`, which today uses a plain two-half z-test, not
a Bayesian credible interval) and was never finished. Flagging as a P3 documentation/architecture gap rather than
asserting either reading as definitive, per the "flag alternative readings" convention — the fix is either an
explicit "standalone, not wired into fit()" docstring note (if (a)), or actually wiring it into the alpha-drift
gate (if (b)).

### TRAINING_COMPOSITE_DISCOVERY-5 (P3): dead duplicate `_KFOLD_SPLIT_CACHE` in `_screening_tiny_perbin.py`

`_screening_tiny_perbin.py` was carved out of `_screening_tiny.py` (both files' docstrings say so verbatim). It
declares its own module-level `_KFOLD_SPLIT_CACHE: dict[...] = {}` and `_KFOLD_SPLIT_CACHE_MAX = 256` at lines
42-43 — byte-identical to the ones already live in `_screening_tiny.py` (lines 47-48) — but then, four lines
later, imports the REAL cache-consulting function from the original module:

```python
from ._screening_tiny import (
    _build_tiny_model,
    _cached_kfold_splits,
    _silence_tiny_model_output,
)
```

`_cached_kfold_splits` reads/writes `_screening_tiny.py`'s own `_KFOLD_SPLIT_CACHE` global (closure over that
module's namespace), never this file's copy. A grep confirms the two lines in `_screening_tiny_perbin.py` are
never referenced again anywhere in that file — pure copy-paste residue from the split that does nothing except
mislead a future maintainer who might reasonably assume editing this file's cache constants affects caching
behavior here. Low risk (no functional impact — it's simply two dead lines) but a real, reportable hygiene finding
per the audit brief's explicit instruction to capture every P3.

## Dimension coverage notes

- **Correctness bugs / crashes**: one confirmed (TRAINING_COMPOSITE_DISCOVERY-1). No off-by-one, mutable-default-argument, or silent-wrong-formula bugs found in the reviewed code — the codebase's own extensive inline commentary (bit-identity contracts, borderline-band re-decision logic for every numba dispatcher, `_dcf`/domain-refinement double-gating) reflects a codebase that has already been through multiple audit/fix cycles for exactly this class of bug.
- **ML correctness (leakage / reproducibility / calibration / sample-weight / honest OOS)**: extensively re-verified — the honest-holdout carve (`_honest_holdout.py`), the base-target leakage guard (`_leakage.py`, `_fit_temporal.py`), the group-disjoint y-scale gate (`_yscale_holdout_gate.py`), the structural-fragility gate, and the causal-lag/grouped-causal-base provenance exemptions (`_causal_lag.py`, `_grouped_causal_bases.py`) all correctly restrict themselves to train-only rows and correctly guard against a holdout/train overlap (explicit `np.intersect1d` assertions in `_fit.py` and `_honest_holdout.py`). No leakage defect found. All RNG usage goes through seeded `np.random.default_rng(...)`; no unseeded global-RNG call found via grep across the whole directory.
- **Computational efficiency**: no new inefficiency found — every hot loop already carries a numba/parallel dispatcher with a documented, benchmarked crossover (or an explicit "no actionable speedup, measured" note), consistent with the CLAUDE.md history of prior perf-audit cycles on this exact codebase.
- **Edge cases and robustness**: TRAINING_COMPOSITE_DISCOVERY-1 is the one gap found. All other size-degenerate paths I traced (empty `train_idx`, all-NaN columns, single-class-like near-constant `y`, `train_idx.size < 50`, zero base candidates, zero usable features, zero-row screening samples) have explicit, tested early-return guards.
- **Test coverage gaps**: no source-inspection-only tests found (the codebase consistently tests behavior/output, e.g. bit-identity assertions between numba kernels and numpy references, biz_value RMSE floors). The one concrete gap is implied by TRAINING_COMPOSITE_DISCOVERY-1: the `mi_gain_bootstrap_n=1`-with-a-failing-replicate branch has no regression test, which is precisely why the bug survived.
- **Code quality/architecture**: two dead-code findings (TRAINING_COMPOSITE_DISCOVERY-3, -5) and one unclear-intent orphaned module (TRAINING_COMPOSITE_DISCOVERY-4). No duplication, misleading naming, or overly-broad `except` clauses beyond the ones already deliberately documented as "never abort discovery on a heuristic gate" (a consistent, intentional pattern throughout, not a hygiene issue).
- **OSS/hygiene**: no stale audit-wave markers, mojibake, or comment-length violations found; comments consistently explain WHY, not WHAT, matching the project's documented convention.

## Meta-test ideas at the cluster level

- A generic "concurrency-safety linter" for this codebase: any module-level mutable cache (`dict`/`OrderedDict`)
  guarded by a `Lock`/`RLock` on SOME access sites should require the lock on ALL access sites in the same file —
  this single AST/grep check would have caught TRAINING_COMPOSITE_DISCOVERY-2 and is broadly reusable across
  `pyutilz`'s other caches (kernel-tuning cache, prebin cache, etc.).
- A generic "size-guard admits zero" fuzz check: for any `x.size >= n // k` (or `>= max(1, n // k)`) pattern
  feeding a downstream reduction that is undefined on an empty input (`np.percentile`, `np.median`, `scipy.stats`
  functions), fuzz `n ∈ {0, 1, 2, k}` with an all-failing inner computation and assert no exception escapes.
- A "public-but-unreachable module" reachability scanner across the whole `mlframe` package: for every non-test,
  non-`__init__.py` module, grep for call-syntax (not just import) references to its top-level public functions
  outside its own file and its dedicated test file; flag modules with zero production call sites so they get
  either wired in, explicitly marked "standalone utility, not integrated," or removed — this would have caught
  both TRAINING_COMPOSITE_DISCOVERY-3 and -4 in one pass, and is very likely to catch similar orphaned modules in
  other `training/` subpackages given this cluster alone had two independent instances.

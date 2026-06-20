# mlframe — project conventions

## Never stop to ask obvious / low-stakes questions — keep working (CRITICAL)

Do NOT pause to ask the user obvious or unimportant questions — execution
order ("which first?", "in what order?"), or "should I also do the deeper
fix / the next item?", or any choice with a sensible default. Pick the
sensible answer yourself and DO ALL the work end-to-end. The cost is
concrete: every needless question burns a full user round-trip during which
nothing useful happens — wasted time the user has explicitly and repeatedly
called out (2026-05-22, 2026-06-02).

In particular, when you finish one thing and there is obvious follow-up work
— follow-ups you yourself just flagged, the next bug in a list, the deeper
root cause after shipping a band-aid — **just do it**. "I shipped the safe
fix; want me to do the deeper one?" is exactly the banned pattern: ship the
safe fix AND do the deeper one, then report both.

Pause for user input ONLY on: (a) a genuine accuracy/functionality tradeoff
with no clear winner; (b) a destructive, irreversible operation; (c) touching
another session's uncommitted WIP or clearly out-of-scope code; (d) a hard
blocker you cannot proceed past. Everything else: keep going.

## Enable corrective mechanisms by default (CRITICAL)

When you build a corrective mechanism that fixes a bug class — DCD
cluster pruning, NaN-sentinel routing, dtype auto-promotion,
MDLP-collapsed unsupervised fallback, edge dedup, frozen recipe `extra`,
etc. — **flip the default to ON in the same change**. Do NOT preserve
the pre-fix default for "bit-stability" or "legacy compatibility"
when the legacy behaviour was silently wrong.

**Why:** legacy bit-stability is NOT a virtue when legacy was wrong.
Every user who doesn't read the changelog / find the opt-in flag keeps
suffering the bug. The default config is the ONE config that matters
for blast radius. Opt-in flags exist for the small minority who need
the legacy behaviour for benchmarks or replay.

**How to apply:**
- After the corrective mechanism lands and existing tests pass with
  the flag on, flip the constructor default to `True`.
- If existing tests break, the test was asserting pre-fix wrong
  behaviour — fix the test, don't disable the mechanism.
- Document the flip in the constructor docstring + commit message;
  list the opt-out parameter for legacy callers.

**Validated 2026-05-30**: `dcd_enable=True` flipped to default after
Layer-6 biz_value confirmed DCD is the production-correct redundancy
mechanism (near-duplicate decoys, collinear clusters, synergistic
groups all benefit). The 0.003× overhead was negligible. A handful of
pre-existing tests that asserted pre-fix wrong behaviour ("no cluster
aggregate in support" / "leak feature directly picked") got updated to
the post-fix contract.

## Fuzz / combo tests are bug DETECTORS, not bug HIDERS (CRITICAL)

**The fuzz/combo suite (`tests/training/test_fuzz_suite.py` +
`_fuzz_combo.py`) exists to find production bugs. Every test failure is a
real prod bug unless you can prove otherwise. NEVER paper a failing combo
over with any of the following without first fixing the underlying
production code:**

1. **Canonicalisation in `_fuzz_combo.py`** — `_canonical_*` methods and
   `canonical_key` rules must collapse only **semantically-equivalent**
   combos (e.g. `imbalance="balanced"` is the same regardless of the
   imbalance-mode flag because at 50/50 the mode is meaningless). They
   must NOT collapse a combo "because it crashes". A canon rule whose
   justification is "fuzz cXXXX hangs / raises / produces empty val" is
   **prohibited** — fix the prod code instead.
2. **Runtime canonicalisations in `test_fuzz_suite.py`** — the
   `*_eff = ... if condition else ...` rewrites at the top of
   `test_fuzz_train_mlframe_models_suite` (before the suite call). Same
   rule: legitimate only when the rewrite preserves semantics; never as
   a guard against a real prod crash.
3. **`pytest.mark.xfail` / `pytest.skip`** — reserved for genuine
   third-party / OS / unfixable issues (Windows symlinks, optional
   dep missing, sklearn API limitation). A test that surfaces an
   mlframe-internal bug must be FIXED, not xfailed-with-TODO.
4. **Defensive guards in production** (trainer.py `_apply_pre_pipeline_-
   transforms`, wrappers.py CV-fold loop, etc.) — `if len(...) == 0:
   skip` patterns are acceptable only when the empty path is a
   legitimate user scenario. When the empty arises from an upstream
   bug, the guard is a band-aid that hides the bug forever — fix
   upstream.

### Concrete examples from this codebase (do not repeat)

- **Bad** (2026-04-26): `_canonical_text_col_count` zeroed text columns
  for CB+small-n+heavy-NaN combos because CB's `occurrence_lower_bound=50`
  produces an empty TF-IDF dictionary on tiny inner-CV folds. Hid a
  real prod hang.
  **Good** (2026-04-27): replaced with
  `training/helpers.compute_cb_text_processing` which scales
  `occurrence_lower_bound` proportionally to the fit-time row count;
  wired into `trainer._train_model_with_fallback` and
  `feature_selection/wrappers.py` RFECV inner-fold. Canon retired.
- **Bad** (still active): `canonical_key:327-335` collapses
  `inject_degenerate_cols=True` to `False` on CB+multilabel because CB
  mis-detects `num_const`/`num_null` as cat features. Hides c0062.
  **Owed fix**: explicit `cat_features=` arg in CB wrapper (or
  type-cast guard) so CB doesn't auto-detect numeric columns as
  categorical when `inject_degenerate_cols=True`.
- **Bad** (still active): `canonical_key:406` forces
  `remove_constant_columns_cfg=True` whenever degenerate / all-NaN
  columns are injected. Hides polars-ds RobustScaler crashing on
  zero-IQR (c0008/c0116).
  **Owed fix**: zero-IQR guard in the polars-ds robust scaler wrapper
  (skip / clip / fall back).
- **Bad** (still active): four layered "0-row val" tolerances in
  `trainer.py` (`_apply_pre_pipeline_transforms` ×2, `_setup_eval_set`,
  `_compute_split_metrics`). All point at the same upstream defect:
  outlier detection / aging-limit collapsing val to 0 rows silently.
  **Owed fix**: `_apply_outlier_detection_global` should raise on
  empty val (mirroring the train-side `min_keep` guard at core.py:1021).
- **Bad** (deferred): two `_rule_cb_*` functions defined in
  `_fuzz_combo.py` lines 833-868 with TODOs but NOT registered in
  `KNOWN_XFAIL_RULES`. Either bugs are fixed (delete dead code) or
  unfixed (silent fail in fuzz). Both were resolved 2026-04-27.

### Process when a fuzz combo fails

1. Read the traceback. Identify the prod-code line.
2. Decide: is this a legitimate user-facing bug (yes → fix in prod) or
   a genuine third-party limitation (yes → xfail with detailed reason
   + open issue / link)?
3. If you find yourself reaching for `_canonical_*`, ask: "would a real
   user with these settings hit this same crash?" If yes, you are
   masking a prod bug — STOP and fix prod instead.
4. Fixing prod often retires multiple canon rules / guards / xfails at
   once (e.g. fixing the splitter empty-val edge retires 4 trainer
   guards + 1 runtime canon + 1 prod-config validator gap).

## Memory / RAM constraints (CRITICAL)

**Frames in mlframe can be 100+ GB.** Never copy them to work around a bug.
Copying a prod DataFrame doubles peak RAM, which on a 200 GB+ workload means
OOM — the user observed this in 2026-04-22 prod logs.

Avoid:
- `df.copy()` (pandas) or `df.clone()` (polars) inside hot paths
- `df[cols] = df[cols].astype(...)` when `df` is the caller's frame (pandas
  broadcasts-copies the sub-frame)
- Constructing a fresh `pd.DataFrame(df)` / `pl.DataFrame(df)` to "get a new
  reference"
- Any fit-transform pattern that returns a mutated input

Prefer:
- Work on views (`.iloc`, column selection, slices)
- Mutate-and-restore: `X[col] = new; try: ... finally: del X[col]`
- Use `with` / context managers that revert the mutation on exit
- Lazy eval via polars `lazy()` + `.collect()` at the leaf call
- Pass `inplace` options where sklearn / the transformer supports them

Fuzz-caught example: MRMR.fit needed to temporarily inject a `targ_<id>`
column into X for MI computation. Original code mutated caller's X
in place, leaked the injected column into downstream sklearn steps, and
tripped `validate_data` on the next transform. Fix in
`feature_selection/filters.py:~2895` must inject + remove the column in a
try/finally (never call `X.copy()`).

If you find a bug that genuinely needs a copy, escalate — the user would
rather ship a design change than accept an unconditional copy on a hot
DataFrame path.

### Caching and batching: use both, but never assume a frame fits in RAM (CRITICAL)

Caching and batching are required optimization patterns in mlframe -- every
new feature must consider both. **But remember that mlframe frames can
reach 100+ GB.** Any cache / batch / eager-conversion design MUST be safe
on a frame that does not fit in RAM (or VRAM):

- **Caching**: prefer caching CHEAP-to-rebuild artefacts (signatures,
  hashes, fitted-state snapshots, small derived arrays). Caching a
  WHOLE frame is allowed ONLY when the carrier is the caller's own
  reference (no extra copy) AND the cache key includes a content-based
  signature so unrelated callers don't accidentally hit a stale entry.
- **Batching**: use mini-batch / streaming iteration patterns when the
  per-element cost is small (PyTorch DataLoader, polars LazyFrame,
  sklearn `partial_fit`). Never materialise the whole frame into a
  single contiguous buffer when batched access is sufficient.
- **Eager conversion** (e.g. polars -> torch.Tensor, pandas ->
  ndarray): always GATE on byte size. The mlframe convention:
  - Frames under ~2 GB: eager-convert is OK -- removes per-batch
    overhead, fits comfortably in any prod host's RAM.
  - Frames above ~2 GB: keep the original carrier and pay the per-
    batch type-check cost. The OOM avoidance outweighs the per-batch
    µs overhead at this scale.
  - Frames of unknown size: treat as small (most callers know their
    data; the typical synthetic / small-test path has no `.nbytes`
    attr and benefits from eager).
- **Serialization / pickle**: never pickle a whole frame as part of an
  optimization (e.g. for inter-process caching). Pickling 100 GB
  doubles disk usage AND blocks for minutes. Use Arrow IPC if you
  truly need cross-process data, but prefer in-process design.

Concrete example (2026-05-11, Wave 22): ``TorchDataset.__init__``
originally deferred tensor conversion to ``__getitem__`` (per-batch
type-check + ``.to(dtype, device)``). The fix hoists conversion to
``__init__`` to save ~67 s per 1M-row MLP fit -- BUT gated on byte
size: above 2 GB the legacy per-batch path stays, so a 100 GB frame
does not OOM the host.

If a new feature needs caching/batching, sketch the byte-size gate
BEFORE writing code. If you cannot easily estimate the worst-case
size, ASK the user before shipping.

### Frame-type conversions are caller responsibility, NOT wrapper auto-magic

Helpers / wrappers / estimators MUST NOT silently down-convert polars
to pandas (or pandas to ndarray, or any other format-shift) on a hot
path "to make the inner library happy". A 100+ GB polars frame turning
into a 100+ GB pandas frame doubles RAM and bypasses every zero-copy
Arrow optimisation the rest of the codebase paid for.

The mlframe convention is unambiguous:

- `train_mlframe_models_suite` and the strategies layer
  (`training/strategies.py`) decide once at the suite boundary whether
  a given inner estimator wants polars or pandas, then pass the
  correct flavour through.
- Downstream wrappers (`CompositeTargetEstimator`, the new pre-pipeline
  estimators, anything else that holds a `base_estimator`) MUST NOT
  re-convert. If the inner estimator chokes on what it gets, that is
  a strategy-layer wiring issue to fix at the suite boundary, not
  inside the wrapper's `fit` / `predict` hot path.
- The ONLY allowed in-wrapper read of a polars column is the targeted
  `_extract_base`-style narrow column pull (one ndarray, not a frame
  copy). Even then the surrounding row mask / row subset must use the
  format-native `.filter(...)` / `.iloc[...]` -- never `to_pandas()`
  on the whole frame.

Concrete example (2026-05-10): `CompositeTargetEstimator` originally
did `X.to_pandas()` inside both `fit` and `predict` to work around
LightGBM 4.5 + sklearn 1.6 not accepting polars (the `feature_names_in_`
write-path bug). On a 100 GB polars frame this would have OOM'd the
host. Removed; the test suite now demonstrates the in-suite pattern
(caller does the conversion at the boundary if the inner needs it).

## /loop fuzz-profile-optimize policy: stop after 100 consecutive rejects (CRITICAL)

When running `/loop` on the fuzz profile-and-optimize prompt (see
`tests/perf/results/_loop_iter_log.md` for the canonical log shape), the
termination rule is NOT a fixed iteration cap. It is a **rejection streak**:

- Each iteration ends with one of: `RESOLVED+<speedup>`, `REJECT+<reason>`,
  `BLOCKED+<reason>`.
- Track a counter of CONSECUTIVE rejects. Reset to 0 on every `RESOLVED`.
- **Stop the loop after 100 consecutive rejects** -- that's the signal the
  suite has nothing more cheaply optimisable at the profiled scales.
- Do NOT treat "max 5 iterations" or similar hard caps as the termination
  signal even if the human's invocation prompt suggested one; the prompt's
  cap is advisory at most, the streak rule is the binding policy.

The rationale: a perf-discovery loop is only useful while it's finding
optimisations. Two rejects in a row is normal (the next profile might surface
a new hotspot). 100 in a row across diverse fuzz cells is statistically
strong evidence the suite is genuinely optimised at that scale -- further
iterations burn compute without gain. Tracking the streak (not the total
count) keeps the loop running through productive stretches and ending
naturally only when it stops being productive.

## REJECTED ≠ DELETED: keep the bench, the verdict, and the option (CRITICAL)

When an idea / optimization / parameter is **REJECTED** after measurement, that
verdict means exactly ONE thing: **it did not go into the defaults**. It does NOT
mean delete the code. Today's reject is tomorrow's win on different hardware /
data / scale — and the reproducible *negative* result is as valuable as a
positive. Every rejected idea KEEPS all of:

1. **The measuring bench / prototype, COMMITTED** (`*/_benchmarks/.../round*_*.py`),
   runnable in one command — INCLUDING agent-created benches (commit them; never
   leave a measured-and-rejected bench uncommitted and lost).
2. **A tracker row with the verdict + the exact numbers + the bench filename** —
   so re-test = re-run the named bench. Never a bare "rejected" without the deltas.
3. **Where the idea was a tunable, the OPTION stays in prod** (e.g.
   `redundancy_aggregator='jmim'`, `tree_rich_ops=("mul",)`, `mrmr_synergy_cap=None`);
   only the DEFAULT is unchanged. Combinatorial ideas tested via bench-local
   subclasses stay in the committed bench, not prod.
4. **Never silent-revert / silent-delete.** If code at a call-site was touched,
   leave a `# bench-attempt-rejected (date): X->Y, reason` note there.

User directive (2026-06-04): "под reject ты понимаешь, что параметр не пошёл в
дефолты, но сам код-то остался — на случай если захотим перетестировать потом" —
yes, and commit the rejected benches too.

## pyutilz hotspots are in scope — optimize them too (CRITICAL)

``pyutilz`` (``D:/Upd/Programming/PythonCodeRepository/pyutilz``) is OUR
first-party library, not a third-party dependency. When a profile attributes
a hotspot to a ``pyutilz`` function (it shows up as
``...\pyutilz\src\pyutilz\...\*.py:LINE`` in the cProfile output), optimize it
in the pyutilz repo with the same standards used for mlframe code: measure
before/after, require bit-identity (or documented numerical equivalence), add
a regression test, and always try ``numba.@njit`` on a numpy hotspot before
declaring it at the floor (see the always-try-njit rule). Commit + push the
pyutilz change to its own repo (``main`` branch). Do NOT dismiss a pyutilz
hotspot as "external / out of scope" — it is ours to fix.

## Profile every new feature with cProfile + optimize hotspots (CRITICAL)

Each new feature added to mlframe must be profiled and its hotspots
optimized before being declared done. This is non-negotiable: features
without an explicit profile pass accumulate latent overhead that only
surfaces as a regression on prod. The standard pipeline:

1. **Write a cProfile harness** for the feature (small-shape + production-
   shape inputs). Save it inside the feature's package — never to
   `D:/Temp` or `/tmp` — so any maintainer can rerun. See
   `mlframe/training/_profile_dummy_baselines.py` as template.
2. **Sort by cumulative time, top 20-30**. The list is the optimization
   plan.
3. **Optimize hotspots where it materially helps**, considering:
   - `numba.njit` (typed JIT for numpy-heavy hot loops)
   - `numba.njit(parallel=True)` + `numba.prange` (multi-core scaling)
   - `numba.cuda.jit` (GPU kernels for very heavy elementwise / reduction
     workloads)
   - `cupy` (drop-in numpy on GPU — often the simplest GPU path when the
     work IS already vectorized numpy)
   - Vectorize loops; replace `apply` with `to_numpy() + ufunc`; cache
     repeated computations; bound input size for unbounded passes (e.g.
     ACF tail-cap pattern in `dummy_baselines._detect_acf_periods`).
4. **Calibrate against cProfile attribution overhead.** cProfile inflates
   pandas/sklearn deep-stack call timings ~10-13× vs standalone wall-
   time microbench. If cProfile flags a hotspot but a wall-time
   microbench on the same function shows <1ms, it's attribution noise
   — document this in the profile harness docstring so future re-runs
   don't re-flag.
5. **Document `no actionable speedup`** when the conclusion is "no
   optimization needed" — same docstring location. Future maintainers
   should see the trail and not redo the analysis.

This rule applies to: new training stages, new metric paths, new
diagnostic passes, new pre-/post-processing transforms, new model
wrappers, and new pipeline integration points. It does NOT apply to
trivial helper additions or pure refactors that don't change the
hot path.

## A measured speedup is a LEAD, not a verdict — never dump it on a hasty measurement (CRITICAL)

When a profile or micro-bench shows ANY speedup, that is a lead to investigate rigorously — NEVER a thing to dismiss
with a hand-wave ("transfer overhead eats it", "marginal", "not worth a dispatcher", "carries some risk"). A 30% — or
even a 1-5% — win is worth pursuing; throwing it away on one hasty/cold measurement is a real, repeated failure mode
(2026-06-08: a GPU argsort was almost discarded on a single cold-cupy shot reading "1.3x, transfer eats it" — a proper
warm multi-size bench showed 1.95x@50k / 3.92x@200k / 4.94x@1M, bit-identical).

Before concluding ANYTHING about an optimization:

1. **Measure properly, not once.** Warm the kernel (numba JIT / cupy NVRTC / cache), loop enough iterations to dwarf
   timer noise, and run it MULTIPLE times. A single cold run is not a measurement — it is noise. The first GPU call
   pays a one-time compile/sweep that has nothing to do with steady-state cost.
2. **Sweep the size range, small AND large.** The crossover is the whole story: a kernel that loses at 10k can win 5x
   at 1M (GPU/transfer-amortised kernels especially). "It's marginal" at one size says nothing about another — find
   where it wins and gate there, do not reject globally from one point.
3. **Validate END-TO-END, then gate — do not reject.** A micro-win that does not survive the full pipeline (per
   "Profile full pipeline always") is still real data: it means the gate threshold is higher, or the path needs
   batching, NOT that the option dies. Find the size/condition where it nets out positive and gate to it.
4. **Hardware-relativity: the dev box can HIDE a win.** A weak laptop GPU (or a contended CPU) can make a real win
   look neutral/negative when it would clearly win on production HW (datacenter GPU, NVLink, uncontended cores). When
   isolated-faster but end-to-end-neutral HERE, you KEEP it as an env/size-gated option (see "REJECTED != DELETED")
   defaulted to the locally-measured-best — you do NOT delete a hardware-relative win. Promote to kernel_tuning_cache
   so each host picks its own crossover.
5. **The only valid rejection is a measured, multi-size, end-to-end one — written down with the numbers.** "Felt
   marginal" is not a rejection; it is a skipped investigation. If you catch yourself reaching for a one-line dismissal
   of a measured speedup, STOP — that is the warning sign you are about to waste the win.

## A/B perf-validation methodology — the procedure for validating every optimization (CRITICAL)

The concrete how-to that operationalizes "a measured speedup is a LEAD". Every perf change (loop iteration or one-off) is validated this way BEFORE it ships. Validated across the 2026-06 perf loop (40+ RESOLVED optimizations, all bit/byte-identical).

1. **Warm, then measure.** The first call pays one-time cost (numba JIT compile, cupy NVRTC, import, cache fill) unrelated to steady state. Warm the kernel + any cache, then time. A cold single shot is noise, never a measurement.
2. **Best-of-N / median over many runs**, never one shot — loop enough iterations to dwarf the timer.
3. **Measure BOTH isolated AND end-to-end.** Isolated (the function alone) proves the local win exists; e2e (the full fit / report / suite) proves it SURVIVES. A real isolated win that is flat-or-slower e2e is a **REJECT**, not a ship — the saving was dwarfed by an inner njit/external kernel, by njit-spawn/contention, or by memory bandwidth (the recurring trap, e.g. iter53). Find the size/condition where it nets out positive e2e, or reject.
4. **Real baseline, not a from-memory rewrite.** The OLD side of the A/B must be the ACTUAL prior code — load it via `git show HEAD:<path>` (or run it in a separate process / a baseline worktree). Compare two real artifacts, never your recollection of the old behavior.
5. **Paired / interleaved A/B on a contended box.** When wallclock swings (±10% on a shared/parallel-loaded machine), alternate OLD vs NEW back-to-back across N trials and compare PAIRED: "faster in K/N trials" + min-to-min + median-to-median, so shared-machine noise cancels and the SIGN is trustworthy. When wallclock is hopelessly noise-dominated, switch the A/B to `process_time` (CPU time).
6. **Separate-process A/B when in-process state would contaminate.** numba `cache=True` on disk, module-level caches/globals, or a warmed JIT can mask a fix or pollute the comparison — run OLD and NEW each in a fresh `python -c` / subprocess. Clear `__pycache__`/`.nbi`/`.nbc` if a numba cache hid the change.
7. **Identity gate ALONGSIDE speed — always.** Prove the output (selection / MI / metric / report) is unchanged: exact `==` where possible, or document the FP reduction-order delta (~1e-9 / single-ULP from a different summation order) AND confirm it cannot move a decision. A selection/score-altering divergence (~1e-3) is NOT acceptable — gate or reject. Pin the identity in a fail-on-regression test (verify it FAILS on pre-fix code).
8. **cProfile attribution caveat.** Compiled-kernel time (numba `@njit`, cython) is mis-attributed to the Python CALLER frame — a 4ms "tottime" Python frame can be ~0.1ms of Python wrapping a 4ms njit body. Before optimizing a flagged frame, microbench the wrapper standalone to confirm it is genuinely plain-Python/numpy, not an njit/external dispatch (and discount cProfile's ~10x deep-stack inflation).
9. **The only valid REJECT is measured + e2e + written down.** Keep the bench committed (REJECTED != DELETED) + a `# bench-attempt-rejected (date): X->Y, reason` note at the site, so the next agent re-runs the named bench instead of re-trying the dead path.

## NEVER kill a near-done background agent to "free the machine" (CRITICAL — 2026-06-20 user complaint)

A/B PAIRED timing (rule 5 above) already cancels machine load, so **concurrent agents do NOT invalidate each other's measurements** — there is NO benchmark reason to serialize-by-killing. Do not stop a running background optimization agent that is in-flight and near completion (already past its core work — emitting its validation/A/B/report/commit) in order to free CPU/GPU, reduce self-contention, or give another agent a "clean" run. Killing a near-done agent throws away its work; `git checkout`-reverting its partial edits on top is a DOUBLE loss. (Incident: killed an lstsq/prewarp-waste agent that had just reported "3 pins green, now the compound gate + A/B" — i.e. done — AND reverted its file. Lost the whole optimization for nothing.) Only stop an agent that is genuinely hung, on a confirmed wrong/superseded path, or editing a file another agent must own — and even then let it finish its report / capture partial output FIRST. Want clean wall-clock numbers? Serialize FUTURE dispatches (don't launch the next until the current returns) — never kill work already in progress. And before blaming "external / parallel-session contention", check whether the load is your OWN over-spawned concurrent agents.

## Audit hot kernels for wasted per-call work — the caller may discard part of the output (CRITICAL)

A kernel can be individually well-tuned and still waste time computing things its CALLER throws away. So a "this kernel
is already optimal" verdict is NOT the end — once a kernel is hot by tottime AND call count, inspect EVERY call site and
ask: does the caller use the kernel's FULL output, or only part of it? If it discards part, write a PRUNED fast-path
kernel that returns ONLY what that caller needs (keep the full kernel for its other callers — see "REJECTED != DELETED"),
and dispatch the discarding caller to the pruned variant. This is bit-identical BY CONSTRUCTION (you remove work, you do
not change numerics), and it pays in exact proportion to the call count — so the hottest callers are the biggest wins.

Concrete win (2026-06-08): `_dcd_metrics.pair_su` (24,270 calls, ~29% of the MRMR full fit) computed the joint
H(X_a, X_b) via `merge_vars`, which allocated a length-n `final_classes` relabel array + a lookup table and walked every
sample twice — all of which the SU path immediately discarded, needing only the pruned joint frequencies. A new
`joint_freqs_2var` njit kernel returns exactly those frequencies with none of that waste: **~23x per joint pair**
(171.9 → 7.4 µs), **+7.1% full-fit wall** (664.8s → 617.4s), selection **BIT-IDENTICAL** (650 selected / 540 engineered),
commit `22b23835`. This surfaced AFTER a whole-suite re-profile had (wrongly) declared convergence — proof that
"converged" without auditing call sites for discarded work is a premature verdict.

Method: profile to rank kernels by tottime AND call count → for each, read its call sites → microbench a pruned variant
in ISOLATION first → only if it shows a real per-call reduction AND the caller genuinely discards that work, wire it in,
run the full-fit wall + bit-identity gate, then keep+push or write down a tested "caller-uses-all-output" verdict (a
tested verdict, never an assumed one).

## Gate a big win on its safe condition (CRITICAL)

When an optimization gives a LARGE speedup but is only bit-identical
under a **detectable** condition (and diverges meaningfully otherwise),
do not reject it wholesale and do not ship it unconditionally — **gate
it**: apply the fast path exactly where it is safe, fall back to the
exact path everywhere else. A conditional win beats no win, and a big
win is worth conditional bit-identity.

- **Divergence magnitude is the deciding factor.** ~1e-9 (FP
  reduction-order under `parallel=True`, half-even ULP) is acceptable
  for a real speedup. **Selection-altering divergence (~1e-3 on an MI /
  score / gain) is NOT** — it silently changes which features get
  picked. If the fast path can move a selection decision, it must be
  gated out of that case, never shipped raw.
- **Find the exact predicate** separating safe from unsafe inputs, make
  it a cheap runtime check, and branch on it. Reference: the bootstrap
  resample-sort in `_orthogonal_bootstrap_mi_fe` — sorting the gather
  indices is a ~6.7× cache-locality win, bit-identical on continuous
  (all-distinct) columns but ~1e-3 MI-divergent on discrete columns
  (equi-frequency `argsort` binning breaks ties **positionally**). Gated
  per matrix on `_all_columns_distinct`: the wide engineered matrix
  keeps the win; discrete matrices keep the exact random-order gather.
- **Verify bit-identity on the UNSAFE case too**, not just the safe one.
  A check that exercises only continuous/no-tie data will pass while the
  discrete/tied path silently diverges. Test tied / discrete / low-card
  inputs explicitly before trusting a "bit-identical" claim.
- **Ship a regression test pinning BOTH sides**: bit-identical on the
  gated-in case AND divergent on the gated-out case, so a future "just
  always do it" cannot slip through unnoticed.
- Same failure class as the rejected njit rank-binning (MI-divergent on
  ties): any row reordering or rebinning is suspect on tied data.

## Every new ML trick gets a `biz_value` synthetic test (CRITICAL)

Every ML feature, parameter, optimizer, or basis added to the project
must come with a synthetic-data test that QUANTITATIVELY asserts the
trick's measurable win. If a future code change silently breaks the
trick -- regressed optimizer, removed parameter, basis disabled, MI
estimator broken -- the test catches it by **failing the win**, not
just by interface or shape checks.

This is non-negotiable for any new ML feature past trivial helpers.

### The pattern

1. **Find a synthetic where the trick should clearly win**
   - For an optimizer: a target where the search space has a known
     canonical optimum the trick should reach faster / better.
   - For a basis: a target whose structure matches the basis's
     natural domain (Fourier -> periodic, RBF -> bumps,
     Sigmoid -> thresholds, Pade -> ratios).
   - For a parameter / kwarg: a target that ONLY produces the
     measurable improvement when the parameter is enabled.
   - For a diagnostic (e.g. nested-CV): a target where the
     diagnostic must produce a known verdict.

2. **Pin a quantitative threshold based on the measured value**
   - Run the trick once during development. Note the measured win
     (e.g. "Fourier MI 0.67 vs Chebyshev 0.53 = 1.27x").
   - Set the assertion floor 5-15% below the measured value to
     absorb measurement noise without losing detection of regressions.
   - **Bad**: `assert res is not None`. Catches nothing if the
     basis silently outputs garbage.
   - **Bad**: `assert res.mi > 0`. Same.
   - **Good**: `assert res.mi >= 0.55, "XOR MI should be >=0.55"`.
   - **Good**: `assert speedup >= 5.0`. Captures a real perf win.

3. **Test alongside the closest baseline**
   - For an optimizer: the previous default optimizer.
   - For a basis: the same target with a different basis.
   - For a kwarg: the same call with the kwarg disabled.
   - The DELTA matters more than the absolute number.

4. **Fast: each test < 5s wall**
   - Use n=2000 max for synthetics.
   - Use n_trials=30-40 (enough for the optimizer to converge on
     the canonical target -- not a full production run).
   - Pre-warm numba in module-level fixture if needed.

### Concrete examples (all in `test_pair_fe_biz_value.py`)

```python
def test_biz_cma_es_at_least_5x_faster_than_optuna():
    """Floor 5x; measured 27x. Catches CMA-ES regression."""
    ...

def test_biz_fourier_wins_on_periodic_target():
    """Fourier mi/Chebyshev mi >= 1.20x on sin target.
    Measured 1.27x; 5% margin."""
    ...

def test_biz_multimode_beats_single_mode():
    """Multi-mode AUC >= single-mode + 0.02 on multimode target.
    Measured 0.9993 vs 0.9677 = 0.0316 delta."""
    ...

def test_biz_triplet_beats_pair_on_3way_xor():
    """Floor 10x; measured 110x.
    A real regression in triplet support drops the ratio to ~1x."""
    ...

def test_biz_honest_baselines_reject_redundant_polynomial_xor():
    """``use_trivial_baseline=True`` MUST reject Hermite poly on
    XOR (where trivial mul wins). Catches regressions in the
    honest-baseline gate."""
    ...
```

### When this rule does NOT apply

- Pure refactors that don't change algorithmic behaviour.
- Trivial helpers (sort key, formatter).
- Bug fixes for crashes (a regression test on the crash itself
  is sufficient).
- Documentation-only changes.

For everything else -- new optimizer, new basis, new diagnostic,
new kwarg that affects behaviour -- the biz_value test is required
PR gate. CI failure on a `test_biz_*` is treated as a real regression,
NOT a flaky test (margins are wide enough that noise doesn't trip them).

### Naming convention

**File**: ``test_biz_val_<подпакет>_<класс_или_семейство>.py``. One
file per CLASS (or tightly-related function family), placed under
``tests/<подпакет>/``. Files contain the full set of biz_val tests
for that class's parameters / features.

**Test function**: ``test_biz_val_<class>_<parameter>_<scenario>``.

Examples (canonical, ship with this convention):

| File | Tests inside |
|---|---|
| ``tests/feature_selection/test_biz_val_filters_hermite_fe.py`` | optimise_hermite_pair param wins (CMA-ES, plug-in MI, basis choice, multi-mode, honest baseline) -- 15 tests |
| ``tests/feature_selection/test_biz_val_filters_mrmr.py`` | MRMR param wins (interactions_max_order, quantization_nbins, min_relevance_gain, n_workers, fe_smart_polynom) -- 5 tests |
| ``tests/feature_selection/test_biz_val_wrappers_rfecv.py`` | RFECV param wins (n_features_selection_rule, stability_selection, must_include, conditional_permutation, checkpoint resume) -- 5 tests |
| ``tests/feature_selection/test_biz_val_filters_permutation.py`` | parallel_mi variants (Besag-Clifford 3-9x speedup, prange reproducibility, npermutations=0 guard) -- 5 tests |
| ``tests/feature_selection/test_biz_val_filters_gpu.py`` | mi_direct_gpu_batched (>=1.5x faster than CPU at n>=10k, scales to n=200k, OOM-safe fallback) -- 4 tests |
| ``tests/training/test_biz_val_training_composite_discovery.py`` | CompositeTargetDiscovery (screening='hybrid' tiny rerank, mi_estimator='bin' default, fail_on_no_gain, transform-selection schema) -- 6 tests |
| ``tests/training/test_biz_val_training_baseline_diagnostics.py`` | BaselineDiagnostics (ablation finds dominant feature, sample_n bounds runtime, enabled=False short-circuit) -- 5 tests |

**Why per-class, not per-parameter**: a per-parameter file pattern
(``test_biz_val_<подпакет>_<класс>_<param>.py``) was considered but
creates 50+ files for a class with 5+ tunable parameters and
inflates test-discovery time. Per-class with parameter-scoped test
functions is the right granularity:
- 1 file per class -> ~10 files total across mlframe
- 3-7 tests per file -> ~50 biz_val tests total
- ~30s wall per file -> full biz_val sweep < 5 min

**Coexistence with legacy ``test_bizvalue_*.py``**: pre-2026-05-10
files (``test_bizvalue_feature_selection.py``,
``test_bizvalue_imbalance_grid.py``,
``test_bizvalue_outliers_earlystop.py``) use the legacy single-token
``bizvalue`` naming. Keep them as-is (they have established history);
new biz_val tests go in the ``test_biz_val_*`` files following the
convention above. A future cleanup pass can rename the legacy files,
but is not required.

`test_biz_*` files live alongside the regular tests for the same
feature (`tests/<package>/test_<thing>_biz_value.py`) so they run as
part of the normal test suite.

## Numerical-kernel acceleration ladder + size-aware dispatch

For numerical hot kernels (polynomial eval, MI estimation, distance
computation, kernel matrix builds, etc.), the project follows a
**measured ladder** of backends and a **size-aware dispatcher** that
picks the right one per call. Do not assume "numpy is already
optimized" — verify with a microbench across `n` first.

### The four backends (in priority order)

1. **numpy / scipy** — the reference. Always implement this first as
   correctness baseline. Often the per-call dispatch overhead
   dominates the actual compute at n<10k, even though the C code
   itself is fast.
2. **`numba.njit` (single-thread)** — wins for n in 100..50k. Removes
   per-call dispatch overhead, inlines into surrounding njit
   functions, runs in machine code with no Python frame transitions.
   Empirically 3-6x over numpy at n=2k for polynomial Horner eval.
   `cache=True, fastmath=True` for max speed; safe when we don't need
   strict IEEE-754 (most ML code).
3. **`numba.njit(parallel=True)` + `prange`** — wins for n in 50k..500k.
   `prange` over array elements parallelizes across cores. Spawn
   overhead (~50us) makes this LOSE to single-thread njit at small n.
   Sweet spot: when each element's work is large enough to amortise
   the spawn (e.g. higher-degree polynomials, MI batches).
4. **CUDA (cupy or RawKernel)** — wins for n >= 500k once
   host->device transfer is amortised. Three flavours, ranked by
   measured throughput on this repo's reference hardware:
   - **`cp.RawKernel`** (custom CUDA C++ inline) -- **WINNER**. One
     thread per output element with state in registers. Compiled by
     NVCC at first call (~30ms), launches in ~30us thereafter.
     Worth the LOC.
   - **cupy elementwise** (drop-in `import cupy as cp` rewrite of
     numpy code): simplest, but allocates intermediate arrays per
     operation. **2-3x slower** than RawKernel for recurrence
     kernels. Use only when the work is genuinely elementwise (no
     temporaries) -- then it ties RawKernel.
   - **`numba.cuda.jit`** -- empirically the LOSER of the three on
     this hardware. Same kernel logic compiled by numba's PTX backend
     ran **6-10x slower than cupy RawKernel** in the polynomial-eval
     bench (1700us per launch baseline at all n<=100k vs ~50us for
     cupy RawKernel; numba's per-call dispatch overhead dominates).
     The Python-as-CUDA syntax is more ergonomic, but the runtime
     cost is real. **Prefer cupy RawKernel for new kernels.** Only
     reach for `numba.cuda.jit` if the kernel must be called from
     inside another `numba.cuda.jit` device function (no FFI between
     numba.cuda and cupy device functions). See bench:
     `feature_selection/_benchmarks/bench_poly_eval_backends.py`.

### CRITICAL: fastest variant must be the DEFAULT, never opt-in

When you add a GPU / numba / cupy variant of an existing function, the
**fastest applicable path must be the default** at the public API. Do
NOT ship the optimization as a public `_gpu` / `_cuda` / `_njit` name
that callers must manually wire in -- callers won't, and the win
stays unrealised forever.

The right shape:

* `do_thing(...)` is the public name and acts as the dispatcher: it
  picks `_do_thing_cpu` / `_do_thing_cuda` / `_do_thing_cupy` based on
  CUDA availability (via `pyutilz.system.gpu_dispatch.is_cuda_available`)
  + data shape + the measured crossover thresholds.
* Each backend lives under an explicit `_do_thing_<backend>` name and
  remains callable directly for tests + benches.
* The dispatcher accepts a `force_backend=` / `prefer_gpu=False` knob
  for tests + diagnostics. Defaults: dispatcher picks fastest.

This is non-negotiable per the user directive (2026-05-19): "самый
быстрый вариант должен быть установлен по дефолту". A correct example
to follow: `dispatch_batch_pair_mi(...)` in
`feature_selection/filters/batch_pair_mi_gpu.py`, with the explicit
`batch_pair_mi_njit_prange / batch_pair_mi_cuda / batch_pair_mi_cupy`
backends each callable on their own.

A WRONG example to avoid: shipping `mi_direct_gpu_batched` as a public
name for >6 months without wiring it into `mi_direct_gpu`; that's the
gap commits 033941b and ba78f04 had to close retroactively.

### Pre-implementation rule: BENCH first, dispatch second

**Always microbench all four backends across `n in {500, 2000, 10000,
100000, 1_000_000}` BEFORE writing the dispatcher.** Save the bench in
`*/feature_*/_benchmarks/bench_<kernel>_backends.py`. Output a JSON
results table to `_results/`. Only then write the dispatcher with
crossover thresholds derived from the actual measurements — never
guess from "GPU is fast".

Reference implementation: `mlframe/feature_selection/_benchmarks/bench_poly_eval_backends.py`
benches numpy / njit / njit_par / cupy / cuda_kernel for 4 polynomial
families. Output → `_results/poly_eval_backends_<basis>.json`.

### Dispatcher contract

The dispatcher is **stateless** and **per-call size-aware**:

```python
def kernel_dispatch(name: str, x: np.ndarray, *args) -> np.ndarray:
    forced = os.environ.get("MLFRAME_<KERNEL>_BACKEND", "")
    n = x.shape[0]
    if forced == "njit" or n < _PAR_THRESHOLD:
        return _NJIT_FUNCS[name](x, *args)
    if (forced == "cuda" or
            (forced == "" and n >= _CUDA_THRESHOLD and _CUDA_AVAILABLE)):
        if _CUDA_AVAILABLE:
            return _CUDA_FUNCS[name](x, *args)
    return _NJIT_PAR_FUNCS[name](x, *args)
```

Conventions:
- Threshold constants exposed as module globals AND override env vars
  (`MLFRAME_<KERNEL>_PAR_THRESHOLD`, `_CUDA_THRESHOLD`). Lets users
  retune for their hardware without code change.
- `MLFRAME_<KERNEL>_BACKEND=njit|njit_par|cuda` forces a specific
  backend (testing, A/B compare).
- Lazy CUDA init: compile RawKernels on first call, not at import.
  `_ensure_<kernel>_kernels()` pattern.
- CUDA path must auto-fallback to njit_par if cupy is unavailable
  OR if the GPU is OOM (catch `cp.cuda.runtime.CUDARuntimeError` and
  log once).

### CRITICAL: integrate with kernel_tuning_cache, do NOT hardcode

For ANY new GPU dispatcher whose backend choice / block size / kernel
variant / threshold depends on data size or hardware, the FIRST move is
to integrate with the existing `pyutilz.system.kernel_tuning_cache`
infrastructure -- the same system already powering `joint_hist_batched`
and `plugin_mi_classif_dispatch`. Hardcoded threshold constants
(`_CUDA_THRESHOLD = 1_000_000`) are wrong on any HW other than the dev
machine. 2026-05-20 incident: I hardcoded `_MI_CUDA_THRESHOLD=1_000_000`
for plug-in MI dispatch by analogy with polyeval; re-measurement on
actual hardware showed the real crossover was n=75k (single) / n=10k
(batch) -- conservative defaults left **2-4x speedups on the table**.

The KTC pipeline (`mlframe/feature_selection/_benchmarks/kernel_tuning_cache/`):

1. `auto_tune.py` -- `_run_sweep_<kernel>()` measures (n, k, block_size,
   variant) grid → returns regions; `ensure_<kernel>_tuning()` consults
   `KernelTuningCache().get_regions("<kernel>")` then runs the sweep +
   persists if missing.
2. `dispatch.py` -- `lookup_<kernel>_backend(...)` wraps the cache
   lookup; on cache miss + `run_auto_tune=True` triggers the sweep;
   provides a measurement-backed hardcoded fallback for the
   no-pyutilz / no-cuda case.
3. Cache file: `~/.pyutilz/kernel_tuning/<hw_fingerprint>.json`
   (schema-v1, cross-process safe via filelock + merge-on-write).
4. Online relearn behind `MLFRAME_KTC_ONLINE_LEARN=1` (every 1000 calls
   re-measure one region; ~50-200ms cost, gated off by default).

When adding a new GPU kernel:
- Mirror `joint_hist_batched` / `plugin_mi_classif_dispatch` end-to-end.
- Add `_run_sweep_<your_kernel>()` + `ensure_<your_kernel>_tuning()` in
  `auto_tune.py`.
- Add `lookup_<your_kernel>_backend(...)` in `dispatch.py`.
- Wire the lookup into your dispatcher (e.g.
  `plugin_mi_classif_batch_dispatch`).
- Keep the env-var force-override (`MLFRAME_<KERNEL>_BACKEND=njit|cuda`)
  as an escape hatch.

The pattern is so well-established that NOT using it for a new GPU
dispatcher is the surprising choice. Per user directive 2026-05-20:
"у нас уже есть методика оптимального подбора параметров/порогов для
cuda kernels, с хранением таблиц per device specs на диске, используй
её!".

### Hoist dispatch out of hot loops

Per-call dispatch overhead is ~4us (env-var get, size check, dict
lookup, function call). At 1000 calls per outer iteration this is
4ms — not free. **In hot loops, hoist the dispatch decision out**:
pick the right backend ONCE before the loop based on the (known,
fixed) array size, then call directly through the loop. See
`optimise_hermite_pair` in `feature_selection/filters/hermite_fe.py`
for the pattern.

### When to skip the ladder

The four-backend ladder has real maintenance cost (4 implementations,
4 cross-correctness tests, the dispatcher, the bench). Skip it
unless the kernel is ALL of:

1. Called >100 times per fit / per request
2. Profile-confirmed >5% of wall on production-shape inputs
3. Has well-defined per-element work (kernels that need cross-element
   coordination — e.g. sorts, k-NN — don't fit the prange model
   cleanly)

Pure-numpy kernels that are <1% of wall do NOT get the ladder.
A trivial helper that runs once per fit does NOT get the ladder.
Document the decision in the kernel's docstring so future maintainers
don't re-raise.

### Reference implementations to copy from

- **Polynomial eval** (4 bases x 4 backends + dispatcher):
  `feature_selection/filters/hermite_fe.py` — `_<basis>val_njit`,
  `_<basis>val_njit_parallel`, `_polyeval_cuda`, `polyeval_dispatch`.
- **Joint histogram + MI** (CPU njit + CUDA RawKernel):
  `feature_selection/filters/gpu.py` — `mi_direct_gpu_batched`,
  `_GpuBufferPool` for persistent device buffers across calls.
- **Permutation MI** (njit single + parallel):
  `feature_selection/filters/permutation.py` — `parallel_mi`,
  `parallel_mi_prange`, `parallel_mi_besag_clifford` (early-stop
  variant).

### Don't over-engineer

If your kernel is called 10 times per fit and runs in 50us, the
ladder buys you nothing. Cache the result, use numpy, move on.
The `feedback_perf_measure_first.md` rule applies first: measure,
then optimize, then dispatch.

### `fastmath=True` + Python-level NaN-gate beats selective fastmath flags

When a numba kernel needs both (a) speed from `fastmath=True` and
(b) correct NaN/inf handling for its callers, the **right pattern is
full `fastmath=True` on the kernel + an `np.isfinite(arr).all()` gate
at the Python wrapper level**, not a selective fastmath flag set.

**Why selective flags fail in practice** (verified empirically
2026-05-14 on `compute_simple_stats_numba`):

`fastmath=True` in numba enables the full LLVM fast-math flag set:
`nnan, ninf, nsz, arcp, contract, afn, reassoc`. Two of those —
`nnan` and `ninf` — let LLVM assume the input contains no NaN/inf,
which lets it elide `np.isfinite()` checks. If your kernel uses
`if np.isfinite(x):` to skip non-finite entries, `fastmath=True`
silently breaks that contract.

The intuitive fix is `fastmath={'reassoc', 'arcp', 'contract', 'afn',
'nsz'}` — everything EXCEPT `nnan`/`ninf`. **This does not work for
hot loops**: without `nnan` the compiler must preserve NaN-propagation
order across the reduction, which blocks SIMD vector reductions on
the accumulator. The "safe-fastmath" kernel ran ~14% SLOWER than a
plain `fastmath=False` kernel with Kahan compensation in our test.

**Working pattern**:

```python
@numba.njit(fastmath=True, cache=True)  # full fastmath
def _kernel_fast(arr): ...               # assumes finite-only input

@numba.njit(fastmath=False, cache=True)  # NaN-aware
def _kernel_compensated(arr): ...        # handles non-finite

def public_wrapper(arr, compensated: bool = False):
    if compensated:
        return _kernel_compensated(arr)
    # Vectorised C check, ~2us per 100k elements. Cheap insurance
    # against the LLVM nnan/ninf assumptions in the fast kernel.
    if not np.isfinite(arr).all():
        return _kernel_compensated(arr)
    return _kernel_fast(arr)
```

Result on `compute_simple_stats_numba` / `compute_moments_slope_mi`
(float64, N=50k, all-finite input — the common case): 1.28x and
1.50x speedup respectively over the Kahan path, with full NaN-safety
preserved via the gate. NaN-containing input automatically falls back
to the compensated kernel.

The `np.isfinite(arr).all()` cost is amortised over the kernel work:
~2us per 100k vs ~100-1000us inner-loop cost. For very short arrays
(N<1k) the gate is a larger fraction; in those cases the Kahan path
is probably the better default anyway.

## Accuracy / performance over legacy / compat / deps

When choosing **defaults** or making **API decisions** in mlframe,
prioritise accuracy and runtime performance over backward-compatibility,
dependency-count minimisation, or "safe" legacy behaviour. The
framework is allowed to evolve aggressively in service of better
results.

**How this applies in mlframe specifically:**

1. **Default knobs flip when a new path measurably wins.** A new
   estimator / strategy / hyperparameter that beats the current default
   on a benchmark becomes the new default. Don't gate behind a feature
   flag "for safety" -- ship it on as the default and mention the
   change in CHANGELOG. Users who relied on the old behaviour can pin
   it explicitly.
   - Example (2026-05-10, R10b): switched
     `CompositeTargetDiscoveryConfig.mi_estimator` default from
     `"knn"` -> `"bin"` because bin is bias-free under monotone
     transforms; switched `screening` from `"mi"` -> `"hybrid"`
     because Phase B catches "wrong base" cases the MI-only path
     misses. Both flips landed without feature-flag gates.

2. **Extra optional deps are fine if they're faster or more accurate.**
   `numba`, `cupy`, `torch`, `lightgbm`, `xgboost`, `catboost`, `dill`,
   `polars`, `scipy.stats.wilcoxon` -- all already pulled in and
   used; users get them via the project's `pip install mlframe[all]`
   extras. A new feature that requires a single additional optional
   dep is welcome if it provides meaningful speedup or accuracy
   improvement. (See user memory `feedback_speed_over_deps`.)

3. **Tighten test assertions to the new path's stronger guarantee.**
   When a test passes on the OLD path with a weaker bound (e.g. RMSE
   tolerance 1.10) and the NEW path delivers 1.05, raise the bound to
   match the new behaviour. Don't keep the loose bound just because
   the old path happens to pass it.

4. **Don't ship "validated only on this fixture" defaults.** When a
   benchmark shows the current default is suboptimal on the
   measured fixture, do the wider benchmark (S1-S16, multi-seed,
   real-data proxy) NOW. Don't ship the suboptimal default with a
   "todo: validate broadly" note.

5. **Hard breakages still need care.** Data-format-incompatibility
   (a saved-model loader that no longer reads v1 pickles) and
   security regressions still warrant the deliberate path with
   explicit fallback / migration. But default-knob flips on quality
   metrics are fair game.

6. **Variant defaults: most ACCURATE first, then FASTEST at equal
   accuracy (CRITICAL).** Whenever there are interchangeable variants of
   an algorithm -- importance metric (impurity / permutation / SHAP),
   MI estimator, kernel, selection rule, shadow construction, scorer,
   optimizer backend -- benchmark them across the wider test bed
   (multi-scenario, multi-seed) and set the DEFAULT to the variant that
   is most accurate on the metric that matters (honest holdout / OOF).
   Only when two variants are within statistical noise on accuracy does
   speed break the tie -> pick the faster one. The accurate-but-slow
   variant must still be the default path for inputs where it wins; use
   a size/cost-aware dispatcher (see "Numerical-kernel acceleration
   ladder") to fall back to the cheaper variant only where it is not
   measurably worse, never as a blanket default chosen for speed alone.
   Profile and try to speed up the accuracy winner (numba / cupy /
   caching / early-stop) BEFORE conceding to a faster-but-worse default.
   Single-seed "wins" do not count -- a variant must win on the MAJORITY
   of scenarios/seeds (selectors here are high-variance; one lucky seed
   has repeatedly misled). Document the benchmark numbers in the
   CHANGELOG when flipping a default.

This rule pairs with the user's general-memory entries
`feedback_accuracy_perf_over_legacy.md` and
`feedback_fastest_default_with_dispatch.md` (accuracy outranks speed;
speed is the tie-breaker, and the dispatcher routes by input).

## Every new feature: unit + biz_value tests + cProfile-driven optimization (CRITICAL)

Every fix or new feature added to mlframe MUST ship with three things,
in this order:

1. **Unit tests** covering the new code paths -- happy path, error
   path, edge cases. Don't merge code without them.
2. **biz_value test** (per the "Every new ML trick gets a biz_value
   synthetic test" rule above) -- a quantitative assertion that the
   trick measurably wins on a synthetic where it should clearly
   succeed. Threshold set 5-15% below the measured value, so
   regressions trip but reasonable seed variation doesn't.
3. **cProfile pass on the hot path** -- profile the new feature at
   a representative input shape, identify the top 3-5 hotspots, and
   optimize anything where the wall-time saving is meaningful (>5%
   of total or >10ms absolute). Document cProfile output and the
   optimizations applied (or "no actionable speedup" with reason) in
   the feature's docstring or a sibling `_benchmarks/` script.

Applies to: new functions / classes, new params on existing estimators,
new branches in hot loops, new file modules, new statistical tests,
new permutation/null variants, new encoding strategies, new optimizer
backends. Does NOT apply to: pure refactors, documentation-only
changes, trivial helper additions, type-annotation passes.

Skip clauses:

- Bug fixes: regression test on the failure mode is sufficient;
  biz_value + cProfile are not required (unless the bug was a perf
  regression, in which case profile + restore).
- Configuration-only changes (e.g. default flips): biz_value tests
  already exist for the feature; verify they pass with the new
  defaults.
- Test-infrastructure additions: skip biz_value (tests testing tests
  don't need quantitative win assertions).

Why all three: unit tests prevent silent breakage, biz_value tests
catch silent quality regressions, cProfile prevents "ships latent
overhead that only surfaces under prod shapes". Skipping any one of
the three has caused real prod regressions in mlframe history.

## Every bug fix ships with a regression unit test (CRITICAL)

A bug fix without a regression test is unfinished work. The fix-and-
move-on pattern accumulates silent failure modes that re-appear at the
worst time (mid-refactor, mid-merge, during a release crunch). A
pinned regression test is the cheapest way to guarantee the specific
failure mode never returns.

The test goes in the SAME commit as the fix, NOT in a follow-up
"add tests later" task. It must:

1. **Exercise the EXACT path the bug travelled.** Reuse the fixture /
   data shape that surfaced the bug if available — synthetic minimal
   data often doesn't reproduce the same conditions (the codepath
   may short-circuit at a guard that the real data tripped past).
2. **Fail on the pre-fix code.** Verify this empirically before
   committing: ``git stash push <fix_file>`` → run the test → expect
   FAIL with the actual pre-fix error signature → ``git stash pop``.
   If the test passes on pre-fix code, it's not a regression sensor —
   rework the test until it catches the bug.
3. **Pass on the post-fix code.** Trivial after step 2.
4. **Be narrowly scoped.** One test per bug, named after the failure
   mode (``test_fe_step_appends_nbins_via_concat_not_elementwise_add``,
   not ``test_mrmr_works``). Reviewers see the bug and the proof
   it stays fixed side by side in the diff.

Applies equally to:
- Bugs YOU introduced in the current session.
- Pre-existing bugs you encountered while doing other work. These are
  the highest-leverage tests because the bug went uncovered precisely
  because no test caught it.
- Bug surfacing during fuzz / metamorphic / regression-sensor runs —
  promote the failing combo into a named unit test so the failure
  mode is pinned independently of the fuzz pool.

Counter-pattern: fix the bug, see green on the existing test suite,
move on. The original-failing-path then drifts back into a future
regression. The 30-second cost of writing a regression sensor pays
for itself the first time someone refactors near the fixed code.

This rule pairs with the user's general-memory entry
``feedback_test_every_bug_fix.md``.

## Multi-agent review: every finding gets explicit disposition (CRITICAL)

When a plan / PR / refactor goes through multi-agent adversarial review
(critique agents covering correctness, performance, edge cases, statistical
rigor, maintenance, etc.) and the agents return N findings -- whether
that's 14, 58, 122, or any other count -- the integration step MUST
address EVERY single finding with an explicit disposition row, NOT just
top-N / "highest-leverage" / P0-P1 subset.

**Why:** silently filtering to a subset is the common failure mode that
causes 6-month-later bug reports and reviewer rework. Lower-severity
findings (doc-rot, edge-case nulls, niche statistical caveats,
pathological inputs) are exactly what cumulative trust hinges on.

**The disposition buckets:**

- **RESOLVED** — fix in this plan / PR.
- **FUTURE** — explicit out-of-scope with a reason; tracked for next
  iteration.
- **DOC** — caveat in docs / docstring / README / CHANGELOG, no
  architectural change.
- **REJECTED** — anti-recommendation with a stated reason (e.g. "hash
  encoding destroys MI by collisions; rejected because plan's whole
  premise is integer factorization").

NEVER use "ignored", "low priority", "deferred without ticket", or any
silent omission. Every ID gets a row.

**Mechanics:**

- Produce a single audit table mapping every finding ID to its
  disposition. Re-emit the full table on every review round (round 1:
  N findings; round 2: N+M; round 3: ...) so the running total is
  always visible.
- Surface deduplication explicitly when the same finding appears from
  two angles (e.g. "I2 ≡ S3 — wrong-null permutation issue, raised by
  both IT-rigor and statistical-adversarial agents"). Don't silently
  drop the duplicate; mark it.
- Every "found by adversarial review" mlframe PR description includes
  the cumulative finding count and disposition rollup
  (`X RESOLVED / Y DOC / Z FUTURE / W REJECTED`).

This rule mirrors the user-level memory `feedback_use_all_agent_findings`
and pairs with the existing `feedback_show_all_agent_findings`,
`feedback_no_premature_closure`, `feedback_summarize_before_fixing`
rules. CI failure on a missing disposition table for a multi-agent-
reviewed PR is a real regression, not a process nit.

## Open work items

(Nothing tracked here currently. Polars support for MRMR — both the
selector core and feature engineering — landed 2026-04-22. See tests:
`tests/training/test_mrmr_polars_fe.py`,
`tests/training/test_bizvalue_feature_selection.py::test_mrmr_drops_uninformative_features_on_polars_input`,
regression sensors in `tests/training/test_fuzz_regression_sensors.py`.)

## Comment line length: up to 160 chars (CRITICAL — repeated user complaint)

Project line-length is **160** for comments and docstrings. Do NOT hard-wrap at 72 or 80 columns. The user has complained about this twice (2026-04-20 and 2026-05-11) — the rule needs to be applied without prompting.

**Why:** short hard-wrapped comments fragment multi-sentence explanations across many lines, bloat diffs, and make grep noisier. The editor / terminal handles 160-char lines fine.

**How to apply:** when writing or editing comments / docstrings in .py / .sql / .md / .ipynb code cells, let each sentence / logical clause flow on ONE line up to ~160 chars. Only break at natural paragraph or clause boundaries, never to meet a narrow width. Long URLs and code examples inside comments may still be broken if they'd overflow visible width.

**Anti-pattern to AVOID** (each sentence wrapped to 50-80 chars):
```python
# Annotate composite-target reports as T-scale.
# Composite targets carry ``MTRESID=`` in the model_name (stamped
# by ``select_target``); this indicates the printed metrics are
# on the RESIDUAL scale, not the raw y-scale.
```

**Correct shape** (one sentence per line, up to 160 chars):
```python
# Annotate composite-target reports as T-scale. Composite targets carry ``MTRESID=`` in the model_name (stamped by ``select_target``); this indicates the printed metrics are on the RESIDUAL scale, not the raw y-scale.
```

If a sentence really exceeds 160, break at a clause boundary (after a comma, semicolon, or "-") - never mid-clause.

## Comments: no audit / phase / refactor junk; default to minimalism (CRITICAL)

Two rules.

**1. NO process / refactor / audit metadata inside source comments.** That information belongs in git history, the PR description, or the issue tracker - never in `*.py` files. Banned in code comments:

- Phase / wave markers: `Phase 3 / 4 / 5 / 7 / 8 / 9`, `wave 2`, `batch 4 followup`, `N3 / N5 / N6`
- Audit-finding IDs: `P0-A1..G34`, `P1-B7/B14`, `F5..F41`, `PR-4 / 5 / 6`, `Perf #5`, `C2 (2026-05-11):` and similar
- Date stamps that record when this was added or changed: `2026-04-21 fix 9.8`, `2026-04-28 batch 4 followup`, `TODO(2026-04-28)`
- Fuzz seed references: `seed=42 fuzz`, `c0093 / c0095`, `c0102 / c0147`
- Refactor-history rationale: "was 4 star imports, now explicit", "split out of the prior monolithic X.py", "Phase 3 module split", "replaces the prior monolithic wrappers.py (1936 lines)", "(kept for downstream callers)" beside `# noqa: F401` (keep the `# noqa` itself, drop the prose)
- Banner-comment section separators (lines of `# -------- HEADER --------`); one blank line between sections is enough

**2. Default to MINIMALIST comments.** Write a comment only when the WHY is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific upstream bug, behavior that would surprise a reader. Do not explain WHAT the code does - well-named identifiers already say that. Do not narrate process ("now we loop over X", "next we filter Y"). Do not write AI-justifying parentheticals like `(natural Python idiom)`, `(elegant)`, `(idiomatic)`, `(obvious choice)`. If removing the comment would not confuse a future reader, do not write it.

**Why:** noise comments accelerate code rot - they go stale, drift out of sync with the code, and bury the few genuine WHY notes in clutter. The 2026-05-14 cleanup pass on `mlframe/feature_selection/wrappers/*.py` removed ~580 lines (16% of the package) of phase / audit / date / seed junk; this rule exists so the next pass does not have to re-do that work.

**How to apply:** before writing a comment, ask "would a reader who has never seen this PR / issue care about this line a year from now?" If no, delete. Anything about a specific fix, ticket, or refactor wave goes in the commit message or PR description, not the source. If a TODO is genuinely needed, write `# TODO: <what>` with no date and no audit ID; the issue tracker handles that.

## Monolith split: AST-audit sibling for unresolved names BEFORE commit (CRITICAL)

When splitting a >1k-line file by moving a function / class body into a new sibling module, Python compiles the moved code fine even if its body references names that live in the original parent module -- name lookup is lazy (resolves at call time, not at module load). So a sibling that's missing `from .parent import _helper` looks healthy at import time and only blows up at the FIRST runtime call, often via a downstream broad `except` that swallows the `NameError` into a recurring WARN ("name '_helper' is not defined; fold reported as NaN" -- the spam pattern that surfaced after waves 92-107).

**Flat sibling vs subpackage (organization):** when a monolith carves into a SINGLE sibling, keep it flat (`name.py` + `name_helpers.py` next to it — a 2-file pair does not need a package). When a monolith fans out into **>=2 siblings**, prefer a **subpackage**: convert `name.py` into `name/__init__.py` (the former parent, now re-exporting the public surface) plus cohesive submodules inside `name/`. This keeps already-crowded directories navigable and groups related code. It is fully backward-compatible — `from ...pkg.name import X` still resolves to the package `__init__`, so external importers are untouched. The same AST gate below applies to every submodule; internal cross-imports between submodules use `.sibling`, and submodules importing the former-parent surface use `from . import X` (resolved from `__init__`). Do NOT convert to a subpackage for a single-sibling carve.

**Required gate before committing any sibling-file split:**

1. Run an AST scope analyser over each new sibling. For every `ast.Name` with `ctx=Load`, check whether the name is bound at the sibling's module scope OR is a builtin OR is a function-local. Anything else is a candidate runtime NameError. The walker MUST chain enclosing function scopes correctly (closure refs are not bugs) and respect deferred annotations (`from __future__ import annotations` makes annotation-only refs strings).
2. For each candidate, grep the parent module for the symbol's home. Add an explicit `from .parent import <name>` at the sibling's module top (or a lazy `from .parent import <name>` inside the function body when the parent is a heavy / cycle-prone dep).
3. Smoke-import every affected sibling AND `hasattr(sibling, '<name>')` on every added symbol after the fix. A clean module import is NOT proof the runtime references resolve.
4. The sensor for each split MUST exercise at least one code path that calls into the moved body, not just `import` the symbol -- import-only sensors pass even when the body has unresolved refs.

`grep` alone is not enough: it misses indirect refs through aliases (`from X import foo as _foo; ... _foo(...)`) and overflags closure variables. Use the AST walker.

This rule is enforced retroactively too: any time a WARN of the form "name 'X' is not defined" appears in mlframe logs, treat it as a P0 sibling-split regression and audit every other sibling for the same class of bug.

## NEVER use destructive git to inspect baseline (CRITICAL)

The user runs **parallel agent sessions** on this repo and frequent in-flight processes (pytest, dev servers, build agents) that read the working tree. Destructive git commands - even with intent to undo - clobber concurrent work.

**Banned when used "just to peek at prior state":** `git stash` (even followed by `git stash pop`), `git checkout -- <path>`, `git checkout <ref> -- <path>`, `git reset --hard`, `git restore`, `git worktree remove`. Even a 5-second window where the working tree is mutated is enough to break another session.

**Use these read-only alternatives instead:**

- `git show <ref>:<path>` - dump file content at a ref to stdout
- `git diff <ref> -- <path>` - diff vs ref without touching files
- `git log -p <path>` / `git log -S '<string>' -- <path>` - history with diffs / pickaxe
- `git worktree add <tmpdir> <ref>` - inspect another commit in an isolated worktree without touching the primary one

**And: don't bother inspecting "pre-existing vs introduced" at all.** It does not matter who originally wrote the bad code. If a linter / type checker / test surfaces an issue, the question is "do we fix this", not "did we introduce this". Just fix what is there.

Sibling rule for processes: never kill processes without explicit user authorisation (the user has separately reinforced that one).

## Test pollution: never rebind module objects without snapshot/restore (CRITICAL)

`del sys.modules['mlframe.X']` + re-import, or `importlib.reload(mlframe.X)`, replaces the module object that `sys.modules['mlframe.X']` points at. Every other test file that did `from mlframe.X import Y` at file-load time keeps a reference to the OLD `Y`; every lazy `from .X import Y` inside a function body (used in `_mrmr_fit_impl.py`, `_predict_main.py`, etc.) resolves to the NEW `Y` on the next call. The two ends of the codebase now disagree on what `Y` is.

The damage:
- Class-attribute caches (`MRMR._FIT_CACHE`, registries, locks, monkey-patched setters) duplicate: writes land on NEW, asserts read OLD, and `len(...)` looks empty even though the fit clearly populated something.
- `isinstance(obj, Y)` returns `False` when `obj` was constructed via OLD and the check uses NEW (or vice versa).
- Idempotent install markers (`_mlframe_feature_names_setter_installed`) silently re-apply, double-wrapping the LGBM `feature_names_in_` shim.

The 2026-05-22 trace identified this as the canonical cross-test pollution pattern. `test_biz_val_filters_mrmr.py::test_biz_val_mrmr_fe_max_polynom_*` did `del sys.modules['mlframe.feature_selection.filters.mrmr']` + `importlib.reload` to dodge a long-gone stale-`.pyc` race, and every later cache-dependent test (`TestMRMRFitCache::*`, `test_mrmr_fit_cache_shared_across_instances`, `test_fit_cache_clear_resets_state`, `test_mrmr_cache_does_not_collide_on_distinct_targets_with_shared_samples`) saw empty caches as a result.

**Rules for tests:**

1. **Do not call `del sys.modules[...]`, `sys.modules.pop(...)`, or `importlib.reload(...)` on an mlframe module unless you genuinely need it.** Most cases that reach for these are doing it to "force a fresh import", which is a smell — the test should use `monkeypatch` for env vars / attributes, or pass parameters directly.

2. **If you absolutely must mutate `sys.modules`, snapshot and restore.** Use an autouse fixture scoped to the single test class / file:

   ```python
   @pytest.fixture(autouse=True)
   def _restore_mlframe_sysmodules(request):
       """Snapshot every mlframe.* entry that this test mutates; restore at
       teardown so module-level imports in other test files keep pointing
       at the same class objects."""
       prefixes = ("mlframe.feature_selection.filters", "mlframe.training.core._setup_helpers")
       snapshot = {n: m for n, m in sys.modules.items() if any(n == p or n.startswith(p + ".") for p in prefixes)}
       yield
       for n in list(sys.modules):
           if any(n == p or n.startswith(p + ".") for p in prefixes):
               if n in snapshot:
                   sys.modules[n] = snapshot[n]
               else:
                   del sys.modules[n]
   ```

3. **`importlib.reload(mod)` is less destructive than `del + re-import`** (the module object stays the same, only its top-level symbols get rebound), but it still resets module-level singletons (caches, registries, locks). If those singletons matter cross-test, use the snapshot+restore pattern too — or snapshot the relevant attribute and reassign it after reload.

4. **For true isolation, use a subprocess.** `subprocess.run([sys.executable, "-c", "..."])` is heavy (~1-3s startup) but it is the only way to guarantee no cross-test state leaks.

5. **Class identity assertions are a tripwire.** If you have a test that does `isinstance(x, SomeClass)` and it intermittently fails, suspect a reload polluter upstream. The class identity check is the cheapest detector.

This rule is enforced retroactively: any test added or modified that touches `sys.modules` or `importlib.reload` without a restore mechanism is a regression. CI / review should reject it.


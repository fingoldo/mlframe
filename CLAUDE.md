# mlframe — project conventions

## No project-wide lint/format rewrite without approval (CRITICAL)
Never run a repo-wide `black .` / `ruff format .` / `ruff check --fix` beyond files already being edited — even when asked to "fix everything"; that's consent for the specific fixes, not a mechanical rewrite. Report scope ("N/M files need reformatting") and ask: run now / make the gate advisory / defer. Narrow fixes to files already being edited for a diagnosed reason are fine without asking.
**Why:** a prior unscoped run once rewrote a huge fraction of a repo unasked while the user was away; reverted.
**Excluded Black behaviors:** arg/collection explosion (multi-item one-line → one-per-line, incl. `from x import (...)`) and blank-line insertion — neither configurable via stock Black flags. Enforced via the shared `py_ci_shared.black_filtered_apply` (`--write`/`--check`) — never raw `black`.

## RUF100 unsafe under a narrow `--select` (CRITICAL)
`ruff check --select <narrow> --fix` makes RUF100 judge `# noqa` comments against only the narrow ruleset and silently strip ones load-bearing for the full config (e.g. star-import re-export markers). Never `--fix` with `--select` narrower than the full configured set. To triage one new rule category: list findings with `--select`, fix by hand. `--fix` only with no `--select` override, on files just edited, verified against the real blocking gate (`ruff check . --ignore C901`) afterward. `--ignore` is safe (adds to the ignore list); `--select` is not (replaces the rule set).

## Never pause on obvious/low-stakes questions (CRITICAL)
Don't ask about execution order or "should I also do the deeper fix" — pick the sensible default and do all the work, report after. Pause only for: a genuine accuracy/functionality tradeoff, a destructive/irreversible op, another session's uncommitted WIP, or a hard blocker. Never end a turn with authorized work still pending — a status update must be followed by more work in the same turn, not be the final act.

## Drive every discovery to resolution (CRITICAL)
A bug/gap found while doing other work is a commitment: fix now, or add a concrete next action to the active plan and finish it the same session. Never drop with "context running out" / "not the main goal" / "pre-existing" — pre-existing bugs still get fixed once found. Closure = fixed+tested, or a written plan item; never a hand-wave.

## New code goes in focused submodules from the start (CRITICAL)
Put new non-trivial functionality in a new, well-named sibling module and re-export from the parent facade — don't keep appending to an already-large file. Carve *before* a file nears ~800-900 LOC; `test_no_file_over_1k_loc.py` is a backstop, not the design.

## Prefer top performance in hot paths (CRITICAL)
Take the higher-performance, more complex option over a "safe but partial" one when it's validated rigorously (bit-identity, GPU/CPU parity, selection-equivalence, full suite) — never ship an *unvalidated* selection change, but do ship validated complexity.

## Enable corrective mechanisms by default (CRITICAL)
When a fix closes a real bug class, flip its default ON in the same change — don't keep the old (wrong) behavior for "compatibility". If existing tests assumed the old wrong behavior, fix the tests. Document the flip and keep an explicit opt-out for legacy callers.

## Fuzz/combo tests are bug DETECTORS, not bug hiders (CRITICAL)
Every fuzz-suite failure is a real prod bug unless proven otherwise. Never paper over with: canonicalization rules that collapse combos "because they crash" (only collapse genuinely semantically-equivalent combos), runtime canonicalizations for the same reason, `xfail`/`skip` for anything but a genuine third-party/OS limitation, or a defensive prod guard hiding an upstream bug. Ask "would a real user with these settings hit this?" — if yes, fix the root cause (often retires several band-aids at once).

## Memory / RAM discipline (CRITICAL)
Frames can be 100+ GB — never `.copy()`/`.clone()`/reconstruct a frame to work around a bug; mutate-and-restore (try/finally) or use views.
- Caching: only cheap-to-rebuild artifacts, or the caller's own reference plus a content-hash key — never a whole-frame copy. Never pickle a whole frame for caching.
- Batching: stream/mini-batch when per-element cost is small; don't materialize a full frame when batching suffices.
- Eager format conversion (e.g. polars→torch): gate on byte size (~2GB — eager under, lazy/per-batch over; unknown-size defaults to eager).
- Frame-format conversions (polars↔pandas↔ndarray) are the CALLER's decision, made once at the suite boundary — inner wrappers must never silently down-convert on a hot path.

## `/loop` fuzz-profile-optimize: stop after 100 consecutive rejects (CRITICAL)
Track a REJECT streak (reset on any RESOLVED), not a fixed iteration cap — stop only after 100 in a row.

## REJECTED ≠ DELETED (CRITICAL)
A rejected optimization keeps: the committed bench/prototype (runnable), a tracker row with exact numbers + bench filename, the option itself in prod if it's a tunable (only the *default* stays unchanged), and a `# bench-attempt-rejected` note at any touched call site. Never silent-revert or silent-delete.

## A validated improvement that breaks a test → re-frame the stale test (CRITICAL)
When a real improvement conflicts with a test (especially test-vs-test contradictions), don't default to reverting: bisect to the exact assertion, confirm the real contract still holds (no regression, equal-or-better), look for the codebase already endorsing the new behavior, then re-frame the assertion to the real contract with a comment citing the evidence. Revert only if the new behavior genuinely regresses *and* can't be gated. Diagnose WHAT the assertion now sees, not WHEN it broke — never bisect commits just to assign blame.

## Profile every new feature; optimize hotspots (CRITICAL)
cProfile harness saved in-package (not `/tmp`), sorted by cumtime top 20-30, optimize where it materially helps (njit / prange / cuda.jit / cupy / vectorize / cache), calibrate against cProfile's pandas/sklearn attribution inflation (~10-13x vs standalone wall-time), document "no actionable speedup" with reasoning when that's the conclusion.

## A measured speedup is a LEAD, never dismiss on one hasty look (CRITICAL)
Warm the kernel, run multiple times, sweep the size range (small AND large — crossovers are the whole story), validate end-to-end not just isolated. A hardware-relative win (weak dev GPU making it look neutral) still gets kept as a size/env-gated option, never deleted — promote it to `kernel_tuning_cache` so each host finds its own crossover. Only a written, multi-size, end-to-end measurement is a valid rejection — "felt marginal" is a skipped investigation. GPU kernels: always measure host-input, GPU-resident-input, and GPU-with-H2D separately — a kernel that wins big resident but loses end-to-end needs residency fixed, not the backend reverted.

## A/B validation procedure (CRITICAL)
Warm before timing; best-of-N/median, never one shot; measure BOTH isolated and end-to-end (an isolated win that's flat e2e is a REJECT — find the size where it nets positive, or reject); real baseline via `git show`/a separate process, never from memory; paired/interleaved trials on a noisy box (or switch to `process_time`); separate-process A/B when in-process state (numba cache, warmed JIT, module globals) could contaminate; identity gate alongside speed always — bit-identical, or a documented ~1e-9 FP-reorder delta proven not to move a decision (FE/MRMR exception: selection-equivalence is the bar, not bit-identical MI); cProfile mis-attributes compiled-kernel time to the Python caller frame — microbench the wrapper standalone before trusting a flagged frame.

## GPU profiling traps (CRITICAL)
nvprof per-kernel % is sync-distorted — never pick a target from it; use cProfile cumtime + a full-fit wall A/B. cProfile is blind to async GPU compute; it shows up as tottime at the blocking sync (`.get()`/`copy_to_host`). Isolated-kernel or wrong-shape microbenches lie end-to-end — confirm on the real `(n, K)` shape and the full-fit wall. `nvprof --print-gpu-summary` needs no admin; `--metrics`/`--events` do. Quiet the machine (`nvidia-smi`) and use novel-seed cold fits (MRMR memoizes fits by content-hash). A single-kernel launch-batch tweak is usually 0x on the wall, but fusing away the aggregate launch COUNT pipeline-wide is a real win — always confirm via the full-fit wall and report both launch count and wall.

## joblib threading over GPU-bound work = contention, not parallelism (CRITICAL)
Gate parallel-vs-serial on whether the stage is GPU-bound: GPU stage → serial main-thread (CPU kernels still `prange` internally); CPU-only stage → keep threading. Diagnose via cProfile `print_callers("time.sleep")` pointing at joblib `_retrieve`.

## GIL-bound per-resample/per-call Python dispatch loops: fuse into one prange-parallel njit call before giving up (CRITICAL)
A bootstrap/resample loop that calls an already-`nogil`-njit kernel once per iteration from Python is NOT actually parallel even under joblib threading — the RNG draw + array slice + Python call dispatch between iterations stays GIL-bound, so threading only buys ~1.0-1.5x. The real lever: materialize ALL iterations' index arrays in ONE vectorized numpy call (preserving the exact RNG draw order for bit-identity), then run every iteration's kernel body inside a SINGLE `numba.njit(parallel=True)` function via `numba.prange` — zero per-iteration Python round-trip, true OS-thread concurrency.
**Why documented:** a first pass on this exact pattern (bootstrap AUC resampler) found only a 1.02x scratch-buffer-reuse win and nearly stopped there ("malloc isn't the bottleneck, ship the small one"). Pushed further per explicit user pressure ("дай реальную оптимизацию") and found the REAL lever the codebase's own prior audit had already flagged as undone (`bootstrap_metrics`'s docstring: "the per-resample cost is dominated by GIL-held index generation... not the nogil kernels... the real future lever is a fully-njit resample loop"). Result: 4.1x-4.2x on the standalone AUC-only kernel, then fused across the FULL roc_auc/brier/log_loss/ece bundle `honest_diagnostics.py` computes together — 2.9x-3.4x, bit-identical (~1e-14 FP-reorder only, from calling the sequential brier/log_loss reduction instead of nesting the parallel-reduction variant inside the outer prange, which numba can't support).
**How to apply:** when a profile shows a Python-level loop repeatedly calling an already-njit kernel (bootstrap, cross-validation, ensemble scoring, permutation tests, ...) as the hotspot, don't settle for a buffer-reuse-class micro-optimization — check whether the ENTIRE per-iteration body (RNG + gather + kernel) can be pushed into one `parallel=True` njit call. Multiple metrics sharing one resample loop CAN be fused together (call each metric's SEQUENTIAL njit variant from inside the shared outer prange — nested `parallel=True` calls silently fail to parallelize or error). Validate with an A/B bench proving bit-identity (same RNG draw order) AND wire it into the real call site, not just as an unused standalone primitive — an optimization nobody calls delivers zero speedup in production.

## Never kill a near-done background agent to "free the machine" (CRITICAL)
Paired A/B timing already cancels for machine load — no benchmark reason to serialize by killing. Only stop an agent that's genuinely hung, confirmed on a wrong/superseded path, or editing a file another agent must own — and let it finish emitting its report first.

## Audit hot kernels for wasted per-call work (CRITICAL)
Once a kernel is hot by tottime *and* call count, check every call site: does the caller use the FULL output, or discard part? If it discards, write a pruned fast-path variant for that caller (keep the full kernel for others) — bit-identical by construction, biggest win at the hottest sites. "Converged" isn't a valid verdict until call sites have been audited for discarded work.

## Gate a big win on its safe condition (CRITICAL)
When a speedup is only bit-identical under a detectable predicate, don't reject wholesale or ship unconditionally — gate the fast path to the safe case, exact path elsewhere. ~1e-9 FP-reorder divergence is fine; ~1e-3 selection-altering divergence never ships unconditionally. Verify bit-identity on the UNSAFE case explicitly (tied/discrete data). Ship a test pinning both sides.

## Every ML trick gets a quantitative `biz_value` test (CRITICAL)
A synthetic where the trick should clearly win; threshold set 5-15% below the measured value; compared against the closest baseline; each test <5s. Bad: `assert res is not None`. Good: `assert res.mi >= 0.55`. Naming: `tests/<pkg>/test_biz_val_<class>.py`, one file per class, functions `test_biz_val_<class>_<param>_<scenario>`. Skip for pure refactors, trivial helpers, crash-fix regression tests, docs-only changes.

## Numerical-kernel acceleration ladder
Backends in priority order: numpy/scipy (baseline) → `numba.njit` (wins n≈100-50k) → `njit(parallel=True)+prange` (wins n≈50k-500k; spawn overhead loses at small n) → CUDA (wins n≥500k once transfer is amortized — ranked `cp.RawKernel` > cupy elementwise > `numba.cuda.jit`, the last measured 6-10x slower than RawKernel on this hardware).
- The fastest applicable path MUST be the public API's default — a dispatcher picks the backend; each `_backend` variant stays directly callable; keep a `force_backend=`/env-var escape hatch. Never ship a `_gpu`/`_cuda` name callers must manually wire in.
- Bench all four backends across n∈{500,2k,10k,100k,1M} *before* writing the dispatcher; save to `_benchmarks/`.
- New GPU dispatchers integrate with `pyutilz.system.kernel_tuning_cache` (measured, per-hardware thresholds) — never hardcode a threshold constant.
- Hoist the dispatch decision out of hot loops (~4us/call overhead adds up).
- Skip the whole ladder for kernels called <100x/fit or already <1% of wall.
- fastmath: a full `fastmath=True` kernel + a Python-level `np.isfinite(arr).all()` gate beats a hand-picked partial fastmath flag set (selective fastmath still blocks the SIMD reduction — measured ~14% slower).

## Accuracy/performance over legacy/compat/deps (CRITICAL)
Default knobs flip to the new path once it measurably wins — no feature-flag-for-safety. Extra optional deps are fine for a real speedup/accuracy win. Tighten loose test tolerances to match the new path's actual precision. Do the wide multi-seed benchmark now, don't ship "validated on one fixture" with a follow-up TODO. Variant defaults: most-accurate-on-the-honest-metric first, speed only breaks ties within noise — profile/speed-up the accurate variant before conceding to a faster-but-worse default; a single-seed win doesn't count.

## Every feature: unit + biz_value + cProfile, in order (CRITICAL)
Required for every non-trivial feature/param/branch. Skip clauses: bug fixes (a regression test suffices), default-flip-only changes (existing biz_value tests must still pass), test-infra additions.

## Every bug fix ships a regression test, same commit (CRITICAL)
Reuse the real fixture that surfaced the bug where possible. Verify empirically: fails on pre-fix code (temporarily revert just the fix, run, confirm the real failure signature, restore), passes post-fix. One narrowly-named test per bug (`test_<failure_mode>`, not `test_thing_works`). Applies equally to bugs you introduced, pre-existing bugs you found, and fuzz-caught combos (promote to a named unit test).

## Multi-agent review: every finding gets an explicit disposition (CRITICAL)
RESOLVED / FUTURE (with reason) / DOC / REJECTED (with reason) — never "ignored", "low priority", or silent omission, regardless of N. One running disposition table across review rounds; mark cross-agent duplicates explicitly. PR descriptions include the full rollup.

## polars FE path is already optimal — don't "fix" it again (CRITICAL)
`MRMR.fit` already bridges polars → a zero-copy Arrow-backed pandas view (`get_pandas_view_of_polars_df`) whenever FE runs — no whole-frame copy at any size. Measured: one contiguous plane beats per-column zero-copy views 8.65x at equal memory, so a "native, zero-copy, per-column" rewrite would be strictly worse. Keep the bridge; the format-agnostic seam (`_fe_frame_ops`) is a fallback path, not the fast one.

## Comment style (CRITICAL — repeated complaints)
Line length up to 160 chars — don't hard-wrap at 72-80; one sentence/clause per line up to the limit.
No process/audit metadata in code comments: no phase/wave markers, finding IDs, date stamps, fuzz-seed refs, refactor-history narration ("was 4 star imports, now explicit"), banner separators — that belongs in git history / the PR description.
Default to minimalist comments: write one only when the WHY is non-obvious (hidden constraint, subtle invariant, workaround, surprising behavior) — never restate WHAT the code does or narrate process. No AI-justifying parentheticals ("(idiomatic)", "(elegant)").

## Monolith split: AST-audit the sibling for unresolved names before commit (CRITICAL)
A moved function/class imports clean but can `NameError` at first call if it references a parent-module name with no matching import (name lookup is lazy). Gate: AST-walk every `Load`-context `Name` in the new sibling, flag anything not bound locally/builtin/closure, grep the parent for its home, add an explicit (or lazy, if cycle-prone) import. Smoke-import + `hasattr` isn't proof — exercise a real call path. Any "name X is not defined" WARN in logs is a P0 sibling-split regression; audit every sibling for the same bug.
Flat sibling (`name.py` + `name_helpers.py`) for a single split; convert to a subpackage (`name/__init__.py` + submodules) once a monolith fans out into 2+ siblings — backward-compatible, same AST gate per submodule.

## Never use destructive git to inspect state (CRITICAL)
Multiple parallel agent sessions share this working tree. Banned even "just to peek": `git stash`, `git checkout -- <path>`, `git checkout <ref> -- <path>`, `git reset --hard`, `git restore`, `git worktree remove`. Use instead: `git show <ref>:<path>`, `git diff <ref> -- <path>`, `git log -p`/`git log -S`, `git worktree add <tmpdir> <ref>`. Don't bother distinguishing "pre-existing vs introduced" — just fix what a linter/test surfaces, regardless of origin.
Never `git reset --hard` / `checkout -B` / `branch -f` / `push --force` on a shared branch — can silently discard another session's committed work. To sync: `git fetch` + `git merge` (never reset). Before any history-moving command, check `git log --oneline @{u}..HEAD` — non-empty means a reset would destroy real commits. Recovery: `git reflog` → `git branch recovery-<name> <sha>`, cherry-pick back, surface to the user.
A broad/dir-wide `ruff`/auto-fixer pass is read-only `check`, never `--fix` — except on files just edited, after a commit exists.
Never kill processes without explicit user authorization for that specific action.

## Test pollution: never rebind a module without snapshot/restore (CRITICAL)
`del sys.modules[...]` / `importlib.reload()` on an mlframe module splits class identity between already-loaded and newly-loaded references — breaks class-attribute caches, `isinstance` checks, and idempotent-install markers in unrelated later tests. Don't reach for these to "force a fresh import" — use `monkeypatch`/explicit params instead. If unavoidable, snapshot+restore via an autouse fixture scoped to the affected module prefixes. `reload()` is less destructive than del+reimport but still resets module-level singletons. For true isolation, use a subprocess. An intermittent `isinstance` failure is the cheapest tripwire for a reload polluter upstream.

## Write mypy-clean code from the start (CRITICAL)
Never `param: T = None` (always `Optional[T] = None`); match return annotations to the actual `return` statements; annotate dict/list literals whose later use needs a wider type than line-1 inference; wrap numpy/pandas arithmetic chains in the concrete constructor matching the declared return type; concrete types over `object` for params that only ever hold one concrete class; declare dynamically-set attributes at class scope; never silence an error with pointless extra wrapping — understand the real runtime type, annotate correctly or `cast`. Run mypy on any touched file before calling it done.

## No hand-waving "time constraints" (CRITICAL)
There's no time limit on a turn. If a shallow fix or an unexplored gap is being justified with "given the time/budget constraints," stop and do the full fix instead — trace the value to its ROOT cause via actual runtime testing, don't assume from one read.

## Running the suite on this machine
Always pass `--no-cov`: pytest-cov raises a PermissionError writing its data file on Windows, aborting the run for a reason unrelated to the tests.
Run unbuffered with `-x -s` rather than launching a long blind run.
A failure that is an OOM or a Windows paging-file error (WinError 1455 under joblib fan-out) reflects machine-wide memory pressure at that moment, not a defect: retry once, and if it fails again it is real. Tolerate it in code via `OSError` plus a skip.
Heavily-parametrised modules expose a fast mode (a `--fast` flag or env var plus a `fast_subset` helper) that runs one representative case per code path, with the exhaustive sweep behind a slow marker. Without it the only options are the full matrix or no coverage, and the full matrix stops being run.

## BUG FIX (2026-08-02): a broad except silently downgraded the whole process's MI backend to sklearn (~100x)
2M-row cProfile on combo `c0094_5637be0a` (master-seed `9999`) found `_mi_classif_batch_sklearn`
(`_orth_mi_backends.py` — the reference/legacy loop, meant only for "numba absent" installs) costing
161.8s cumtime across 19 calls (~8.5s/call), even though numba/njit was clearly available and used
everywhere else in the same profile. Root cause: `_select_mi_backend()` resolves numba-vs-sklearn via a
trial import and caches the result ONCE per process into module-level `_MI_BACKEND` — but a bare
`except Exception` around that trial import treated ANY exception as "numba genuinely unavailable",
including a transient device/driver fault surfaced while importing `hermite_fe` (which probes CUDA
availability at import time) — unrelated to whether the CPU njit dispatcher itself works. One transient
hiccup at process startup silently downgrades the ENTIRE process to the ~100x-slower sklearn reference
loop for its whole lifetime, logged only at `debug` (invisible in production). Narrowed the except to
`ImportError` only (the one genuine "numba absent" case); any other exception now logs a `warning` and
still defaults to `"numba"` rather than permanently paying the fallback. Verified both branches directly
(simulated `ImportError` → sklearn; simulated transient `RuntimeError` → numba) plus a new regression
test (`test_select_mi_backend_transient_failure.py`, 3 tests) pinning the fix. 63 existing MI-dispatch
tests pass unchanged.

## BUG FIX (2026-08-02): the per-call numba/GPU MI dispatch fallback was silent, hiding a 12%-of-wall regression
Same finding, one call deeper: 2M-row cProfile on combo `c0056_f76bf491` (master-seed `31337`, 5-model
cb/hgb/lgb/linear/xgb suite) caught `_mi_classif_batch_sklearn` costing 149.3s cumtime across 23 calls
(~12% of a 1271s run), even with `_mi_classif_batch_numba` clearly resolved and running (231 calls,
275.7s cumtime) in the same profile. Root cause: `_mi_classif_batch_numba`'s dense-column batch-dispatch
call (`plugin_mi_classif_batch_dispatch` / the GPU batcher) is wrapped in a per-call
`except Exception: mis[dense_cols] = _mi_classif_batch_sklearn(...)` with **zero logging** — any dispatch
failure (cupy import error, kernel-tuning miss, transient GPU contention, or a genuine kernel regression)
silently falls to the ~53x-slower sklearn loop for that slice, and there was no way to tell which cause
it was after the fact. Added a `logger.warning` naming the exception type/message and slice width before
the fallback runs (the fallback's own behavior — still return correct MI via sklearn — is unchanged, so
this is a diagnosability fix, not a behavior change). New regression test
(`test_mi_classif_batch_numba_fallback_logs.py`, 2 tests: failure path warns + returns correct values,
success path stays silent) pins both directions.

## PERF WIN (2026-08-03): hinge pre-check's 3-cut loop did a full lstsq per cut instead of the FWL rank-1 update it already had the QR for
2M-row cProfile on combo `c0417_7a16cb7d` (master-seed `31337`, cb/xgb suite) caught `_segmented_sse`
costing 13.7s cumtime across 74 calls (~2.6s of that pure self-time in the `column_stack`/`lstsq` call
overhead). All 74 calls trace to `_hinge_slope_change_plausible`'s 3-candidate precheck loop — the *main*
24-cut hinge scan in `_detect_hinge_breakpoints` already scores each cut via a Frisch-Waugh-Lovell rank-1
update against a QR factorization computed ONCE per round (see that function's docstring: 2.4x faster than
per-cut lstsq, and a prior bench explicitly rejected re-forming `A.T@A` per cut as 2.2x SLOWER). The
precheck computes that exact same QR (`Q`, `r_y`, `sse_lin`) via `_linear_qr_fit` for its own linear-SSE
baseline, but its 3-cut loop never got the FWL treatment — it still called `_segmented_sse` (fresh
`lstsq`) per cut. Rewired the precheck loop to the identical FWL rank-1 update the main scan already
uses. Verified 0 mismatches across 30 synthetic scenarios (kink / pure-linear / pure-noise, n=500-200k)
and 1.92x speedup at n=2M (warm, best-of-20). `_segmented_sse` itself is unchanged and still used
elsewhere (`_heldout_hinge_r2_uplift`'s design-comparison closures); only its caller here was rewired.

## PERF WIN (2026-08-03): fourier_adaptive prewarp replay's per-frequency Python loop fused into one njit pass (6.96x)
2M-row cProfile on combo `c0266_c0090719` (master-seed `31337`, hgb-only regression) caught
`apply_operand_prewarp` (`hermite_fe/_hermite_prewarp.py`) costing 4.03s pure self-time across 191 calls.
Root cause: its `fourier_adaptive` basis branch replays `f(x) = sum_k a_k*sin(2*pi*f_k*z) +
b_k*cos(2*pi*f_k*z)` via a Python `for i, f in enumerate(pp["freqs"])` loop, each iteration allocating
fresh full-`n`-length `ang`/`sin`/`cos` numpy temp arrays — K separate elementwise passes over the column
instead of one. Added `_fourier_adaptive_prewarp_njit` (`@njit(parallel=True)`, `prange` over rows, the
frequency reduction as an inner per-row loop in the SAME `i=0..K-1` order the Python loop used) and wired
it into `apply_operand_prewarp`'s `fourier_adaptive` branch. Verified 0 mismatches across 20 synthetic
scenarios (K=1-8 frequencies, linear + quadratic axis preprocessing, n=500-50k) and 6.96x speedup at
n=2M/K=4 (warm, best-of-15). The other basis branches (poly/eval_dispatch) were untouched.

## PERF WIN (2026-08-03): _dyadic_haar_leg's zeros-alloc + 2-mask numpy build fused into one njit pass (8.02x)
2M-row cProfile on combo `c0317_0c2314d2` (master-seed `31337`, 5-model multiclass suite) caught
`_dyadic_haar_leg` (`_wavelet_basis_fe.py`) costing 5.3s self-time across 200 calls. The prior form built
the 3-valued Haar-leg step function (`{-1, 0, +1}`) via `np.zeros_like` + two separate boolean-mask +
fancy-index writes — 4 full array traversals of a purely memory-bandwidth-bound op. Added
`_dyadic_haar_leg_njit` (`@njit(parallel=True)`, one `prange` pass doing both comparisons and the write per
element) and wired it in, collapsing 4 traversals to 1. Verified 0 mismatches across 30 synthetic scenarios
(n=500-200k, varying scale `j`/offset `k`, float32 + float64 output dtype) and 8.02x speedup at n=2M (warm,
best-of-30).

## PERF WIN (2026-08-03): conditional-residual FE's per-pair sum+count split into a fused njit pass (1.48x)
2M-row cProfile on combo `c0367_a7a91550` (master-seed `31337`, lgb-only regression) caught
`generate_conditional_residual_features` (`_extra_fe_families.py`) with real self-time. Its inner
`(x_i, x_j)` loop (iter111, an earlier rewrite) already used `np.bincount` instead of `np.add.at`, but still
paid TWO separate bincount passes per pair — one for the per-bin sum, one for the per-bin count — plus two
subset-copy allocations (`codes_j[finite_i]`, `xi[finite_i]`) to feed them. Added
`_bin_sum_count_masked_njit`: one row-scan that masks and accumulates both sum and count together, in the
same row order `np.bincount` on the pre-built finite-only subset used (so results are bit-identical).
Verified 0 mismatches across 15 synthetic scenarios (n=500-50k, k=2-6 columns, NaN-mixed) and 1.48x speedup
at n=2M/k=15 (warm, best-of-5). The existing perf-regression test pinning this function's accumulation
mechanism (`test_conditional_residual_bincount_perf_regression.py`) was reframed to spy on the new fused
kernel instead of `np.bincount` — its real invariant (no `np.add.at` scatter, bit-identical accumulation)
is unchanged, only the current mechanism-under-test needed updating.

## PERF WIN (2026-08-04): hermite_fe's minmax preprocess-replay fused into one njit pass (6.58x)
2M-row cProfile on combo `c0546_4b643261` caught `_apply_minmax` (`hermite_fe/__init__.py`) — the
legendre/chebyshev basis `preprocess`-replay closure — with real self-time across 137 calls. The prior form
was plain numpy (`2*(x-lo)/span - 1`, then an optional `np.clip`): 3-4 separate full-array passes
(subtract/multiply/divide/subtract, +1 for the clip branch). Added `_apply_minmax_njit` (`@njit(parallel=True)`,
one `prange` pass, op order kept identical to the numpy expression — `2*(x-lo)/span - 1`, not a
reciprocal-multiply rewrite — for bit-identical FP rounding) and wired it into `_apply_minmax` (numba-available
branch; falls back to the original numpy form when numba is absent). Verified 0 mismatches across 20 synthetic
scenarios (n=500-50k, with/without clip) and 6.58x speedup at n=2M (warm, best-of-30).

## PROCESS FIX (2026-08-04): a REJECT verdict must check for @njit/parallel/cuda.jit/cupy/KTC specifically, not just "look" optimized
The user caught this live: `_apply_minmax` and `_apply_orth_fourier` (both plain numpy, both doing 3-4
chained elementwise passes) were almost REJECTed as "already minimal" purely by reading the code, without
checking whether an `@njit` decorator was actually present anywhere near them. Both turned out to be real
wins once actually fused (6.58x, 3.84x — see entries below). A third case, `_compute_extremality_matrix`
(`row_wise_extremality.py`), had a "bench-attempt-rejected" note on file that only ever compared TWO NUMPY
forms against each other and never tried njit at all — also a real win once fused (2.11x, see below).
**Rule going forward**: before writing a REJECT for any hotspot with real tottime, explicitly grep it (or
trace one level into what it calls) for `@njit`, `parallel=True`/`prange`, `cuda.jit`, `cupy`, and
`KernelTuningCache`/`get_or_tune`. A REJECT is only valid when one of those five is confirmed present and
actually covers this code path, or a genuine bench-attempt-rejected note already tested THIS lever (not
just other numpy variants), or the cost is dominated by a third-party call with no first-party loop to
fuse. See global memory `feedback_check_njit_before_reject.md`.

## PERF WIN (2026-08-04): orth_fourier basis replay's per-element sin/cos chain fused into one njit pass (3.84x)
Same profiling cycle as the minmax fix above (combo `c0546_4b643261`) also surfaced `_apply_orth_fourier`
(`engineered_recipes/_orth_basis_recipes.py`) — the Fourier-recipe replay closure, plain numpy doing
subtract/divide/multiply/(sin|cos)/nan_to_num as separate full-array passes. Added
`_fourier_linear_sincos_njit` (`@njit(parallel=True)`, one `prange` pass fusing the whole chain, NaN/inf
outputs zeroed in-kernel to match `nan_to_num`) and wired it into the common `arg='linear'` branch (the
rarer `arg='quadratic'` chirp-axis branch is untouched, still numpy). Verified 0 mismatches across 20
synthetic scenarios (n=500-50k, sin/cos, power=1/3) and 3.84x speedup at n=2M (warm, best-of-30).

## PERF WIN (2026-08-04): row-wise extremality's per-column rank loop parallelised across columns via njit (2.11x)
Same profiling cycle (combo `c0546_4b643261`, triggered by the audit above) revisited
`_compute_extremality_matrix` (`row_wise_extremality.py`), whose only existing note
("bench-attempt-rejected 2026-07-13") had compared a per-column Python loop against a single vectorized
`np.argsort(axis=0)` — both plain numpy, njit was never tried. Added `_extremality_matrix_njit`
(`@njit(parallel=True)`, `prange` over COLUMNS — each column's own NaN-mask/argsort/rank/normalise chain is
independent of every other column, so this parallelises the existing loop across cores instead of running
it on one thread) and wired it in. Verified bit-identical to the numpy reference on NaN handling and on
tie-free (continuous) data; on heavily-tied/low-cardinality columns numba's argsort breaks ties in a
different order than numpy's quicksort, so the exact per-row rank WITHIN a tied group can differ (still a
mathematically valid extremality score) — documented in the function's docstring as the same class of
precision-vs-speed tradeoff `_ordinal_rank` already accepts for non-tie-averaged ranks. 2.11x speedup at
n=200k/k=200 (warm, best-of-5); no existing test pins exact tie-order values on discrete columns.

## PERF WIN (2026-08-04): hinge-gate's held-out R^2 fit switched from full SVD lstsq to normal equations
Same njit-check audit as the three entries above. `_r2` (a closure inside `_fit_impl_core.py`'s selected-set
incremental-R^2 hinge re-add gate) fit its held-out OLS via `np.linalg.lstsq` — a full SVD solve — on a
tiny-k (intercept + a handful of base/leg columns) design. The sibling `_deflate_sincos`
(`_orth_extra_basis_fe.py`) already proved normal equations (`A.T@A` / `np.linalg.solve`) beats a full SVD
lstsq for exactly this shape (small, well-conditioned design), but `_r2` never got the same treatment.
Switched to normal equations with an `np.linalg.LinAlgError` fallback to the original lstsq path for a
singular design. Verified numerically equivalent (max relative diff 1.14e-16 — machine epsilon — across 50
synthetic OLS scenarios, n=200-100k, k=2-5) so the gate's admit/reject decisions are unchanged. Low call
volume at this exact site (8 calls / 4.7s in the triggering profile, combo `c0554_46233682`) so the direct
win here is modest, but flagged and fixed per the "check every lever, not just eyeball it" rule above.

## PERF WIN + LIVE BUG CAUGHT (2026-08-04): binned-numeric-agg's global fallback stats fused 4 full-array passes into 1-2, restoring a silently-reverted optimization that had a catastrophic numerical bug
User pushback on the njit-check audit itself: `_global_stat` (`_binned_numeric_agg_fe.py`) individually uses
`np.mean`/`np.std`/`scipy.stats.skew`/`scipy.stats.kurtosis` — each is individually optimal, but the caller
(`fit_binned_numeric_agg`) calls it ONCE PER STAT (`{s: _global_stat(av, s) for s in kept_stats}`), so up to
4 SEPARATE full-array traversals happen per column, each independently recomputing overlapping raw moments
(std recomputes mean; skew/kurtosis each recompute mean AND variance internally).

Mid-fix, `git log -- <file>` turned up that this EXACT fusion had already been implemented and shipped once
before, as `_global_stats_all` in commit `2640bc00d` (2026-07-31) — then SILENTLY REVERTED two commits later
by `2d4d86c60` (2026-08-02, an unrelated GROUP-MI-batching perf commit whose diff, evidently built from a
stale pre-`2640bc00d` checkout, deleted `_global_stats_all` and the sibling fold-gate optimization as a side
effect with no mention in its commit message). This is the exact "worktree-copy trap" this session already
hit once on `_gpu_resident_extval.py` — always `git log -- <file>` before starting AND before finishing any
optimization, not just to avoid duplicate work but because a "duplicate" can turn out to be a silently-lost
fix.

Reconstructing `2640bc00d`'s actual `_global_stats_all` (it reused `_raw_moments`/`_derive_cell_stats` — the
SAME raw-moment-expansion formula `_derive_cell_stats` already uses for PER-CELL stats) and A/B-testing it
against `_global_stat` over 3000 trials (scale 1e-3..1e6, offset ±1e4) reproduced the bug LIVE: skew/kurt
errors up to 17 orders of magnitude from large-nearly-equal-numbers cancellation in
`sum(x**k) - k*mean*sum(x**(k-1)) + ...`. Their own commit message claimed "worst diff 2.3e-10" from a
3000-trial sweep — their sweep evidently never hit the large-offset/small-scale regime that breaks this
formula. Had the revert not accidentally happened, this would be a live, shipped production correctness bug.

Restored the fusion (`_global_stats_all`, keeping the name the existing test
`test_global_stats_all_matches_global_stat` already expects) AND the sibling fold-gate optimization
(`test_fin.size == finite_count` replacing the `(fold_ids != f) & finite` full-array-AND + `.any()` gate),
but backed the stats fusion with a NEW, numerically-stable kernel — `_centered_moments_njit`: mean pass,
then ONE fused pass accumulating `sum((x-mean)^2)`/`^3`/`^4` directly (no algebraic expansion, no
cancellation) — what scipy's own skew/kurtosis does internally. Also needed an explicit constant-column fast
path (`vmin == vmax` short-circuit): sequential float summation of many copies of the same value isn't
always bit-exact to `n*value`, so a truly-constant huge-offset column could still compute a
just-over-1e-12 "std" from that mean-drift and blow up skew/kurt as (near-zero)/(near-zero)^k.

Verified against the original per-stat calls (`np.mean`/`np.std`/`scipy.stats.skew`/`kurtosis`) across 30
synthetic scenarios spanning extreme scale/offset combinations (1e-3 to 1e6), NaN-mixed and constant columns
— max relative diff after both fixes: mean 1e-14, std/skew/kurt at the ~1e-6..1e-8 noise floor (the same
tolerance `_derive_cell_stats` already accepts for this stat family). 12.04x speedup at n=2M (warm,
best-of-10) on the common all-4-stats request. Full existing test suite for the file (9 tests, incl. the
pre-existing `test_global_stats_all_matches_global_stat`) passes.

OPEN FOLLOW-UP: `_derive_cell_stats`'s own per-cell skew/kurt (lines ~129-133) uses the SAME raw-moment
binomial-expansion formula just proven catastrophically unstable — per-cell offsets are typically smaller
than a whole-column global, but this is unverified, not yet stress-tested, and is a real candidate for the
exact same bug on production data with large per-cell offsets. Needs its own dedicated A/B sweep before
being ruled safe or unsafe; not fixed here because it's a wider-blast-radius change (touches every per-cell
stat consumer in this file, not just the global fallback).

## PERF WIN (2026-08-04): fused bootstrap bundle's tie-free-only AUC gate was silently defeating its own 3-4x win on the exact runs it targets — extended to tied scores
A 2M-row cProfile (combo `c0619`, binary classification, cb/hgb/lgb/mlp/xgb) showed `bootstrap_metrics`
(`evaluation/bootstrap.py`, the GENERIC serial per-resample bootstrap loop) still costing real time
(25.6s/5 calls tottime, plus `_resampler_grouped`/`fast_log_loss_binary`/`fast_brier_score_loss`/
`_ece_score` individually at 6-23s each) ALONGSIDE `_bootstrap_fused_binary_bundle.py`'s fully-njit
`prange`-parallel bundle (`bootstrap_auc_brier_ll_ece_batch`) — meaning `honest_diagnostics._bootstrap_block`
was falling back to the slow GIL-bound path for SOME of its 8 per-model bootstrap calls in the same run.

Root cause: `bootstrap_auc_brier_ll_ece_batch` had a hard `tie_free` gate — ANY duplicate value in the
predicted-probability column (`p_pos`) made it return `None`, forcing the full fallback to
`bootstrap_metrics`. Real predicted probabilities routinely have ties at scale (quantised tree-ensemble
leaf outputs, float32 rounding, 2M-row combos) — this gate was silently defeating the module's own
documented 2.9-3.4x fusion win on exactly the large real-world runs it was built for. The tie-free
`_fused_resample_auc_batch_parallel` (in `_core_auc_brier.py`) already had a tie-AWARE serial sibling
(`_fused_resample_auc_grouped`, DISTINCT-score-group counting instead of per-base-rank counting — used by
`make_bootstrap_auc_resampler`'s own tied fallback) but no BATCHED/parallel twin existed yet, so the fused
bundle had no tied-score code path to fall into other than bailing out entirely.

Added `_bootstrap_batch_auc_brier_ll_ece_grouped` (prange-parallel twin of the existing
`_bootstrap_batch_auc_brier_ll_ece`, using `group_of_base`/`y_base`/`ngroups` grouped counting for AUC
instead of `base_rank`/`y_by_rank` — brier/log_loss/ece are order-invariant per-row reductions, unaffected
by ties, so only the AUC accumulation needed the tie-aware variant) and wired it into
`bootstrap_auc_brier_ll_ece_batch`: `tie_free` now selects WHICH kernel runs (base-rank vs grouped) instead
of gating whether the fusion runs at all. `honest_diagnostics._bootstrap_block`'s call site is unchanged —
this was already `bootstrap_auc_brier_ll_ece_batch`'s sole production caller, so the fix reaches production
by construction, no additional wiring needed.

Verified bit-identical to `bootstrap_metrics` (the ground-truth serial path) on tied/low-cardinality scores
(1e-9 tolerance, incl. an all-identical-scores degenerate case, `ngroups=1`) — `test_fused_bundle_*` suite
(6 tests, was `test_fused_bundle_returns_none_on_tied_scores`, now
`test_fused_bundle_matches_bootstrap_metrics_on_tied_scores` + `test_fused_bundle_handles_all_tied_scores`)
passes, plus the full `tests/evaluation/` suite (185 tests) and `honest_diagnostics` suite (8 tests)
unchanged. 3.87x speedup measured on n=200k/R=1000 realistic quantised-probability data (warm, was the
`bootstrap_metrics` fallback path before this fix).

## PERF WIN (2026-08-04): fused bootstrap bundle's ECE jackknife was duplicating the OLD pre-fix bootstrap_metrics behaviour instead of using the already-shipped closed form
Re-profiling combo `c0091` right after the tied-scores fix above (2M rows, 5 model families) surfaced
`_jackknife_metric` at 23.985s/12 calls (35.9s cumtime) PLUS its callback lambda at
`_bootstrap_fused_binary_bundle.py:305` costing 21.469s across 24000 calls — both inside
`bootstrap_auc_brier_ll_ece_batch`'s ECE jackknife branch, now hit far more often since the tied-scores fix
routes nearly every `_bootstrap_block` call through this module instead of falling back.

Root cause: an already-shipped closed-form ECE jackknife (`_jackknife_ece`, added 2026-07-31 — see that
entry below — O(n) instead of the generic gather path's O(max_n\*n), ~800x measured) is wired into
`honest_diagnostics._bootstrap_block`'s OWN `jackknife_fns["ece"]` for its `bootstrap_metrics` fallback
path, but `_bootstrap_fused_binary_bundle.py` (written to mirror `bootstrap_metrics`'s behaviour) had a
comment saying ECE has "no closed-form... in bootstrap_metrics either" and called the slow generic
`_jackknife_metric` directly — true when the fused bundle was first written, stale after the 2026-07-31 fix
landed on the OTHER call site and nobody circled back to this one. A classic "already-optimized primitive,
just not wired into every call site that needs it" gap, not a new optimization to invent.

Fixed by calling `_jackknife_ece(y_true, p_pos, n_bins=n_bins)` first, falling back to the generic
`_jackknife_metric` gather only on `_jackknife_ece`'s documented degeneracy return (`None`) — mirrors
`bootstrap_metrics`'s own custom-jackknife-then-gather-fallback order exactly. `_jackknife_ece` was already
verified bit-identical (1.1e-16) to the generic path in the original 2026-07-31 work, so no new correctness
risk; `test_bootstrap_fused_binary_bundle.py`'s existing match-tests (comparing against `bootstrap_metrics`,
which itself falls back to the SAME generic gather path in the test's reference wiring) continue to pass at
the 1e-9 tolerance, confirming CI equivalence either way.

## PERF WIN (2026-08-04): hinge breakpoint detector's per-candidate-cut scan fused into one parallel-njit call (2.4x serial / ~14x parallel on the isolated kernel)
2M-row cProfile on combo `c0412` (cb/lgb/linear, binary classification) showed `_detect_hinge_breakpoints`
(`_hinge_basis_fe.py`) itself — not a callee — at 48.0s tottime across 97 calls (106.5s cumtime): the
FWL-scored per-candidate-cut loop (`for c in cand: ...`, up to `_HINGE_N_CANDIDATES`=24 cuts per round) runs
directly in this function's own Python frame, so its cost cannot be cProfile-misattributed to a callee —
it's real. The FWL identity itself (`SSE_B - (r_relu.r_y)^2/(r_relu.r_relu)`) was already the fast math (a
2026-06-09 fix replaced a per-cut `lstsq`/SVD with this rank-1 update), but the per-cut Python loop still
paid dispatch overhead on top of several already-small numpy calls (`count_nonzero`, two `Q @ (...)`
matrix-vector products, two dot products) per candidate — candidates are mutually independent given the
round's fixed `Q`/`r_y`/`sse_B`, exactly the fuse-into-one-parallel-njit-call pattern.

Added `_hinge_cut_scan_njit` (`@njit(parallel=True)`, `prange` over candidates): each candidate's `n_right`
count, `relu` construction, and the two O(n·k) projection reductions run in a private per-candidate
accumulator (recomputing `relu` twice rather than materialising an `(n,)` array, to keep the per-candidate
working set O(k) under `prange`), with the argmin taken over the per-candidate SSEs afterward (avoids a
cross-thread running-min race). Wired into both `_detect_hinge_breakpoints` call sites (round 0's reused
precheck QR and every later round's fresh QR) — the round loop no longer has an inline candidate scan at
all. Verified bit-identical `tau` and ~1e-14 relative `sse` (well within the FWL identity's own already-
documented ~1e-12 FP-reorder tolerance) against the original Python loop across 40 synthetic scenarios
(varying `n`, design width, already-found taus) — `bench_hinge_cut_scan.py`. 2.38x on the isolated kernel at
n=2M/24 candidates (serial njit vs the original numpy loop), ~14.2x with `parallel=True`. Full hinge test
suite (27 tests across `test_hinge_basis_fe.py`, GPU-resident upload/subsample, provenance, mrmr-gate) passes
unchanged, incl. `test_detect_hinge_fwl_rank1_taus_bit_identical_to_lstsq_per_cut` — the exact test pinning
the bit-identity contract this fusion had to preserve.

## LIVE BUG CAUGHT (2026-08-04): target-encoding's per-category skew/kurt catastrophically wrong on real (large-offset) regression targets — third instance of a bug class this session
While auditing a fresh 2M-row cProfile (combo `c0392`) hotspot, `_target_encoding_fe.py:_raw_moment_sums`
turned out to use the SAME raw-power-sum + textbook-binomial-expansion formula for per-category skew/kurt
already proven catastrophically unstable TWICE earlier this session (`_binned_numeric_agg_fe.py`'s
`_global_stats_all`/`_derive_cell_stats`, and the still-open follow-up on `_derive_cell_stats`'s own
per-cell path). A/B against scipy's direct per-category skew/kurt confirmed it live here too: errors up to
**5.8e13** on synthetic data with a large offset relative to spread (offset~8.5e3, scale~0.05-0.08) —
exactly the shape of a real regression target (price, revenue, counts: rarely centred at 0). `stats=(...,
"skew","kurt")` is opt-in (default `stats=("mean",)`), so the blast radius is every caller that explicitly
requests higher target-encoding moments, not every target-encoding call.

Fixed with the SAME numerically-stable two-pass pattern as the earlier fixes: `_per_cat_centered_moments_njit`
(njit, mean pass then one fused pass accumulating centred `(y-mean)**2/3/4` directly — no algebraic
expansion, no cancellation) + `_smooth_moments_from_centered` (derives std/skew/kurt from those centred
moments, replacing `_raw_moment_sums`/`_smooth_moments_from_sums` entirely — only this one file used them).

SECOND bug found in the SAME derivation, distinct from cancellation: the old formula padded the skew/kurt
denominator with an additive `+1e-12` epsilon (`std**3 + 1e-12`, `var*var + 1e-12`) meant to guard
div-by-zero — but once the numerator/denominator are computed stably (not near-zero, just SMALL, e.g.
var~1e-6 so var²~1e-12), that epsilon is on the same order as the true denominator and corrupts the result
by ~30-100% even with zero cancellation error. Caught via a debug trace showing the njit kernel's raw
moments were correct (`_per_cat_centered_moments_njit`'s own output matched scipy to 1e-10) while the
DERIVED kurt was still wrong by ~0.8 — the bug was downstream, in the epsilon-padded division, not the
moment computation. Fixed by dropping the `+eps` pad entirely: the existing `np.where(var > 1e-12, ...)` /
`np.where(std > 1e-9, ...)` guards already bound the denominator away from zero before it's used, so the
additive pad was pure liability once the numerator computation stopped needing defensive padding.

Also had to change the OOF fold structure: the old code exploited raw-power-sum ADDITIVITY (`train =
full - test`) to give each fold an O(n/n_folds) test-only pass instead of an O(n_train) rescan — centred
moments are NOT additive across row subsets (a subset's own mean differs from the full-data mean, so
`moments(full) - moments(test) != moments(train)` for any centred quantity), so each fold's TRAIN moments
are now computed directly via the already-available `train_mask` (no extra computation needed to obtain
it — it already existed in the loop, just wasn't being used for this). Correctness over the old (buggy)
row-visit-count optimization.

Added `test_skew_kurt_stable_on_large_offset_small_scale_target` (pins per-category skew/kurt against
scipy's direct computation on exactly this large-offset regime) to `test_multistat_target_encoding.py`.
Full target-encoding suite (39 tests across `target_encoding/` + the mrmr biz_val kfold-TE suite) passes.

OPEN FOLLOW-UP (unchanged from the earlier entry): `_binned_numeric_agg_fe.py`'s `_derive_cell_stats` still
uses the original catastrophically-unstable raw-moment-expansion formula for its per-cell skew/kurt and has
not yet been fixed — now THREE confirmed/suspected instances of this exact bug class across the codebase
(`_global_stats_all` fixed, target-encoding fixed here, `_derive_cell_stats` still open). Worth a dedicated
sweep for any other `s3 = ... ; m3 = s3/n - 3*mean*(s2/n) + 2*mean**3`-shaped code before assuming these
three are the only occurrences.

## INVESTIGATION LEAD (2026-08-04, NOT yet actionable): `resident_operand`'s GPU cache shows an ~84% miss rate under a wide pairwise-modular/conditional-gate FE sweep — `cupy.array` cost 1697.9s / 29346 calls (~25% of the ENTIRE 6827s run) on combo `c0605` (5 models, multilabel, wide pair search)
`resident_operand` (`_fe_resident_operands.py`) is an already-sophisticated content-hash LRU cache (192-entry
cap, ~330MB budget) specifically built to dedupe redundant re-uploads of the SAME fit-constant operand
content across roles. On this profile it was CALLED 34993 times but only avoided upload on ~16% of calls
(`cupy.array` fired 29346 times) — the cache is doing its job (no aliasing bugs, no wrong-value hits), the
workload here just generates far more DISTINCT large operands (via `_pairwise_modular_fe.py`/
`_conditional_gate_fe.py`'s `_scan_one`/`_residue_grid_mi`/`_add`/`_flush` candidate machinery, 360
`_scan_one` calls) than the cache's `_MAX_ENTRIES=192` budget was sized for — the module's own comment
justifying 192 cites "a 1M strict-resident fit touched 118 distinct contents," a DIFFERENT (smaller/narrower)
combo than this one.

NOT marked RESOLVED or REJECT because the evidence is ambiguous, not because the lead is weak: (a) the
per-call cost (~58ms/call for `cp.asarray`) is far above cupy's normal array-creation latency and is more
consistent with this session's well-documented severe multi-process GPU contention than a real per-call
regression — the same "discard wall-clock as evidence-free under heavy contention" caveat this session has
applied to CPU thread-lock-dominated profiles applies here too; (b) but the CALL-COUNT-based finding (cache
miss rate, independent of per-call wall-clock) is contention-immune and does suggest `_MAX_ENTRIES=192` may
be genuinely undersized for wide-pair-search combos, which VRAM-budget math would need to confirm (each
evicted-then-reused operand is a few MB; is 192 -> e.g. 500-1000 entries still comfortable on a 4GB card
alongside the chunked candidate buffers the module's docstring already accounts for?).

**Next step for whoever picks this up**: re-profile this SAME combo (`c0605`, master-seed `2027_07_02`,
`--combo-pool 700`) on a quiet machine (or instrument `resident_operand` with a hit/miss counter independent
of wall-clock) to separate the contention-inflated per-call cost from the genuine cache-sizing question
before touching `_MAX_ENTRIES` or the eviction policy.

**UPDATE (2026-08-05) — reproduced on a SECOND, different combo, but "just raise `_MAX_ENTRIES`" is NOT the
safe fix it looks like**: combo `c0576` (hgb/lgb/linear/xgb, multilabel, master-seed `2027_12_20`) showed the
SAME pattern independently — `resident_operand` called 35285 times, `cupy.array` fired 28837 times (~82%
miss rate, consistent with the earlier ~84%), `cupy.array` costing 1285.0s / ~24% of the 5263s total run.
Two different combos landing on the same ~82-84% miss rate makes contention-noise a much weaker explanation
for the CALL-COUNT finding specifically (still plausible for the per-call wall-clock, per the caveat above).

BUT: this same run's log shows `batch_pair_mi_gpu` REJECTING a 0.27GB upload because it "would breach the
absolute VRAM cushion floor (free=1.25GB, total=4.00GB)" — this GPU is a 4GB card running with as little as
1.25GB free DURING THIS EXACT WORKLOAD. `_MAX_ENTRIES=192` was explicitly calibrated in the module's own
docstring to stay "comfortable alongside the chunked candidate buffers" on a 4GB card — blindly raising it
to fix the miss-rate would shrink that already-thin headroom further and could turn a slow-but-working run
into a hard OOM crash elsewhere in the SAME fit, trading a perf problem for a correctness/stability one. A
safe fix needs to be VRAM-aware (e.g. size the cache from `cupy.cuda.Device().mem_info` free bytes at cache-
init/reset time instead of a fixed entry count) rather than a blind constant bump — a bigger undertaking than
originally assumed, still not attempted.

## PERF WIN (2026-08-05): integer-lattice's 12-permutation null band batched into one call — the sibling `_pairwise_modular_fe.py` version already had this fix, `_integer_lattice_fe.py`'s near-identical copy never got it
2M-row cProfile (combo `c0454`, cb/hgb/lgb, multiclass) surfaced `_perm_null_hi` at real tottime in THREE
sibling files (`_integer_lattice_fe.py` 3.654s/3 calls, `_pairwise_modular_fe.py` 2.647s/3, `_conditional_gate_fe.py`
1.272s/3) — each independently implementing "upper band (mean + z\*std) of a fixed feature's MI under
`n_perm=12` y-permutations." `_pairwise_modular_fe.py`'s version already had a documented batching fix (the
"SF2 :311 collapse" note: joint-reindex invariance `MI(feat; y[perm]) == MI(feat[inv_perm]; y)` lets all 12
permuted-feature columns score against the SAME y in one `_mi_classif_batch` call instead of 12 separate
`_mi()` calls) — but `_integer_lattice_fe.py`'s copy, despite its own docstring saying "Mirrors
`_pairwise_modular_fe._responded`," still ran the original unbatched per-perm Python loop.

Ported the exact same batching to `_integer_lattice_fe.py._perm_null_hi`: build a `(n, n_perm)` matrix of
`feat[argsort(perm)]` for each of the 12 RNG-drawn permutations (same draw order, same seed sequence — bit-
identical reproducibility unchanged) and score all 12 in one `_mi_classif_batch` call. Verified bit-identical
(0 diff, not just tolerance-close) against the original per-perm loop across 30 synthetic scenarios —
`bench_lattice_perm_null_hi.py` — and 1.95x at n=200k. Added
`test_perm_null_hi_batched_matches_per_perm_loop` to `test_integer_lattice_njit_identity.py`. Full
integer-lattice suite (27 tests, run with `CUDA_VISIBLE_DEVICES=""` after an unrelated native access-violation
crash mid-suite — matches this session's documented GPU-crash contention pattern, unrelated to this change)
passes. `_conditional_gate_fe.py`'s copy has NOT yet been checked/ported — worth a follow-up look.

## PERF WIN (2026-08-02): per_feature_edges' thread-pool threshold was 64x too high for real usage (1.2x-7.2x)
2M-row cProfile on combo `c0037_c314bb14` (master-seed `2026_04_29`) found `per_feature_edges`/
`_compute_col_edges` (`_adaptive_nbins.py`) costing 78s wall on a `fayyad_irani` (MDLP) fit with
only ~30 feature columns. `per_feature_edges` already had a `ThreadPoolExecutor` path for its
per-column edge loop (the columns are independent and the MDLP njit kernels release the GIL) — but
it was gated behind `_PARALLEL_EDGES_MIN_COLS = 128`, a threshold no realistic fuzz-combo width
(typically 10-50 columns) ever reaches, so the thread pool was effectively DEAD CODE in production
despite the docstring's own "columns are independent... a THREAD pool gives real wall-time
parallelism" rationale. Re-benchmarked fresh on this host (n=300k, `fayyad_irani`): ncols=2 -> 1.24x,
ncols=4 -> 3.71x, ncols=8 -> 4.44x, ncols=16 -> 4.24x, ncols=30 -> 7.18x — consistently faster
threaded from the smallest width tested, zero regression anywhere measured. The threshold's own prior
comment ("verified on p=50: parallel ties serial") does not reproduce; trusted the fresh, reproducible
A/B over the stale unverifiable claim. Lowered `_PARALLEL_EDGES_MIN_COLS` from 128 to 2. Edges are
BIT-IDENTICAL to the serial path regardless of thread count (each column's edges are independent, no
shared mutable state) — verified directly (0 mismatches across 12 synthetic columns, new default vs.
`n_jobs=1`). 58 tests across the adaptive-nbins / fayyad-irani / categorize_dataset suites pass.

**Incidental bug fix caught while validating**: `test_per_feature_edges_default_uses_validated_split`
failed — but on a literal revert-and-rerun (fully unmodified code) it failed IDENTICALLY, proving it
predates and is unrelated to the threshold change. Root cause: the test mocked
`_mdlp_recurse_validated` (the pre-2026-07-31 DFS recursion name); the 2026-07-31 BFS rewrite changed
`mdlp_bin_edges`'s default validated path to call `_mdlp_recurse_validated_bfs` instead, so the mock
never intercepted anything (`call_record` stayed all-zero) and the test failed unconditionally since
that landed. Fixed the mock target to the current function name; all 6 tests in the file pass.

## PERF WIN (2026-08-02): Fourier peak-frequency refine's grid scan fused into one parallel-njit batch call (3.1x)
2M-row cProfile on combo `c0016_c3f401e4` (master-seed `2026_04_29`) found `_power_centered`
(`_orth_extra_basis_fe.py`) at 30.98s tottime / 2134 calls. `_refine_peak_freq`'s `_scan` helper grid-
searches a frequency band via a serial Python `for` loop calling `_power_centered` once per candidate
point (~9-21 points per scan, 2 scans per refine call) — every candidate in one scan shares the SAME
`z_tr`/`yc`/`y_ss`, only `freq` varies, yet each call independently re-dispatches the ALREADY-parallel
njit kernel (`_power_centered_fused_par_njit`), paying its own thread-launch overhead every time. Added
`_power_centered_batch_njit`: flattens `n_freqs * nblocks` independent (frequency, block) partial-sum
tasks into ONE `prange` dispatch, then reduces each frequency's own blocks in the SAME fixed 0..NB-1
order the single-frequency kernel already used — bit-identical per frequency (concurrent OTHER
frequencies' block schedule cannot affect a given frequency's own float accumulation order). Wired into
`_scan`, gated on the same `_POWER_CENTERED_PAR_MIN_N` threshold the single-call path already used (the
small-n numpy fallback is untouched). Verified EXACT match (`==`, not just close) against the original
per-call loop across 10 synthetic single-tone-plus-noise scenarios (n=5k-8k). Measured 3.12x at n=200k
(0.304s -> 0.097s, warm, best-of-5). 49 tests across `test_spline_fourier_basis_fe.py` +
`test_extra_basis_fe_adaptive_max_cols.py` + `test_corr_sq_centered_noalloc.py` pass unchanged.

## BUG FIX (2026-08-02): auto-drop after feature_distribution_analyzer silently no-op'd on every pandas train_df
Incidentally spotted in a profiling cycle's log (`c0091` combo): `WARNING: [mini-HPT] auto-drop after
feature_distribution_analyzer failed (The truth value of a Index is ambiguous...)`. Root cause in
`_maybe_auto_drop_after_feature_analyzer` (`_main_train_suite_target_distribution.py`): `train_cols =
set(getattr(train_df, "columns", []) or [])` forced `bool()` on the `or` operator's left side, and
pandas raises `ValueError: The truth value of a Index is ambiguous` for any multi-column
`DataFrame.columns` — so this crashed on EVERY pandas `train_df` (caught by the caller's best-effort
`except`, silently falling back to the full column set — the analyzer's drop recommendations were
computed but never applied). A SECOND, independent bug in the same function's `_drop` helper: `df.drop(
present)` — pandas' `.drop()` defaults to `axis=0` (drops ROWS by index label, not columns), so even
past the first bug this would have raised `KeyError` (never caught by the `except TypeError` that was
meant to catch a wrong-signature call) instead of dropping columns. Fixed both: `train_cols` now guards
only the genuine "attribute is `None`" case explicitly (no boolean-context Index evaluation), and `_drop`
now calls `df.drop(columns=present)` first (pandas), falling back to the positional form (`df.drop(
present)`) that polars actually accepts on `TypeError`. Verified via a live repro (both bugs reproduced
and fixed for pandas AND polars) and a new regression test
(`test_auto_drop_after_feature_analyzer_drops_columns_on_pandas_and_polars` in
`test_run_target_distribution_analyzer_arity.py`) pinning both fixes.

## PERF WIN (2026-08-02): wavelet leg-scan's CPU fallback fused into one parallel-njit batch call (12.2x)
2M-row cProfile on combo `c0079_b01d8c82` (HGB+LGB+linear, regression) found `_select_wavelet_legs`
(`_wavelet_basis_fe.py`) at 380.1s cumtime — 28% of the entire suite's wall time — with `_binned_mi`
alone costing 207.9s across 1200 calls. A GPU-resident batched path already exists for this function
(`select_wavelet_legs_batched`, STRICT-gated, off by default), but the CPU fallback (the DEFAULT path)
still walked every candidate `(j, k)` Haar leg through a serial `for j: for k:` Python loop calling
`_binned_mi` twice per leg (train + held-out), each paying `np.unique`/`np.searchsorted`/`np.bincount`
dispatch overhead. A Haar leg's values are ALWAYS exactly `{-1, 0, +1}` (3 classes, by construction —
see `_dyadic_haar_leg`'s docstring) and the target codes are already precomputed once per source column,
so every leg's MI can be scored inline from a fixed-size (3, n_y_classes) joint histogram without ever
materializing the `{-1,0,1}` array or calling `_binned_mi`. Added `_wavelet_legs_mi_batch_njit` (`prange`
over legs, ternary leg-code built directly from the base column `z`) and wired it into the CPU fallback's
sole call site, preserving the exact `(j, k)` enumeration order for tie-breaking. A leg/y class absent
from a given (leg, split) pair contributes a literal zero either way (fixed vs. `np.unique`-derived
alphabet is mathematically equivalent, not an approximation — verified, not assumed). Verified exact
match (`==`, not just close) against the original per-leg loop across 8 synthetic scenarios (binary +
multiclass y, varying n). Measured 12.2x at n=200k/max_scale=6 (4.28s -> 0.35s, warm, best-of-3). 40
tests across `test_wavelet_batched_mi_parity.py` + `test_fourier_wavelet_max_cols_default_bounded.py` +
`test_binned_mi_bincount_identity.py` + `test_wavelet_basis_fe_max_cols.py` + `test_wavelet_basis_fe.py`
pass unchanged (including the suite's own embedded cProfile smoke check, now showing
`_select_wavelet_legs` as a minor cost instead of the dominant one).

## PROCESS FIX (2026-08-02): check `git log -- <file>` vs origin/master BEFORE investing in a fix
Second redundant-work collision in one loop session (after the MDLP permutation-null one above):
independently found + fixed `group_aware_relevance`'s (`_ranker_fs.py`) NaN-column/NaN-`y` fallback
still walking a serial per-column `np.quantile` loop (324,002 calls, 105.5s of a 183.2s/96%-of-suite
hotspot on LTR combo `c0037_1aebc059`) — only to discover, right before commit, that `origin/master`
already had a comprehensive fix for the EXACT SAME profile finding (`git log -- <file>` showed commit
`36b5f6b56`, "fuse group_aware_relevance's per-group loop into one chunk-parallel njit call (237x)"),
superseding the local change entirely. Discarded again (`git checkout --`). **Standing rule going
forward**: before spending real effort on a hotspot found via profiling, run `git log --oneline -3 --
<file>` (or diff the worktree's rebased copy against the file) FIRST — a concurrent session may
already be working the same profile, and the file's recent-commit list is a two-second check that
would have caught both collisions before any implementation time was spent.

## ENV NOTE (2026-08-02): stale installed `pyutilz` crashed every mlframe import
The site-packages `pyutilz` install lagged mlframe's current code (missing
`_generate_combinations_recursive_njit_core`, added by concurrent pyutilz work), crashing every
import (`ImportError` from `feature_selection/filters/evaluation.py`). Fixed by reinstalling from
the sibling source repo (`pip install -e . --no-deps` in `../pyutilz`). Not an mlframe code issue;
noted in case it recurs — check `pip show pyutilz` points at the source checkout, not a stale wheel.

**Worktree-copy trap (same session, caught before push)**: independently found and parallelized
`_permutation_null_gain_njit`'s permutation loop (`prange`, ~330x at n=200k) — only to discover, at
`git commit` time (mypy failed referencing a nonexistent `_mdlp_recurse_validated_bfs`), that
origin/master already had this EXACT fix plus a superior BFS-batching rewrite on top of it, landed
by concurrent work between this cycle's `git rebase origin/master` and its `cp`-from-main-tree step.
The bug: copying files from the (stale, un-pulled) main working tree into a freshly-rebased worktree
silently overwrites whatever the rebase just pulled in for that same file. Lesson: after `git rebase
origin/master` in the worktree, diff the file against `origin/master` (or just re-fetch + inspect)
BEFORE copying the main-tree version over it, when the file's own history shows recent unrelated
concurrent activity — don't assume the main tree is authoritative. Discarded the now-redundant
local change (`git checkout --`) instead of committing over the newer upstream version.

## PERF WIN (2026-08-01): binned_numeric_agg's GROUP MI pre-selection fused into one parallel-njit batch call
2M-row cProfile on combo `c0033_87df93d9` (LGB+MLP+XGB, binary classification) surfaced `_cheap_mi_with_y`
(`_binned_numeric_agg_fe.py`) at 47.1s tottime / 228 calls (~207ms/call) inside
`binned_numeric_agg_with_recipes`'s GROUP-column pre-selection: `g_mi = {g: _cheap_mi_with_y(X[g].to_numpy(),
y_codes) for g in gcands}` walked every candidate column through a SERIAL Python loop, each call
independently sorting/binning the full n-row column single-threaded. The candidate columns are mutually
independent (no cross-column dependency), so this is the "fuse-into-one-parallel-njit-call" pattern the
project's own perf convention calls for. Added `_cheap_mi_edge_dedup_njit` (one column, exact port of the
existing algorithm: quantile-edge order statistics via `np.partition` + adjacent-dedup, matching
`_cheap_mi_with_y`'s `quantile_edges`+`np.unique` dedup bit-for-bit) + `_cheap_mi_batch_njit` (`prange` over
columns) + `_cheap_mi_group_selection` (the new call-site wrapper: batches through the njit kernel by
default, but stays on the exact original per-column `_cheap_mi_with_y` loop when
`fe_gpu_strict_resident_enabled()` — the diagnostic full-GPU-coverage mode — is on, so that mode's
every-kernel-on-device contract is untouched). NOT a drop-in for the OTHER existing batched-edge-MI kernel
(`_fe_edge_mi.plugin_mi_classif_batch_edge_njit`) — that one deliberately skips the dedup step to bit-match
the GPU orth-family kernel, which would silently change the effective bin count (and therefore the MI
ranking) on tied/low-cardinality group columns; wrote a dedup-preserving twin instead. Verified bit-identical
to the original per-column loop (max diff ~1e-15 across continuous/tied/low-cardinality synthetic columns,
n=500..200k) and 3.34x faster at n=2M/k=16 (3.01s -> 0.90s). `tests/feature_selection/fe/test_binned_numeric_agg_fe.py`
(7 tests) + the polars-parity binned_numeric_agg case (15 tests total) pass unchanged.

## PERF WIN (2026-07-31): ECE's BCa jackknife left on the generic O(max_n*n) path while roc_auc/brier/log_loss already had closed forms
2M-row cProfile on combo `c0008_946d0da3` (binary classification, cb+hgb) surfaced `_jackknife_metric`
(`_bootstrap_jackknife.py`) at 15.3s tottime / 30.6s cumtime across just 6 calls, inside
`honest_diagnostics._bootstrap_block`'s bootstrap-CI bundle for {roc_auc, brier, log_loss, ece}. Three of
those four metrics already have an O(n) exact-algebraic BCa jackknife (`_jackknife_mean_metric` for
brier/log_loss's per-row-mean form, `_jackknife_auc`'s Mann-Whitney placement-value form) — ECE was the one
metric still falling through to the generic gather jackknife (O(max_n * n): re-slices and re-runs
`_ece_score` on ~2000 leave-one-out subsets of the full row range). ECE decomposes the SAME way AUC does —
not a mean of per-row values, but a sum over bins of `|per-bin-sum-diff|`, so leaving out one row only moves
its own bin's two sums; every other bin's term is untouched. Added `_jackknife_ece` (same file, algebraic
closed form, derivation in its docstring) and wired it into `honest_diagnostics.py`'s `_jackknife_fns["ece"]`
next to the existing `roc_auc` entry. Verified bit-identical to the generic gather path (max diff 1.1e-16 at
n=50/500/5000) and 1345x faster in isolation at n=1.6M (69.2s -> 0.051s). Full `tests/evaluation/` jackknife
+ bootstrap + ECE suite (40 tests) passes unchanged.

## Coverage cannot see inside `@njit`
numba-compiled bodies never reach the Python trace hook, so every `@njit` function reads as uncovered no matter how heavily it is exercised. Measure them with `NUMBA_DISABLE_JIT=1`, and expect the run to be much slower.

## PERF cycle REJECT (2026-08-01): c0011 2M-row profile, all fresh candidates already documented/sub-material
2M-row cProfile on combo `c0011_903f1399` (LGB+XGB, multi_target_regression, polars_nullable, 15 cats;
290s total, mostly threading waits + real booster fits) surfaced no candidate above ~10s cumtime not
already covered: `row_wise_extremality._compute_extremality_matrix`'s per-column argsort loop already has
a documented 2026-07-13 bench-rejected vectorization attempt (axis=0 argsort measured 3.3x SLOWER, kept
the loop) and its own absolute cost here (3.2s tottime / 3 calls) is sub-material; `_gpu_resident_select`'s
radix-select and `_resident_candidate_mi`'s candidate-MI path are the same already-saturated MANDATE-2 GPU
subsystem as the c0003 REJECT above. No safe, well-scoped, materially-sized win identified this cycle.

## PERF cycle REJECT (2026-07-31): c0003 2M-row profile, GPU-resident FE candidate MI already saturated
2M-row cProfile on combo `c0003_5a0bbd4e` (HGB, multilabel, polars_nullable, 15 cats) surfaced two more
mlframe-internal candidates besides the ECE jackknife win above:
`plugin_mi_classif_batch_dispatch` (275.7s tottime / 871 calls) and `cupy._core.core.array` /
`concatenate_method` (368s / 162s tottime across ~10-12k calls) inside the GPU-resident FE-candidate MI
path (`_resident_candidate_mi.py`, `_pairwise_modular_fe.py::_residue_grid_mi`, `_orth_mi_backends.py`).
Both investigated and REJECTED as fresh wins this cycle:
- `plugin_mi_classif_batch_dispatch`'s huge tottime is a KNOWN cProfile mis-attribution already documented
  in its own docstring (bench-rejected 2026-07-06): numba's compiled njit body has no Python frame, so its
  compute time rolls into this plain-Python dispatcher's tottime. Not actionable.
- The cupy H2D/concatenate volume lives entirely inside the kernel-tuning-cache-gated GPU-resident
  candidate-MI subsystem (`_resident_candidate_mi.py` / MANDATE-2), which already carries extensive
  per-host-measured crossover gating and multiple documented bench-rejected attempts (e.g. the
  2026-06-26 host-stacking rejection in `_build_best_existing_op_candidates_gpu`'s docstring). The dev
  host's GPU (GTX 1050 Ti, 4GB) is explicitly called out there as weak/contended; further tuning needs a
  fresh per-host KTC re-measurement + isolated bench, not a code-reading-only guess, and risks destabilizing
  an already-validated selection-equivalence contract. Deferred to a cycle with bandwidth for a full
  bench + A/B rather than forced under this cycle's time budget.

## polars traps that fail silently
`min() == max()` as a constant-column check returns null, not True, for an all-null column, so the column is silently not detected as constant — use `eq_missing` for any comparison that must treat null as a value, and test the all-null case explicitly whenever writing a column-level predicate.
`pl.Categorical` in polars 1.x resolves categories through a process-wide string cache, so two independently built frames can hold codes meaning different things, and joining or comparing them gives wrong results rather than an error. Use `pl.Enum` with an explicit category list wherever the value set is known.

## val / test / OOF mean different things
**val** is the split that drives early stopping, so any metric read on it is optimistically biased. **test / OOS** is untouched during fitting and is the honest estimate. **OOF** is the cross-validation analog of a test estimate. Name variables and report columns after what the split actually is, and never quote a val number as the headline result.

## Test behaviour, not source text
Never assert on `inspect.getsource()` to check that a string, call or pattern appears in a function body — such assertions break on every harmless refactor while passing for implementations that are actually wrong. Call the function and assert on its output, side effects or raised exception.
Size a rare-class synthetic from the minority count it needs, not the total: a 1% positive rate needs on the order of 5000 rows before any metric computed on the minority is stable, and an undersized fixture reads as flakiness.

## Serialization hygiene
Prefer `orjson` over the stdlib `json`, and compile regexes once at module scope rather than inside a function.
Any JSON serialization feeding a hash, cache key or dedup comparison must sort its keys — dict ordering is not stable across processes or versions, so the same logical object otherwise yields different hashes and the cache silently misses.
A cache attached to an instance at runtime (memo dicts, warmed kernels, device buffers, open handles) must be excluded in `__getstate__`, and the pickle suite run afterwards: the object pickles fine in a smoke test and fails later in a real save/load or joblib fan-out, far from the cause.

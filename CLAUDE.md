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

## Plain `python`/subprocess runs need explicit PYTHONPATH on this machine (CRITICAL)
This machine's global editable-install `.pth` (`__editable__.mlframe-*.pth`) points at a stale/unrelated worktree (`mlframe-sync-worktree`), not this repo's `src/`. `pytest` still resolves the real code correctly ONLY because `pyproject.toml` sets `pythonpath = ["src"]` in `[tool.pytest.ini_options]` — that setting does NOT propagate to `subprocess.run([sys.executable, ...])` calls a test spawns, so a subprocess-based import-smoke-test can fail with `ModuleNotFoundError: No module named 'mlframe...'` for reasons having nothing to do with the code under test. A profiling/bench script run as plain `python script.py` needs `PYTHONPATH=<repo>/src` set explicitly, every time. **Never "fix" this by editing the shared global `.pth` file** — it's machine-wide and other concurrent sessions' worktrees may depend on its current target; only pytest's own `pythonpath` ini setting or an explicit per-invocation `PYTHONPATH` env var are safe fixes.

## An optimization only helps if EVERY call site is wired, not just the one you found first (CRITICAL)
A hot recursion/function often has more than one independent call site reaching the same underlying logic (e.g. two separate public entry points that both duplicate a "sort + dedupe + recurse" sequence instead of one delegating to the other). Optimizing the first one you profile and calling it done leaves the others silently un-sped-up — a fresh cProfile run later can still show the OLD implementation as the #1 hotspot by tottime even though a validated faster replacement already exists in the same file. Before declaring a hotspot "already optimized, move to the next", `grep` every call site of the slow function/recursion by name across `src/`, not just the one the current profile happened to point at.
**Why documented:** the MDLP BFS rewrite (`_mdlp_recurse_validated` → `_mdlp_recurse_validated_bfs`) shipped 2026-07-31 and was wired into `mdlp_bin_edges_validated()` (`_mdlp_validated_split.py`) — but `mdlp_bin_edges()` (`supervised_binning.py`), a SEPARATE, independent call site that duplicates the same sort/dedupe/recurse sequence and is the entry point MRMR/FE code actually calls, still called the old DFS function directly. A fresh 2M-row cProfile on a *different* combo caught it: 21.3s tottime / 1212 calls on `_mdlp_recurse_validated`, i.e. the just-shipped optimization was real but unused in production. Fixed by wiring the same already-validated BFS function into the second call site — no new algorithm, purely a missed grep.

## FIXED BUG (2026-07-31, root-caused): `test_support_indices_within_feature_names_in_and_transform_runs` was a broken TEST FIXTURE, not an MRMR bug
`TestNeverEmptyRescueSupportIndexSpace::test_support_indices_within_feature_names_in_and_transform_runs`
(`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fe_encodings/test_kfold_target_encoding.py`)
failed with `transform()` emitting `['cat_region']` instead of `['cat_region__te']`. Root cause (confirmed
live via `verbose=True` fit logging, not guessed): the test called `_make_mrmr(fe_ntop_features=3)` believing
`fe_ntop_features` alone turns TE on ("TE ON by default" in the old docstring) — it does not.
`make_fast_mrmr`'s base preset is `fe_max_steps=0` with every FE family off; TE requires the explicit
`fe_kfold_te_enable=True, fe_kfold_te_cols=(...)` kwargs (as the sibling `TestOOFNoLeak`-style tests in the
same file already do). With TE never enabled, `cat_region__te` was never engineered at all (0 recipes; the
"MRMR+ selected 1 out of 3 features: cat_region" log line before the fix confirms selection was raw-only) —
nothing in the never-empty raw-representative re-attach path (`_fit_impl_core.py` ~line 8685,
`min_features_fallback`) was ever exercised or defective. Fix: added the missing `fe_kfold_te_enable=True,
fe_kfold_te_cols=("cat_region",)` kwargs to the test's `_make_mrmr` call. All 39 tests in
`tests/feature_selection/fe/target_encoding/` + this file now pass (was 38 passed / 1 failed).

## Coverage cannot see inside `@njit`
numba-compiled bodies never reach the Python trace hook, so every `@njit` function reads as uncovered no matter how heavily it is exercised. Measure them with `NUMBA_DISABLE_JIT=1`, and expect the run to be much slower.

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

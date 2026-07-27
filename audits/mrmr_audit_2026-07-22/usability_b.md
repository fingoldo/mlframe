# MRMR audit 2026-07-22 — cluster usability_b

This cluster covers the "usability surface" siblings of MRMR: fit-time artifact export for cross-selector
reuse (`_mrmr_artifacts.py`), the degenerate-column diagnostic scan (`_mrmr_degenerate.py`), the one-call
human-readable `explain_selection()` (`_mrmr_explain.py`), FE provenance tracking (`_mrmr_fe_provenance.py`),
top-level fingerprint/hash/cache-replay helpers (`_mrmr_fingerprints.py`), the incremental `partial_fit()`
streaming API (`_mrmr_partial_fit.py`), embedding/free-text passthrough detection (`_mrmr_passthrough.py`),
the SIS front-gate glue + kernel (`_mrmr_sis_apply.py` / `_mrmr_sis_screen.py`), the `MRMRTreeRescued`
selection-collapse rescue subclass (`_mrmr_tree_rescue.py`), `validate_inputs`/`transform` (`_mrmr_validate_transform.py`),
several self-contained FE-step sub-blocks (`_mrmr_fe_step_helpers.py`), the opt-in RelaxMRMR/FJMI 3-D-MI
research extension (`_relaxmrmr_3d.py`), and the vendored third-party InfoNet neural MI estimator
(`_vendored/infonet/**`). None of the 13 mlframe-original files exceed ~725 LOC (well under the 800-900 LOC
guideline; no split needed), and `mypy --cache-dir=.mlframe_mypy_cache_shared` reports zero issues across all
13 of them. Item 9 of the checklist (SQL/HTTP/UI best practices) is confirmed N/A for this cluster: no
database, network, or browser-facing surface exists anywhere in these files — `explain_selection()` /
`get_fe_report()` emit plain-text strings for a REPL/notebook, not HTML, and the vendored InfoNet loader only
prints a `gdown` download command, it never performs the download itself. Most prior-audit cross-references
resolve to "still open, unchanged" (the `_mrmr_tree_rescue.py:125` reversed seed-precedence P2 and the
`_mrmr_partial_fit.py:270` dead `is_first is False` P2, both from `c1_core_class.md`) rather than newly-fixed;
the real yield this pass is in angles the 2026-07-20 audit did not apply file-by-file — computational
efficiency/memory-discipline on two always-on validation/diagnostic passes, a cache-replay side effect on the
user's own returned object, and vendored-code hygiene (licensing, sys.path pollution).

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| USABILITY_B-1 | P1 | efficiency | `_mrmr_degenerate.py:139-247` (called from `mrmr/_mrmr_class.py:3359` and `mrmr/_mrmr_class_fit_helpers.py:573`) | `audit_degenerate_columns(X)` runs unconditionally on **every** `MRMR.fit()` call (single-target and multi-output paths alike) with no size gate or opt-out flag, despite the module's own docstring documenting a measured 43.0s cost on a real p~518-column production frame; the collinearity pass allocates a dense `(p, p)` correlation matrix (`_gram_matrix`, no cap), which on a genuinely wide raw frame (tens of thousands of raw columns — not implausible for mlframe's stated wide-p use cases) becomes a multi-GB-to-tens-of-GB allocation purely for a diagnostic that never influences selection. | New — not raised in `mrmr_audit_2026-07-20` (that audit's `c1_core_class.md:49` only checked GPU-residency correctness of this same function and called it "clean" for that angle; it did not apply the efficiency/edge-case checklist item here). |
| USABILITY_B-2 | P1 | efficiency / memory-discipline | `_mrmr_validate_transform.py:225-262` | `_validate_inputs`'s +/-inf guard unconditionally materializes a **full copy** of X's numeric columns via `X.select_dtypes(include=["number"]).to_numpy()` (upcasts a mixed int/float frame to one common float64 block) on every single `.fit()` call, then separately does a second full-frame materialize + a per-cell pure-Python `np.frompyfunc` scan over every object-dtype column. Neither pass is gated by size, and the cost is paid even when the numeric block is all-integer (ints can never hold inf, so the whole densify-and-upcast copy is pure waste in that common case — the dtype check that would have skipped it only runs *after* the copy). This directly violates the project's "never `.copy()`/reconstruct a 100+GB frame" convention; the pre-existing `_footprint_bytes` RAM-headroom check earlier in the same function only budgets for the eventual int32-binned working set, not this separate, unaccounted validation-time copy, so a frame sized right at that guard's boundary can still OOM here. | New — not raised in the 2026-07-20 audit or the 2026-07-21 repo-wide audit (grepped both; no hits for `_validate_inputs`/`select_dtypes` in either). |
| USABILITY_B-3 | P1 | bug (silent-wrong-result) | `_mrmr_partial_fit.py:268-280` | On a recompute, `partial_fit`'s sample_weight reconciliation only raises when the caller-supplied `sample_weight` is **too short**; when it is longer than the new batch (the common case when `partial_fit_window` is left at its default `None`, so no window truncation of the current batch can ever legitimately explain a length mismatch), the code silently takes `sw_new[-kept_new:]` and applies it — misattributing weights to the wrong rows with zero warning, instead of raising the same actionable `ValueError` the "too-short" branch already provides. Zero test coverage exists for `partial_fit`'s `sample_weight` argument at all (`test_coverage_api_partial_fit.py` never exercises it). | New — not raised in the 2026-07-20 audit (`edge_cases.md:173` covers a different `partial_fit` dtype-upcast edge case in the same file, not this one). |
| USABILITY_B-4 | P2 | efficiency / code quality | `_mrmr_partial_fit.py:129-153` (`_apply_rolling_window`) | The first loop (building `new_sizes`) already correctly computes the post-truncation batch-size registry (once `drop_remaining` hits 0 it appends every subsequent batch's full size unchanged, so trailing batches ARE handled) — but its result is discarded entirely, and a second, essentially-identical loop rebuilds the same list into `rebuilt` (the one actually returned), justified by a comment ("walk again to pick up everything past the consumed prefix") that is not actually necessary given the first loop's own logic. Dead computation + duplicated logic from what reads like an incomplete refactor. | New. |
| USABILITY_B-5 | P2 | design / API contract | `_mrmr_fingerprints.py:562-591` (`_replay_fitted_state`) | When a second `MRMR().fit(X, y)` call hits the `_FIT_CACHE` (identical params + content as an earlier fit `A`), the replay logic freezes (`v.flags.writeable = False`) `A`'s own large ndarray fitted attributes in place, as a side effect purely of servicing the *second* instance's replay. This is deliberate and already regression-tested from the replay target's perspective (`tests/feature_selection/mrmr/core/test_replay_fitted_state_isolation.py::test_large_internal_ndarrays_still_shared_for_density`), but the retroactive effect on the *source* — a previously fully-mutable, already-returned user object silently losing in-place-write capability on its own attributes because of an unrelated later `.fit()` call elsewhere in the same process — is not documented anywhere in `MRMR`'s public docstring/API contract. A user relying on `A.some_ndarray_attr[i] = x` working (as it does immediately after a fresh fit) can see it start raising `ValueError: assignment destination is read-only` with no code change on their part. | New. |
| USABILITY_B-6 | P2 | efficiency | `_mrmr_fe_provenance.py:280-301` (`_origin_from_rosters`) | For every produced-but-unmatched engineered name (can be hundreds on a kitchen-sink wide FE fit whose greedy screen keeps only a small survivor subset), the fallback roster lookup re-converts each of the 13 `_ROSTER_ATTR_TO_ORIGIN` attributes to a fresh `list(...)` and does an O(len(roster)) `in`-membership test from scratch, per name, instead of building the 13 memberships once (as sets) before the loop. Straightforward O(n·m) → O(n+m) fix with no behavioural change. | New. |
| USABILITY_B-7 | P2 | test coverage | `_mrmr_sis_apply.py` (whole file, 94 LOC) | `_apply_sis_screen`'s polars-specific branches (the `X.to_numpy()` + non-numeric-factorize fallback, and the final `X[:, survivors.tolist()]` subsetting) have zero test coverage. `tests/feature_selection/mrmr/core/test_mrmr_sis_screen.py` exercises the pandas-DataFrame path (including its P0-2 non-numeric-column fix) and the raw-ndarray `sis_screen` kernel directly, but never drives `_apply_sis_screen` with a polars `DataFrame`, despite polars being a fully-supported, actively-optimized MRMR input type elsewhere in the codebase. | New — not raised in the 2026-07-20 audit's `test_coverage.md` (no mention of `_mrmr_sis_apply.py` at all). |
| USABILITY_B-8 | P2 | logging discipline | `_mrmr_explain.py:232-251` (`explain_selection`) | Five `except Exception as exc:` blocks embed the caught exception's type name into the returned narrative string (e.g. `"Surviving features: (unavailable: KeyError)."`) but never call `logger` at all, not even at debug level — unlike every other file in this cluster, which logs on every swallowed exception per the project's "no silent except-Exception swallowing without logging" convention. The deliberate never-raise contract for this reporting helper is reasonable, but a production failure here (e.g. a corrupted `fe_provenance_` DataFrame) is invisible to log-based monitoring; only a human reading the returned string in a notebook would ever see it. | New. |
| USABILITY_B-9 | P2 | licensing (vendored code) | `_vendored/infonet/__init__.py:1`, `_vendored/__init__.py:1`, whole `_vendored/infonet/` tree | No `LICENSE`/`NOTICE` file exists anywhere under `_vendored/` for the vendored InfoNet code (Hu et al., ICML 2024, `github.com/datou30/InfoNet`) — only a one-line attribution comment naming the source repo. `pyproject.toml` references the vendored path only for packaging (`configs/*.yaml` data files) and the import-linter isolation contract, neither of which records the upstream license terms. CLAUDE.md's own review checklist calls out licensing specifically for vendored code; this is a genuine compliance gap for anyone redistributing mlframe as a library. | New. |
| USABILITY_B-10 | P2 | security / design (sys.path hygiene) | `_vendored/infonet/infer.py:6-9` (consumed via `sys.path.insert(0, ...)` in `_neural_mi.py:249-251` / `:347-349`, outside this file set) | `infer.py`'s top-level absolute imports (`from model.decoder import Decoder`, `from model.encoder import Encoder`, `from model.infonet import infonet`, `from model.query import Query_Gen_transformer`) only resolve because the caller in `_neural_mi.py` permanently prepends `_vendored/infonet/` to the process-global `sys.path` on first use and never removes it. This makes the generic top-level names `model`, `util`, `query`, `decoder`, `encoder`, `attention`, `attention_block`, `gauss_mild` importable process-wide for the rest of the interpreter's lifetime, at `sys.path[0]` (highest priority) — any other code in the process that later does e.g. `import model` or `import util` expecting an unrelated same-named module would silently resolve to the vendored InfoNet internals instead. No collision exists in mlframe's own tree today (`find`-verified), but this is a live landmine for the embedding host application or a future first-party module of the same name. | New. Not a bug in this file per se (the vendored `model/*.py` submodules correctly use *relative* imports internally; only `infer.py`'s own top-level imports are absolute, which is what forces the sys.path hack at the call site) — flagged here because the hazardous consumption pattern is required specifically by this file's import style. |

### Explicitly checked, no issues found

- **`_relaxmrmr_3d.py`** (RelaxMRMR/FJMI 3-D-MI, Vinh 2016): formula, njit kernels, and code-range guards
  (`_assert_codes_in_range`) all check out; its documented `O(|S|^2)` per-candidate cost is an intentional,
  already-disclosed research-grade tradeoff (opt-in, gated behind a pruned candidate pool per the module's own
  docstring), not a hidden complexity bug. Reasonable test coverage exists (`test_relaxmrmr_3d_redundancy_signs.py`,
  `test_relaxmrmr_hoist_equivalence.py`, plus several biz_val research-knob tests).
- **`_mrmr_artifacts.py`**: `compute_mrmr_artifacts` / `validate_artifact_dict` handle `n_samples=0`,
  `n_features_in=0`, all-NaN/constant target and feature columns correctly (denominator floors avoid div-by-zero,
  NaN is left rather than a fabricated 0). No new issues beyond the already-known, still-open P2 coverage gap for
  `export_artifacts()`'s two raise paths (`c1_core_class.md`).
  test coverage.
- **`_mrmr_sis_screen.py`** (the SIS kernel itself, as opposed to its glue in `_mrmr_sis_apply.py`): chunk-width
  selection is RAM-derived via `kernel_tuning_cache` (no hardcoded threshold), the redundancy-dedup pass correctly
  recovers original column indices through the string-keyed `corr_clusters` join, and `p=0`/`n=0` edge cases degrade
  gracefully via the existing per-block try/except.
- **`_mrmr_tree_rescue.py`** / **`_mrmr_fe_step_helpers.py`**: no new correctness issues found beyond the
  already-known, still-open P2 reversed-seed-precedence finding (`c1_core_class.md:25-29`, unchanged since
  2026-07-20 — verified via `git log`, no commit since has touched `_mrmr_tree_rescue.py`'s seed line). The FE-step
  helpers module is exceptionally well-instrumented (every swallowed exception logs a warning with `exc_info=True`);
  no silent-failure gaps found.
- **`_mrmr_passthrough.py`**: embedding/free-text detection sampling is correctly bounded (`O(_SAMPLE_ROWS)` per
  column regardless of frame height), degrades safely on all-NaN columns, and is exercised by dedicated tests
  (`test_mrmr_embedding_passthrough.py`, `test_biz_val_mrmr_embedding_passthrough.py`).
- Vendored InfoNet `model/*.py` (`attention.py`, `attention_block.py`, `decoder.py`, `encoder.py`, `gauss_mild.py`,
  `infonet.py`, `query.py`, `util.py`): grepped for `eval(`/`exec(`/`os.system`/`subprocess`/`pickle`/`__import__` —
  none present. `infer.py`'s `torch.load(..., weights_only=True)` is already hardened against a tampered checkpoint
  executing arbitrary code via unpickling (with a documented, justified fallback for torch<1.13). `load_config`
  uses `yaml.safe_load`, not the unsafe `yaml.load`. No security findings in the vendored logic itself.

## Proposals

- **Gate `audit_degenerate_columns` on `p`** (relates to USABILITY_B-1): cap the collinearity pass to the first
  N columns (or skip it) above a configurable `p` threshold, or make the whole diagnostic opt-in via a
  constructor flag (mirroring `retain_artifacts`) rather than always-on — the scan already documents itself as
  "PURELY DIAGNOSTIC", so a size-gated/opt-in default costs nothing for existing users who never inspect
  `degenerate_columns_`.
- **Chunked/columnar inf-check** (relates to USABILITY_B-2): replace the whole-frame `.to_numpy()` copy with a
  per-column (or per-block) scan that never materializes more than one column/block at a time, and skip
  non-float dtype columns before any array construction (not just before the `isinf` call) — this alone removes
  the always-on cost that any all-integer frame currently pays for nothing.
- **Raise unconditionally on `partial_fit` sample_weight length mismatch when no window truncation occurred**
  (relates to USABILITY_B-3): only apply the trailing-slice recovery when `window is not None and len(X_df) >
  batch_sizes[-1]` (i.e. an actual truncation of the current batch happened); otherwise any length mismatch is a
  genuine caller error and should raise immediately. Add a dedicated regression test.
- **Document the cache-replay source-freeze side effect** (relates to USABILITY_B-5) in `MRMR`'s class docstring
  under a "Caching" section: "a previously-fitted MRMR instance's internal ndarray attributes may become
  read-only if a later `.fit()` call elsewhere in the process replays from it via `_FIT_CACHE`".
- **Direct polars test for `_apply_sis_screen`** (relates to USABILITY_B-7): a small polars-DataFrame fixture
  (mixed numeric + one string categorical column) run through `MRMR()._apply_sis_screen(pl_df, y)` directly,
  mirroring the existing pandas non-numeric-column test.
- **Vendor a LICENSE file alongside the InfoNet code** (relates to USABILITY_B-9): copy the upstream
  `github.com/datou30/InfoNet` repository's license file (once confirmed) into `_vendored/infonet/LICENSE` and
  add a short `NOTICE` naming the exact commit/version vendored, so the compliance chain is self-contained
  rather than resting on a single source-URL comment.
- **Scope the InfoNet `sys.path` injection** (relates to USABILITY_B-10): wrap the `from infer import ...` call
  in a `try/finally` that restores `sys.path` to its prior state immediately after the one-time import completes
  (the loaded module objects stay cached in `sys.modules`/the model cache regardless), removing the permanent
  global namespace exposure of `model`/`util`/`query`/etc.

# Meta-test proposals: catching these bug classes by class, not by instance

Produced by a read-only research pass over all 39 reports in this audit, after the full fix-wave closed
every individual finding. Goal: static/behavioral repo-wide checks that would have caught whole
*classes* of the bugs found here automatically, so the next occurrence doesn't need a full manual audit.
Ordered by value/cost (cheapest + highest historical hit-count first). None of these are implemented yet
-- this is the proposal list; implementing them is a separate, explicitly scoped follow-up.

## 1. Duplicate seed-derivation / hash-based helper functions
AST-grep functions whose body uses `hashlib.sha256`/`blake2b`/`md5` + a numeric mask/modulo, or whose
name matches `_?derive.*seed|_?per_.*_seed|_?seed.*from`; cluster by structural similarity; flag any
cluster of size >=2 that isn't literally the same function reused. Same class as the already-known
`store_params_in_object` postfix bug (13 call sites): one canonical helper, N ad-hoc reimplementations.
Would have caught: `x_architecture_api_consistency.md` F7. Cost: cheap, static, no execution.

## 2. `except ...:` around a test's actual call-under-test (skip masks regression)
AST-walk `tests/**/*.py` for try/except blocks whose except body calls `pytest.skip(...)` and whose try
body contains more than an import (i.e. calls the function under test), plus module-level
`except Exception: pytest.skip(..., allow_module_level=True)` wrapping multiple imports.
Would have caught: `x_test_suite_architecture.md` F1, F2, F3. Cost: cheap, static AST over `tests/`.

## 3. Tautological assertions (`assert x is not None` as the only check)
AST-walk every `test_*` function; flag ones whose only assert(s) are `is not None`/`!= None`/bare
truthiness with no attribute/behavioral assertion after. Finishes the job of the grep
`x_test_suite_architecture.md` already ran (661 hits) by adding the "is this the ONLY assertion" check.
Would have caught: `x_test_suite_architecture.md` F7 (4 stopfile-callback smoke tests). Cost: cheap.

## 4. Filler/auto-generated docstrings satisfying a coverage gate without content
Grep `tests/**/*.py` for docstrings matching `"""Helper that <verb>."""` or near-verbatim restatements of
the function name (token-overlap threshold); fail CI on *new* filler-pattern docstrings (ratchet).
Would have caught: `x_test_suite_architecture.md` F9 (499 occurrences / 369 files). Cost: cheap regex.

## 5. `sample_weight` silently dropped along a call chain
For every public function accepting `sample_weight`/`sw`, AST-trace whether it's used inside the body
and forwarded on every call to an inner estimator/metric that itself accepts it. Flag accept-but-ignore
and any internal call to a `sample_weight`-aware sibling omitting it despite a non-None value in scope.
This is the single highest-yield class found -- 10+ independent findings across 8 report files:
`x_ml_correctness_meta.md` F1/F2/F4/F7, `training_core_a.md` F1, `training_composite_loose_b.md` F3,
`training_loose_b.md` F2/F8, `models_all.md` F3/F4, `training_neural.md` F4,
`training_feature_handling.md` F6. Cost: medium (dataflow script, no execution needed) -- worth a
dedicated script given the repeat-offender count.

## 6. `from .submodule import *` with no curated `__all__` anywhere in the chain
For every `__init__.py` doing `from .x import *`, verify either the `__init__.py` itself defines
`__all__` or every star-imported submodule does. Flag if neither.
Would have caught: `x_architecture_api_consistency.md` F5 (8 packages), `fe_top_b.md` F5/F6/F7,
`reporting_charts.md` F6. Cost: cheap, static AST + `__all__` presence check.

## 7. Layering violations -- lower-level package importing a higher-level one
Explicit layer-order list (`core < preprocessing/metrics/calibration < feature_engineering < models <
training`); AST-scan every module-level import; flag anything pointing "upward". Already proposed inside
the audit itself as PR3 (`import-linter`/`tach`-style contract, ~10 lines of config, off-the-shelf tool).
Would have caught: `x_architecture_api_consistency.md` F1, F2/F3, F4. Cost: cheap-medium, no custom code.

## 8. Missing `__getstate__`/`__setstate__` on classes holding unpicklable live resources
AST-walk every class `__init__` for attribute assignments matching `threading.Lock()`/`RLock()`,
`torch.cuda.*`, open file handles, or other live third-party resource types; if found, verify the class
(or an ancestor) defines `__getstate__`. Matches memory note `feedback_runtime_caches_break_pickle`.
Would have caught: `x_security_robustness.md` SEC-3 (`FeatureCache._lock`) -- the exact same class of bug
already patched once in `training/neural/ranker.py`'s trainer/CUDA-tensor exclusion, meaning that fix was
never generalized into a repo-wide guard. Cost: cheap, static AST scan.

## 9. Hardcoded seed instead of threading the caller's `random_state`
Grep/AST for numeric-literal `random_state=`/`seed=0`/`np.random.RandomState(0)`/`random_state=42` inside
functions that themselves receive a `random_state`/`config.random_state` param but don't forward it to an
inner call that also accepts `random_state`.
Would have caught: `training_composite_discovery.md` F1/F2, `training_composite_blocks.md` F3,
`training_composite_loose_b.md` F8/F9, `training_baselines.md` F9/F10. Cost: cheap-medium (grep + light
dataflow: is there an unused `random_state` in scope at this call site).

## 10. Silent broad `except Exception:` swallowing real bugs (general census)
Repo-wide AST scan for bare `except:`/`except Exception:` blocks whose body returns a default, `pass`s,
or `continue`s without a `logger.warning`/`logger.error` call inside. `x_architecture_api_consistency.md`
already ran a partial version (9 hits, 3 packages) but didn't turn it into an enforced ratchet.
Would have caught: `fe_top_b.md` F3, `reporting_charts.md` F3, `training_composite_loose_b.md` F11,
`x_security_robustness.md` SEC-2. Cost: cheap static AST scan, same pattern as the existing
`test_no_bare_except.py` -- extend its scope repo-wide instead of leaving it partial.

## 11. Docstring/contract drift on return-tuple arity
Cheap partial version only: AST-extract each public function's docstring "Returns:" tuple-arity claim,
compare against actual `ast.Tuple` arity in `Return.value`. Flag mismatches. (Deeper semantic drift, e.g.
"documented as block-diagonal, actually dense," isn't automatable -- stays a periodic manual-audit item.)
Would have caught (arity subset): `fe_top_a.md` F16 (3-tuple documented, 2-tuple returned). Cost: medium.

## 12. Missing `timeout-minutes` on high-stakes CI jobs
YAML-parse every `.github/workflows/*.yml` job; flag any job lacking `timeout-minutes`. Already proposed
inside the audit as PR1, matching this repo's existing `test_meta/` ratchet-test culture.
Would have caught: `x_cicd_dependencies.md` F2 (`ci.yml` build), F3 (`release.yml` publish -- the
highest-stakes job in the repo). Cost: trivial, ~15-line PyYAML script.

## Deliberately excluded from the top-12 (noted, not silently dropped)
- **Perf-only findings** (unvectorized per-row loops, unnecessary `.copy()`/`.deepcopy()` inside loop
  bodies) recur constantly (`fe_top_a.md` F19/F21, `fe_top_b.md` F2/F8/F9, `preprocessing.md`
  F3/F5/F6, `training_neural.md` F12, `training_loose_b.md` F1) but don't reduce to one crisp static
  pattern without a high false-positive rate. A plausible #13 (`.copy()` nested inside a `for`/`while`
  AST node) exists but should be staged only after the top 12 prove out.
- **`metrics.py` facade-bypass** (`x_architecture_api_consistency.md` F9: facade exists but most call
  sites import submodules directly) folds into #7's import-linter contract for free once that's in place
  -- no separate tool needed.

## Cross-referenced against pyutilz / py-ci-shared / mlframe's existing meta-tests

A second read-only pass inventoried the existing meta-test infrastructure across the three sibling repos
before treating any of the 12 proposals above as genuinely net-new. `pyutilz/src/pyutilz/dev/code_audit/`
is a reusable AST-scanner registry (~34 scanners, run via `run_all(root, checks, exclude_dirs)`) that
`py-ci-shared/src/py_ci_shared/code_audit_meta.py` wraps into a baseline-gated harness, which mlframe's
own `tests/test_meta/test_code_audit_baseline.py` already wires in. This is the natural home for new
cross-repo rules rather than one-off mlframe-local AST scripts.

| # | Proposal | Verdict | Where |
|---|---|---|---|
| 1 | Duplicate seed-derivation helpers | NET-NEW | no matching scanner in pyutilz `code_audit/` |
| 2 | `except:`+`pytest.skip` masking the real call-under-test | NET-NEW | closest is mlframe's own one-off instance-pinning test, not a generic rule |
| 3 | Tautological `is not None`-only assertions | EXISTS-PARTIALLY | `pyutilz/.../code_audit/vacuous_assertions.py` catches `assert True`/same-target boolops, but not "only assertion is bare `is not None`" specifically |
| 4 | Filler/auto-generated docstrings | NET-NEW | `docstring_args.py`'s `docstring_args_incomplete` checks a different axis (Args-section coverage) |
| 5 | `sample_weight` dropped along a call chain | NET-NEW | no parameter-forwarding dataflow scanner exists anywhere in the three repos |
| 6 | `import *` with no `__all__` | NET-NEW | mlframe's own `x_architecture_api_consistency` fixes file already logs this as assessed-proposal-only, no tooling built |
| 7 | Layering/import-linter | NET-NEW | same file confirms: assessed as a tooling suggestion, no `import-linter`/`tach` config exists anywhere yet |
| 8 | Missing `__getstate__` on classes holding live resources | NET-NEW, highest-leverage | no scanner anywhere checks this; directly matches the SEC-3 `FeatureCache._lock` bug this wave already fixed once by hand |
| 9 | Hardcoded seed instead of threaded `random_state` | NET-NEW | no seed-forwarding scanner exists |
| 10 | Broad `except Exception:` silent-swallow census | **ALREADY DONE** | `pyutilz/.../code_audit/broad_except.py`'s `scan_broad_except_swallows` is repo-wide, P1-severity, and already wired into mlframe via the baseline-gated harness; mlframe's own `test_no_bare_except.py` extends it further (verbose-gated logging). This proposal is effectively closed already -- just wasn't labeled as such when originally proposed. |
| 11 | Docstring/return-tuple arity drift | EXISTS-PARTIALLY | `return_annotation.py`'s `scan_return_annotation_mismatch` checks the type ANNOTATION vs actual returns, not docstring "Returns:" text vs actual arity -- adjacent, not the same check |
| 12 | Missing `timeout-minutes` on CI jobs | NET-NEW (adjacent tool exists) | `py-ci-shared/.../ci_workflow_gate.py` is the same regex-YAML-scanner family (currently only checks `continue-on-error: true` review status) -- natural place to add a sibling function |

### Reverse direction: existing shared-repo checks mlframe should adopt but hasn't wired in

1. `py_ci_shared.loc_budget.assert_no_new_oversized_file` -- mlframe has a **hand-rolled duplicate**
   (`tests/test_meta/test_no_file_over_1k_loc.py`) instead of consuming the shared harness this module
   was specifically extracted to prevent drifting from.
2. `py_ci_shared.ci_workflow_gate.assert_continue_on_error_is_reviewed` -- not wired into mlframe despite
   9 workflow files under `.github/workflows/` and `x_cicd_dependencies.md`'s own finding history of
   exactly this silent-gate-defeat failure mode.
3. `py_ci_shared.entry_points_resolvable` -- not wired despite mlframe's `pyproject.toml` declaring CLI
   entry points this checker would validate.
4. `py_ci_shared.config_drift_check` / `config_call_site_parity` -- not cross-checked against mlframe's
   own bespoke `test_subconfig_wiring_parity.py`; worth a diff-read to see if one subsumes the other.
5. `py_ci_shared.phantom_markdown_links` -- not wired despite mlframe's large `audits/`/README/CHANGELOG
   Markdown surface with internal links that can rot.
6. pyutilz's `test_lock_discipline_consistency.py` / `test_resource_reassignment_without_close.py` /
   `test_thread_daemon_or_lifecycle_managed.py` -- no mlframe counterpart, yet SEC-3's `FeatureCache._lock`
   is exactly the risk class these three guard against in pyutilz. mlframe's own
   `test_resource_handle_safety.py` only tracks `open`/`Popen`/tempfile, not `threading.Lock`/`RLock`.
7. pyutilz's `test_cache_bounded_or_invalidatable.py` / `test_dict_cache_has_eviction.py` -- no mlframe
   equivalent, despite mlframe having multiple in-process dict-caches (`kernel_tuning_cache`,
   `FeatureCache`, MRMR's fit cache) that are exactly this risk shape.

Points 6/7 and proposal #8 all trace back to the same `FeatureCache`/lock class this wave already fixed
once by hand (SEC-3) -- implementing #8 as a pyutilz `code_audit` scanner would close the reverse-direction
gap and the forward proposal in a single addition. This is the single highest-leverage item on this list.

## Status
Proposal-only, per the original request ("think about what meta-tests could be designed to catch these
bug classes in advance"). None implemented yet except #10, which turned out to already exist and be wired
in -- discovered only through the cross-reference pass, not assumed. Highest-priority next-implementation
set, folding in both passes: **#8** (closes 3 gaps at once, cheapest to build as a pyutilz scanner), then
#2, #3 (already a partial scanner to extend), #6, #12 (adjacent tool to extend).

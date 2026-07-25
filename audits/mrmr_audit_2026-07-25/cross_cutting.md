# Cross-cutting (whole-MRMR-surface) — audit (2026-07-25)

Surface-wide checks over `src/mlframe/feature_selection/filters/` (337 `.py` files, excluding `_benchmarks/`
and `_vendored/`): zero test coverage, file-over-1k-LOC gate, mypy, ruff, leftover audit-metadata comments,
security/API surface, implicit-Optional signatures, and `--`-in-prose. No single file-cluster agent owns these.
Verified against current source at the working tree (git `d8091a138` base).

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| CROSS-1 | P1 | house-conv / CI gate | `filters/evaluation.py` (1003 LOC) | The enforced 1k-LOC gate `tests/test_meta/test_no_file_over_1k_loc.py` is **currently RED**: `evaluation.py` is at 1003 LOC and is NOT in `LOC_BUDGET_EXEMPT`. Prior wave (`x_efficiency_architecture.md`) recorded it at 997 (Tier-2, under gate); it regrew past 1000 since. | `python -m pytest tests/test_meta/test_no_file_over_1k_loc.py` → `AssertionError: 2 mlframe .py file(s) exceed 1000 LOC` listing `filters/evaluation.py` 1003. Carve via sibling re-export (e.g. lift the report-assembly/formatting block into `_evaluation_report.py`). |
| CROSS-1b | P1 | house-conv / CI gate | `feature_selection/shap_proxied_fs/_shap_proxied_fit.py` (1012 LOC) | Second breach in the same RED gate run. Just outside `filters/` but on the MRMR feature-selection surface, so flagged here so it isn't missed. Not in the exempt set. | Same gate test lists it at 1012 LOC. Carve the holdout/clustering-prefilter or per-model scoring block into a sibling. |
| CROSS-2 | P3 | house-conv (comment metadata) | 273 files / 2254 lines across `filters/` | Leftover audit-metadata comments (date stamps `2026-0*`, `Wave N`, `loop iter N`, `Layer N` audit markers) that CLAUDE.md bans from code comments. The prior ~178-file cleanup was partial; a large residue remains. Concrete per-file counts below — this is the implementer's cleanup master list. | e.g. `_fit_impl_core.py:171` `# 2026-05-30 Wave 9.1 fix (loop iter 36): ...`; `:401` `# 2026-05-31 Layer 23 — hybrid orthogonal-polynomial...`. These are process/history narration, not WHY comments. |
| CROSS-3 | P3 | house-conv (implicit-Optional) | `filters/_confirm_predictor_context.py:97,102,103,104` | Four dataclass fields declared `candidates: list = None` / `selected_vars: list = None` / `selected_interactions_vars: list = None` / `partial_gains: dict = None`, each suppressed with `# type: ignore[assignment]`. CLAUDE.md bans `param: T = None`; the correct form is `Optional[list] = None` (or `field(default_factory=list)`), removing the need for the ignore. | Grep-confirmed only site in `filters/`; mypy passes only because of the suppressions. |
| CROSS-4 | P3 | house-conv (`--` in prose) | 5141 lines across `filters/` | `--` (double-dash) used in prose comments/docstrings, banned by CLAUDE.md ("single ` - ` or recast"). Worst offenders: `mrmr/_mrmr_class.py` (246), `_fit_impl_core.py` (239), `_pairs_score.py` (50), `_pairs_core.py` (48), `_step_score.py` (42). Informational / large-scale; a mechanical recast risks touching many files, so treat as a sweep, not per-line. | e.g. `_mrmr_class.py` docstring `... the ``MRMR`` estimator -- after ...`. |

### CROSS-2 detail — leftover audit-metadata comments, top files by line count

(pattern set: `finding #`, `backlog #`, `mrmr_audit_20`, `Wave N`/`wave N`, `loop iter`, `Layer N`, `2026-0*` date stamps, `.md finding`, `suppressed in <file>:<line>`)

```
 232  _mrmr_fit_impl/_fit_impl_core.py
  91  mrmr/_mrmr_class.py
  42  _mrmr_fe_step/_step_core.py
  38  _screen_predictors.py
  35  _orthogonal_univariate_fe/__init__.py
  32  _mi_greedy_cmi_fe.py
  32  _feature_engineering_pairs/_pairs_core.py
  30  _prewarm.py
  27  discretization/__init__.py
  26  evaluation.py
  26  _mrmr_fe_step/_step_score.py
  25  _gpu_resident_fe.py
  22  _orthogonal_scorer_auto_fe.py
  22  _orthogonal_meta_scorer_fe.py
  22  _orthogonal_cmim_fe.py
  22  _fe_batched_mi.py
  22  _dynamic_cluster_discovery/_dcd_swap.py
  21  _feature_engineering_pairs/_pairs_score.py
  21  _dynamic_cluster_discovery/__init__.py
  20  engineered_recipes/_recipe_dispatch.py
  20  engineered_recipes/_orth_basis_recipes.py
  20  _cat_pair_fe.py
  19  permutation.py
  18  batch_mi_noise_gate_gpu.py
  18  _orthogonal_hsic_fe.py
  18  _orthogonal_cluster_basis_fe.py
  17  _orthogonal_quadruplet_fe.py
  17  _hermite_fe_optimise_pair.py
  17  _fe_raw_redundancy_drop.py
  16  _usability_aware_selection.py
```
Total: 273 files, 2254 matching lines. `Layer N` date-stamped stage markers dominate `_fit_impl_core.py`; if the
team decides `Layer N` is a legitimate architecture name (not an audit marker) the count drops, but the co-located
`2026-0*` date stamps and `Wave N`/`loop iter N` on the same lines are unambiguously banned and must go regardless.

## Non-findings / confirmed-clean angles

- **Test coverage — CLEAN (major result).** Enumerated all 337 `filters/` modules; grepped `tests/` for each module
  basename AND every `def`/`class` name (public and private) it defines. **Zero modules have zero test references** —
  every module is reached by basename or by at least one module-unique defined name. A stricter pass (modules whose
  basename is absent AND no *module-unique* def-name appears, i.e. covered only incidentally via shared/generic names)
  also returned **0**. The prior `x_test_coverage_quality.md` gaps appear closed. (Caveat: this measures module-reference
  granularity, not per-branch/per-bug-path coverage — the per-cluster agents own specific untested bug paths.)
- **mypy — CLEAN.** `python -m mypy --ignore-missing-imports --cache-dir=../.mlframe_mypy_cache_shared src/mlframe/feature_selection/filters/` → `Success: no issues found in 348 source files`.
- **ruff — CLEAN.** `python -m ruff check src/mlframe/feature_selection/filters/ --ignore C901` → `All checks passed!`.
- **Security / API surface — CLEAN.** No SQL, no HTTP/requests/urllib/socket, no `os.system`, no `subprocess` on any
  input, no `eval()`/`exec()`/`__import__`/`yaml.load` on untrusted data (the only `eval` hits are docstrings describing
  a basis `eval(z,c)` closure contract and `torch...().eval()`), no `pickle.loads` of network state. All 161 env-var
  reads are `MLFRAME_*`-prefixed config toggles (plus standard `CUDA_VISIBLE_DEVICES`) — config-only, no injection.
- **File-over-1k — checked exhaustively.** 11 `filters/` files exceed 1000 lines; 10 are correctly in the exempt set
  (`_fit_impl_core.py` 10056, `mrmr/_mrmr_class.py` 4173, `_mi_greedy_cmi_fe.py` 1876, `_gpu_resident_basis.py` 1599,
  `discretization/__init__.py` 1355, `_pairs_core.py` 1292, `_pairs_score.py` 1276, `_gpu_resident_fe.py` 1184,
  `_step_score.py` 1181, `_step_core.py` 1060). Only `evaluation.py` (1003) is unexempt → CROSS-1. `_cat_confirm_permutation.py`,
  flagged by the prior wave at 1097, is now carved to 952 (RESOLVED).

## Prior cross-cutting items — resolution status

- `x_efficiency_architecture.md` **X_EFFICIENCY_ARCHITECTURE-1** (`_cat_confirm_permutation.py` 1097 LOC breaching the gate): **RESOLVED** — now 952 LOC. But the same gate is red again for a *different* file (`evaluation.py`, regrown 997→1003) → CROSS-1; the "carve-then-regrow to the edge" pattern the prior wave warned about recurred exactly as predicted.
- `x_test_coverage_quality.md` zero-coverage gaps: **RESOLVED** at module granularity (0 unreferenced modules this pass).
- `x_security_api_packaging.md` (no injection/DB/HTTP surface): **CONFIRMED still clean.**

## Proposals (perf / refactor / test — not bugs)

1. **CROSS-2 cleanup sweep**: script the removal of `2026-0*` date-stamp / `Wave N` / `loop iter N` fragments from
   comments across the 273 files (leave the WHY prose, strip the metadata prefix). Gate with a new meta-test
   (`test_no_audit_metadata_in_comments.py`) grepping the banned pattern set so the residue can't regrow — mirrors the
   1k-LOC gate's role. ~2254 lines; do as one reviewed pass, not incrementally.
2. **`_neural_mi.py` network-deserialization note (informational)**: `check_neural_mi`/`estimate_mi_mist` calls
   `MISTForHF.from_pretrained("grgera/MIST"|"grgera/MIST-QR")` which downloads a safetensors checkpoint from the HF hub
   on first use (opt-in, not a default MRMR path; safetensors is code-execution-safe). Not a finding, but worth a
   one-line docstring note that the estimator performs a network fetch + on-disk cache to `~/.cache/huggingface/hub`.

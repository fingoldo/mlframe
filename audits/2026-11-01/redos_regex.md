# REDOS2: Regex and ReDoS audit, mlframe (2026-11-01)

Scope: every `re.compile` / `re.match` / `re.search` / `re.sub` / `re.findall` / `re.split` / `re.finditer` call under `src/mlframe` (about 110 call sites found by grep, including `_re.` aliases). Read-only. No payloads were run. Each call site was classified by whether its input can come from outside the process, then its pattern was checked for nested or overlapping quantifiers.

Summary: 0 high, 0 medium, 2 low, 4 info. No runtime site takes an attacker-controlled pattern into `re.compile` without a config-trust boundary. Two findings are real quadratic backtracking on crafted strings, but both are reachable only through config or environment strings.

## Findings

### RDS2-1 (low): mlflow userinfo redaction, quadratic on long alphanumeric runs
- Location: `src/mlframe/integrations/mlflow.py:36`
- Pattern: `(?i)([a-z][a-z0-9+\-.]*://)[^/@\s]+@`
- Input source: MLflow tracking/registry URI (environment variable or config). Logged or redacted before output.
- Why it blows up: the scheme group `[a-z][a-z0-9+\-.]*` is unanchored under `re.sub`. A long run of scheme-class characters with no `://` after it makes every start position consume the rest of the run and then backtrack, so the cost is O(n^2) in the run length. Only strings with a very long scheme-like token trigger it.
- Proposed disposition: FUTURE. Reason: real but needs a long crafted URI from config or env. Cheap fix: anchor with `\A` / `re.match`, or use `urllib.parse.urlsplit` for the netloc redaction.

### RDS2-2 (low): plot_outputs DSL clause regex, overlapping whitespace quantifiers
- Location: `src/mlframe/reporting/output.py:55`
- Pattern: `^\s*([A-Za-z]+)\s*\[\s*([A-Za-z0-9_,\s]+?)\s*\]\s*$`
- Input source: the `plot_outputs` DSL string, from the suite config (also settable by callers). Each clause is passed to `_CLAUSE_RE.match` after splitting on `+`.
- Why it blows up: the lazy class `[A-Za-z0-9_,\s]+?` includes `\s`, and it is followed by `\s*\]`. A whitespace run inside the bracket can be divided between the lazy class and the trailing `\s*` in many ways. With no closing `]`, the match backtracks O(k^2) in the length k of the whitespace run. The `^` anchor limits it to one start position.
- Proposed disposition: FUTURE. Reason: quadratic only on a very long whitespace run inside an unterminated bracket, from a config string. Fix: drop `\s` from the class (`[A-Za-z0-9_,]+?`) and strip whitespace in Python before matching. The format list is already `.strip()`-ed on split, so behaviour is unchanged.

### RDS2-3 (info): column-exclusion regex from user config
- Location: `src/mlframe/preprocessing/_cleaning_helpers.py:188` (compile), `:351` (`.search(col)`)
- Pattern: user-supplied `exclude_mask` string, compiled as-is.
- Input source: the caller's `exclude_mask` argument (config). The column names are the untrusted-ish part, not the pattern.
- Why it blows up: depends entirely on the pattern author. A nested pattern such as `(a+)+$` would backtrack on long column names. Column names are short and the pattern is supplied by the same operator who runs the pipeline.
- Proposed disposition: DOC. The pattern is an operator-trusted regex, not data. Document that `exclude_mask` is a regex and that nested quantifiers are the caller's responsibility.

### RDS2-4 (info): composite-target forbidden base patterns from config
- Location: `src/mlframe/training/composite/discovery/__init__.py:163` (compile), used by `src/mlframe/training/composite/discovery/_filter.py:298` and `:360` (`p.search(c)` on feature column names).
- Pattern: `config.forbidden_base_patterns` (a `List[str]` on the pydantic model, `_composite_target_discovery_config.py:166`).
- Input source: config, not data. The matched strings are column names.
- Why it blows up: same as RDS2-3. Patterns are compiled once per discoverer with no safety check.
- Proposed disposition: DOC. Operator-supplied regex. Note in the field description that patterns must be linear-time (no nested quantifiers).

### RDS2-5 (info): calibration include/skip patterns, lru_cached compile
- Location: `src/mlframe/calibration/post.py:31-33` (`_compile_pattern`, `lru_cache(maxsize=128)`), used at `:313` and `:315` with `.search(name)`.
- Pattern: `include` / `skip` lists supplied to the postcalibrator selection API.
- Input source: caller-supplied config. The names are internal calibrator names.
- Why it blows up: as RDS2-3. The cache has no effect on backtracking.
- Proposed disposition: DOC. Operator-trusted regex. Same documentation note as RDS2-3.

### RDS2-6 (info): per-group aggregate column pattern (MLP extreme-AR protection)
- Location: `src/mlframe/training/core/_phase_train_one_target_mlp_helpers.py:43-58`, `_get_per_group_pattern` (IGNORECASE compile, memoized) and `_identify_per_group_columns` (`rx.match(str(c))`).
- Pattern: `pattern` argument from the training config.
- Input source: config for the pattern; column names as the matched data.
- Why it blows up: as RDS2-3. The memo dict is keyed by pattern string and unbounded, but the set of config patterns is small.
- Proposed disposition: DOC. Operator-trusted regex.

## Reviewed and rejected (no finding)

| Site | Reason |
|---|---|
| `src/mlframe/reporting/renderers/_render_timings.py:31` `_SUFFIX_RE` | Unanchored alternatives, but each `_` start consumes a run that cannot contain `_`, so total work is linear. Input is internal. |
| `src/mlframe/utils/numba_cache_guard.py:20-21`, `src/mlframe/_numba_cache_deps.py:28-31` | Run over repo source files, not runtime input. `^\s*` under MULTILINE is polynomial in the worst case but only over trusted source. |
| `src/mlframe/training/feature_handling/providers.py:66` `_URI_RE` | Anchored `^...$`, one greedy `.+`, linear. |
| `src/mlframe/integrations/...` other patterns, `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_explain.py:107` | `re.match` on a fixed prefix, `[^,\]]+` is a single class, linear. |
| `re.escape(c)` embedded in `rf"(?<!...)...(?!...)"` (`_benchmarks/bench_sf3_jmim_exponent_selection.py:62`, `_benchmarks/four_pillars_contracts.py:29`) | Column name is escaped, so it is a literal, not a pattern. Benchmark code only. |
| `training/composite/discovery/_coord_names.py:23-25`, `preprocessing/_cleaning_helpers` splits, `_fe_*_TOKEN_SPLIT` family, `providers.py:64` | Single negated character class with `+`, linear. Used as split/sub on column names. |
| `training/_eval_helpers.py:344-348` (`BTTR=`, `MTTR=`, `MLTR=` patterns) | Fixed literal prefixes with simple `\d+` / `[\d,]+`, linear. |
| `training/reporting/_reporting_probabilistic_helpers.py:50,56` | Fixed alternation, anchored groups, applied to internal metric names. |
| `training/reporting/_diagnostics_budget.py:33` | Simple alternation, internal names. |
| `training/composite/cache_store.py:35` `_HEX_KEY_RE` | Anchored `\A[0-9a-fA-F]+\Z`, linear. |
| `training/baselines/_dummy_metrics_pick_plot.py:355,359` | `_p(\d+)` / `_w(\d+)` on internal names. |
| `feature_selection/importance.py:38` `_SAFE_FILENAME_RE` | Negated class, linear. |
| `feature_selection/filters/_fe_family_budget.py:121`, `:242` | Fixed-prefix pattern, linear. `re.sub` replaces each unsafe character. |
| `training/core/_phase_train_one_target_schema.py:178` | Negated class with `+`, linear. |
| `feature_selection/_benchmarks/profiling/nvprof_range_summary.py` (`(.*)` on profiler text) | Dev tool reading profiler output, not runtime input. Linear in practice. |
| `feature_selection/_benchmarks/kernel_arg_dtype_audit.py`, `fe_gpu_coverage_inventory.py`, `fs_hybrid/*`, `exp2_general.py`, `dbg_case2.py` | Benchmark and audit tooling, run on repo files. Not runtime input. |
| `models/ensembling/score_gate.py:23` `\[[^\]]+\]` | Single negated class, linear. |
| `feature_selection/filters/_mrmr_class_transform.py:79`, `inference/postanalysis.py:32`, `training/strategies/*` `_SLUG_PATTERN` | Fixed or negated-class patterns, linear. |

## Notes on scope
- No `re.compile` call takes a value that a remote caller can supply directly. The only operator-supplied patterns (RDS2-3 to RDS2-6) are config-trusted and are classed DOC.
- `regex` (third-party module) is not used in src.
- The fuzz and benchmark scripts were checked only for direct input from data; they are not runtime paths.

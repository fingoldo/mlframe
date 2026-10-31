# Command / subprocess execution audit (mlframe, src/mlframe)

Read-only audit. No code or git state changed.

## Summary

- Scope: every `subprocess.*`, `os.system`, `os.popen`, `shell=True`, `Popen`, and command string in `src/mlframe`.
- `os.system`, `os.popen`, `os.exec*`/`os.spawn*`, and `shell=True`: **zero occurrences** in `src/mlframe`.
- The raw grep (about 66 hits) contains many false positives: the word "empty" and the identifier `call` match the pattern. Roughly 40 real `subprocess` invocation sites remain, plus 9 import lines.
- Every real invocation is list-form argv with no shell. No command string is built by interpolating data into a shell.
- One real data-to-interpreter path exists in a helper: `utils/native_import_probe.py` builds `python -c "import {name}"` with an f-string. All current callers pass the literal `"hnswlib"`, so it is not reachable from user data today.
- Shell-free `git` calls in `_staleness.py` take a SHA read from a manifest file. A leading `-` would be parsed as a git option. The impact is limited to `rev-list`, so it is Low.
- Severity counts: **High 0, Medium 0, Low 3, Info 9** (12 findings in total).

## Findings table

| ID | File:line | Call | Severity | Proposed status |
|----|-----------|------|----------|-----------------|
| CMD-01 | `src/mlframe/utils/native_import_probe.py:56` | `subprocess.run([sys.executable, "-c", f"import {name}"], ...)` | Low | FUTURE |
| CMD-02 | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_staleness.py:89` (+ `:122-125`) | `_git(["rev-list","--count", f"{sha}..HEAD", ...])`, `sha` from manifest JSON | Low | FUTURE |
| CMD-03 | `src/mlframe/training/_gpu_state_probe.py:88` and `:99` | `subprocess.run([exe, "--query-gpu=...", ...])`, `exe` from `shutil.which` with bare-name fallback | Low | FUTURE |
| CMD-04 | `src/mlframe/training/_polars_native_support.py:90` | `subprocess.run([sys.executable, "-c", _PROBE_SOURCE, library], timeout=120)` | Info | RESOLVED |
| CMD-05 | `src/mlframe/system/kernel_tuning_cache/_ensure.py:164` | `subprocess.run([sys.executable, "-m", "mlframe.system.kernel_tuning_cache", "refresh", kernel_name], timeout=...)` | Info | RESOLVED |
| CMD-06 | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_manifest.py:74` | `subprocess.run(["git","rev-parse","HEAD"], cwd=..., timeout=20)` | Info | RESOLVED |
| CMD-07 | `src/mlframe/utils/native_import_probe.py:36, 73` | env var `MLFRAME_*` (from `_env_key`) pre-seeds the probe verdict | Info | DOC |
| CMD-08 | `src/mlframe/training/pipeline/_pipeline_extensions_pysr.py:12, 246` | `import subprocess`; `subprocess.CalledProcessError` in an `except` tuple, no invocation | Info | RESOLVED |
| CMD-09 | `src/mlframe/feature_selection/_benchmarks/profiling/family_batch.py:45`, `family_scale.py:167` | child benchmark runs via list argv, `-I` (family_batch has no timeout) | Info | DOC |
| CMD-10 | 17 `_benchmarks/` files that call `git show` / `git rev-parse` / `git -C` (see details) | `subprocess.check_output`/`run` with constant argv | Info | RESOLVED |
| CMD-11 | `feature_selection/_benchmarks/bench_iter87_cumulative.py:119`, `bench_iter100_stratified_anchors_contour.py:113`, `bench_iter101_stratified_anchors_2d.py:119` | worker subprocess, list argv, timeouts set | Info | RESOLVED |
| CMD-12 | `feature_selection/filters/_benchmarks/bench_sf3_jmim_exponent_selection.py:80` | `subprocess.run([sys.executable, "-m", <module>, "--arm"], env=...)` with no timeout | Info | DOC |

## Details

### CMD-01: Name interpolated into `python -c` (Low, FUTURE)

`utils/native_import_probe.py:52-56`:

```python
def _probe_child(name: str) -> subprocess.CompletedProcess[str] | None:
    return subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
    )
```

- Data source: `name`, passed by `native_module_importable(name)` (line 91).
- Callers, all literal: `feature_engineering/transformer/_key_bank.py:209`, `feature_engineering/transformer/_knn_helper.py:56`, `feature_engineering/transformer/_row_attention_ann.py:47`, `feature_selection/_benchmarks/kernel_tuning_cache/auto_tune.py:216`. Each passes `"hnswlib"`.
- Reachability: none from user input today. A value such as `"os; import x"` would run arbitrary Python in the child, but nothing supplies such a value.
- No shell is involved, so this is not shell injection. It is code injection into a child interpreter, and it would only become reachable if a future caller passes config or data.
- Mitigating factors: argv list form, `sys.executable`, timeout 120 s.
- Proposed fix (FUTURE): pass the name through argv as `polars` does (`-c "import sys; __import__(sys.argv[1])" name`), or validate with `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name)` before use.
- Reason for FUTURE: the fix is cheap, but no current path exists. Do it when the next caller is added.

### CMD-02: Manifest SHA reaches `git rev-list` argv unvalidated (Low, FUTURE)

`feature_selection/_benchmarks/fs_hybrid/_staleness.py`:

- `:89` `_git(["rev-list", "--count", f"{sha}..HEAD", "--", *paths], repo)`
- `:122-125` `sha` comes from the run manifest JSON via `assess(manifest_path)`, and then `commits_since(sha, root)`.

- Data source: a manifest file on disk. It is produced by the benchmark runner, so it is local-file input, not network input.
- Shell: none. `_git` (lines 66-82) uses list argv, `cwd=repo`, `timeout=30`, `check=False`.
- Reachability: a manifest whose `commit` field begins with `-` (for example `--output=...`) makes the value an option token. `git rev-list` accepts few options, so the effect is limited to an error or a changed count. Writing such a manifest requires write access to the results directory.
- Fix: validate the SHA before use, e.g. `if not re.fullmatch(r"[0-9a-f]{7,40}", sha): return None`, and add `--end-of-options` before the revision range where git supports it.
- Reason for FUTURE: real but low impact. It is a robustness fix for a benchmark tool.

### CMD-03: `nvidia-smi` resolved via `PATH`, bare-name fallback (Low, FUTURE)

`training/_gpu_state_probe.py:83-103`:

```python
exe = shutil.which("nvidia-smi")
if exe is None:
    exe = "nvidia-smi"
out = subprocess.run([exe, "--query-gpu=...", ...], timeout=_SMI_TIMEOUT_S, check=True)
```

- Data source: `PATH` environment. No user-supplied argument is involved. Arguments are fixed.
- Shell: none. Timeout 5 s on both calls.
- Reachability: the fallback branch is reached only if `shutil.which` returns `None` at call time after `_BACKEND` was set to `"smi"`. In that case, on Windows, a bare `nvidia-smi` name is searched in the current working directory first, so a planted `nvidia-smi.exe` in the cwd would run. This requires an attacker who can write into the working directory, which is a low-prevalence scenario.
- Fix: when `which` returns `None`, return `None` from `_probe_smi` instead of falling back to the bare name. Alternatively, cache the absolute path resolved in `_pick_backend`.
- Reason for FUTURE: small, local fix, low exposure.

### CMD-04: Polars probe child (Info, RESOLVED)

`training/_polars_native_support.py:90-93`: `subprocess.run([sys.executable, "-c", _PROBE_SOURCE, library], capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS, check=False)`.

- The code is the fixed constant `_PROBE_SOURCE`. `library` is passed as `argv[1]`, not as code (`_PROBE_SOURCE` reads `sys.argv[1]`, line 46).
- `library` values come from `accepts_polars` callers that pass literals `"catboost"` / `"lightgbm"`.
- No shell, timeout set, `OSError`/`TimeoutExpired` handled. This is the safe form. The `_PROBE_SOURCE` pattern is the model for CMD-01.

### CMD-05: Kernel-tuning child (Info, RESOLVED)

`system/kernel_tuning_cache/_ensure.py:164`: `subprocess.run([sys.executable, "-m", "mlframe.system.kernel_tuning_cache", "refresh", spec.kernel_name], timeout=timeout_s, check=False)`.

- `spec.kernel_name` comes from the in-repo kernel registry. It is a list argument, so no shell parsing applies.
- Timeout set and `TimeoutExpired` handled. The lock file serialises runs.

### CMD-06: Benchmark git SHA (Info, RESOLVED)

`feature_selection/_benchmarks/fs_hybrid/_manifest.py:74-82`: list argv, `cwd` derived from `__file__`, `timeout=20`, `(OSError, SubprocessError)` caught and returned as `"unavailable"`.

### CMD-07: Probe verdict env override (Info, DOC)

`utils/native_import_probe.py:34-36, 73-75`: an inherited environment variable of the form `<prefix><MODULE>` set to `"0"` or `"1"` forces the verdict for module `<MODULE>`, skipping the child probe.

- This is not command execution. It trusts the environment. Anyone who can set the environment of the process can already run code, so there is no added privilege.
- Proposed disposition: DOC. Document in the module docstring that the variable overrides the probe, so operators know it exists.

### CMD-08: PySR import guard (Info, RESOLVED)

`training/pipeline/_pipeline_extensions_pysr.py:12` imports `subprocess` with the comment "subprocess used below with list args only". Line 246 catches `subprocess.CalledProcessError` from a Julia import. No subprocess is invoked in this file. The `# nosec` comment is stale. Cleanup candidate only (no security impact). Proposed status: RESOLVED, no action needed for security.

### CMD-09: Benchmark family drivers (Info, DOC)

- `feature_selection/_benchmarks/profiling/family_batch.py:45`: `subprocess.run(cmd, capture_output=True, text=True, check=False)`. `cmd` is a list built from `sys.executable -I <file> run --n --p --mode --family --timeout --out`. Operator CLI arguments are passed through as argv. No shell. **No timeout**, so a hung child blocks the batch. Benchmark-only.
- `feature_selection/_benchmarks/profiling/family_scale.py:167`: `Popen(cmd, stdout=PIPE, stderr=STDOUT, ...)` with `communicate(timeout=a.timeout)`. The timeout is enforced. Same argv model.
- Proposed status: DOC for both (operator-trusted CLI args, benchmark tooling). Optional follow-up: add `timeout=` to `family_batch.py`.

### CMD-10: Benchmark `git show` helpers (Info, RESOLVED)

All of these use a constant argv list with a constant repository path (or one derived from `__file__`), no shell, and `check_output`/`run`. Several have no timeout, which is a hang risk for a benchmark, not a security issue.

- `feature_engineering/_benchmarks/ab_grouped_10m.py:31, 35`
- `feature_engineering/_benchmarks/bench_anchor_ewm_recurrence.py:30`
- `feature_engineering/_benchmarks/bench_anchor_slope_cache.py:31`
- `feature_engineering/_benchmarks/bench_ensemble_disagreement_shared_binning.py:40`
- `feature_engineering/_benchmarks/bench_knn_dispersion_vectorize.py:31`
- `feature_engineering/_benchmarks/bench_spectral_centroid_e2e.py:23`
- `estimators/_benchmarks/bench_cpx39_decorrelator.py:31`
- `feature_selection/filters/_benchmarks/bench_batch_pair_mi_cupy_getsync.py:36`
- `feature_selection/filters/_benchmarks/bench_conditional_permutation_strata_arrs.py:34`
- `feature_selection/filters/_benchmarks/bench_polynom_pair_loky_cpu_only.py:30`
- `feature_selection/filters/_benchmarks/bench_ratio_delta_fe.py:38`
- `feature_selection/filters/_benchmarks/bench_target_encoding_apply.py:25`
- `feature_selection/filters/_benchmarks/bench_temporal_agg_fe.py:38`
- `feature_selection/_benchmarks/bench_mrmr.py:50` (`git` rev, `DEVNULL` stderr)
- `feature_selection/_benchmarks/bench_pairwise_modular_cost.py:61`
- `reporting/_benchmarks/_e2e_slice_finder_ab_iter71.py:48` (`f"{ref}:src/..."`, where `ref` is a CLI value)
- `training/composite/_benchmarks/bench_venn_abers_ivap_envelopes.py:23`

Note on `ref`: in `_e2e_slice_finder_ab_iter71.py:48`, `ref` is a benchmark CLI argument passed as a single argv element. The same `-`-prefix option issue as CMD-02 applies, but the operator controls it. Accepted as DOC if needed.

### CMD-11: Benchmark worker subprocesses (Info, RESOLVED)

- `feature_selection/_benchmarks/bench_iter87_cumulative.py:119`: `[python_exe, "-c", WORKER, json.dumps(regime), cache_dir, ...]` with `timeout=PER_CONFIG_TIMEOUT_S`. JSON travels as argv, not as code.
- `feature_selection/_benchmarks/bench_iter100_stratified_anchors_contour.py:108, 113`: `[sys.executable, worker_path, str(...)...]`, `timeout=PER_FIT_TIMEOUT_S`, `TimeoutExpired` handled.
- `feature_selection/_benchmarks/bench_iter101_stratified_anchors_2d.py:114, 119`: same pattern.

### CMD-12: S-F3 benchmark driver (Info, DOC)

`feature_selection/filters/_benchmarks/bench_sf3_jmim_exponent_selection.py:80-82`: `subprocess.run([sys.executable, "-m", "<fixed module>", "--arm"], env=..., capture_output=True, text=True)`. The module name is a constant. No shell. **No timeout.** Benchmark-only. Proposed status: DOC (or add `timeout=`).

## Non-findings checked

- `training/_unpickle_policy.py:26-27`: names such as `"system"`, `"popen"`, `"call"`, `"Popen"` appear in a denylist of unsafe callables. This is a security control, not an execution site.
- `feature_engineering/_recursion_autotune.py`, `feature_selection/filters/_batch_mi_noise_gate_bits.py`, `feature_selection/filters/_fe_interaction_prerank_kernels.py`: `_run_sweep` runs in-process. No subprocess.
- `training/callbacks/progress_widget.py:97`: `get_ipython()` local variable, not a shell call.
- `training/lgb_shim.py:435`, `system/kernel_tuning_cache/auto_tune.py:209`: comments and docstrings only.
- Remaining raw-grep hits are English words ("empty", "call") and are not sites.

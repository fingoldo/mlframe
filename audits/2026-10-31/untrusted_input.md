# Untrusted-input parsing and resource exhaustion audit (2026-10-31)

Read-only. No code or git state changed. Scope: src/mlframe, sampled via grep over file readers
(read_csv/read_parquet/json/orjson/np.load/zipfile/zstd/pickle), regex compiled or applied to caller strings,
and caller-controlled allocation sizes. This is a sampled audit, not exhaustive: the `np.empty/zeros` sweep
was grep-level, and only the sites below were read in full.

## Summary

- 9 findings: 0 High, 2 Medium, 3 Low, 1 Info-with-action (INP-07 DOC), plus 2 Info closed (REJECTED, RESOLVED).
- Decompression and pickle loading are already hardened: bounded zstd, `np.load(allow_pickle=False)` default in the feature cache, `torch.load(weights_only=True)`, `validate_trusted_path` on joblib.
- Largest real risks are caller-controlled dense allocations that scale as a product of two caller sizes (bootstrap x n, 2^n_models Shapley coalitions) and a dense n_nodes^2 adjacency matrix.

## Findings table

| ID | file:line | severity | proposed status |
|---|---|---|---|
| INP-01 | src/mlframe/calibration/policy.py:461 | Medium | FUTURE |
| INP-02 | src/mlframe/votenrank/shapley_blend.py:183 | Medium | FUTURE |
| INP-03 | src/mlframe/data_valuation/_mc_sampling.py:102 | Low | FUTURE |
| INP-04 | src/mlframe/feature_engineering/graph_spectral_features.py:56 | Low | FUTURE |
| INP-05 | src/mlframe/reporting/output.py:55 | Low | FUTURE |
| INP-06 | src/mlframe/preprocessing/_cleaning_helpers.py:188 | Low | DOC |
| INP-07 | src/mlframe/training/targets_performance.py:284 | Info | DOC |
| INP-08 | src/mlframe/calibration/post.py:30-33 | Info | REJECTED |
| INP-09 | src/mlframe/feature_engineering/_grouped_segments.py:92-96 | Info | RESOLVED |

RESOLVED checks (no finding, cited for the record):
- Zstd decompression bomb: src/mlframe/training/_bounded_zstd.py:43-84 (streaming byte ceiling plus frame_content_size precheck, max_output_size).
- npz/pickle loading: src/mlframe/training/feature_handling/cache.py:557 (allow_pickle gated, default False; pickle fallback refused at :566); src/mlframe/feature_engineering/transformer/_key_bank.py:179-182 (np.load default allow_pickle=False; metadata via safe_load).
- Torch checkpoint loads: src/mlframe/training/neural/_recurrent_wrapper_base.py:581 and src/mlframe/training/_unpickle_policy.py:145, both weights_only=True.
- joblib/dill loads: src/mlframe/core/helpers.py:22-33 (validate_trusted_path requires trusted_root).

## Details

### INP-01 Medium: dense bootstrap index matrix scales as n_bootstrap x n
- Location: src/mlframe/calibration/policy.py:461 `out = np.empty((n_bootstrap, _total_n), dtype=np.int32)`.
- Input source: n_bootstrap and row count, both caller-controlled.
- Outcome: memory blow-up. 1000 x 1M rows x 4 bytes is about 4 GB, allocated up front.
- Realistic caller: yes, with default-sized bootstrap counts on large calibration sets.
- Note: src/mlframe/calibration/_benchmarks/bench_resample_index_ram_chunked.py already measured a chunked alternative; the production path does not use it.
- Fix: chunk over n_bootstrap (fixed byte budget, reused buffer), or reuse the chunked helper; raise or chunk above a documented byte cap.
- Disposition: FUTURE. Real; touches the bootstrap contract, needs bit-identity tests.

### INP-02 Medium: exact Shapley enumerates 2^n_models coalitions
- Location: src/mlframe/votenrank/shapley_blend.py:183 `v = np.empty(n_coalitions, dtype=np.float64)`, with n_coalitions = 2^n_models and exponential loops at :163-187.
- Input source: number of ensemble members passed by the caller.
- Outcome: hang and memory blow-up. 25 models = 33M coalitions (268 MB, tens of millions of scoring calls); 30 models = 1e9 entries, 8 GB.
- Realistic caller: yes for a large candidate pool. Not verified whether an upstream cap exists; check before acting.
- Fix: above a small n_models threshold, switch to permutation Monte Carlo (sibling _mc_sampling.py exists) and log a warning naming the threshold.
- Disposition: FUTURE.

### INP-03 Low: Monte Carlo coalition buffer sized by caller count
- Location: src/mlframe/data_valuation/_mc_sampling.py:102 `v = np.empty(n_coalitions, ...)`; also values_sum/beta at :51 and :107 sized by n_rows.
- Input source: n_coalitions and n_rows, caller-supplied.
- Outcome: allocation linear in the requested count. Only abusive counts hurt, so Low.
- Fix: validate n_coalitions against a documented ceiling tied to row count.
- Disposition: FUTURE (bundle with INP-02).

### INP-04 Low: dense adjacency matrix for spectral graph features
- Location: src/mlframe/feature_engineering/graph_spectral_features.py:56 `A = np.zeros((n_nodes, n_nodes), dtype=np.float64)`.
- Input source: node count from the caller's edge list.
- Outcome: quadratic memory. 50k nodes = 20 GB; 10k nodes = 800 MB.
- Fix: scipy.sparse adjacency with a sparse eigen solver, or a node-count cap raising with a pointer to the sparse path.
- Disposition: FUTURE.

### INP-05 Low: plot-output DSL regex with overlapping whitespace classes
- Location: src/mlframe/reporting/output.py:55 `^\s*([A-Za-z]+)\s*\[\s*([A-Za-z0-9_,\s]+?)\s*\]\s*$`.
- Input source: plot_outputs DSL string from configuration.
- Outcome: polynomial backtracking. `\s` appears both inside the lazy class and in the surrounding `\s*` tokens, so a long whitespace run with no closing bracket costs about quadratic time per clause. Not timed in this pass.
- Realistic caller: only with a pathological config string; config is developer-supplied, so Low.
- Fix: drop `\s` from the inner class (strip whitespace before matching) and/or cap input length (e.g. 1 KB) with ValueError.
- Disposition: FUTURE.

### INP-06 Low: caller-supplied regex compiled without guard
- Location: src/mlframe/preprocessing/_cleaning_helpers.py:188 `re.compile(exclude_mask)`.
- Input source: exclude_mask argument of the cleaning API.
- Outcome: ReDoS if the pattern is attacker-controlled. Python re has no timeout.
- Realistic caller: only with an untrusted pattern source; this is a developer-facing argument.
- Disposition: DOC. State in the docstring that exclude_mask is a trusted regex.

### INP-07 Info: CSV read with no size limit
- Location: src/mlframe/training/targets_performance.py:284 `pd.read_csv(path)`.
- Input source: a path the caller chooses (this suite's own saved output).
- Outcome: loads the whole file, proportional to the file pointed at.
- Disposition: DOC. Intended local reader.

### INP-08 Info: runtime regex cache
- Location: src/mlframe/calibration/post.py:30-33, `@lru_cache(maxsize=128)` around re.compile.
- Outcome: bounded cache, no unbounded growth. Pattern safety is INP-06.
- Disposition: REJECTED as a finding (bounded; no new risk).

### INP-09 Info: counting sort over integer group ids
- Location: src/mlframe/feature_engineering/_grouped_segments.py:92-96. The span allocation is gated by span <= 4n + 1M; sparse or huge-span keys fall back to argsort.
- Disposition: RESOLVED (already gated).

## Not checked

- Excel and image loaders: none found in src/mlframe by grep.
- Recursion depth: not examined.
- Full read of each np.empty/zeros hit: most are sized by internal data and proportional to input; not findings.

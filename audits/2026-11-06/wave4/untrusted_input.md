# Wave 4: untrusted input and resource exhaustion (2026-11-06)

Read-only check of HEAD. INP-01..INP-09 from audits/2026-10-31/untrusted_input.md are not re-reported.

## Findings

| ID | Severity | file:line | Failure scenario | Proposed disposition |
|---|---|---|---|---|
| W4-INP-01 | Medium | src/mlframe/feature_engineering/categorical_powerset_concat.py:78,83-88 | Default `max_order=None` enumerates 2^n-n-1 subsets. With n=20 key columns that is about 1.05M composite string columns. At 100k rows and about 20 bytes per string, that needs about 2 TB, so the process dies with OOM. Also, `out = concat_categorical_group(out, ...)` rebuilds the frame on every subset, which adds quadratic frame-rebuild time even at n=12 (4083 columns). The docstring warns about this, but there is no hard cap. | RESOLVED-needed: raise a ValueError when the subset count exceeds a cap (for example 4096) unless the caller passes `max_order`, and build all the columns first, then do one concat. |
| W4-INP-02 | Medium | src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_interactions.py:253,267-269 (caller _shap_proxied_fit_interactions.py:143-145) | When P <= `exhaustive_max` (16 by default) and `max_features=None`, every subset is scored. At P=16 that is 65535 subsets. Each call to `interaction_margin` (line 236) copies an n x r x r slice. At n=100k and an average r of 8, each copy is about 51 MB, so the full pass copies about 3.4e11 elements. That takes tens of minutes to more than an hour of single-threaded time for what is meant to be a quick extra step. | RESOLVED-needed: limit the exhaustive pass to cardinality <= 3 (or a budget on the subset count), and compute the margin from a precomputed pairwise sum rather than by slicing Phi. |
| W4-INP-03 | Low | src/mlframe/feature_selection/filters/_usability_aware_selection.py:239 | `list(itertools.combinations(base_names, 2))` builds the full pair list before the `max_pairs` cap is applied on line 240. With 20k base columns that is 2e8 tuples, about 16 GB of Python objects, so the process dies with OOM before the cap runs. | FUTURE: rank columns first, or stream the pairs with `heapq.nlargest`/`islice` so the full list is never built. |

## Checked and safe

- feature_selection/filters/_row_stat_fe.py:110: the pair list is bounded by `MAX_POOL=16` (line 37, applied at line 97), so it holds at most 120 pairs x 7 statistics.
- feature_selection/filters/_mrmr_fit_impl/_selection_compaction.py:115: guarded by `3 <= len(spanned) <= 8`, so at most 56 triples.
- feature_selection/filters/_mrmr_fe_step_helpers.py:485: streams the combinations instead of building a list.
- evaluation/leak_scan.py:149: the number of combinations has a cap.
- core/helpers.py:22-33 and estimators/pipelines.py:100,159: joblib and dill loads go through `validate_trusted_path`. Deserialization is out of scope for this class and is already covered.
- I found no decompression (zipfile, tarfile, gzip) on input paths, no unbounded `lru_cache(maxsize=None)`, and no nested-quantifier regex in src/mlframe outside `_benchmarks`. The pickle loads in `_benchmarks` are dev-only caches the scripts write themselves.

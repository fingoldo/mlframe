# Wave 5: untrusted input and resource exhaustion (2026-11-07)

Read-only check of HEAD. INP-01..INP-09 (2026-10-31) and W4-INP-01..03 (2026-11-06) are not re-reported.
Status note: W4-INP-01 looks fixed on HEAD. `categorical_powerset_concat.py:83-88` now raises a ValueError when the subset count goes over `max_subsets`, so the tracker row can move to RESOLVED once that is confirmed.

## Findings

| ID | Severity | file:line | Failure scenario | Proposed disposition |
|---|---|---|---|---|
| W5-INP-01 | Low | src/mlframe/utils/_param_oracle.py:525-531 | `_all_combos` builds the whole Cartesian product of `param_space` as a list of dicts. With 8 knobs of 10 values each that is 1e8 dicts, about 30+ GB of Python objects, so the process dies with OOM before any epsilon-greedy pick runs. Today `param_space` is written by developers (in-package dispatcher configs), not by end users, so a real trigger needs a misconfigured caller. | FUTURE: either refuse a product above a cap (for example 1e5) when the oracle is built, or sample combos lazily (one random index per axis) instead of building the full list. |
| W5-INP-02 | Info | src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_interactions.py:528,557-558 | `nb[c] = bins.max()+1` comes from the data. The joint bincount length is `nb_a*nb_b`. If the bins were unbounded raw integer codes (for example, a code of 1e6 on each side), one call would allocate 1e12 int64 cells (8 TB). On HEAD the `bins` are the output of the quantile binning step earlier in the function, so `nb` is bounded by its bin budget. No input reaches this today. | DOC: add a comment or assert that `nb[c] <= n_bins_max`, so a future caller that passes raw codes fails loudly. |

## Checked and safe

- models/rf_proximity.py:35,55: the n x n proximity matrix is capped by `_DEFAULT_MAX_N=20000` (about 3.2 GB).
- metrics/classification/_weighted_kappa.py:62-75: the matrix size comes from the distinct labels (`np.unique`), not the largest label value, and an explicit `n_classes` is bounds-checked. A label of 1e9 does not allocate 1e18 cells.
- reporting/charts/slice_finder.py:386-395: pairs go through `_cap_combos_with_pairs` with `max_combos`. Triples come from `min(three_way_top_features, p)` features and are truncated to the remaining room.
- shap_proxied_fs/_shap_proxy_interactions.py:543,566: the pair scan is bounded by `max_screen_cols`, so it is O(max_screen_cols^2).
- feature_selection/filters/friend_graph.py:284: the n x n matrix is over the selected features, not the rows.
- feature_selection/filters/_conditional_gate_fe.py:359, _integer_lattice_fe.py:231: (n, n_perm) with n_perm fixed at 12.
- training/pipeline/_categorical_composite_fe.py:97,151: powerset `max_order` defaults to 2, and the source-column count is capped by `categorical_composite_max_source_columns`.
- engineered_recipes/_recipe_unary_binary.py:29,39 and _offset_product_fe.py:62: the `functools.cache` keys are a small finite set of op names, not data.
- reporting/output.py:56 `_CLAUSE_RE`: anchored, and the repeated group needs a literal comma separator, so there is no catastrophic backtracking.
- I found no zipfile/tarfile/gzip decompression of caller input in src/mlframe outside `_benchmarks`. This repeats the wave 4 result.

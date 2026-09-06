# Reporting audit 2026-09-06 — rendering speed

Scope: `src/mlframe/reporting/` only. Bug class: anything that makes chart generation slow.
Read-only audit; no source file was modified.

## Method

All timings on this machine (Windows 11, py 3.14.3), single process, no other load.
Benchmarks were throwaway scripts in the session scratchpad (not saved into the repo).
Every number below is tagged **MEASURED** or **INFERRED**. Where a fix is proposed with a
number, the alternative was actually run and its output compared against the current code's.

Two things the audit did NOT find, worth recording because they were the primary hypothesis:

* The known `add_annotation` O(n^2) defect is **already fixed everywhere it occurs**:
  `renderers/_plotly_heatmap.py:229-247`, `renderers/_plotly_network.py:111-127` and
  `renderers/_plotly_scatter.py:78-99` all use the first-call-then-batched-tuple pattern.
  The only remaining per-item `add_*` loops are PERF-12/PERF-13, both currently 1-2 items.
* The whole-frame-copy class is largely handled: `pdp_ice.py` / `_pdp_carrier.py` use
  in-place single-column mutation, `diagnostics_dispatch` caps columns before gathering rows.
  The one surviving instance is PERF-14, and it is bounded in practice by those caps.

The real remaining costs are compute in the chart *builders*, not mark emission in the renderers.

---

## PERF-01 — P1 — `charts/binary.py:269-286` — AP bootstrap runs 500 serial numpy passes over 50k rows

`bootstrap_ap_ci`'s per-resample loop does a `bincount` + two `cumsum` + a `dot` over
`m = _AP_BOOTSTRAP_ROW_CAP = 50_000` rows, 500 times.

* Hurts at: any binary figure with `ap_ci=True` (the default) and n >= `_AP_BOOTSTRAP_MIN_N`;
  cost is flat in n above the 50k cap, so it hits *every* binary chart.
* MEASURED: `compose_binary_figure` at n=2M costs 3.55 s total, of which `bootstrap_ap_ci`
  is 1.83 s cumulative / 1.17 s self — **52% of the whole builder**. Standalone at n=200k: 2.58 s.
* MEASURED (rejected alternative): batching 32 resamples into a 2-D `cumsum(axis=1)` is
  *slower* (2.91 s vs 1.72 s) — the loop is memory-bandwidth bound, not call-overhead bound.
  A `bincount`-into-a-flat-offset variant is a wash (1.77 s).
* MEASURED (proposed fix): an `@njit(parallel=True, fastmath=True)` kernel with `prange` over
  the 500 resamples, each thread doing its own multiplicity count and a single fused
  running-sum pass, costs **0.133 s vs 1.72 s = 12.9x**. First three resample APs were
  identical to the current code's to all printed digits (0.10569373 / 0.10174294 / 0.10501416).
* Fix: move the resample loop into a `@njit(parallel=True)` kernel taking the fixed-rank
  `pos_desc` and one chunk of drawn indices, returning the chunk's APs. Keep the existing
  `_AP_BOOTSTRAP_IDX_CHUNK` chunking so the index matrix stays bounded (the RNG draw order,
  and therefore the interval, is unchanged). Expected saving ~1.6 s per binary figure.

## PERF-02 — P1 — `charts/error_analysis.py:200` — depth-3 `DecisionTreeRegressor` is the single largest cost in the package

`_top_split_features` fits an exact-splitter sklearn tree to rank features; sklearn sorts
every feature at every node, so cost is O(p * n log n) even at `max_depth=3`.

* Hurts at: `weak_segment_heatmap` (default-ON diagnostic) on any wide frame. The dispatch
  caps are `DIAG_ROW_CAP=100_000` rows / `DIAG_MAX_FEATURES=200` columns, and
  `DEFAULT_TREE_FIT_CAP=50_000` rows — i.e. the *capped* case is already the expensive one.
* MEASURED: `weak_segment_heatmap` on 100k x 200 = **9.22 s, of which the tree fit is 8.70 s
  (94%)**. Bare sklearn at 100k x 200 depth 3: 19.9 s; float32 does not help (18.8 s).
* MEASURED (proposed fix): a quantile-binned depth-1 split-gain ranker (bincount + cumsum of
  per-bin count/sum, best variance-reduction split per feature) over 50k x 200 costs
  **1.87 s total (1.69 s binning + 0.18 s ranking) vs 8.70 s = 4.6x**, and it recovered the
  planted discriminating feature as rank 1. Inside a dispatch run the binning is *already*
  computed by `slice_finder._bin_matrix` on the same matrix, so sharing it makes the ranking
  step **0.18 s = 48x**.
* MEASURED (secondary, changes the answer — flagged, not recommended alone):
  `max_features="sqrt"` on the same fit is 1.49 s (13x) but makes the ranking stochastic.
* Fix: replace the impurity-importance ranker with the binned split-gain ranker and feed it
  the codes `_bin_matrix` already produces; keep the sklearn path as the fallback that the
  `except (ValueError, ImportError)` branch already contemplates.

## PERF-03 — P1 — `renderers/_plotly_heatmap.py:195,197,263` — `z=p.matrix.tolist()` forces plotly's Python-list validator

The scatter renderer explicitly documents that "ndarrays pass through to plotly natively
(faster + smaller than `.tolist()`)" (`renderers/_plotly_scatter.py:110`); the heatmap path
does the opposite for `z`, for `text` (cell hovertext) and for the contour overlay's `z`.

* Hurts at: every heatmap panel; cost grows with cell count because the nested-list validator
  is per element. Affected panels include the drift PSI grid (features x buckets), confusion
  matrices, `class_structure_heatmap`, `calibration_heatmap_2d`, the regression density panel,
  and `shap_interactions` (40x40).
* MEASURED (`go.Heatmap` construction + `to_json`, warmed):

  | grid | ndarray | tolist | ratio |
  |---|---|---|---|
  | 100x100 | 0.0026 s | 0.043 s | 16x |
  | 300x300 | 0.020 s | 0.416 s | 21x |
  | 600x600 | 0.052 s | 1.469 s | 28x |

  Output JSON was byte-length identical in every pair, so this is pure waste.
* MEASURED (corroboration): in a real `compose_regression_figure` render profile,
  `_plotly_utils/basevalidators.py:to_scalar_or_list` shows 13,480 calls / 0.138 s — that is
  this conversion on the density panel alone.
* Fix: pass `p.matrix`, `p.cell_hovertext` and `mat` straight through (drop `.tolist()`).
  `list(p.col_labels)` / `list(p.row_labels)` are label sequences and should stay as they are.

## PERF-04 — P1 — `charts/drift.py:164-199` vs `:210-214` — PSI is computed for every column, then most of the work is discarded

`compute_psi_matrix` loops over *all* frame columns computing per-bucket PSI, and only
afterwards keeps `max_features=40` by peak (`:210`). The chart draws `40 * n_buckets` marks.

* Hurts at: any wide frame. The docstring's "one O(n) pass per feature" is accurate but the
  pass is run for features that are then thrown away.
* MEASURED: 1M rows, 20 cols -> 2.03 s; 1M rows, 200 cols -> **18.60 s to draw a 40x10 grid**
  (~93 ms per column, linear in column count).
* INFERRED: on the 100+ GB frames this repo targets (500 cols, tens of millions of rows) the
  same code is many minutes for 400 drawn cells.
* Fix: two-pass. Rank on a bounded row screen (e.g. 100k rows, stratified over the same time
  buckets) to pick the top `max_features` by peak PSI, then recompute those 40 exactly on all
  rows. MEASURED-derived estimate at 1M x 200: 200 * 9.3 ms + 40 * 93 ms = 5.6 s vs 18.6 s
  (3.3x), and the ratio improves linearly with column count. Secondary: the per-column body is
  independent, so the exact pass parallelises over columns.

## PERF-05 — P2 — `charts/slice_finder.py:394-400` — Python append loop over every candidate cell, for a 7-row chart

After the vectorised decode, a `for ci, cid in enumerate(cell_ids)` loop appends five Python
lists per positive-degradation cell, across every enumerated combo (up to
`DEFAULT_MAX_COMBOS = 5_000` combos x up to `nbins^2` cells each). Only `top_k` rows survive.

* MEASURED: `find_weak_slices` on 100k x 200 (arity 2) = 4.65 s cumulative with **1.42 s of
  self-time in the function body** — the enumerate loop is the only per-element Python work in
  it (`_aggregate_combo` is accounted separately at 0.19 s). The rendered table had **7 rows**.
* Attribution of the 1.42 s to this loop is INFERRED from the profile shape; the 1.42 s
  self-time and the 7-row output are MEASURED.
* Fix: accumulate per-combo `np.ndarray`s (means/support/score/cell ids) in a list, one
  `np.concatenate` at the end, `np.argpartition` to `top_k`, and decode + build tuples for the
  survivors only. This is the same "defer formatting until after top_k" discipline the module
  already applies to the label strings (its own comment at `:452`).

## PERF-06 — P2 — `renderers/_trend.py:29` — the Theil-Sen trend fit is recomputed once per backend from the same frozen spec

`robust_fit_endpoints` is a pure function of `(x, y, method)` but is called from inside each
renderer (`_plotly_heatmap.py:303`, `_plotly_scatter.py:221`, `matplotlib.py:477`,
`_matplotlib_scatter.py:121`). The default `plot_outputs` renders BOTH backends from the same
`FigureSpec`, so every trend line is fitted twice with identical inputs and identical output.

* MEASURED: instrumented `compose_regression_figure` at n=2M, then rendering the same spec on
  both backends — plotly render 1.10 s with **1 trend fit costing 0.24 s**, matplotlib render
  0.78 s with **1 trend fit costing 0.25 s**. Same panel, same data, two fits.
* The module's own docstring records this fit was ~786 ms/call before `_TREND_FIT_CAP` was cut
  to 3000, so the duplication used to cost ~1.6 s per figure.
* Fix: memoise on the panel spec (panel specs are frozen) — e.g. an id-keyed cache keyed by
  `(id(panel), method)`, or compute the endpoints in the builder and carry them on the panel
  spec so the renderers only draw. Saves ~0.25 s of CPU per trend panel per figure; the two
  backends run concurrently in `render_and_save`, so the wall-clock win is smaller than the CPU
  win but the two fits do contend.

## PERF-07 — P2 — `diagnostics_dispatch.py:260` and `:690` — the same frame is densified and the same error recomputed twice per run

`render_split_error_diagnostics` and `render_slice_finder_diagnostic` each independently
compute `_per_row_error`, call `_bounded_sample_idx` (same seed, same loss -> identical
indices), `_select_feature_columns` and `_subset_rows`; then `weak_segment_heatmap`
(`charts/error_analysis.py:263`) and `find_weak_slices` (`charts/slice_finder.py:297`) each
call `_resolve_feature_matrix` on that identical sub-frame.

* MEASURED: `_resolve_feature_matrix` on 100k x 200 all-numeric = 0.29 s; on 100k x 200 with
  20 object/string columns = **2.55 s**. Doubling that is 0.6-5.1 s per diagnostics run.
* Fix: compute `(loss, sample_idx, sub_df, names)` once in the dispatch and thread the
  densified `(mat, names)` through both builders as an optional pre-resolved argument — the
  package already has this convention (`charts/ltr.py`'s `shared: Optional[dict]` layout cache).

## PERF-08 — P2 — `charts/error_analysis.py:116` (+ `engineered_separability.py:169`, `_error_analysis_shared.py:74`) — per-row Python scan and a slow string encode on every object column

`any(isinstance(v, (list, tuple, np.ndarray)) for v in arr)` walks EVERY row of every
object-dtype column in Python and, for the normal case (a plain string column), finds nothing
after a full pass. The encode that follows is `np.unique(arr.astype(str), return_inverse=True)`.

* MEASURED at n=100k, one 50-level string column: isinstance scan 0.051 s, `astype(str)`
  0.028 s, `np.unique` 0.031 s = **0.110 s/column**; `pd.factorize` on the same column is
  **0.010 s** (11x). Over 20 such columns that is 2.2 s of the 2.55 s in PERF-07.
* Fix (safe, no semantic change): the code already documents the exact numpy error
  `astype(str)` raises on list-valued cells ("setting an array element with a sequence"), so
  wrap the `astype(str)` in `try/except (ValueError, TypeError)` and drop the pre-scan
  entirely — same behaviour, one full Python pass per object column removed (~1.9x on those
  columns). Replacing the encode with `pd.factorize(..., sort=True)` gets the remaining ~6x but
  needs a check that sorted-code identity is preserved, because `_error_analysis_shared.py:58`
  promises bit-identical codes between the two code paths.

## PERF-09 — P2 — `charts/multiclass.py:101` — `_stratified_subsample` is O(n * K)

`for c in np.unique(y_pos): idx_c = np.flatnonzero(y_pos == c)` makes one full-length pass per
class.

* MEASURED at n=2M: K=10 -> 0.164 s, K=50 -> 0.376 s, K=200 -> **1.323 s** (linear in K).
* Fix: one `np.argsort(y_pos, kind="stable")` + `np.bincount` gives every class's index block
  as a contiguous slice; draw within each slice. One pass instead of K. INFERRED <= 0.4 s at
  n=2M (a 2M argsort measured 0.31-0.75 s elsewhere in this report) and independent of K.

## PERF-10 — P3 — `charts/binary.py:116` — `argsort(kind="stable")` costs 2.4x a quicksort

`_ScoreSort.__init__`'s full-n argsort is the largest non-bootstrap cost in
`compose_binary_figure`.

* MEASURED at n=2M float64: `kind="stable"` 0.745-0.774 s vs `kind="quicksort"` 0.308 s (2.4x);
  MEASURED in situ: 0.79 s of the 3.55 s builder.
* Caveat: stability determines tie ordering, which changes the drawn ROC/PR/KS curves when
  scores tie (common with tree models). Only safe behind a gate, e.g. sort with quicksort,
  check `np.any(np.diff(sorted_scores) == 0)`, and re-sort stably when ties exist. That gate
  costs 0.31 s + one O(n) scan on the tie-free path and ~1.05 s on the tied path, so it wins
  only for models with continuous scores. Recorded as a lead, not a recommendation.

## PERF-11 — P3 — `charts/binary.py:735` — `np.unique(ys).size` to answer "is the score constant?"

Full O(n log n) sort to test a property two O(n) reductions answer. `charts/calibration.py:230`
already documents the `min == max` idiom for exactly this test.

* MEASURED at n=2M: `np.unique(ys).size` 0.069 s vs `ys.min() == ys.max()` 0.003 s (**23x**).
* INFERRED: at 100M rows this is seconds of a figure's build time for one boolean.
* The sorted scores are already available on the `sort` object at that point, so the answer is
  also derivable there for free.

## PERF-12 — P3 — `renderers/plotly.py:740-760` — per-span `add_vrect` + `add_annotation`, per-vline `add_shape` + `add_annotation`

Structurally the same O(n^2) shape as the fixed heatmap-annotation defect: each call
re-validates the whole growing `layout.shapes` / `layout.annotations` tuple, and both loops are
per item with no cap.

* Current inputs are 1-2 items (`charts/drift.py:579-580`, `calibration_drift.py:252`,
  `training_curve.py:136-137`, `_error_analysis_splits.py:110-111`, `temporal.py:182`), so
  there is **no measurable cost today** — this is a latent-shape finding, not a live one.
* Fix if touched: same pattern as `_plotly_heatmap.py:229-247` — first call through
  `add_vrect`/`add_shape` to resolve xref/yref, remaining items assembled as
  `go.layout.Shape` / `go.layout.Annotation` and appended in one tuple assignment.

## PERF-13 — P3 — `renderers/matplotlib.py:822` and `:836` — uncapped per-edge and per-node `ax.annotate`

The plotly twin caps arrows at `_NETWORK_MAX_ARROWS = 500` (`_plotly_network.py:28,106`); the
matplotlib path has no equivalent cap and also annotates every node label individually. The
in-code comment leans on "the friend-graph max_nodes guard keeps edge counts modest", which is
a property of one caller, not of the renderer.

* `Axes.annotate` is not the quadratic case `add_annotation` is, so this is O(edges), not
  O(edges^2); the concern is an unbounded constant, not a complexity class.
* Fix: apply the same `_NETWORK_MAX_ARROWS` ceiling on the matplotlib side so the two backends
  agree about what is drawn (a parity issue as much as a perf one).

## PERF-14 — P3 — `charts/error_analysis.py:90` vs `:113` — docstring says "without a full frame copy"; `np.column_stack` is one

The columns are pulled narrowly, but `np.column_stack(mats)` then materialises a dense float64
copy of the whole selected frame (and holds `mats` alive alongside it, so peak is ~2x).

* MEASURED: 0.202 s / ~160 MB at 100k x 200. Bounded in the dispatch by `DIAG_ROW_CAP` /
  `DIAG_MAX_FEATURES`, but a direct library caller gets no such bound and the docstring tells
  them there is no copy.
* Fix: either correct the docstring, or fill a preallocated `np.empty((n, p), order="F")`
  (F-order also suits the per-column consumers, cf. `slice_finder._bin_matrix`'s F-order note)
  so the `mats` list can be released column by column.

## PERF-15 — P3 — `charts/ltr.py:223`, `charts/ltr.py:385`, `charts/_error_analysis_splits.py:38` — one full-length boolean mask per group

`for b in np.unique(bin_idx): m = bin_idx == b` (ltr, ~log2(max query size) bins),
`mask = y_true_arr == g` (ltr, <= 12 grades), `vals[labels == lab]` (splits, 2-4 labels).
O(n * groups) for an O(n) answer.

* No standalone measurement taken; group counts are small (<= ~20), so INFERRED impact is
  ~10-20 extra full-array passes, i.e. tens of ms at n=2M.
* Fix: single `argsort` + `bincount` grouping, same shape as the PERF-09 fix. Low priority, but
  it is the same idiom and should move with it.

## PERF-16 — P3 — `charts/regression.py:172-182` — Python set of coordinate tuples + a full scan per worst-K row

`present = set(zip(s_pred.tolist(), s_true.tolist()))` materialises a Python set over the whole
plotted subsample, and the `already` comprehension then runs a full
`np.flatnonzero((s_pred[:base] == ...) & (s_true[:base] == ...))` per already-present worst-K
row — O(K * base).

* INFERRED: at base=50k and K=50 that is ~2.5M element comparisons plus 50k tuple hashes; tens
  of ms, not seconds. Not visible in the regression profile, which was dominated by
  `_uniform_bin_index` (0.203 s) and `_resid_vs_pred_panel` (0.202 s self).
* Fix: `np.searchsorted` / `np.isin` over a lexicographically sorted `(pred, true)` view, or key
  on the subsample's original row indices, which `subsample_preserving_extremes` already returns.

## PERF-17 — P3 — `charts/slice_finder.py:469` — three `.iloc[i]` lookups per row inside a generator

`f"{table['bounds'].iloc[i]} (n={int(table['support'].iloc[i]):_}, {table['error_ratio'].iloc[i]:.2g}x)"`
re-resolves three Series and does three positional lookups per row.

* Table is `top_k` rows (MEASURED: 7 rows in the 100k x 200 run), so the cost is negligible
  today. Recorded because it is the "per-element DataFrame access in a loop" idiom, and the fix
  is one line: hoist `.to_numpy()` for the three columns and `zip` them.

## PERF-18 — P3 — `charts/slice_finder.py:176` — 200 separate `np.quantile` calls, each partitioning its own column

`_bin_matrix` calls `np.quantile` once per column; each call does its own `np.partition`.

* MEASURED at 100k x 200 in situ: 0.96 s in `np.quantile` (0.83 s of it in
  `ndarray.partition`) plus 0.68 s in `searchsorted`, out of `_bin_matrix`'s 2.39 s.
* MEASURED (alternative): `np.quantile(X, qs, axis=0)` batched = 1.53 s vs 2.03 s per-column
  (1.3x) on a clean 100k x 200 array.
* Only 1.3x, and batching is awkward because the current loop applies a per-column NaN mask and
  `row_mask`. Worth doing mainly if PERF-02's shared-binning fix makes this matrix the input to
  two consumers, at which point the 2.4 s is paid once instead of twice anyway.

---

## Suggested order of work

1. PERF-03 (one-line change, 16-28x on every heatmap panel, zero behavioural risk).
2. PERF-01 (13x measured, self-contained kernel, values verified identical).
3. PERF-02 + PERF-07 together (they share the densified matrix and the binning; 8.7 s -> ~2 s).
4. PERF-04 (largest absolute win on wide frames, needs a screening design decision).
5. PERF-05, PERF-06, PERF-08, PERF-09 (each a contained fix worth 0.2-2 s).
6. The P3s as they are touched.

---

## Parent verification notes (measured on this host, after the fixes landed)

**PERF-03 confirmed, and larger than reported.** Independently re-measured with a bare
`go.Figure().add_trace(go.Heatmap(...))`, warm, best-of-3: 300x300 `344.0 ms` via `.tolist()` against
`2.9 ms` via the ndarray (**117x**), 600x600 `1403.9 ms` against `6.1 ms` (**230x**). `to_json()` output is
byte-identical between the two, so nothing downstream can tell them apart. Fixed at all three sites in
`_plotly_heatmap.py` (`z`, `text`, and the contour `z`).

**PERF-01 confirmed as a defect, but the reported speedup is the KERNEL's, not the function's.** The
`prange` kernel does replace the per-resample Python loop and the numbers agree to reassociation (AP
identical, both CI bounds within 3e-15). End to end through `bootstrap_ap_ci`, though, warm and
paired: n=50k `1.551 s -> 0.720 s` (**2.15x**), n=400k `1.687 s -> 1.442 s` (**1.17x**). The gap is that
`rng.integers(0, m, size=(chunk, m))` dominates once the arithmetic is compiled - at the row cap that draw
is ~25M int64 per figure, and it is unchanged. Shipped anyway (a real win, no accuracy cost, and it removes
the GIL-bound loop), but the honest figure for the PR panel is 1.2-2.2x, not 12.9x. Making the draw itself
cheaper means changing the generator's consumption order, which moves every existing seed's interval - that
needs its own decision, not a silent rewrite.

**PERF-05 fixed; the win is smaller than the reported self-time suggests.** The per-cell Python loop is
gone (per-combo numpy blocks, with the combo/bin identity of a row recovered from block offsets only for
the rows that reach the table). Old-vs-new through `find_weak_slices`, warm, best-of-2, tables compared
with `DataFrame.equals`: n=60k/p=40 `0.504 -> 0.487 s` (1.04x), n=50k/p=150 `1.895 -> 1.754 s` (1.08x),
n=50k/p=200 `2.451 -> 2.062 s` (**1.19x**), identical tables at every size. The 1.42 s of self-time was
real but overlapped with the kernel work it sits between, so removing it does not subtract that much from
the wall. It scales with the candidate pool, so it grows with feature count - kept on that basis plus the
identical output, not on the headline number.

**PERF-11 fixed and independently measured:** `np.unique(ys).size` vs `ys.min() == ys.max()` for the
constant-score check - 154.1 ms vs 7.2 ms at n=2M (**21.4x**), 13.9 vs 0.77 at 200k (18.1x). Verified the
warning still fires on a constant column and stays off on a varying one.

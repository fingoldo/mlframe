# fe_step — mrmr_audit_2026-09-14

## Scope

| File | LOC |
|---|---|
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_score.py` | 1189 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core.py` | 1060 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py` | 809 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py` | 535 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pool.py` | 251 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/_helpers.py` | 60 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step/__init__.py` | 21 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_step_helpers.py` | 738 |

Cross-read for evidence (not audited as scope): `_fe_batched_mi.py`, `_fe_auto_escalation.py`,
`_fe_deadline.py`, `_mi_greedy_cmi_fe.py`, `_fe_cmi_redundancy_gate.py`, `_mrmr_fingerprints.py`,
`mrmr/_mrmr_class.py`, and the `tests/feature_selection/{fe,mrmr,gpu}` trees.

**Severity counts: P0 0 · P1 3 · P2 10 · P3 9 (22 total).**

---

## Findings

### FE_STEP-1 — batched-CPU pair-MI retry writes NON-canonical `cached_MIs` keys, the one thing the primary path explicitly canonicalises  [P1]
**Where:** `_mrmr_fe_step/_step_pairmi.py:423` vs `:198-206`

**What:** The primary batch-prefill canonicalises every key and carries a six-line comment saying why:

```python
_p = tuple(sorted((int(_pair_a_arr[_i]), int(_pair_b_arr[_i]))))     # :204
if _p not in cached_confident_MIs and _p not in cached_MIs:
```

The loky-failure retry path, ~220 lines below, does the same loop but drops the `sorted`:

```python
_p = (int(_retry_pair_a[_i]), int(_retry_pair_b[_i]))                # :423
if _p not in cached_confident_MIs and _p not in cached_MIs:
```

**Why it is wrong / costly:** `dispatch_batch_pair_mi_chunked` is fed `_retry_ids_arr =
np.fromiter(numeric_vars_to_consider, ...)` — a **set**, whose iteration order is not ascending once it
mixes small raw indices with large later-appended engineered indices (exactly the case the `:199-204`
comment describes for `fe_max_steps>1`). So the retry can emit `(b, a)` with `b > a`. Consequences, all
in the same fit-persistent `cached_MIs`:
- the `_p not in cached_MIs` de-dup check misses an already-present `(a,b)` → **the same logical pair is
  stored twice under two keys with two different MI values**;
- `_step_pairs_rank.score_prospective_pairs` iterates `sort_dict_by_value(cached_MIs)` and will visit both
  orientations → the pair is ranked twice and, if admitted, enters `prospective_pairs` twice;
- `checked_pairs.add(raw_vars_pair)` (`_step_score.py:775`) records only the orientation actually seen, so
  the de-dup for the NEXT FE step fails as well → the same pair gets a second expensive operator search and
  can emit a duplicate `..._2` engineered column;
- `_pair_mm_bias.get(tuple(sorted(raw_vars_pair)), 0.0)` (`_step_pairs_rank.py:319,596`) IS canonical, so
  the two orientations at least share a bias — the divergence is purely in the MI dict.

Bites exactly when the loky pool times out, which the module's own comments say is a reproduced
production event (`_step_pairmi.py:326-339`, "reproduced live at n=3M/p=423").

**Fix:** `_p = tuple(sorted((int(_retry_pair_a[_i]), int(_retry_pair_b[_i]))))`. Better: extract the
prefill loop (it is byte-identical otherwise) into one `_prefill_cached_pair_mis(pair_a, pair_b, mi,
cached_MIs, cached_confident_MIs)` helper used by both sites so the two cannot drift again.

**Test:** `test_pair_mi_retry_prefills_canonical_keys` — force the loky pool to raise (the harness in
`tests/feature_selection/fe/test_pair_mi_loky_failure_retries_batched_cpu.py` already does this), feed a
`numeric_vars_to_consider` set whose iteration order is descending, and assert
`all(k[0] <= k[1] for k in cached_MIs if len(k) == 2)`. The existing test asserts only call counts
(`:113-118`) — it would pass today with the bug present.

---

### FE_STEP-2 — the FE-step's per-candidate CMI-gate loops call the single-column binner + MI once per candidate, while the batched twins are already shipped and unused  [P1]
**Where:** `_mrmr_fe_step/_step_score.py:143-163` and the near-duplicate at `:846-867`

**What:** Both loops are the repo's signature shape — a Python `for` per candidate, each iteration doing
an RNG-free gather + a single-column device bin + a single-column MI kernel:

```python
for _rp, (...) in prospective_additions.items():
    for _jc, _cname in enumerate(_ncols):
        _vals = np.asarray(_tvals[:, _jc], dtype=np.float64)
        ...
        _vb = _quantile_bin_gpu_resident(_vals, int(self.quantization_nbins))   # one column
        if _vb is None: _vb = _quantile_bin(_vals, nbins=...)                   # one column
        _marg = float(_cmi_from_binned(_vb, _y_dense_g, None, kx=...))          # one column
        _cmi_cands[_cname] = (_vals, _marg)
```

**njit/GPU check performed as the brief requires:** `_mi_greedy_cmi_fe.py` is fully `@njit`/`cupy`-backed
(`:305,371,410,478,637,666`, `parallel=True` at `:410`, cupy at `:97,155,185,607`) — so the *inner kernels*
are optimal and a naive REJECT would be defensible on that basis alone. It is **not** valid here, because
batched twins of both stages already exist and are simply not wired to these two call sites:
- `_fe_batched_mi.batched_quantile_bin_gpu(x_cols, nbins)` (`:715`) — bins a whole `(n, K)` block;
- `_fe_batched_mi.batched_cmi_gpu(x_cols, y, z=None, kx=, ky=)` (`:816`) — "Miller-Madow plug-in
  CMI(x_k; y | z) in nats for **EVERY column of `x_cols`, in ONE device workload**… Matches
  `_mi_greedy_cmi_fe._cmi_from_binned` per column (selection-equivalent)" (its own docstring, `:818-825`).

This is the precise "already-optimized primitive, just not wired into every call site that needs it" gap
CLAUDE.md records for `_jackknife_ece` (2026-08-04 entry). Every candidate pays a separate H2D/launch and,
on the host fallback, a separate `np.percentile`+`searchsorted` pass. `K` here is the number of surviving
engineered candidates across all prospective pairs — the wide-pool case the whole step is built for.

**Fix:** stack the surviving candidates' (already-strided) columns into one `(n_g, K)` float block, call
`batched_quantile_bin_gpu` once, then `batched_cmi_gpu(codes, _y_dense_g, None, kx=quantization_nbins,
ky=n_classes)` once; unpack into `_cmi_cands`. Same for the escalation admitted-pool build at `:846-867`.
Keep the existing per-candidate host path as the documented fallback.

**Bench plan (no number is claimed here — nothing was measured):** `_benchmarks/bench_step_score_cmi_cands.py`,
K ∈ {8, 32, 128, 512}, n ∈ {50k, 250k (the gate stride cap), 1M}, warm + best-of-10, CPU-only and
CUDA-present, asserting max |Δmarginal MI| == 0 against the current loop before reporting any speedup.

**Test:** `test_step_score_cmi_cands_batched_matches_per_candidate` — same candidate block through both
paths, assert bit-identical `_cmi_cands` marginals and identical gate admit/drop set.

---

### FE_STEP-3 — escalation admitted-pool marginal MI is estimated at full n, then compared inside the S5 gate against survivor MIs estimated on the escalation subsample  [P1]
**Where:** `_step_score.py:846-867` (build) → `_fe_auto_escalation.py:644-649` (row subsample) →
`:837-842` (the mixed pool)

**What:** `_step_score.py` builds `_esc_admitted_pool[name] = (full_n_values, full_n_marginal_mi)`. Inside
`run_fe_auto_escalation` the pool's **values** are re-sliced to the escalation subsample but the **MI
scalar is carried over unchanged**:

```python
admitted_pool = {k: (np.asarray(v)[_esc_idx], m) for k, (v, m) in (admitted_pool or {}).items()}   # :649
```

and the gate then mixes the two populations in one dict:

```python
for nm, (vals, marg) in (admitted_pool or {}).items(): pool[nm] = (..., float(marg))   # :838
for c in survivors:                                    pool[c["name"]] = (..., c["mi"]) # :841
accepted, _diag = apply_cmi_redundancy_gate(pool, ...)                                  # :842
```

**Why it is wrong:** `apply_cmi_redundancy_gate` sets its relative bar as `retain_frac ×` the **weakest
admitted feature's** MI. Plug-in MI carries a finite-sample bias of order `(k_x-1)(k_y-1)/2n`, so an MI
estimated on `_esc_ss_n` rows is systematically **larger** than the same feature's MI at full n. Mixing the
two scales means the bar is set from differently-biased estimates: the full-n admitted marginals look
artificially weak relative to the subsampled survivors, which **systematically favours admitting escalation
candidates** over the already-admitted support. This is a silent distortion of the mRMR redundancy
trade-off, in the one direction that disables the check.

Note this is also inconsistent with the gate's own sibling at `:130-133`, which strides candidate values
**and** `y` together precisely so "the observed CMI + null decide on one consistent slice".

**Fix:** recompute (or re-scale) the admitted-pool marginals on `_esc_idx` inside
`run_fe_auto_escalation`, right where the values are sliced — one `batched_cmi_gpu` call on the sliced
block (see FE_STEP-2) makes this nearly free. Alternatively pass the raw values only and have the gate own
the single MI estimation pass for the whole pool.

**Test:** `test_escalation_admitted_pool_mi_matches_subsampled_rows` — run the escalation with
`fe_escalation_subsample_n` forced well below n, and assert each admitted-pool entry's stored MI equals
`_cmi_from_binned` recomputed on the sliced rows (today it equals the full-n value instead).

---

### FE_STEP-4 — the gate-composite / cross-group over-materialisation prunes key on a regex that only matches single-letter fixture names, so they are inert on real column names  [P2]
**Where:** `_step_score.py:277-280` (`_bare_tokens`), `:427-429` (`_bare_tokens_fsc`), used at
`:294,310,334,348,462`

**What:** Both "raw variable" extractors are

```python
re.findall(r"(?<![A-Za-z0-9_])([a-z](?:[a-z]?\d+)?)(?![A-Za-z0-9_])", _nm)
```

i.e. exactly one lowercase letter, optionally another letter, optionally digits, with word boundaries.
It matches `a`, `b`, `x12` — the synthetic fixture names every comment in the block cites
(`y=a**2/b+log(c)*sin(d)`). It matches **nothing** in `mul(log(revenue),sin(customer_age))`.

**Why it is wrong / costly:** on any real frame `_clean_cov` is empty and `_cov` is empty, so
`if not (_cov and _cov <= _clean_cov and ...)` short-circuits at `:336` and the gate-composite prune never
fires; the cross-group branch (B) exits at `:463` on `len(_toks) < 2`. The block's docstring/comments
(`:257-272`, `:400-423`) present it as the general over-materialisation control (and it is described as
running "unconditionally" since 2026-06-22). It is in fact a fixture-shaped no-op in production. The
failure mode is benign (no wrong drop) but the documented protection does not exist, so a wide production
fit can still emit the 9-engineered-column over-materialisation the block was written to cap.

A second, narrower defect in the same block: `_gate_cols_in` (`:282-284`) matches gate columns by plain
**substring** (`_gc in _nm`), so a gate named `gate_mask__b__d` also matches a different column
`gate_mask__b__d2`, mis-attributing its source vars.

**Fix:** replace name-parsing with the provenance the step already has — operand identity comes from
`this_pair_features`' `(transformations_pair, bin_func_name, i)` configs, which carry `var_a_idx` /
`var_b_idx` directly (used for exactly this at `:660-665`). Build the coverage sets from those indices
(recursively through `engineered_recipes`' `src_names` for nested parents) instead of from the rendered
name; match gate columns by exact set membership, not substring.

**Test:** `test_gate_composite_prune_fires_on_realistic_column_names` — replay the CASE1 fixture with
columns renamed to `alpha/beta/gamma/delta` and assert the same set of composites is pruned as with
`a/b/c/d` (fails today: nothing is pruned).

---

### FE_STEP-5 — `compute_pair_maxt_floor` still materialises ~5 full `C(k,2)` arrays with no chunking (prior FE_STEP-1, only half fixed)  [P2]
**Where:** `_mrmr_fe_step_helpers.py:416-426`

**What:** The prior wave's `list(combinations(...))` tuple blowup is gone, but the index arrays are not:

```python
_k_vars = np.fromiter(numeric_vars_to_consider, dtype=np.int64, count=len(...))
_ia, _ib = np.triu_indices(_k_vars.shape[0], k=1)     # 2 × C(k,2) int64
_maxt_pa = _k_vars[_ia]; _maxt_pb = _k_vars[_ib]      # 2 more × C(k,2) int64
_bias_vec = pairwise_mm_joint_bias(data, _maxt_pa, _maxt_pb, nbins, _k_y)   # 1 more × C(k,2) float64
```

**Why it is costly:** at k=5000 (`_step_pairmi.py:126` and the prior audit both use that width),
C(k,2)=12,497,500, so `_ia`/`_ib`/`_maxt_pa`/`_maxt_pb` are 4×100 MB = 400 MB live simultaneously, plus
100 MB for `_bias_vec` when `fe_mm_debias_prevalence` is on — ~0.5 GB in one allocation burst, before the
null kernel's own working set. This is the default path (`fe_pair_maxt_null_permutations=25 > 0`,
`n_pairs >= fe_pair_maxt_min_pairs=30`). The sibling in the SAME package — the "auto" prevalence debias at
`_step_pairmi.py:509-512` — was fixed to `_lazy_chunks(combinations(...), _auto_chunk_size)` and carries a
comment explicitly naming this anti-pattern; this one still has it.

**Fix:** chunk the pair space (`_lazy_chunks` or a strided `triu_indices` block walk) and accumulate the
per-shuffle max across chunks — the maxT statistic is a max, so chunk-and-combine is exact; the
`_pair_mm_bias` dict is built per chunk. Note `_pair_mm_bias` itself is an unbounded per-pair Python dict
(`:481-483`) built only under `mm_debias`, and it too is O(k²) entries — cap it to the pairs the gate will
actually look up, or key it lazily.

**Test:** `test_pair_maxt_floor_peak_rss_bounded_at_wide_pool` — monkeypatch the null kernel to record the
largest single array it receives; assert it never exceeds `chunk_size`, and assert the returned floor is
bit-identical to the unchunked reference at k=200.

---

### FE_STEP-6 — `_all_pairs_precomputed` re-enumerates the whole O(k²) pair space eagerly, even when the branch it guards is already decided  [P2]
**Where:** `_step_pairmi.py:243`, consumed at `:252`

**What:**

```python
_all_pairs_precomputed = n_pairs > 0 and all((p in cached_MIs or p in cached_confident_MIs)
                                             for p in combinations(numeric_vars_to_consider, 2))
...
if n_jobs <= 1 or n_pairs < max(2, n_jobs) or _all_pairs_precomputed or _below_perm_floor:
```

**Why it is costly:** Python's `or` short-circuits *inside* the `if`, but `_all_pairs_precomputed` is
computed on the preceding line, unconditionally. On the common `n_jobs <= 1` or `_below_perm_floor` fit
(the latter is the *default*, since `fe_npermutations` is typically 3 and the floor is 20, `:39,251`) the
result is never needed, yet the full `C(k,2)` generator is walked with two dict lookups per pair: ~25M dict
lookups at k=5000, several seconds of pure waste per FE step. And `all()` short-circuits on the **first**
miss, so the cost is fully paid precisely in the case where the answer is False and nothing is saved.

**Fix:** make it lazy — move the `all(...)` into the condition after the cheap predicates
(`if n_jobs <= 1 or n_pairs < max(2, n_jobs) or _below_perm_floor or _all_pairs_precomputed()`), or cheaper
still, compare counts: the batch prefill already returns `_n_pairs_batch` and counts `_batch_prefill_count`,
so `_batch_prefill_count + <pre-existing cached pairs> == n_pairs` answers the same question in O(1).

**Test:** `test_all_pairs_precomputed_not_evaluated_when_serial_branch_already_taken` — spy on
`combinations` (the package re-exports it for exactly this kind of introspection, `__init__.py:11`) and
assert it is not consumed for the precomputed check when `n_jobs == 1`.

---

### FE_STEP-7 — `_operand_cache` / `_single_corr_cache` memoize full-n float columns per operand with no bound  [P2]
**Where:** `_step_pairs_rank.py:460-493`, filled from `_prepass_gate_and_usability_candidates`
(`:420-421`) and `_maybe_relax_prevalence_for_tail_concentrated_pool` (`:197-198,228`)

**What:** `_cached_operand` stores `_usability_operand_continuous(self, X, cols, _idx)` — a full-length
float array — for every operand index touched, and never evicts.

**Why it is costly:** the prescan is capped (`fe_pair_usability_prescan_max_pairs`, default 256 →
≤512 operands), but `_prepass_gate_and_usability_candidates` calls `_cached_operand` for **every** pair that
fails the normal gates and has `pair_mi > 0` (`:418-422`) — i.e. potentially every operand in the pool. At
k=5000 raw operands and n=1M float64 that is 5000 × 8 MB = **40 GB**, on a code path whose module docstring
elsewhere notes frames are 100+ GB. The cache's own comment (`:451-459`) justifies the memoization on CPU
grounds and never addresses its footprint. `_batch_usability_admission_verdicts` then `np.vstack`s the
*strided* subset (`:102`), which is bounded — the unbounded part is the full-resolution cache behind it.

**Fix:** cache the **strided** operand (`_corr_stride` is already applied at every consumer, `:98,228`) and
store it as `_crit_np_dtype()` (f32) rather than raw f64 — that is what both consumers cast to anyway, so
it is bit-identical and ~8× smaller; additionally bound the cache with an LRU sized from
`fe_pair_usability_prescan_max_pairs`.

**Test:** `test_score_prospective_pairs_operand_cache_bounded` — a wide synthetic pool, assert
`len(_operand_cache)` stays within the configured bound and that each cached array's `nbytes` matches the
strided/f32 expectation. (`tests/.../test_score_prospective_pairs_operand_cache.py` today only pins that
the cache is *used*.)

---

### FE_STEP-8 — no FE-step stage consults the wall-clock deadline; the escalation / CMI-gate / stability-vote tails run unbounded past `max_runtime_mins`  [P2]
**Where:** whole package — `grep -r "fe_deadline" src/mlframe/feature_selection/filters/_mrmr_fe_step/`
returns **zero** hits; contrast `_feature_engineering_pairs/_pairs_dispatch.py`, `polynom_pair_fe.py`,
`_hinge_basis_fe.py` and 16 other FE modules which do consult it.

**What:** `_fe_deadline.py`'s docstring (`:9-11`) scopes the deadline to the "optional pre-FE enrichment
generators" and asserts "the `check_prospective_fe_pairs` pair-search and the `_confirm_predictor` greedy
step … carry their own budget". That is true of the pair-search call itself, but **not** of the stages
`_step_score` runs after it, all of which are per-candidate loops on the same fit:
- the CMI-gate candidate build (`:143-163`) and its escalation twin (`:846-867`);
- `run_fe_auto_escalation` (`:877-887`) — proposers + a full gate cascade per failed pair;
- `propose_additive_fusions` (`:992-1004`);
- `confirm_recipes_cross_fold` (`:1126-1139`) — K plug-in replays per surviving recipe.

Nor does `_step_pairs_rank`'s O(pairs) pre-pass (`:406-425`), which can run a conditional-permutation null
per asymmetric-synergy pair (`:331-366`).

**Why it matters:** a fit given `max_runtime_mins` can enter `_step_score` with the budget already spent
and still run every tail stage to completion. Also flag for the record (the brief asks): the thread-local
does **not** cross the loky boundary used at `_step_pairmi.py:321-324,350-377`, and no
`fe_deadline_scope` re-publish is done for those workers — consistent with `_fe_deadline.py:20-23`'s own
warning, and correct today only because the pooled `compute_pairs_mis` does not consult the deadline.

**Early-break desync check (the brief's specific concern):** no pre-sized parallel array is at risk if a
break were added to `_step_score`'s main materialise loop — `_data_chunks` (`:554,594`), `cols`, `nbins`
and `_newly_engineered_indices` all grow together per iteration and the single `np.concatenate` flush at
`:780-781` happens after. The escalation and fusion blocks, by contrast, pre-size
`_esc_new_codes = np.empty((len(X), len(_esc_admitted)))` (`:907-909`) and `_fz_codes` (`:1015`) and then
append `nbins` entries by `len(...)` — a break inside those fill loops WOULD leave
`data.shape[1]` / `len(cols)` / `len(nbins)` desynchronised. Any deadline work must break *between*
blocks, not inside those two fill loops.

**Fix:** add `if fe_deadline_passed(): break` between the FE-step's top-level stages (after the CMI gate,
before escalation, before fusion, before the vote), each with a `logger.warning` naming the skipped stage
so a truncated FE step is never silent. Do not add it inside the two `np.empty`-prefilled loops.

**Test:** `test_fe_step_stages_skipped_after_deadline` — set an already-elapsed `set_fe_deadline`, run one
`_run_fe_step`, assert `run_fe_auto_escalation` / `propose_additive_fusions` /
`confirm_recipes_cross_fold` were not called and that the returned `(data, cols, nbins)` widths agree.

---

### FE_STEP-9 — the CMI redundancy gate and the escalation twin densify `classes_y` with two different idioms; the escalation one truncates  [P2 — prior FE_STEP-5, still open, plus a new ravel divergence]
**Where:** `_step_score.py:118-120` vs `:840-844`

**What:**
```python
_y_codes = np.asarray(classes_y).ravel()                    # :118  gate — ravel, no cast
_, _y_dense = np.unique(_y_codes, return_inverse=True)

_esc_y = np.asarray(classes_y)                              # :840  escalation — no ravel
if not np.issubdtype(_esc_y.dtype, np.integer):
    _esc_y = _esc_y.astype(np.int64)                        # :842  truncating cast
_, _esc_y_dense = np.unique(_esc_y, return_inverse=True)
```

**Why it is wrong:** two defects in one pair of siblings that are supposed to build the same class codes.
(a) The truncation (prior FE_STEP-5): a fractional `classes_y` collapses distinct labels into one bucket
on the escalation path only. Masked today because `classes_y` is always the discretised integer screening
target — latent, not live. (b) **New:** the missing `.ravel()`. Under numpy ≥ 2.0 `np.unique(...,
return_inverse=True)` returns an inverse shaped like the input, so an `(n, 1)` `classes_y` yields an
`(n, 1)` `_esc_y_dense` while `_y_dense` is `(n,)`. `_esc_y_dense` is then passed straight into
`_esc_mi(_cb, _esc_y_dense, None)` at `:867`. **Unverified** whether any production path hands a 2-D
`classes_y`; what would settle it is a grep of `_run_fe_step`'s callers in `_mrmr_fit_impl` for the shape
of the `classes_y=` argument plus the installed numpy version.

**Fix:** one helper `_dense_y_codes(classes_y)` used by both sites: `np.unique(np.asarray(classes_y).ravel(),
return_inverse=True)[1].astype(np.int64)`.

**Test:** `test_fe_step_y_densification_identical_across_gate_and_escalation` — feed a `(n,1)` and a
float-valued `classes_y`, assert both sites produce the same 1-D int64 codes.

---

### FE_STEP-10 — `_ls_anchor` replays a nested-parent recipe over the full frame once per log-side operand, per candidate, with no memo  [P2]
**Where:** `_step_score.py:720-738`

**What:** for every admitted candidate whose `unary_a_name == "log"` (and again for side b), `_ls_anchor`
does a full `apply_recipe(_p, X)` of the nested parent over the whole frame just to compute one scalar
(`np.nanmin`).

**Why it is costly:** the same parent recipe is nested by many sibling candidates within one step (that is
what the feed-forward at `:626-643` exists to produce), and each of them re-replays it end to end over
full `n`. The result is a pure function of `(src_name, nested_recipe_identity)`, both fit-constant for the
whole call. The frozen anchor itself is correct and necessary (BUG2, `:711-719`); only the recomputation is
waste.

**Fix:** memoize by `(src_a_name_raw, id(_nested_a))` in a dict local to
`materialise_and_finalise_fe_candidates`; raw-operand anchors additionally reduce to a single
`np.nanmin(X[col])` per column, also worth memoizing. Cheap and bit-identical.

**Test:** `test_ls_anchor_memoized_across_candidates` — spy on `apply_recipe`, emit three candidates
nesting the same parent with `log` on side a, assert exactly one replay.

---

### FE_STEP-11 — the whole-frame `X.copy()` on the pandas materialise path  [P2]
**Where:** `_step_score.py:546-549` (`X = X.copy()`), guarded re-copies at `:897-899`, `:1007-1009`

**What:** on the pandas path a full `DataFrame.copy()` is taken once per FE step that admits any candidate,
to avoid mutating the caller's frame (`X[col] = ...` below is in place).

**Why it is costly:** CLAUDE.md's Memory/RAM discipline ("frames can be 100+ GB — never `.copy()`/
reconstruct a frame to work around a bug; mutate-and-restore (try/finally) or use views"). The `_x_is_owned`
flag already collapses three copies into one, which is the right direction but stops short. Note the polars
path is genuinely free (`with_columns` shares buffers, `:614-617`) and the documented
`get_pandas_view_of_polars_df` bridge means the *common* production path is the polars one — so this is a
real but not universal exposure.

**Fix:** either (a) mutate-and-restore: record the appended column names and `X.drop(columns=..., inplace=True)`
in a `try/finally` at the FE-step boundary, so the caller's frame is restored; or (b) build the engineered
columns into a small side-frame and `pd.concat([X, side], axis=1, copy=False)` once at the end, which
copies only the new columns' worth of blocks. Whichever is chosen, keep the "fit must not mutate caller
input" contract explicit and tested.

**Test:** `test_fe_step_does_not_copy_whole_pandas_frame` — wrap `pandas.DataFrame.copy` with a spy on a
frame large enough to be distinguishable, assert zero full-frame copies while
`test_mrmr_fit_does_not_mutate_caller_frame` (the existing contract) still passes.

---

### FE_STEP-12 — escalation marginal MI drops the `kx` hint its sibling passes, forcing a blocking device scalar sync per candidate  [P2]
**Where:** `_step_score.py:867` vs `:162`

**What:**
```python
_marg = float(_cmi_from_binned(_vb, _y_dense_g, None, kx=int(self.quantization_nbins)))   # :162 gate
_esc_admitted_pool[_cname] = (_cv, float(_esc_mi(_cb, _esc_y_dense, None)))               # :867 escalation
```

**Why it is costly:** `_fe_batched_mi.py:860-867` documents exactly this: without `kx`/`ky` the kernel
falls back to `int(X.max())` / `int(dy.max())`, "each drains the GPU queue ~ms; the kernel-timeline gap
analysis put ~4,900 such scalar D2H as the dominant remaining GPU-idle source". The escalation pool build
runs under the same `_gate_resident` residency flag (`:858`), i.e. exactly the regime where the sync hurts.
A pure sibling-divergence: the fix that landed on one call site never reached the other.

**Fix:** pass `kx=int(self.quantization_nbins)` (and `ky=` the class count) at `:867`. Subsumed if
FE_STEP-2's batching lands.

**Test:** covered by `test_step_score_cmi_cands_batched_matches_per_candidate` plus an assertion that the
escalation call site forwards a positive `kx`.

---

### FE_STEP-13 — the escalation pool scores marginals at full n while the gate strides to `MLFRAME_FE_GATE_MAX_ROWS`  [P2]
**Where:** `_step_score.py:130-133` (gate stride) vs `:852,867` (escalation, no stride)

**What:** the gate caps its scoring rows (`_gate_stride`, default 250k) with a comment explaining the
decision is selection-equivalent under a large strided subsample. The escalation admitted-pool build, whose
output feeds the **same** `apply_cmi_redundancy_gate`, uses the full `_tvals` column and the full
`_esc_y_dense`.

**Why it matters:** it is both the cost asymmetry (full-n binning + MI per admitted column at n=1M+) and
the estimator-scale inconsistency behind FE_STEP-3. The two numbers end up in one `pool` dict compared
against one relative bar.

**Fix:** apply the same `_gate_stride` (and the same `_y_dense_g`) to the escalation pool build; or, per
FE_STEP-3, push the MI estimation down into `run_fe_auto_escalation` so one population is used throughout.

**Test:** `test_escalation_pool_uses_same_scoring_rows_as_cmi_gate` — set `MLFRAME_FE_GATE_MAX_ROWS` low
and assert both builds score the same row count.

---

### FE_STEP-14 — pair ranking is by operand-reuse count, so ties resolve to the LOWEST-MI pair first  [P3]
**Where:** `_step_pairs_rank.py:762` (value written), `_step_core.py:456` (`sort_dict_by_value(...,
reverse=True)`), ordering source `_step_pairs_rank.py:525,580`

**What:** `prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[a] + vars_usage_counter[b]` —
the dict **value** is a cache-locality counter, not the MI. `_step_core.py:456` then sorts by that value
descending. The pair MI survives only inside the key tuple.

**Why it matters:** (a) `sort_dict_by_value` is stable, and the insertion order here is
`_sorted_pairs` = `sort_dict_by_value(cached_MIs)` **ascending** (`:525`, the docstring at `:179` confirms
"default is ASCENDING, so the highest-MI pairs are at the END"). So among pairs with equal reuse counts —
which is the overwhelming majority early in the walk, when every counter is 0 — **the weakest pair is
processed first**. (b) The counters themselves are accumulated in ascending-MI order, so the reuse ranking
is itself seeded by the weakest pairs. Under any downstream truncation (`fe_max_pair_features`, the rung
schedule's `min_pairs` floor, or a time budget), the lowest-MI members of a tie group get the expensive
operator search first. The rung screen at `_step_core.py:472-482` does re-rank by `key[1]` and mitigates
this when it fires (≥ `fe_rung_min_pairs`=6 pairs), but it does not when it self-gates off.

**Fix:** make the tie-break explicit — sort by `(usage_count, pair_mi)` descending, or simply iterate
`reversed(_sorted_pairs)` in the main loop so both the counter accumulation and the insertion order are
MI-descending. Either is a selection-ordering change and needs the usual selection-equivalence check.

**Test:** `test_prospective_pairs_tie_break_prefers_higher_mi` — construct pairs with identical reuse
counts and distinct `pair_mi`, assert the emitted order is MI-descending.

---

### FE_STEP-15 — `apply_interaction_information_routing`'s `getattr` default contradicts the constructor default and every comment  [P3]
**Where:** `_mrmr_fe_step_helpers.py:610` vs `mrmr/_mrmr_class.py:1898` and `_step_core.py:411,418`

**What:** `if not bool(getattr(self, "fe_ii_routing_enable", True))` — default **True**. The ctor declares
`fe_ii_routing_enable: bool = False` (`_mrmr_class.py:1898`), `_step_core.py:411` says "bench-rejected as a
DEFAULT (now `fe_ii_routing_enable=False`)", and `_interaction_information.py:137` says "this whole path is
default-off".

**Why it matters:** unreachable on a real `MRMR` instance (the attribute always exists), so no live bug —
but the fallback encodes the opposite of the bench-rejected verdict for any duck-typed `self` (tests,
partial mocks, a future carve that forgets to set it), and it silently enables a path whose own comment
records it as NEUTRAL-at-best and harmful on the weak-F2 fixture.

**Fix:** `getattr(self, "fe_ii_routing_enable", False)`.

**Test:** `test_ii_routing_default_off_for_bare_namespace` — call `apply_interaction_information_routing`
with a `SimpleNamespace` lacking the attribute; assert the pairs come back unchanged.

---

### FE_STEP-16 — `_gate_cache[raw_vars_pair]` is an unguarded lookup whose only safety is filter-chain duplication  [P3]
**Where:** `_step_pairs_rank.py:634`, populated by `_prepass_gate_and_usability_candidates:406-414`

**What:** the main loop does `... = _gate_cache[raw_vars_pair]` with no `.get`. Its correctness rests on
the pre-pass's four filter conditions (`len == 2`, `not in checked_pairs`, both operands in
`numeric_vars_to_consider`, `_ies > 0`) mirroring the main loop's own chain (`:581-584,598`) exactly — a
coupling the pre-pass docstring (`:388-391`) acknowledges by asking the reader to keep them in sync by
hand. Any future edit to either chain turns into a raw `KeyError` mid-FE-step.

Separately, both the pre-pass (`:411`) and the main loop (`:585`) index `cached_MIs[(idx,)]` unguarded — a
missing single-var MI is a `KeyError` rather than a diagnosable skip. (Not observed; screening does
populate these.)

**Fix:** `_gate_cache.get(raw_vars_pair)` with an explicit `if _gate_state is None: continue` plus a
`log_throttle` warning naming the pair, so a chain drift degrades to a skipped pair with a trace rather
than an aborted fit. Better still, have the pre-pass return the filtered pair list and have the main loop
iterate *that*, so one chain exists.

**Test:** `test_score_prospective_pairs_survives_gate_cache_miss` — monkeypatch the pre-pass to omit one
pair; assert the fit completes and logs a warning naming it.

---

### FE_STEP-17 — `_cached_single_corr` does not memoize its `None` result  [P3]
**Where:** `_step_pairs_rank.py:484-493`

**What:** when `_yc_cont_ is None` or shapes mismatch, the function `return None` **without** writing to
`_single_corr_cache`, so every subsequent call for that operand re-enters, re-resolves `_cached_operand`
and re-tests. Cheap per call, but on the no-continuous-y path (classification) it is every operand of
every pair in `_need_usability`.

**Fix:** `self`-consistent early return — hoist the `_yc_cont_ is None` test out of the per-operand
function entirely (it is call-constant), and cache the per-operand `None` for the shape-mismatch case.

**Test:** `test_single_corr_cache_memoizes_none` — count `_single_operand_usability_corr` /
`_cached_operand` entries when `_fe_prewarp_y_continuous_` is absent.

---

### FE_STEP-18 — `run_cluster_aggregate_emission` and the two `selected_vars` promotions use O(n²) list membership  [P3]
**Where:** `_mrmr_fe_step_helpers.py:569-570`; compare `_step_score.py:956-959`

**What:** `selected_vars = _sv + [i for i in _ca_indices if i not in _sv]` — `not in _sv` is a linear scan
of a list, inside a comprehension. `_step_score.py:957-959` does the same promotion correctly, with an
`_sv_set = set(_sv)` hoisted out. Same divergence again: one site got the fix, the sibling did not.

**Fix:** hoist `_sv_set = set(_sv)` in `run_cluster_aggregate_emission` exactly as `_step_score` does.
(`selected_vars` is small, so this is quality, not a live cost.)

**Test:** covered by a `test_conventions`-style lint, or simply folded into the cluster-aggregate unit test
proposed below.

---

### FE_STEP-19 — process/audit metadata still in comments (prior FE_STEP-3, partially open)  [P3]
**Where:** `_step_score.py:522` ("ROOT CAUSE 5 fix"), `:711` ("BUG2 FIX"), `:746` ("ND-1"), `:816`
("MM-DEBIAS (2026-06-09, + #4)"), `:1160` ("BUG2:"); `_step_core.py:68` ("FIX3:"), `:801,906` ("OPT-A"),
`:1017` ("ND-1"); `_step_pairmi.py:144,232,402` ("finding-#21 cap", "finding-#21 fix")

**What:** CLAUDE.md's comment rules ban phase/wave markers, finding IDs and date stamps in code comments
("that belongs in git history / the PR description"). The prior wave flagged a different set of IDs which
have since been stripped; these remain. Note `_step_pairmi.py:232,402` cite a bare "finding #21" that no
longer resolves to anything in-tree.

**Fix:** strip the markers, keep the WHY prose. Purely mechanical, but must be scoped to these files (see
CLAUDE.md's no-repo-wide-rewrite rule).

**Test:** extend `tests/test_conventions.py` with `test_no_finding_ids_in_fe_step_comments` matching
`\b(FIX\d|BUG\d|ROOT CAUSE \d|OPT-[A-Z]|ND-\d|finding[- ]#?\d+)\b` in comment lines of this package.

---

### FE_STEP-20 — `_step_core._run_fe_step_impl` is a 940-line single function; `_step_score.py` is 1189 LOC, over the 1k ceiling  [P3]
**Where:** `_step_core.py:118-1060`; `_step_score.py` (1189 LOC total)

**What:** `_step_score.py` is over the 1000-LOC budget that `test_no_file_over_1k_loc.py` backstops (the
module's own docstring says it was carved *out* of `_step_core.py` to bring **that** file under the ceiling,
and has since grown past it itself). `materialise_and_finalise_fe_candidates` is one ~1130-line function
with six distinct stages (CMI gate, gate-composite prune, cross-group prune, materialise+recipe,
escalation, fusion, vote).

**Why it matters:** CLAUDE.md: "Carve *before* a file nears ~800-900 LOC". The six stages have clean
boundaries (each consumes and re-emits `prospective_additions` / `data,cols,nbins,X`), so this is a
mechanical carve, and it is the precondition for the deadline work in FE_STEP-8 (stage boundaries need to
be callable).

**Fix:** split into `_step_score_prunes.py` (the two over-materialisation prunes + the CMI gate) and
`_step_score_tail.py` (escalation + fusion + vote), re-exported from `_step_score`. Per CLAUDE.md, AST-audit
each new sibling for unresolved `Load`-context names before commit.

**Test:** the existing `test_no_file_over_1k_loc.py` is the gate; add
`test_step_score_siblings_have_no_unresolved_names` mirroring the monolith-split AST rule.

---

### FE_STEP-21 — `_gate_cols_in` / `_gate_cols_in_fsc` are duplicated closures rebuilt per call, and the two `_bare_tokens` are byte-identical duplicates  [P3]
**Where:** `_step_score.py:277-284` vs `:427-439` (plus `import re as _re_gate` / `import re as _re_fsc`
at `:275` and `:425`)

**What:** `_bare_tokens` and `_bare_tokens_fsc` have identical bodies and identical regexes; `_gate_cols_in`
and `_gate_cols_in_fsc` differ only in which map they close over. Both blocks also re-`import re` under a
distinct alias.

**Fix:** one module-level `_BARE_TOKEN_RE = re.compile(...)` plus one `_bare_tokens(name)` and one
`_gate_cols_in(name, gate_map)`; subsumed if FE_STEP-4's provenance-based rewrite lands. Also note the
regex is recompiled implicitly by `re.findall` cache lookups once per name — a module-level compile is the
repo's own convention (`feedback_orjson_compile_regex`).

**Test:** n/a (pure dedup); guarded by FE_STEP-4's test.

---

### FE_STEP-22 — five FE-step sub-blocks still have zero direct unit coverage (prior FE_STEP-7, still open)  [P3]
**Where:** `_mrmr_fe_step_helpers.py:20` (`apply_synergy_bootstrap`), `:190` (`apply_surrogate_gbm_seeder`),
`:380` (`compute_pair_maxt_floor`), `:502` (`run_cluster_aggregate_emission`), `:586`
(`apply_interaction_information_routing`)

**What:** a grep of `tests/` for these five names still returns nothing; coverage is transitive through
full-fit integration tests only. Concretely untested contracts: the maxT floor's self-gate
(`n_pairs >= fe_pair_maxt_min_pairs`) and its mm-bias map keying; the II router's
`marginal_mi_a/b=np.zeros(len(_keys))` null-floor argument (`:629-630` — passing zeros as the marginal MIs
into an *interaction-information* null floor deserves a pinned test in its own right); the synergy
bootstrap's three-way cost/cap/pre-rank branch table (`:107-186`).

**Fix / tests:** see "Proposed tests" below.

---

## Proposed tests (beyond the per-finding ones)

1. `test_pair_maxt_floor_selfgates_below_min_pairs` — pool of 20 pairs with `fe_pair_maxt_min_pairs=30`;
   assert the floor is exactly `0.0` and `_pair_mm_bias == {}` (the byte-identical-narrow-pool contract).
2. `test_pair_maxt_floor_mm_bias_keys_are_canonical` — with `fe_mm_debias_prevalence=True`, assert every
   key of the returned bias map satisfies `k[0] <= k[1]` and that `score_prospective_pairs` finds a bias for
   every pair it gates (guards the `tuple(sorted(...))` coupling at `_step_pairs_rank.py:319,596`).
3. `test_ii_routing_null_floor_zero_marginals_contract` — pin that
   `pooled_pair_ii_null_floor` is called with zero marginal vectors and document (in the test's assertion
   message) why that is the correct null construction; today nothing states or checks it.
4. `test_synergy_bootstrap_cost_gate_branch_table` — parametrise over
   `(n_raw ≤/> cap) × (sweep_cost ≤/> budget) × (prerank on/off)` and assert which of the four log/branch
   outcomes fires; the three-way `if/elif/elif` at `_mrmr_fe_step_helpers.py:107-159` plus the second
   `if/elif/elif` at `:161-186` has no coverage.
5. `test_synergy_max_sweep_cost_zero_disables_not_unlimits` — the `:56-60` comment says a legitimate `0.0`
   must disable the sweep (not be widened to `inf`); assert it.
6. `test_fe_step_engineered_feed_forward_cap_tie_break_deterministic` — equal marginal MI on several
   engineered operands; assert the kept set is identical across two runs with different set-insertion
   histories (pins the `(-mi, v)` key at `_step_pairmi.py:110`).
7. `test_fe_step_checked_pairs_prevents_reprocessing_across_steps` — two FE steps, assert no pair appears in
   `prospective_pairs` twice and no `..._2`-suffixed duplicate engineered column is emitted (this is the
   integration-level detector for FE_STEP-1).
8. `test_step_score_nnb_length_mismatch_warns_once_per_block` — the three `log_throttle` sites
   (`_step_score.py:246,376,498`) have their warning text tested nowhere; assert the warning fires and that
   the pass-through `_nnb` is the original object.
9. `test_cluster_aggregate_emission_promotes_indices_once` — assert `selected_vars` gains each `_ca_indices`
   entry exactly once even when it is already selected (covers FE_STEP-18's site behaviourally).

---

## Prior-wave findings touching this cluster (2026-07-25 `fe_step.md`)

| Prior ID | Verdict in current source |
|---|---|
| FE_STEP-1 (O(k²) maxT pair build) | **PARTIALLY FIXED.** The `list(combinations(...))` tuple materialisation is gone — `_mrmr_fe_step_helpers.py:417` now uses `np.triu_indices`. The array half is **NOT** fixed: four `C(k,2)` int64 arrays (`_ia`,`_ib`,`_maxt_pa`,`_maxt_pb`) plus `_bias_vec` are still allocated unchunked. Re-filed as **FE_STEP-5** above with the corrected memory arithmetic (~0.5 GB at k=5000, not ~300 MB). |
| FE_STEP-2 (`_non_numeric_column_indices` silent except) | **FIXED.** `_helpers.py:39-43` now `logger.debug(..., exc_info=True)` with an explanatory comment. |
| FE_STEP-3 (audit metadata in comments) | **PARTIALLY FIXED.** The cited `FE_STEP_*` / `FE_ORCH_BUDGET-8` IDs and the stale line-number citations are gone. A different set remains — re-filed as **FE_STEP-19**. |
| FE_STEP-4 (`--` in prose at `_step_core.py:172`) | **FIXED.** `_step_core.py:170-177` now reads cleanly with no ` -- ` and no finding ID. |
| FE_STEP-5 (escalation `classes_y` truncation) | **STILL HOLDS**, re-filed as **FE_STEP-9** with an additional newly-found `.ravel()` divergence at the same pair of sites. |
| FE_STEP-6 (`_FE_FAMILY_WALL` cross-fit) | Out of this wave's file scope (`_fe_family_timing.py`); not re-verified. |
| FE_STEP-7 (five sub-blocks untested) | **STILL HOLDS** — re-filed as **FE_STEP-22**; grep of `tests/` for all five names still returns zero hits. |
| FE_STEP_B-1 (dtype-wrap, marked FIXED) | **CONFIRMED STILL FIXED.** All three materialise sites pre-widen via `_safe_code_dtype` (`_step_score.py:583-585`, `:905-909`, `:1013-1015`) and narrow at the append (`:920`, `:1024`). No raw-`quantization_dtype` `np.empty` code buffer remains in the file. |
| FE_STEP_B-2 (auto-debias O(k²), marked FIXED) | **CONFIRMED STILL FIXED** — `_lazy_chunks` at `_step_pairmi.py:509`. |
| FE_STEP_B-3 (`cached_MIs` canonicalisation, marked FIXED) | **ONLY HALF TRUE.** The primary prefill at `_step_pairmi.py:204` is canonical, but the retry prefill at `:423` is not — the prior wave verified one site and not the sibling. This is **FE_STEP-1**, the most serious finding of this wave. |
| FE_STEP_B-4 (double sort, marked FIXED) | **CONFIRMED STILL FIXED** — hoisted once at `_step_pairs_rank.py:525`, threaded via `sorted_pairs=` at `:531,571`. |
| FE_STEP_B-8 (`_nnb` pass-through, marked FIXED) | **CONFIRMED STILL FIXED** — all three prune blocks warn (`_step_score.py:246,376,498`). |
| FE_STEP_A-1 (mempool teardown, marked FIXED) | **CONFIRMED STILL FIXED** — `try/finally` in the `_run_fe_step` wrapper (`_step_core.py:112-115`). |
| "Deadline / budget: CLEAN" | **DISPUTED.** The prior wave judged `_fe_deadline.py` itself, which is indeed clean. It did not check whether the FE-step *consumes* it — nothing in `_mrmr_fe_step/` does. See **FE_STEP-8**. |

---

## Verified-clean

- **Rejection-ledger drain, serial vs joblib.** Both paths correctly pop the three reserved result keys
  before `prospective_additions` is treated as a pair map (`_step_core.py:939-959` per chunk, `:966-986` for
  the serial single dict), and `prewarp_specs_out=None`/`gate_med_specs_out=None` are passed on the joblib
  path with the specs recovered from the result payload — no list is mutated across the worker boundary.
  The spec-merge clobber the comment at `:926-938` describes is genuinely fixed.
- **Prewarp / gate-med spec persistence across FE steps.** Backed by `self._prewarp_specs_accum_` /
  `self._gate_med_specs_accum_` (`_step_core.py:672-696`) and re-bound to the SAME dict objects in
  `_step_score.py:67-74`, so a spec fit in an earlier step is still resolvable at recipe-build time. The
  `KeyError`-at-replay failure the comment describes cannot recur through this path.
- **`_safe_code_dtype` pre-widening + `.astype(data.dtype, copy=False)` narrowing** at all three materialise
  sites (see prior-wave table).
- **Stability-vote / C2-fusion drop bookkeeping.** Vote-failed names are removed from `selected_vars`, popped
  from `engineered_recipes`, AND recorded on `self._fe_stability_vote_dropped_` (`_step_score.py:1151-1173`);
  C2-subsumed fragments are preserved in `self._fe_subsumed_recipes_` **before** the pop (`:1056-1069`) and
  the nested-parent resolver consults that store (`:676-678`). The ordering constraint the comment asserts
  (C2 must run before the vote) is actually satisfied by the code layout (`:961` before `:1121`).
- **`_x_is_owned` copy-once bookkeeping.** The flag correctly prevents the escalation (`:897`) and fusion
  (`:1007`) blocks from re-copying a frame the materialise block already privatised. (The copy itself is
  FE_STEP-11; the *bookkeeping* is correct.)
- **`_step_score` main materialise loop parallel-array integrity.** `_data_chunks`, `cols`, `nbins` and
  `_newly_engineered_indices` grow in lockstep per iteration with a single `np.concatenate` flush at
  `:780-781`; `nbins` uses `np.concatenate` (not `+`, which would broadcast-add) with an explicit comment.
  No desync possible at this site.
- **Engineered feed-forward cap tie-break** (`_step_pairmi.py:110`): `sorted(..., key=lambda v:
  (-cached_MIs.get((v,), 0.0), v))` — the secondary index key makes the cap survivors deterministic despite
  the pool being a set. Correct, and the comment accurately states why.
- **`_pair_maxt_floor` order-independence.** The floor is a quantile over per-shuffle maxima across the whole
  pair set, so the non-deterministic iteration order of `numeric_vars_to_consider` (a set) does not affect
  it; `np.fromiter` is correctly used because `np.asarray` cannot convert a set.
- **`_step_pool.build_fe_operand_pool` name→index maps.** Both `_cols_idx` (`:73`) and `_fni_idx`
  (`_step_core.py:621`) are built once, replacing O(K·F) `.index()` rescans. Correct and well-commented.
- **`compute_pair_maxt_floor`'s `assert _pair_maxt_floor is not None`** (`_mrmr_fe_step_helpers.py:498`) is
  genuinely provable from the two assignment paths; not a mypy-silencing hack.
- **Batch-precompute failure logging** (`_step_pairmi.py:214-228`) is at WARNING with the shape context,
  not `debug` — this is the one silent-fallback class the repo has shipped twice, and this site is correct.
  Same for the loky-failure (`:392-397`) and retry-failure (`:433-438`) handlers.
- **`fe_gpu_strict_resident_enabled` forwarding** (`_step_core.py:180-204`) uses an explicit kwarg
  allowlist rather than a `locals()` snapshot, with an accurate comment on why.
- **mypy surface**: no implicit-`Optional` parameters and no return-annotation mismatches in the eight
  scoped files; the `dict | None` / `float | None` annotations at `_step_pairs_rank.py:264`,
  `_step_core.py:672,693`, `_mrmr_fe_step_helpers.py:399` are all explicit.

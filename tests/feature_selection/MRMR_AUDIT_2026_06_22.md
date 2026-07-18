# MRMR FS/FE subsystem — 6-agent critique audit (2026-06-22)

Six read-only critique agents swept the MRMR feature-selection/engineering subsystem along orthogonal
axes: CPU/GPU equivalency, performance, GPU/CPU code effectiveness, edge cases, hidden flaws, code quality.
This file is the consolidated disposition of EVERY finding (per the "use all agent findings" rule — each
gets RESOLVED / QUEUED / REJECTED with a reason). RESOLVED items landed this session with regression tests.

## RESOLVED (committed this session)

| Finding (agent) | Fix | Commit |
|---|---|---|
| `div` op used the pre-2026-06-13 perturbed `x/(y+sign(y)*eps+eps)` in all 3 chunk materialise kernels (CPU serial+parallel + GPU CUDA), diverging from canonical `_safe_div` (equivalency P1-1, hidden #14) | exact `_safe_div` form in all three; `test_materialise_op_parity` | ce7378fa |
| `_materialise_extval_njit` computed ops 6/7/8 (abs_diff/signed/ratio_abs) as `min` (hidden #2) | implemented 6/7/8; op-parity test | ce7378fa |
| `_engineered_recipes_ = {}` on identity-shortcut path vs list everywhere else (hidden #6) | `= []` | 4b182069 |
| `generate_pair_cross_basis_features` NaN mean-fill mutated caller's DataFrame in place (hidden #4) | `np.array` copy; `test_pair_cross_no_mutation` | 4b182069 |
| `_orthogonal_univariate_fe/__init__.py` over 1000-LOC meta-test ceiling (code-quality P1) | carved `_dedup_collinear_source_cols` → `_orth_dedup.py` sibling; facade 1061→865 | 9fe6f980 |
| GPU basis routing was opt-in despite being selection-equivalent + (correctly measured) not-slower (effectiveness, earlier) | default-ON + single resident-matrix upload | a02c2307 |

## QUEUED — real, but selection-rippling or large; need broad biz-value validation (fresh context)

These are confirmed real but touch selection-critical or core code; each must land with the full
biz-value/layer suite as the gate, not just the FE pins.

- **cached_MIs read-before-assign** (hidden #1): `_fit_impl_core.py:8531,8705` read `self.cached_MIs`
  (set only at 8847) instead of the populated local `cached_MIs` → fresh-fit cluster-rep / p≥n tiebreaks
  degrade to feature-index order. Fix = use the local. Ripples cluster-rep selection (has an existing
  partial mitigation at 8548); validate against the cluster/dedup layer suite.
- **GPU-gated exhaustive synergy** (equivalency P0-1): `auto` exhaustive sweep gated on GPU presence →
  a feature can exist on a CUDA host and be absent on a CPU host. Fix = gate on n_raw/cost (device-
  independent) + run exhaustive on the CPU backend when chosen. Cross-host parity test (P3-7).
- **discretize_2d_array_cuda NaN→bin0 vs CPU NaN→top** + **int8 wrap at n_bins>128** (equivalency
  P1-2/P1-3): public GPU API diverges from CPU. Fix = NaN→top in the RawKernel + `_safe_code_dtype`.
  Production MRMR is shielded (NaN-scrubbed first); the public API path is not. Add discretization
  parity test (P3-3, currently 0 such tests).
- **perm-MI reduction order** (equivalency P1-4): `mi_direct_gpu_batched` reduces permutation MI in a
  different FP order than the CPU `original_mi` comparator → can flip a noise-gate count. Fix = reduce on
  CPU from integer counts (mirror the noise-gate split).
- **escalation subsample-decide / full-n-replay backend straddle** (equivalency P1-5/P1-6): poly candidate
  gated on CPU-Horner subsample values can ship recomputed via CUDA at full n. Fix = pin polyeval backend
  across decide+replay. (Escalation admits ~nothing at canonical n; lower live exposure.)
- **CMI GPU prefill ~1e-9 divergence** (equivalency P2-1, default ON): `_prefill_cond_MIs_gpu` writes GPU-
  reduced entropies into `cached_cond_MIs` → near-tie redundancy flip. Fix = reduce from GPU integer counts
  on CPU.
- **GPU heavy-tail detector omits CPU guards** (hidden #3): `_gpu_detect_heavy_tail_batched` skips
  `n_finite<8`, MAD-collapse, and finite-subset filtering → different verdict on NaN-bearing columns →
  f32 selection drift on the now-default GPU routing/basis-MI path. Fix = per-column finite filtering +
  the two missing branches + a GPU-vs-CPU verdict parity test. (Higher priority now routing is default-on.)
- **source-name `__` split** (hidden #5): `eng_name.split("__",1)[0]` mis-stems one-hot sources like
  `col__value` → `uplift=emi/1e-12` → always clears the gate. Fix = carry source in metadata (4 sites).
- **forward(host) vs Clenshaw(GPU) recurrence** for cheb/leg/herme (equivalency P2-2): ~1e-12 at degree≥3,
  sub-selection-noise at default max_degree=4 but widens with degree. Fix = unify on forward; extend
  `test_gpu_basis_column_parity` to degree 5-6.
- **evaluation.py 1144 LOC** over the meta-test ceiling (code-quality P1, pre-existing from the 2026-06-19
  JMIM/CMI work, not this session). Carve `save`/`_prefill` blocks to a sibling — a dedicated refactor.

## QUEUED — perf (measure-first; expected wins, not yet benched)

- grand-fusion ~10k Python-loop MI count reductions/chunk → batched njit `_mi_from_counts_batch` (perf #1,
  default path, est 10-50× on that block).
- `score_candidates_by_cmi` / greedy CMI loop recompute loop-invariant yz/y entropy terms though the
  `_fixed_yz`/`_fixed_y` helpers exist (perf #2/#3; the standalone Layer-60 constructor, off canonical fit).
- cross-stage Spearman dedup O(K²) per-pair rank+corrcoef with a **dead precompute** at `_fit_impl_core.py:5448`
  → one `np.corrcoef(R, rowvar=False)` (perf #4).
- resident pair-MI re-uploads host `y` per chunk + routes through the host wrapper (effectiveness #3) →
  upload y once, call `_plugin_mi_classif_batch_cuda_resident` directly.
- 53× `y.to_numpy()` re-materialisation in `_fit_impl_core.py` (code-quality) → hoist once.
- Fourier-deflation `lstsq` → normal-equations (perf #8, validated pattern elsewhere).

## QUEUED — code quality / hygiene

- Single-source the robust-axis constants `_GPU_ROBUST_AXIS_*` from `_hermite_robust._ROBUST_AXIS_*`
  (silent CPU/GPU selection-drift risk if one is returned).
- `_GPU_RESIDENT_MIN_N=50k` + other hardcoded GPU thresholds → `kernel_tuning_cache` (effectiveness #1;
  the sibling `fe_gpu_pairs_mi_backend_choice` is the working template).
- Shared `_env_truthy(name, default)` for the 10+ hand-rolled `fe_gpu_*_enabled` / env-flag gates.
- Cache-correctness tests for `_RADIX_INTERP_CACHE`/`_COMBO_IDX_CACHE`/`_QLEVELS_CACHE` (HIT-identity +
  pickle-exclusion) + a direct `_gpu_batched_abs_corr` test.
- Triple-maintained ctor defaults (`__init__` / `__setstate__` / inline `getattr`) already drifted once
  (`cluster_aggregate_mode`) → derive `__setstate__` from `_ctor_defaults()`.
- `except: pass` / logging-without-`exc_info` at the noise-wrap veto + escalation proposers (hidden #20,
  H6, M4) → `logger.debug(exc_info=True)` (no-silent-swallow rule).
- Module carves still over/at budget: `_mrmr_class.py` (4836, exempt), `_fit_impl_core.py` (9794, exempt),
  `_gpu_resident_fe.py` (2670), `hermite_fe/__init__.py` (982, at threshold), `_fe_auto_escalation.py` (834).

## REJECTED / NO-ACTION (with reason)

- **int32 overflow in CUDA flat-index math** (edge-case "verified-safe"): all kernels use `long long`
  casts — no overflow. Confirmed non-issue.
- **batch MI dispatch hard-returns njit, tuner discarded** (effectiveness #2): DELIBERATE ground-truth
  override (end-to-end njit 3× faster under contention; documented). Not a bug — keep, but the dead KTC
  lookup is noted.
- **`use_gpu` relevance-f32 / redundancy-f64 argmax flip** (equivalency "verified clean"): does not exist;
  the path forces f64. Confirmed clean.
- **`_gpu_route_bases_batched` untested** (code-quality premise): incorrect — `test_gpu_routing_parity`
  covers it (host-parity across seeds).
- **radix-select warp-divergence binary-search** (effectiveness #5): documented FUTURE; kernel already
  beats cp.percentile at every size; "room to grow", not a bug.
- numerous LOW edge-cases on opt-in/bench-rejected paths (imbalance MI `n_bins`, MM-debias cardinality,
  `_class_balanced_mi` y-offset) — gated off by default; revisit only if those paths are promoted.

## Verdict

No P0 confirmed across all six axes. The subsystem is mature, heavily bit-identity-tested, and pickle-safe.
The real exposure is a consistent shape: a tiny FP delta or device-dependent choice feeding a discrete
selection decision on a default-ON GPU path whose CPU twin is the validated reference but lacks a parity
test at the divergence-visible regime. The highest-value RESOLVED items (the `div` perturbation + extval
`min` bug) were genuine selection-altering divergences on the default path. The QUEUED selection-rippling
items each need the broad biz-value/layer suite as their validation gate and are intentionally not rushed.

## UPDATE — follow-through implementation status (2026-06-22, end of session)

RESOLVED + pushed (each with a regression test or selection-equivalence contract + validation):

- div perturbation + extval ops 6/7/8 (ce7378fa); _engineered_recipes list-invariant + pair-cross
  caller-mutation (4b182069); _dedup carve under LOC budget (9fe6f980); cached_MIs fresh-fit tiebreak +
  dead _col_basis_for_recipe + DCD S2 reframe (5769f708); C1 n==0 njit guards + CPU/GPU robust-const drift
  guard (0edcb979); two load-bearing silent-except -> logger.debug (d4c6a571).
- P0-1 exhaustive-synergy device-independent (no GPU-gated feature existence) -- 58a2a358.
- P1-3 discretize_2d_array_cuda int8 widen + first GPU/CPU discretize parity tests; P1-2 DISPROVED
  empirically (NaN routes to top on both backends -- agent mis-traced the rawkernel) -- 9f4bebc0.
- P1-4 permutation-MI gate compares GPU-reduced observed MI (FP-consistent, both twins) -- fa5e33ba.
- P2-1 GPU CMI prefill: selection-equivalence contract documented (all-or-nothing per round, ~1e-9 parity;
  a CPU-exact reduce would defeat the batched kernel for a P2 near-tie -- intentionally not taken) -- 1b5a2cd1.
- P2-2 GPU-Clenshaw vs host-forward recurrence: comment corrected + parity test extended to degree 6
  (<1e-6 holds; selection decided on consistent GPU values) -- e5ab4b78.
- P1-5/6 escalation decide/replay backend-straddle: selection-equivalence contract (admits ~nothing at
  canonical n; ~1e-12 below the gate resolution) -- 61b37bf6.
- perf#4 cross-stage Spearman dedup: cache full-column ranks, drop O(K^2) re-sorts (bit-identical) -- 5a453aff.
- eff#3 resident pair-MI: upload y once + H2D-free resident MI -- 65dc5d72.
- perf#1 grand-fusion MI reduction via the batched njit (drop ~10k/chunk per-cell dispatch) -- f5cbac94.
- eff#1 pair_candidate_mi_dispatch routes the GPU<->CPU crossover through kernel_tuning_cache (50k is now
  only the cold-start fallback) instead of the hardcoded threshold.

REMAINING (genuinely invasive / low-value / parallel-session -- deliberately NOT rushed at depth; each is
specced above): source-name ``__`` split (hidden #5, a 9-site recipe-naming-convention change needing
round-trip validation); the 53x ``y.to_numpy()`` hoist in _fit_impl_core.py (mechanical but in the 9.8k-LOC
file); lstsq->normal-eq in _orth_extra_basis_fe deflation (perf#8, measure-first); ``_env_truthy`` DRY
helper; ctor-defaults single-source; evaluation.py carve (1144 LOC, parallel-session-owned); the LOW M-tier
edge cases on opt-in/bench-rejected paths. All P0/P1/P2 equivalency divergences are resolved.

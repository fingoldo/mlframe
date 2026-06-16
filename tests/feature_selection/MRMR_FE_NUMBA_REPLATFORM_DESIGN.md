# MRMR FE — Unified zero-copy numpy/numba re-platform (DESIGN)

Status: PLAN (2026-06-16). No prod wiring yet — phased, each phase ships behind a default-off
flag and flips to default only after a multi-seed bit-parity + biz-value gate passes. Authored as
the "plan it separately" deliverable; implementation begins after the full-suite run on the big
machine confirms the current master baseline.

---

## 1. Why (the problems this collapses into one)

Five separately-observed issues share ONE root: the FE candidate-**generation** path is pandas-bound
and materialises every family's candidates into a single growing frame, and the cheap pair screen
ranks by a marginal-ish statistic it cannot trust on weak interactions.

| # | Symptom (observed this campaign) | Today |
|---|---|---|
| 1 | **polars input** silently skips FE (families guard `isinstance(X, pd.DataFrame)`) | no native path |
| 2 | **OOM at ~1M rows** — 46 families append candidates into one growing pandas `X` | all candidates live at once |
| 3 | **FE-scan speed** — conditional-gate is profile hotspot #1 (400k profile) | python/pandas per-family |
| 4 | **GPU under-used** — `apply_gpu_*_batched` exists but is not the unified default path | partial, opt-in-ish |
| 5 | **weak-interaction detection** — rung-0 / I4 / I5 lose low-marginal needles (`log(c)·sin(d)`); rung is now no-drop-by-default at the cost of its speedup | cheap pair_mi screen can't separate weak-genuine from weak-noise |

A single matrix-native core with a kernel ladder + a real joint-synergy screen closes all five.

## 2. Non-goals

- NOT rewriting the MI/CMI math: `permutation.mi_direct`, `info_theory.mi`, joint-histogram batches
  are already njit/cuda + kernel_tuning_cache-dispatched. They stay; the re-platform FEEDS them a
  matrix instead of a frame.
- NOT changing the SELECTION algorithm (Fleuret greedy, synergy bootstrap, maxT null, redundancy
  drop). Those operate on MI scores and stay identical. Only candidate generation + the cheap
  pre-screen change.
- NOT a public API change. `MRMR.fit`/`transform` signatures, `get_feature_names_out`, recipes,
  pickle contract all preserved bit-for-bit on the pandas path.

## 3. Current entry points (what gets replaced vs reused)

- `feature_engineering.py` — `create_unary_transformations` / `create_binary_transformations`
  (the family registry), `apply_gpu_unary_batched` / `apply_gpu_binary_batched` /
  `gpu_compatible_{unary,binary}_names` (the EXISTING partial GPU path — folded into the ladder),
  `_can_hoist_shared_buffer` / `_estimate_fe_shared_buffer_bytes` / RAM budgeting (reused as the
  ladder's memory governor), `compute_pairs_mis`.
- `_mrmr_fe_step/_step_core.py` (2212 LOC) — the per-step FE driver: builds prospective pairs,
  applies the rung-0 screen (`apply_rung_schedule`), runs the operator search. This is the main
  consumer that switches from frame-append to matrix-fill.
- `_mrmr_fit_impl/_fit_impl_core.py` (8816 LOC — already over the 1k limit; do NOT add here, carve
  siblings) — recipe materialisation, redundancy drop, fallback. Reads the engineered columns; gets
  a thin adapter so it can read from either the legacy frame or the new matrix.

## 4. Architecture — the ladder

```
INPUT (pandas | polars | pyarrow)
   │  zero-copy: pa.Array buffers -> np.ndarray view (no to_numpy copy; numba reads pyarrow
   │  buffers directly per the confirmed zero-copy entry). Nullable -> validity mask + sentinel.
   ▼
RAW MATRIX  X_raw: (n_rows, n_raw) float32 (usability_feature_dtype default), C-contiguous,
   plus a parallel int8 categorical/codes plane where needed.
   ▼
CANDIDATE GENERATION (per family, streamed in BLOCKS — never all-at-once -> fixes OOM)
   for each family f:
     pick kernel by dispatch(f, n_rows, n_cands, hw) via kernel_tuning_cache:
        ladder = [njit_serial, njit_prange, numba_cuda, cupy]
     write candidates into a REUSED block buffer (max_block_cands columns), score against y
     (joint MI / pair MI) IN-PLACE, keep only survivors -> free the block.
   ▼
JOINT-SYNERGY SCREEN (replaces the cheap rung-0 pair_mi cut)
   score each prospective pair by its TRUE joint MI(pair; y) on a cheap fixed binning, NOT a
   marginal proxy; a low-marginal high-joint needle (c,d) ranks correctly -> survives without the
   full operator search. This is the no-drop-AND-fast screen the rung default lacked.
   ▼
SURVIVOR MATRIX  X_eng: only kept candidates (bounded, not the full pool)
   ▼
OUTPUT  back to pandas OR polars (native), same column names / recipes as today.
```

### Kernel dispatch (the iron rule)
Every family's kernel choice routes through `pyutilz.system.kernel_tuning_cache` keyed on
`(family, n_rows, n_cands, dtype, hw)` — mirror `joint_hist_batched` / `plugin_mi_classif_dispatch`.
NEVER hardcode a block size / threshold / variant. Keep ALL kernel versions (njit / prange / cuda /
cupy) under distinct names; the dispatcher picks, so we can re-bench per hardware and roll back.

## 5. Phases (each independently shippable, default-off → default-on after its gate)

- **P0 — Arrow zero-copy entry + matrix adapter** (new module `_fe_matrix_io.py`). pandas|polars|
  arrow -> (X_raw float32, codes int8, validity). Round-trip back. Gate: byte-parity vs current
  `X.to_numpy()` on the test fixtures; nullable + categorical covered. No behaviour change yet.
- **P1 — Unary families on the matrix, njit-serial** (`_fe_kernels_unary.py`). Port
  `create_unary_transformations` ops to njit kernels writing into a block buffer. Fold in the
  existing `apply_gpu_unary_batched`. Gate: engineered VALUES bit-identical (≤1e-6) to the pandas
  path, multi-seed; biz-value unchanged.
- **P2 — Binary families + block streaming** (`_fe_kernels_binary.py`). The OOM fix: generate +
  score + prune per block, reusing the buffer. Gate: 1M-row fixture completes under the RAM budget;
  survivors identical to pandas.
- **P3 — Joint-synergy screen** (`_fe_synergy_screen.py`). Replaces the rung-0 cheap cut with a true
  cheap joint-MI rank. Gate: the rung IDENTICAL fixture stays no-drop AND recovers the (c,d) needle
  WITHOUT keep_frac=1.0 (i.e. the screen is now both safe and fast) -> rung default can re-enable a
  speedup. Also re-checks I4/I5 cases at n=8k (the screen should detect the needle at lower n than
  the marginal path could).
- **P4 — prange / cuda / cupy ladder + kernel_tuning_cache wiring**. Bench prange vs cuda vs cupy on
  the FE families (the GPU bench-off). Gate: dispatcher picks fastest per hw; bit-parity on every
  rung of the ladder; pickle suite green (no live-kernel objects cached without __getstate__).
- **P5 — polars-native exit + remove the `isinstance(X, pd.DataFrame)` FE guards**. Gate: a polars
  input produces the SAME engineered features as the pandas input (names + values), end to end.

Order rationale: I/O first (everything depends on it), then correctness-only single-thread kernels
(easy bit-parity), then the OOM streaming, then the screen (the accuracy win), then GPU (pure speed),
then polars (pure reach). Each phase keeps the suite green; a phase that can't hit bit-parity stays
default-off with its delta documented in-code (bench numbers), never force-flipped.

## 6. Validation & rollback

- **Bit-parity harness**: for each ported family, assert engineered column values match the pandas
  path within usability dtype tolerance across ≥5 seeds and {uniform, normal, lognormal, heavytail}.
- **Biz-value gate**: the downstream HGB/Ridge uplift on the canonical + realistic fixtures must not
  regress (reuse the I5 / biz_value tests).
- **Pickle**: any cached compiled kernel / cupy handle excluded via `__getstate__` IN THE SAME change
  (the runtime-caches-break-pickle rule); run the pickle suite after P4.
- **Rollback**: every phase behind `MLFRAME_FE_MATRIX_<PHASE>` (default off until its gate passes).
  Legacy pandas path stays intact until P5; flipping the flag off restores it instantly.
- **Per-feature discipline**: each new kernel ships with a unit test + a biz-value test + a cProfile
  hotspot pass (the standing per-feature rule).

## 7. Open questions to resolve in P0/P3

- Arrow zero-copy for the NULLABLE / categorical planes: confirm numba reads the validity bitmap
  buffer without a copy, or whether a one-time densification is cheaper (measure, don't assume).
- The joint-synergy screen's fixed binning: MDLP vs fixed-quantile. The I4/I5 finding was that
  marginal MDLP collapses an interaction-only operand to 1 bin; the screen must bin the PAIR jointly
  (or use a 2D fixed grid) so a low-marginal operand still gets resolution. Prototype both, bench
  detection-vs-noise at n ∈ {8k, 25k, 100k}.

## 8. Relationship to shipped work

This supersedes the rung opt-in tradeoff (commit 82462930): once P3's joint-synergy screen lands, the
rung can prune aggressively AND stay no-drop, so `_fallback_keep_frac` can return a fraction again
(gated on the screen being active). It also retires the I4/I5 "tracked-red, needs re-platform"
caveat by giving the selection path a screen that detects weak interactions at realistic n.

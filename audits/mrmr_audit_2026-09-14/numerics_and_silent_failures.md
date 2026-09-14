# numerics_and_silent_failures — mrmr_audit_2026-09-14

Cross-cutting lane. Not a file cluster: this doc sweeps the WHOLE MRMR surface for exactly two bug
classes and ignores everything else. Findings are tagged `NUM-<n>`.

## Scope

The 77 `*mrmr*` `.py` files (28 145 LOC excluding `_benchmarks/`), **plus** the MI / entropy /
discretisation / correlation / moment primitives those files call into even where the primitive lives
outside an `*mrmr*` path:

- `filters/mrmr/` (8 files), all 12 top-level `filters/_mrmr_*.py`, `_relaxmrmr_3d.py`,
  `training/composite/discovery/_mrmr_base_rank.py`
- `filters/_mrmr_fit_impl/` (incl. `_friend_graph_and_redundancy/`, `_hybrid_orth_family_variants/`),
  `filters/_mrmr_fe_step/`, `_mrmr_fe_step_helpers.py`
- MI/entropy primitives: `filters/info_theory/`, `_fastmi.py`, `_ksg.py`, `_renyi_alpha.py`,
  `_interaction_information.py`, `_neural_mi.py`, `_mi_aggregator.py`, `_fe_batched_mi*.py`,
  `_orthogonal_univariate_fe/_orth_mi_backends.py`, `batch_pair_mi_gpu.py`, `_gpu_resident_pair_mi.py`,
  `_resident_candidate_mi.py`
- discretisation: `_adaptive_nbins.py`, `discretization/`, `supervised_binning.py`, `_mdlp_validated_split.py`
- moment/correlation primitives reached from the MRMR FE loop: `_orthogonal_univariate_fe/_orth_dedup.py`,
  `_feature_engineering_pairs/_pairs_core.py` / `_pairs_score.py` / `_pairs_setup.py`,
  `_usability_njit_pool.py`, `_gpu_resident_basis.py`, `_hinge_detect_gpu_resident_batch.py`,
  `_stability_cluster.py`, `stability.py`, `hermite_fe/`, `bases.py`,
  `_cat_target_encoding_and_weighted.py`, `_surrogate_interaction_seeder.py`

---

## Bug class 1 — cancellation

### NUM-1 — the FE collinearity-dedup correlation matrix computes variance from raw power sums, in ALL THREE backends  [P0]

**Where:** `src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_dedup.py:92-106`
(numpy/BLAS, the default), `:118-144` (njit), `:172-183` (cupy).

**What:** numpy backend, lines 92-105:
```python
n = Qmf @ Rmf.T
Sx = Q0 @ Rmf.T
Sy = Qmf @ R0.T
Sxx = (Q0 * Q0) @ Rmf.T
Syy = Qmf @ (R0 * R0).T
Sxy = Q0 @ R0.T
with np.errstate(divide="ignore", invalid="ignore"):
    cov = Sxy - Sx * Sy / n
    varx = Sxx - Sx * Sx / n
    vary = Syy - Sy * Sy / n
    corr = np.abs(cov / np.sqrt(varx * vary))
corr[(n < 8) | (varx <= 1e-24) | (vary <= 1e-24)] = np.nan
```
njit backend, lines 133-143:
```python
sxx += a * a
syy += b * b
sxy += a * b
...
cov = sxy - sx * sy * inv
vx = sxx - sx * sx * inv
vy = syy - sy * sy * inv
out[i, j] = np.nan if (vx <= 1e-24 or vy <= 1e-24) else abs(cov / np.sqrt(vx * vy))
```
cupy backend, lines 178-182: identical algebra (`vx = Sxx - Sx * Sx / n`).

**Why it is wrong / costly:** this is `var = E[x²] − E[x]²` at k=2 — the exact form CLAUDE.md's
2026-09-02 follow-up says to grep for at any k >= 2. The absolute error in `varx` is
~`eps · Sxx ≈ eps · n · mean²`, so the computed variance is meaningless once
`|mean| / std ≳ 1/sqrt(eps) ≈ 1e8` in float64, and materially wrong from `~1e6` upward. Regime that
bites: a candidate column carrying a large offset relative to its spread — an epoch-second column, a
price/revenue column, a count, or any engineered recipe of the `x + const` / `log(x)+c` / cumulative
shape. Two failure directions, both silently actionable:
1. `varx` collapses below the `1e-24` gate → the entry is set to `nan`, and this function's own
   docstring (`:189`) says `nan` means **'not a duplicate'** — so a genuinely collinear engineered
   column is declared distinct and both copies survive into the selected set.
2. `varx` survives but is wrong by orders of magnitude → `|r|` is arbitrary, so a good feature can be
   dropped as a duplicate of an unrelated one.

This is the dedup gate for every FE family's candidate block (the file's own comment at `:154-156`
lists orth / extra-basis / gpu-resident / wavelet / hinge, <=6 calls per `MRMR.fit`), so the blast
radius is the whole engineered-feature surface. The sibling primitive
`_feature_engineering_pairs/_pairs_core.py:34-40` documents this exact bug being found and fixed on
the CPU pair-correlation kernel ("at offset/spread 1e7 a true |r| of 0.300 was reported as 0.767, and
on one minute of epoch-second ticks a true |r| of 0.497 came back as EXACTLY 0.0") — `_orth_dedup.py`
never received the same treatment.

**Fix:** the pairwise-complete structure (per-pair means, because of per-pair NaN masks) blocks a
literal two-pass matmul, but a **row-wise pre-shift is exact and free**: Pearson `r` is invariant to
any per-row translation, so subtract each row's own finite mean from `Q` and `R` once
(`Q = Q - np.nanmean(Q, axis=1, keepdims=True)`, likewise `R`) before the six matmuls. The raw-sum
algebra is then applied to data whose offset is ~0, and the residual cancellation is bounded by the
gap between the row-global mean and the pairwise-complete mean — negligible except under extremely
structured missingness. For the njit backend do the honest thing: a true per-pair two-pass loop, the
form `_pairs_core._abs_corr_finite_njit:41-72` already uses (it is documented as SLOWER than BLAS and
never auto-selected, so the extra pass costs nothing in production). Separately, make the degeneracy
gate **scale-relative** (compare `varx` against `n * mean_sq_scale * tol`) instead of the absolute
`1e-24`.

**Test:** `test_orth_dedup_corr_stable_on_large_offset_small_scale` — build two exactly-collinear
columns at offset 8.5e3 / spread 0.05 (the `_target_encoding_fe` regression regime), assert
`_pairwise_complete_abs_corr` returns `>= 0.999` (not `nan`) for all three backends, and assert the
three backends agree to 1e-9 with `np.corrcoef` on the same data.

---

### NUM-2 — the GPU-resident basis screening correlation RawKernel uses raw power sums  [P1]

**Where:** `src/mlframe/feature_selection/filters/_gpu_resident_basis.py:755-783` (`_ABS_CORR_SRC`).

**What:**
```c
psx += v; psxx += v * v; psxy += v * yy;
...
double cov = nf * sxy - sx * Sy;
double vx = nf * sxx - sx * sx;            // = n * sum((v-mean)^2)
double vy = nf * Syy - Sy * Sy;
double den = sqrt(vx * vy);
double corr = den > 1e-300 ? fabs(cov / den) : 0.0;
bool ok = isfinite(corr) && (sqrt(vx / nf) > 1e-12);
out[c] = ok ? corr : -1.0;
```

**Why it is wrong / costly:** the inline comment `// = n * sum((v-mean)^2)` is the algebraic identity,
not the numeric one — same k=2 cancellation as NUM-1, and this is a **GPU twin**, the exact shape
CLAUDE.md's 2026-08-04 `_derive_cell_stats` / `_per_cell_moments_stable_gpu` entry names. The
substituted value on failure is `-1.0`, which `_gpu_batched_abs_corr`'s docstring (`:799-800`) says
makes the host argmax skip the column — so a destroyed variance is indistinguishable from a genuinely
constant candidate and the basis column is silently never selected. Regime: candidate columns here are
basis evaluations. The hermite/legendre paths z-score first (mean~0, safe), but the `eval_dispatch` /
minmax / clip paths do not centre, and an even-degree basis value (`z**2`, `z**4`) is all-positive with
mean far above its spread — `mean/std` of `z**2` for standard-normal `z` is ~0.7, but for a clipped or
minmax-mapped axis it climbs, and any non-centred `preprocess` replay pushes it further. Unverified
how far into the 1e6 danger band production data reaches; what IS verified is that the formula has no
margin where the CPU twins deliberately do.

**Fix:** pass the per-column mean in, or run a two-kernel pass (mean reduction, then a centred
`psxx += (v-mean)*(v-mean)` accumulation) — the kernel already does a two-stage
shared-memory/atomicAdd reduction so a second launch is structurally cheap. `Sy`/`Syy` for the target
are host-computed once and can be replaced by a host-side centred `yc` at no cost. Note
`_hinge_detect_gpu_resident_batch.py:55-64` is the in-repo model: it centres `Xc`/`yc` on device first
and then accumulates.

**Test:** `test_gpu_abs_corr_kernel_matches_numpy_on_offset_column` — `cupy`-gated; feed a candidate
block containing one column at offset 1e7 / spread 1.0 that correlates 0.5 with `y`, assert the kernel
returns 0.5 ± 1e-6 rather than `-1.0` or a corrupted value.

---

### NUM-3 — Rényi-α MI's RBF Gram matrix uses the `‖a‖²+‖b‖²−2a·b` trick on uncentred input  [P2]

**Where:** `src/mlframe/feature_selection/filters/_renyi_alpha.py:82-84`.

**What:**
```python
sq = np.sum(x * x, axis=1)
d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
np.maximum(d2, 0.0, out=d2)
```

**Why it is wrong / costly:** the Gram trick is the k=2 raw-power-sum form: absolute error scales with
`eps·‖x‖²` (offset²) while the true `d2` scales with spread². The `np.maximum(d2, 0.0)` clamp exists
only because the form produces negative squared distances — it is the tell, not the fix. Worse than a
plain moment bug here because `sigma` comes from `_silverman_sigma` (`:69`, spread-sized), so
`exp(-d2/(2σ²))` divides an offset-sized error by a spread-sized bandwidth, amplifying it by
`(offset/spread)²` in the exponent and corrupting the whole Gram matrix, its eigenvalues (`:119`), and
the returned MI/CMI. `renyi_alpha_mi` ingests raw user columns (`_as_2d`, no centring anywhere in the
file; `renyi_alpha_cmi:188-190` the same), so a price/epoch column reaches it unmodified. Bites at
`mean/spread ≳ 1e6`.

**Fix:** centre before the trick — `xc = x - x.mean(axis=0, keepdims=True)` then the same three lines.
The RBF kernel depends only on differences, so centring is exactly invariant: a free fix. (Or use
`scipy.spatial.distance.pdist(x, 'sqeuclidean')`, which takes direct differences — that is what the
sibling `_fastmi.py:89` already does.)

**Test:** `test_renyi_alpha_mi_stable_on_large_offset_column` — assert
`renyi_alpha_mi(x, y) == renyi_alpha_mi(x + 1e8, y)` to 1e-9 (shift invariance is the invariant the
current code violates).

---

### NUM-4 — the surrogate-interaction self-gate divides by `perm_std + 1e-9` after explicitly setting `perm_std = 0.0`  [P1]

**Where:** `src/mlframe/feature_selection/filters/_surrogate_interaction_seeder.py:292-302, 318`.

**What:**
```python
else:
    # permuted runs all failed: fall back to the majority-class / 0-R^2 baseline + a
    # nominal spread so the z-gate still applies.
    ...
    perm_std = 0.0
    info["oof_perm"] = perm_mean
info["self_gate_z"] = float((oof_real - perm_mean) / (perm_std + 1e-9))
...
pairs_pass = bool(z >= float(self_gate_min_z) and oof_real > perm_mean + float(self_gate_margin))
```

**Why it is wrong / costly:** two defects compounding, and it is both class 1(b) and class 2.
1. The comment claims the fallback supplies "a nominal spread so the z-gate still applies" — but the
   value assigned is literally `0.0`, so the only spread is the additive `1e-9` pad. `z` becomes
   `(oof_real - perm_mean) * 1e9`, i.e. ~1e6 for a difference of 0.001. `z >= self_gate_min_z` is then
   unconditionally true and the OOF significance gate collapses to the `self_gate_margin` check
   alone. The substituted value is non-neutral in exactly the direction that **disables** the check it
   feeds, and nothing is logged at any level.
2. The same explosion reaches the **normal** path at `:290`: `perm_std = float(perm_arr.std())` over
   `reps` permutations (small by design). If those permuted surrogate fits agree exactly — a
   degenerate/majority-class target, a tree that learns nothing on any permutation — `perm_std` is
   genuinely `0.0` and `z` explodes identically, with no fallback branch involved. This is the
   additive-pad-in-a-denominator defect verbatim: once the numerator is computed correctly the
   denominator can be legitimately zero/small and the pad is the same order as the truth.

Consequence: pair emission (`pairs_pass`, `info["gated"]`) is the proposer for downstream interaction
FE. A disabled z-gate admits noise pairs into engineering, which the module's own comment at
`:305-317` says is the thing the gate exists to prevent ("On pure noise z ~ 0 -> no pair pollution").

**Fix:** make "no usable null spread" a tri-state, not a number. Replace the `perm_std + 1e-9` division
with
```python
if perm_std > _PERM_STD_TOL * max(1.0, abs(perm_mean)):
    z = (oof_real - perm_mean) / perm_std
else:
    z = None   # null spread unmeasurable -> gate cannot be evaluated
```
and have `pairs_pass` fail **closed** on `z is None` (or fall back to the margin check only, but
loudly). Log at warning via `log_throttle(logger, "surrogate_seeder_null_std", logging.WARNING, ...)`
naming `reps`, `len(perm_scores)` and `perm_mean`, so a degenerate null is visible rather than
disguised as an overwhelming z. The repo already has the right precedent:
`_mrmr_fit_impl/_fit_impl_core.py:616-622` (`None == probe could not measure`, only a genuine measured
value acts).

**Test:** `test_surrogate_self_gate_does_not_pass_on_degenerate_perm_null` — force all permuted fits to
return the same score, assert `info["gated"] is False` (currently it is `True` with `z ≈ 1e6`), and a
second case forcing every permuted fit to raise, asserting the same.

---

### NUM-5 — `uplift = mi / (baseline + 1e-12)` with `baseline` defaulted to `0.0` makes a missing baseline the top-ranked feature  [P2]

**Where:** ~20 sites across the orthogonal FE families, all the same expression. The canonical one is
`_orthogonal_univariate_fe/__init__.py:539-551`:
```python
baseline = float(raw_mi_map.get(source, 0.0))
emi = float(eng_mi[j])
uplift = emi / (baseline + 1e-12)
...
df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
```
Same expression at `_mi_greedy_fe.py:467`, `_orthogonal_adaptive_arity_fe.py:349,473`,
`_orthogonal_bootstrap_mi_fe.py:237`, `_orthogonal_cluster_basis_fe.py:529`,
`_orthogonal_copula_mi_fe.py:315`, `_orthogonal_cmim_fe.py:457`, `_orthogonal_dcor_fe.py:323`,
`_orthogonal_diff_basis_fe.py:414`, `_orthogonal_elasticnet_fe.py:210`,
`_orthogonal_jmim_fe.py:309`, `_orthogonal_quadruplet_fe.py:327`, `_orthogonal_routing_fe.py:413`,
`_orthogonal_scorer_auto_fe.py:487`, `_orthogonal_total_correlation_fe.py:445`,
`_orthogonal_triplet_fe.py:312`, `_orthogonal_three_gate_mi_fe.py:438`, `_orth_auto_scorer_fe.py:382`,
`_orthogonal_univariate_fe/_orth_gpu_resident.py:234`, `_orthogonal_univariate_fe/_orth_pair_cross_fe.py:331`.

**Why it is wrong / costly:** class 1(b). An MI baseline is legitimately SMALL, not zero — a
near-independent source column scores ~1e-4 and a perfectly independent one ~0 up to plug-in bias.
When `baseline → 0` the pad does not "guard a division", it manufactures `uplift ≈ emi · 1e12`, which
sails past every `min_uplift` gate in the codebase (`1.05`, `1.10`, `0.95` — see
`mrmr/_mrmr_config_dataclasses.py:105,170,174`) and sorts FIRST in the ranking at `:551`. So the FE
families systematically prefer engineered columns built on **noise source columns**, which is the
opposite of the relative-uplift criterion's intent.

The `.get(source, 0.0)` default at `:539` compounds it: `_source_from_engineered_name` does
longest-raw-prefix stemming (the `D1` comment at `:536-537` records a prior mis-stemming bug), and any
stemming miss yields `baseline = 0.0` and therefore a 1e12 uplift for that column. A name-resolution
failure becomes a top-of-ranking promotion.

**Fix:** guard, don't pad, and make an unresolved baseline explicit:
```python
if source not in raw_mi_map:
    log_throttle(logger, "orth_uplift_unresolved_source", logging.WARNING,
                 "uplift: no raw baseline for engineered %r (stemmed to %r); uplift left undefined", eng_name, source)
    uplift = float("nan")
else:
    baseline = float(raw_mi_map[source])
    uplift = emi / baseline if baseline > _MI_BASELINE_TOL else float("nan")
```
and sort/gate on `uplift` with `nan` ordering last rather than first. Where a ratio must still be
produced for a near-zero baseline, pair it with the absolute MI floor the class already exposes
(`min_abs_mi_frac`, `hybrid_orth_mi_fe:587`) rather than letting the ratio alone decide. Because this
is ~20 copies of one expression, the real fix is one shared `_relative_uplift(emi, baseline)` primitive
that every family calls.

**Test:** `test_uplift_not_inflated_by_zero_baseline` — two engineered columns with identical
`engineered_mi`, one whose source has baseline 0.0 and one with baseline 0.30; assert the zero-baseline
column does not rank above the genuinely-uplifted one, and
`test_uplift_unresolved_source_does_not_rank_first` for the `.get(..., 0.0)` path.

---

### NUM-6 — z-score / basis standardisation pads the std denominator additively instead of guarding it  [P2]

**Where:** `hermite_fe/__init__.py:493, 498`; `hermite_fe/_hermite_prewarp.py:299, 302`;
`hermite_fe/_hermite_prewarp_gpu_resident.py:144, 147`; `_gpu_resident_basis.py:274, 683`;
`_gpu_resident_fe.py:597`.

**What:** e.g. `hermite_fe/__init__.py:497-499`
```python
mean = float(np.mean(x))
std = float(np.std(x) + 1e-12)
return (x - mean) / std, dict(mean=mean, std=std)
```
and the GPU twin `_gpu_resident_basis.py:683`:
```python
std = cp.where(std > 1e-12, std, M.std(axis=0) + 1e-12)
```
and `_gpu_resident_fe.py:597`: `z = 2 * (xf - pp["lo"]) / (pp["hi"] - pp["lo"] + 1e-12) - 1`.

**Why it is wrong / costly:** `np.std` itself is stable (two-pass, numpy-internal), so there is no
cancellation here — this is purely the additive-pad half of the class. The pad is an **absolute**
1e-12 against a quantity whose natural scale is the column's own. For a column whose spread is
genuinely ~1e-13 — a normalised residual, a difference of two nearly-equal engineered columns, a
small-unit physical measurement — the pad dominates the true std by ~10x and every z-score is shrunk by
that factor. The Hermite/Legendre basis is then evaluated on a collapsed axis, its MI reads near-zero,
and the column is dropped as uninformative. The pad is also persisted into `params` (`dict(mean=mean,
std=std)`) and replayed at transform time, so fit and transform stay consistent — the harm is a
silently wrong basis, not a train/serve skew.

Note `hermite_fe/__init__.py:493` already shows the correct shape half-way
(`std = std if std > 1e-12 else (...)`) and then pads the fallback branch anyway.

**Fix:** the `np.where(var > tol, ...)` pattern the repo settled on — and make `tol` relative:
```python
scale = max(1.0, abs(mean))
std = float(np.std(x))
if std <= 1e-12 * scale:
    return np.zeros_like(x), dict(mean=mean, std=0.0, degenerate=True)
return (x - mean) / std, dict(mean=mean, std=std)
```
An explicitly degenerate column should return a constant-zero axis (and be skipped by the caller),
not a silently rescaled one.

**Test:** `test_hermite_standardise_preserves_tiny_scale_column` — a column with spread 1e-13 at offset
0; assert `np.std(z) == pytest.approx(1.0, rel=1e-6)` (currently ~1e-1) and that a Hermite basis built
on it recovers the same MI as the same column scaled by 1e13.

---

### NUM-7 — the stability-cluster correlation matrix z-scores with `std + 1e-12`  [P2]

**Where:** `src/mlframe/feature_selection/filters/_stability_cluster.py:164, 167, 183`.

**What:**
```python
_ysd = float(_yv.std()) + 1e-12
...
_marg_corr = np.abs((_xc * _yv[:, None]).mean(axis=0) / ((_sub.std(axis=0) + 1e-12) * _ysd))
...
Z = (_Xk - _Xk.mean(axis=0)) / (_Xk.std(axis=0) + 1e-12)
C = np.abs((Z.T @ Z / n).astype(np.float64))
```

**Why it is wrong / costly:** same mechanism as NUM-6, but the consumer is a **clustering threshold**.
`C` is compared against a correlation cut to form the redundancy clusters; a tiny-scale column's `Z`
is shrunk by `1e-12/std`, so its row of `C` is uniformly deflated and it never clusters with anything
— it survives as its own singleton cluster and is kept as a distinct feature even when it is an exact
rescaling of a column already selected. The `_marg_corr` line has the same deflation on the
relevance-ordering side. (The centring itself is correct two-pass; only the pad is at fault.)

**Fix:** compute `sd = _Xk.std(axis=0)`, build `good = sd > 1e-12 * np.maximum(1.0, np.abs(_Xk.mean(axis=0)))`,
divide by `np.where(good, sd, 1.0)` and zero out (or explicitly drop) the `~good` columns before the
Gram product — the pattern `_mrmr_degenerate.py:251-256` already uses in this repo.

**Test:** `test_stability_cluster_groups_rescaled_duplicate` — feed `x` and `x * 1e-13`; assert they
land in the same cluster.

---

### NUM-8 — degeneracy gates on variance use scale-ABSOLUTE thresholds where a relative one is required  [P2]

**Where:** `_feature_engineering_pairs/_pairs_core.py:66` (`va <= 1e-24 * n or vy <= 1e-24 * n`) and
`:120` (same form); `_mrmr_fit_impl/_eng_dedup_scan.py:73, 119, 132` (`.std() <= 1e-12`);
`_orth_dedup.py:105, 143, 182` (`varx <= 1e-24`); `_gpu_resident_basis.py:780`
(`sqrt(vx / nf) > 1e-12`); `_usability_njit_pool.py:322` (`if var <= 1e-18`).

**What:** all of these declare a column constant when its variance falls below a fixed absolute bound.

**Why it is wrong / costly:** a column whose values are genuinely ~1e-13 in magnitude has a true
variance ~1e-26 and is unconditionally declared constant. `_pairs_core.py:39-40`'s own comment states
what the resulting `0.0` means downstream: *"'not redundant, keep' in the dedup gate and 'no signal,
drop' in the y-gate, so the wrong answer was silently actionable in both directions."* That comment was
written about the cancellation bug it fixed, but the absolute threshold left behind reproduces the same
wrong answer for a different input class. Note `_pairs_core.py` is at least `n`-scaled; `_eng_dedup_scan.py`
and `_orth_dedup.py` are not scaled at all. These are the least severe of the class because they fail
*safe* in the dedup direction, but not in the relevance direction.

**Fix:** scale every such threshold by the column's own magnitude, e.g.
`va <= (1e-12 * max(1.0, abs(ma)))**2 * n`. Better, make it one shared helper
(`_is_numerically_constant(values, mean)`) so all six call sites agree.

**Test:** `test_abs_corr_finite_detects_correlation_at_1e_13_scale` — two perfectly correlated columns
whose values are ~1e-13; assert `|r| ≈ 1.0`, not `0.0`.

---

### NUM-9 — the remaining `x / (std + eps)` / `x / (ptp + eps)` normalisation family  [P3]

**Where:** `_fe_pure_form_retention.py:396`; `_fe_pure_form_retention_gpu_resident.py:171`;
`_usability_gpu.py:144`; `_orthogonal_routing_fe.py:327`; `_extra_basis_fe_proto.py:33`
(`(c - c.min()) / (np.ptp(c) + 1e-12)`); `bases.py:122, 124`
(`std = float(np.std(x) + 1e-12)`, `bandwidth = float(1.06 * std * n**-0.2) + 1e-12`);
`_hermite_fe_optimise.py:634` (`/ (np.linalg.norm(...) + 1e-12)`).

**What / why:** the same additive pad as NUM-6/NUM-7, on normalisers whose consumers are less
decision-critical (an RBF bandwidth, a direction-vector normalisation, a routing z-score, a
`[0,1]` rescale). Each is individually a 30-100 % corruption once the denominator is legitimately
~1e-12, but none of them is the primary gate for a selection decision the way NUM-1/NUM-5 are.
Listed in full per the brief's report-everything rule.

**Fix:** same `np.where(scale > tol, ...)` pattern; a single shared `_safe_scale()` helper would retire
all of them at once.

**Test:** `test_safe_scale_helper_rejects_degenerate_instead_of_padding` once the helper exists.

---

### NUM-10 — `sqrt(p*(1-p)/n + 1e-9)` floors a confidence margin  [P3]

**Where:** `src/mlframe/feature_selection/filters/_cat_confirm_bandit.py:221`
`margin = 1.96 * math.sqrt(p_j * (1 - p_j) / n_j + 1e-9)`.

**What / why:** the epsilon is inside the `sqrt`, so it acts as a floor on the margin
(`>= 1.96 * 3.16e-5`) rather than a denominator pad — a conservative direction (a wider interval means
fewer confident arms). It is still an absolute additive constant on a quantity whose scale is
`p(1-p)/n`; at `n_j` large and `p_j` near 0 or 1 the true term drops below 1e-9 and the margin is
entirely the epsilon. Cosmetic; flagged for completeness, no behavioural harm identified.

**Fix:** `max(p_j * (1 - p_j) / n_j, _MIN_VAR)` with `_MIN_VAR` named and justified, so the floor is
explicit rather than buried in a sum.

---

### NUM-11 — `u / (np.abs(v) + 1e-6)` in candidate generation  [P3 — by design, not a defect]

**Where:** `_conditional_gate_fe.py:214`; `_resident_candidate_mi.py:56, 123`; `fe_baselines.py:37, 38,
199`; `feature_engineering.py:585`.

**What / why:** these are **feature definitions** (a protected ratio operand), not statistical
denominators — the epsilon is part of the engineered column's semantics and is replayed identically at
transform time. Explicitly NOT class 1(b). Recorded so the next wave does not re-flag them.
`_mrmr_fit_impl/_fe_stage_cascade_early_b.py:592,625` (`fe_pairwise_ratio_eps`) is the same category.

---

### Verified-clean (bug class 1)

Checked and found genuinely stable — do not re-audit blind.

**Already-fixed two-pass / centred kernels (the model forms in this repo):**
- `_feature_engineering_pairs/_pairs_core.py:26-72` (`_abs_corr_finite_njit`) and `:75-126`
  (`_abs_corr_zerofill_njit`) — explicit two-pass centred, with the cancellation trap named and the
  measured failure numbers recorded in the docstring.
- `_mrmr_fit_impl/_eng_dedup_batch_corr.py:50-76` — two-pass mean-then-centre; `denom` unpadded, guarded
  at `:72` by `saa <= 1e-24 * n`. The correct 1(b) pattern.
- `_usability_njit_pool.py:303-324` (and the identical comment blocks at `:359, :426, :485, :708`) —
  converted to centred two-pass with an exact `vmax == vmin` constant short-circuit; the comment
  documents the exact `-1.0` false-negative this replaced.
- `_hinge_detect_gpu_resident_batch.py:55-64` — `Xc = X_batch - xbar`, `yc = yg - ybar` BEFORE
  `sxx = (Xc*Xc).sum(0)`. Properly centred on device; the model for fixing NUM-2.
- `_binned_numeric_agg_resident.py:81` — docstring records the replacement of the raw-power
  `(cnt,s1,s2,s3,s4)` form; centred now.
- `_mrmr_degenerate.py:251-256` — `M -= M.mean(axis=1, keepdims=True)` then `sqrt((M*M).sum(1))`, with
  `np.where(stds == 0, 1.0, stds)` — correct guard, not a pad.
- `_mrmr_fit_impl/_friend_graph_and_redundancy/_group2.py:204-206, 236, 450, 479-494` and
  `_group3.py:100-108` — `ss = np.sum((yv - yv.mean())**2)` with `if ss < 1e-24: return 0.0` and
  unpadded division.
- `_mrmr_sis_screen.py:124-131` (`_zscore`: `np.std` + `if sd <= 0.0: return zeros`), `:162-166`
  (median/MAD only).
- `_mrmr_fingerprints.py:373-392` — delegates to `np.corrcoef` with explicit `std == 0` guards.
- `_mrmr_artifacts.py:168-182` — entropy from `np.bincount` proportions; `denom > 1e-12` is a
  *threshold*, not a pad.
- `mrmr/_mrmr_class_fit_helpers.py:326` — `<= 1e-12 * max(1.0, abs(float(sw.mean())))`: a genuinely
  **relative** tolerance. The form NUM-8 should adopt.
- `_mrmr_fit_impl/_eng_dedup_scan.py:134` — `np.corrcoef` (numpy centres internally).
- `stability.py` — no moment arithmetic at all.

**MI / entropy / discretisation primitives:**
- `_ksg.py:257-273` — `c00 += u*u; c01 += u*v; c11 += v*v` is NOT cancellation: `knn_xy` is already
  `points[knn] - point`, centred at the focus point by construction (`:235-236`, `:243-244`), and no
  mean is subtracted afterwards. `:205` / `:481` `np.maximum(eps, 1e-12)` are clamps, not pads.
- `_fastmi.py:89` — direct pairwise differences `(zx[:,None]-zx[None,:])**2`, not the Gram trick. The
  correct form NUM-3 should copy. `:198` `sigma = 1.0` is analytic (probit marginals).
- `_renyi_alpha.py:69` (`np.std(...).mean()`), `:81` (`max(sigma, 1e-12)` clamp), `:120`
  (`eigvals[eigvals > 1e-12]` filter) — all clean; only `:82-84` (NUM-3) is at fault.
- `info_theory/_entropy_kernels.py:289, 388, 460`; `_batch_kernels.py:454, 795, 860`;
  `_class_mi_kernels.py:77, 267` — all `if denom <= 1e-12: return 0.0` / `if lam > 1e-12: term /= lam`,
  i.e. the prescribed threshold-guard pattern, never `/(denom + eps)`.
- `_neural_mi.py:193, 199, 779, 784, 789` — `+ 1e-12` inside `torch.log(...)`: a log-domain probability
  floor, not a variance denominator. Out of class. `:296` `np.std` feeds a jitter scale only.
- `_mi_aggregator.py:113` — `np.diag(v + ridge)` is documented Tikhonov regularisation of a QP system
  matrix (`:100, :111-112`), not a variance pad.
- `_adaptive_nbins.py:128-132, 169, 228` — `np.percentile` IQR with an `if iqr <= 0` guard; order
  statistics only.
- `discretization/_discretization_edges.py:126, 129, 173, 263` — `1e-9` is a right-edge *offset* for a
  half-open bin; `1e-12 * _span` is span-**relative**.
- `supervised_binning.py`, `_mdlp_validated_split.py`, `discretization/*` — count/MDL criteria, no
  moments.
- `_fe_batched_mi.py`, `_fe_batched_mi_cmi.py`, `_fe_edge_mi.py`, `_gpu_resident_pair_mi.py`,
  `_resident_candidate_mi.py`, `batch_pair_mi_gpu.py`, `_batch_pair_mi_cuda_*.py`,
  `info_theory/_cmi_cuda*.py`, `_group_mi.py`, `_class_encoding.py`, `_state_and_dispatch.py` —
  histogram/count-based MI only; no second moments anywhere.
- `_relaxmrmr_3d.py:43-122, 180-210` — integer-bin counts and log-ratios, every division pre-guarded by
  an explicit `<= 0.0: continue`. No moments, no pads.

**Class 1(c) — subset-additive centred moments: ABSENT across the whole surface.**
- `_mrmr_partial_fit.py` — no streaming accumulator; buffers rows (`:245-246`) and calls
  `self.fit(X_buf, y_buf, sample_weight=weights)` for a **full recompute** (`:298`). No `train = full − test`.
- `_mrmr_stability_report.py` — each bootstrap re-indexes stored bin codes (`:131-136`) and recomputes
  `_marginal_mi_codes` from scratch; frequencies are integer counts / K (`:140, :216`).
- `_cat_target_encoding_and_weighted.py:138-152` — `cell_sum = full_sum - test_sum` IS a
  full-minus-test subtraction, but of **raw sums feeding a MEAN**, which is exactly additive and
  loses no precision (the remainder is ~(K−1)/K of the total, no near-cancellation). No centred
  quantity is combined. Clean — this is the correct residue of the 2026-08-04 `_target_encoding_fe`
  fix, not a survivor of it.
- No analogue found in `_mrmr_fit_impl/`, `_mrmr_fe_step/`, or the MI primitives.

**Non-defects flagged so they are not re-reported:** `_mrmr_partial_fit.py:61, 124`
(`_WEIGHT_FLOOR = 1e-9` is a `max()` clamp on a sample *weight*, a numerator, never a denominator);
`_mrmr_fit_impl/_group4.py:265, 304`, `_fit_impl_core.py:728, 730`, `_finalise.py:259`,
`_step_pairs_rank.py:681` (`max(_anchor_mi, 1e-9)`) — comparison tolerances and `max()` floors on MI
ratios, not variance denominators; `_group2.py:224-225` normal equations with an explicit
`LinAlgError` → lstsq fallback (the already-endorsed pattern).

---

## Bug class 2 — silent fallbacks

### NUM-12 — the order-2 maxT permutation-null FLOOR is substituted with `0.0`, which is the literal "gate off" sentinel  [P0]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step_helpers.py:491-497`.

**What:**
```python
        except Exception:
            logger.warning(
                "MRMR FE: order-2 maxT permutation-null floor failed; continuing without it.",
                exc_info=True,
            )
            _pair_maxt_floor = 0.0
            _pair_mm_bias = {}
```

**What downstream consumes it / why nobody would notice:** `0.0` is exactly the disabled sentinel —
confirmed at both consumers: `_mrmr_fe_step_helpers.py:484` (`if _pair_maxt_floor != 0.0 and verbose >= 1:`,
the "floor is active" test) and `_mrmr_fe_step/_step_pairs_rank.py:369` (`"No-op when floor==0.0."`).
So any transient fault in the permutation sweep — a GPU TDR, an OOM, an njit recompile fault, a dtype
edge on one candidate pool — silently removes the **entire** best-of-p chance-max rejection for that FE
step and pure-noise pairs are admitted into engineering. The warning does fire (level is right), but it
names neither the exception type, nor the pair count, nor which FE step, and the handler binds no
exception variable at all (`except Exception:`) — so after the fact there is no way to tell whether the
floor was measured-as-zero or crashed. "floor == 0.0 because measured" and "floor == 0.0 because it
blew up" are indistinguishable downstream by construction.

**Fix:** make it tri-state, the way `_fit_impl_core.py:616-622` already does
(`None == probe could not measure`, only a genuine measured value acts):
`except Exception as exc:` → log `type(exc).__name__`, `exc`, `n_pairs`, `_pair_maxt_perms`, the step
index → set `_pair_maxt_floor = None` and have the consumers treat `None` as "floor unavailable" with
an explicit policy (fail closed: reject the pool; or fail open with a per-gate warning, not one
per-step). At minimum set `self._pair_maxt_floor_failed_ = True` so the two zeros stop aliasing.

**Test:** `test_pair_maxt_floor_failure_does_not_read_as_measured_zero` — monkeypatch the permutation
sweep to raise; assert the FE step does NOT admit a pure-noise pair that a measured floor rejects, and
that the failure is exposed on the estimator rather than only in the log.

---

### NUM-13 — the circuit breaker that makes NUM-12 recoverable is itself swallowed to `debug` + `pass`  [P0]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step_helpers.py:462-467`.

**What:**
```python
                try:
                    trip_pair_maxt_gpu_circuit_breaker()
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed: %s", e)
                    pass
                _pair_maxt_floor = None
```

**Why:** this is `except: pass` on a path whose whole job is to ENFORCE something — the block's own
comment at `:454-456` says it must *"TRIP the process circuit breaker so every later pair-maxT floor
skips the poisoned GPU context"*. If the trip fails, every subsequent pair re-faults the poisoned CUDA
context, re-enters the `:452` handler, and the file's own documented consequence is *"a ~1h CPU floor
invisible"*. `logger.debug("suppressed: %s", e)` names neither the function nor the consequence. (The
`_pair_maxt_floor = None` at `:467` is correct — the exact CPU path at `:469` still runs; only the
breaker silently fails to arm.)

**Fix:** `logger.warning("MRMR FE: failed to trip the pair-maxT GPU circuit breaker (%s: %s); every later pair will re-fault the poisoned CUDA context", type(e).__name__, e)`.
Drop the dead `pass` after the log.

**Test:** `test_pair_maxt_breaker_trip_failure_is_warned` — make `trip_pair_maxt_gpu_circuit_breaker`
raise; assert a WARNING record naming the exception type is emitted.

---

### NUM-14 — a prewarp held-out validation returns `True` ("accept") on any exception  [P1]

**Where:** `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_setup.py:148-150`.

**What:**
```python
            except Exception as e:
                _module_logger.debug("prewarp held-out correlation validation failed, falling back to accepting the warp: %s", e)
                return True  # validation failure -> fall back to accepting the warp
```

**What downstream consumes it / why nobody would notice:** this closure IS the held-out validation gate
for an ALS-fitted operand prewarp — the same function returns `False` at `:133` and `:138` for its
genuine rejection cases and compares `|corr| >= _pw_min_val_corr` at `:147`. `True` is the value that
switches the check off, at `debug`. The `try` wraps `apply_operand_prewarp` (a numba path) and
`np.corrcoef`, so a numba typing/recompile fault, a NaN-producing warp, or a shape mismatch on the
validation mask all silently promote an **unvalidated, distorted** re-expression of the operand into
the engineered column set. Production runs at `verbose=0` see nothing.

**Fix:** fail closed — `return False` for anything that is not a deliberate rejection, and log at
`log_throttle(logger, "pairs_setup_prewarp_validation", logging.WARNING, ...)` naming
`type(e).__name__`, the operand pair and `_val_mask.sum()` (this sits inside a per-pair loop, hence the
throttle). If the current permissive polarity is deliberate, it must at minimum be counted and
surfaced once per fit ("N/M prewarp validations could not be evaluated; those warps were accepted
unverified").

**Test:** `test_prewarp_validation_failure_does_not_accept_the_warp` — force `apply_operand_prewarp`
to raise; assert the prewarp is rejected (and that a WARNING is emitted).

---

### NUM-15 — a broad `except Exception` around a trial import, CACHED at module level  [P1]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_degenerate.py:44-50`.

**What:**
```python
_xxh3_64: Optional[Callable] = None
try:
    import xxhash as _xxhash

    _xxh3_64 = _xxhash.xxh3_64_intdigest
except Exception as e:  # nosec B110 - xxhash optional: falls back to pandas hash_array below
    logger.debug("xxhash unavailable, falling back to pandas hash_array for duplicate-column detection: %s", e)
```

**Why:** this is verbatim the shape CLAUDE.md's 2026-08-02 entry describes. Evaluated ONCE per process
at import. Only `ImportError` means xxhash is genuinely absent; a partially-installed wheel, a Windows
DLL hiccup, or an `AttributeError` from a version that renamed `xxh3_64_intdigest` all read as
"unavailable" and pin the whole process onto the slower `pd.util.hash_array` duplicate-column path for
its entire lifetime, at `debug` — invisible in production, exactly as the MI-backend regression was.

**Fix:** `except ImportError as e:` → debug (genuine absence). Add a separate
`except Exception as e: logger.warning("xxhash present but unusable (%s: %s); duplicate-column detection downgraded to pandas hash_array", type(e).__name__, e)`.

**Test:** `test_xxhash_transient_failure_is_warned_not_silently_downgraded` — simulate an
`AttributeError` on the attribute lookup; assert a WARNING is emitted (mirrors the existing
`test_select_mi_backend_transient_failure.py`).

---

### NUM-16 — a swallowed polars Struct-column validator substitutes an empty reject-list  [P1]

**Where:** `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_fit_helpers.py:176-184`.

**What:**
```python
                except Exception as exc:
                    logger.debug("mrmr: polars Struct-column detection failed; assuming none: %r", exc, exc_info=True)
                    _struct_cols = []
```

**Why:** `_struct_cols = []` is non-neutral in the direction that disables the guard at `:183`
(`if _struct_cols: raise ValueError(...)`). The `try` covers a `dt == _pl.Struct` dtype comparison, and
polars 1.x dtype-equality semantics shifting under a version bump is a real, recurring hazard (the
repo's own CLAUDE.md has a "polars traps that fail silently" section). On that path an unsupported
Struct column sails into MI estimation, which has no scalar value for it — a downstream crash or
silently garbage bins, with only a `debug` line naming the cause.

**Fix:** `except ImportError` only for "polars absent"; on any other exception `logger.warning` and
**raise** — this is a validator, not an optimisation, and the whole point of `:183` is to fail early
with an actionable message.

**Test:** `test_struct_detection_failure_does_not_disarm_the_guard` — monkeypatch the dtype comparison
to raise; assert `fit` raises rather than proceeding.

---

### NUM-17 — MI substituted with `0.0` on any per-column exception, at `debug`, in two places  [P1]

**Where:** `src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_mi_backends.py:43-45`
and `:204-206`.

**What:**
```python
        except Exception as e:
            logger.debug("mutual_info_score failed for column %d, treating MI as 0.0: %s", j, e)
            mis[j] = 0.0
```
```python
            except Exception as e:
                logger.debug("plugin_mi_classif_batch_dispatch failed for column %d, treating MI as 0.0: %s", j, e)
                mis[j] = 0.0
```

**Why:** MI is a **maximised** relevance score; `0.0` is its floor and therefore the "never select this
column" value. A quantile/searchsorted/`mutual_info_score` fault (dtype surprise, object array, memory
pressure) silently deletes a genuinely strong feature from MRMR's relevance ranking. The second site is
worse: the swallowed call is the numba/GPU dispatcher — the *same* call whose sibling failure at `:166`
was explicitly upgraded to `warning` by the 2026-08-02 fix. **This call site was missed by that fix.**

**Fix:** `logger.warning` naming the column index, dtype, finite count and `type(e).__name__`; and
substitute `np.nan` rather than `0.0` so the caller's non-finite handling reads the column as
*unscored* rather than as *scored zero*. If the array contract forbids NaN, re-raise — a per-column MI
failure is a bug, not an expected regime.

**Test:** `test_orth_mi_backend_column_failure_warns_and_does_not_score_zero`.

---

### NUM-18 — the subsumed-operand check substitutes an empty "columns to drop" set  [P1]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_assign_support.py:151-153`.

**What:**
```python
            except Exception as exc:
                logger.debug("mrmr: subsumed-operand computation failed; falling back to MI-only pick (best-effort): %r", exc, exc_info=True)
                _subsumed_operand_names = set()  # best-effort: fall back to MI-only pick
```

**Why:** the literal "empty set for columns-to-drop" pattern. Consumed at `:160`
(`_eligible_idxs = [_oi for _oi in _operand_idxs if cols[_oi] not in _subsumed_operand_names and ...]`)
— an empty set makes EVERY operand eligible, so the never-empty re-attach resurrects precisely the
fully-subsumed raw operand the subsumption check exists to exclude (the `a` in `a**2/b` case
`_fe_raw_redundancy_drop.py` documents at length). Anti-conservative: it re-injects redundant features.
`debug` + `exc_info` means a `verbose=0` production fit shows nothing.

**Fix:** `logger.warning` naming the exception type and the candidate count, and fail **closed** — skip
the never-empty re-attach entirely when the subsumption verdict is unavailable, rather than
re-attaching on an unverified basis.

**Test:** `test_subsumption_failure_does_not_reattach_subsumed_operand`.

---

### NUM-19 — `prevalence_debias_auto` disabled for the WHOLE fit by one transient fault  [P1]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py:531-533`.

**What:**
```python
        except Exception as e:
            logger.debug("prevalence auto-debias computation failed, disabling it for this fit: %s", e)
            _prevalence_debias_auto = False
```

**Why:** non-neutral in the **loosening** direction. `_step_pairs_rank.py:374`
(`_obs_for_prevalence = _pair_mi_floor_cmp if _prevalence_debias_auto else pair_mi`) combined with
`_step_core.py:212-214` — which states debiasing *"can ONLY LOWER the observed joint MI, so the gate
can only TIGHTEN"* — means `False` re-admits exactly the finite-sample-noise pairs the user opted into
`"auto"` to reject. Silent at default verbosity, and `_prevalence_debias_auto` is returned at `:535`,
so the disable is **latched for the whole fit**: one transient fault in one chunk degrades every
subsequent pair.

**Fix:** `logger.warning` naming the exception and the chunk; and latch per-chunk, not fit-wide.

**Test:** `test_prevalence_debias_failure_is_not_latched_for_the_whole_fit`.

---

### NUM-20 — `NameError` → `True` for a routing/permission guard, with NO logging at all  [P1]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_early_a.py:291-294`.

**What:**
```python
        try:
            _extra_basis_scorer_ok = _default_scorer == "plug_in"
        except NameError:
            _extra_basis_scorer_ok = True
```

**Why:** `True` for a support/permission guard, silent. Consumed at `:295`
(`if _univ_fourier_on and _univ_basis_on and _extra_basis_scorer_ok and ...`), so an unbound
`_default_scorer` OPENS the gate that `:288-290` explicitly says must stay shut under alternate
routing (*"adding it under alternate routing would emit columns the routed scorer never selected and
diverge from a direct call to that scorer"*). The fallback value is the exact opposite of the
documented intent, and nothing in the log records that it fired. This is also the
`NameError`-on-a-conditionally-bound-local shape CLAUDE.md's monolith-split rule warns about — a name
that resolves lazily and is silently absent on some paths.

**Fix:** bind `_default_scorer` unconditionally at function top
(`_default_scorer = getattr(self, "fe_hybrid_orth_default_scorer", "plug_in")`) and delete the
try/except. If the guard must stay, default to `False` and `logger.warning` the unbound name.

**Test:** `test_extra_basis_gate_closed_when_scorer_unresolved`.

---

### NUM-21 — the pre-FE screened-raw safety net is emptied on failure  [P1]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core.py:326-328`.

**What:**
```python
        except Exception as exc:
            logger.debug("mrmr: capturing the pre-FE screened-raw safety net failed; support finalisation cannot re-add these raws: %r", exc, exc_info=True)
            self._prefe_screened_raw_ = []
```

**Why:** empty-collection substitution that disables the recovery mechanism `:318-321` says exists to
stop genuine raw columns being dropped. The log TEXT is good (it names the consequence); the defect is
the LEVEL — this fires at most once per fit (`num_fs_steps == 0`), so a warning costs nothing.

**Fix:** promote to `logger.warning`.

**Test:** `test_prefe_safety_net_failure_is_warned`.

---

### NUM-22 — `_config_corr` returns `-1.0` on failure, which DISABLES the clean-form demotion  [P2]

**Where:** `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_score.py:783-785`,
consumed at `:794`.

**What:**
```python
                except Exception as e:
                    logger.debug("_config_corr: column re-materialisation/correlation failed for config %r, treating it as unrecoverable: %s", _cfg, e)
                    return -1.0
...
            if _clean_corr >= 0.0 and _pw_corr < _clean_corr * 1.05:
                best_config, best_mi = best_nonprewarp_config, best_nonprewarp_mi
```

**Why:** asymmetric. If `_pw_corr` fails → `-1.0` → demotion fires (conservative, fine). If
`_clean_corr` fails → `-1.0` → `_clean_corr >= 0.0` is False → **the demotion never fires** and the
distorted prewarp form is kept unchallenged. Since `_config_corr` re-materialises the column, a fault
is more likely on the *less recently touched* config, so the disabling direction is not the rarer one.
`debug` only.

**Fix:** return `None` for "could not measure" and `-1.0` only for "measured and rejected"; treat
`None` on the clean side as "demote by default" (fail closed toward the simpler form). Log at
`log_throttle(..., logging.WARNING, ...)` naming the config and exception type.

**Test:** `test_clean_form_demotion_still_fires_when_clean_corr_unmeasurable`.

---

### NUM-23 — the MI-ceiling bound disables itself, self-documented  [P2]

**Where:** `src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/__init__.py:560-575`,
consumed at `:732`.

**What:** the docstring says it outright — *"Returns 0.0 when the entropy cannot be established, which
disables the bound rather than guessing"* — and:
```python
    except Exception:
        logger.debug("_target_entropy_nats: could not establish H(y); MI-ceiling bound disabled", exc_info=True)
        return 0.0
```

**Why:** `H(y) = 0.0` is the hard ceiling on any `MI(X; y)`, so a zero ceiling makes the sanity bound
(which exists, per `:557-559`, to detect an MAD-derived MI floor that has drifted above what any column
could score) unsatisfiable-detection-proof: the check that was meant to catch a drifted floor is
exactly the one that is switched off. The realistic trigger is `pd.qcut` on an exotic/object `y` at
`:568` or `np.unique` on unorderable mixed types. `debug` only.

**Fix:** return `None` and have `:732` treat it as "ceiling unknown" with an explicit branch; log at
`warning` naming `type(exc).__name__` and `y.dtype`.

**Test:** `test_mi_ceiling_bound_not_silently_disabled_on_entropy_failure`.

---

### NUM-24 — interaction-information routing manufactures synergy from a cache miss  [P2]

**Where:** `src/mlframe/feature_selection/filters/_interaction_information.py:266-275, 284-285`.

**What:**
```python
        mi_a = float(cached_MIs.get((va,), 0.0))
        mi_b = float(cached_MIs.get((vb,), 0.0))
            logger.debug("mrmr: interaction-information routing found no cached marginal MI for column %r; treating as 0.0.", va)
```

**Why:** the code's own comment at `:266-269` states the mechanism: `0.0` **inflates**
`ii = pair_mi − mi_a − mi_b`, pushing the pair past `ii_floor` into `ROUTE_SYNERGY` (`:284-285`). The
substituted value is non-neutral precisely in the direction that defeats the routing gate — a cache
defect manufactures synergy. The only diagnostic is at `debug`.

**Fix:** throttled `logger.warning` (`_mi_aggregator` already uses `log_throttle` for exactly this),
and/or treat a missing marginal as **unroutable** — force `ROUTE_ADDITIVE` (the conservative route)
instead of feeding `0.0` into the arithmetic.

**Test:** `test_ii_routing_missing_marginal_does_not_route_synergy`.

---

### NUM-25 — `return True` ("the upload fits") when the VRAM probe itself fails  [P2]

**Where:** `src/mlframe/feature_selection/filters/batch_pair_mi_gpu.py:338-340` and `:369-371`.

**What:**
```python
    except Exception as e:
        logger.debug("%s._gpu_upload_fits: memGetInfo failed (%s); permissive", context, e)
        return True
```

**Why:** `True` for a capability/sizing guard is the canonical disabling value, at `debug`. The
module's own docstring at `:326-327` promises *"A REJECTION is always logged at WARNING… never a silent
fallback"* — that promise covers the rejection path but not this bypass path, so the one case where the
guard cannot do its job is the one case it is silent about. A near-full 4 GB card then takes the
launch fault the guard exists to prevent.

**Fix:** promote both to `warning` with the requested byte count; fail **closed** (`return False`) for
anything other than a genuine `ImportError` (cupy/psutil absent).

**Test:** `test_gpu_upload_fits_fails_closed_when_memgetinfo_raises`.

---

### NUM-26 — the CMI-CUDA absolute free-VRAM floor is skipped on probe failure  [P2]

**Where:** `src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py:826-828`.

**What:**
```python
    except Exception as e:
        logger.debug("swallowed exception in _cmi_cuda.py: %s", e)
        pass
```

**Why:** this wraps the **absolute free-VRAM floor** check at `:818-825`. If the probe raises, the
guard's `return False` never executes and the function falls through to ALLOW CUDA — non-neutral for a
guard whose stated purpose (`:812-814`) is to prevent a launch fault on a near-full shared 4 GB card.
A tripped launch then poisons the context and latches `_CMI_GPU_FAILED` for the whole process. The
message names neither the guard nor the sizing inputs.

**Fix:** `except ImportError` → permissive with a named debug line; any other exception → `logger.warning`
naming `bytes_needed`, `free`, `total` and `return False` (fail closed on a safety guard).

**Test:** `test_cmi_cuda_vram_floor_fails_closed_on_probe_error`.

---

### NUM-27 — the memory-headroom OOM guard degrades to permissive  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py:176-181`, consumed at `:183`.

**What:**
```python
            except Exception as e:
                logger.debug("psutil.virtual_memory() probe failed: %s", e)
                _available_bytes = 0
```

**Why:** `0` makes `_headroom_bytes == 0`, so `if _headroom_bytes > 0 and ...` never fires — the
substituted value is exactly the one that turns the check off. The comment concedes it ("disables the
headroom check (permissive)"). psutil is a hard dependency here so this only masks a *runtime*
`virtual_memory()` failure, but on a container-sandboxed host that failure is persistent and OOM
protection is lost for the process lifetime, at `debug`. A sibling in the same cluster chose the
opposite default: `_mrmr_sis_screen.py:66-68` substitutes a conservative 2 GB rather than "unlimited".

**Fix:** keep permissive behaviour if preferred, but log once at `warning` via `log_throttle` so the
missing protection is visible; better, adopt the `_mrmr_sis_screen.py` conservative-constant default.

**Test:** `test_headroom_guard_failure_is_warned`.

---

### NUM-28 — the inf/NaN and constant-`y` input validators silently skip themselves  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py:279-282` and `:298-301`.

**What:**
```python
    except ValueError:
        raise  # re-raise our own ValueError
    except Exception:
        logger.debug("MRMR.fit: inf/NaN input validation scan failed unexpectedly; skipping the guard.", exc_info=True)
```
(and the identical shape for the constant-`y` guard.)

**Why:** the `except ValueError: raise` correctly preserves the guards' own rejections, but any other
exception disables the entire `±inf` rejection (`:261`: *"the discretization step produces undefined
bins on inf"*) or the single-class-`y` rejection (`:295`) for that fit. Realistic triggers:
`select_dtypes`/`to_numpy()` on an exotic extension dtype, the `np.frompyfunc` object scan at `:274` on
a ragged object column, `np.unique` sorting an object `y` with unorderable mixed types. The fit then
produces undefined bins / all-zero MI instead of an actionable `ValueError`, and the message names
neither the column nor the exception type.

**Fix:** log at `warning` with `type(exc).__name__` and the column/dtype being scanned; scope the `try`
to the per-column body so one bad column does not disarm the guard for all the others; for the
`np.unique` case fall back to a `set()`-based uniqueness count rather than abandoning the check. The
model is in this same file at `:507-513`, which already logs at `warning` and states the consequence
in plain terms.

**Test:** `test_inf_guard_not_disarmed_by_one_bad_column`, `test_constant_y_guard_survives_unorderable_dtype`.

---

### NUM-29 — multi-output detection defaults to `False`, the fit-path-changing direction  [P2]

**Where:** `src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py:163-167`.

**What:**
```python
    except Exception as exc:
        logger.debug("mrmr: multi-target detection np.asarray(y) failed; treating as single-target: %r", exc, exc_info=True)
        return False
```

**Why:** `False` routes a possibly-multi-target `y` away from `_fit_multioutput` into the single-target
path, which mis-handles the extra columns instead of raising. `np.asarray(y)` failing on a
ragged/exotic container is precisely the moment you know least about `y`. `debug` only.

**Fix:** `logger.warning`; on an inconclusive probe prefer raising a `TypeError` naming `type(y)` over
guessing a fit path.

**Test:** `test_multioutput_probe_failure_raises_rather_than_guessing`.

---

### NUM-30 — `_defaults = {}` silently turns the whole `fast_search` profile into a no-op  [P2]

**Where:** `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_config.py:184-188`.

**What:**
```python
        except Exception as exc:
            logger.debug("mrmr: ctor-default introspection failed in _apply_fast_search_profile; treating all knobs as user-set: %r", exc, exc_info=True)
            _defaults = {}
```

**Why:** `_override_if_at_default(...)` compares against `_defaults`; an empty dict means no attribute
ever matches its default, so **no** knob is overridden — the user asked for `fe_fast_search=True` and
silently gets nothing. Empty-dict-for-an-ownership-map, the listed non-neutral shape.

**Fix:** `logger.warning` stating that the fast-search profile was NOT applied, so the missing speedup
is attributable.

**Test:** `test_fast_search_profile_failure_is_warned`.

---

### NUM-31 — `return {}` makes every usability verdict in the step default to `False`  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py:115-117`,
consumed at `:749`.

**What:**
```python
    except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
        logger.debug("usability-verdict batch lookup failed, every candidate defaults to False: %s", e)
        return {}  # any failure: every lookup defaults to False (strict rank-MI decision stands)
```

**Why:** empty-collection → `False` for an admission path (`_usability_verdict.get(raw_vars_pair, False)`).
Conservative in direction and the message DOES state the consequence — but it is a **whole-batch**
substitution at `debug`: one fault silently removes the tail-concentration admission route for every
pair in the step, which is the exact rescue path `:739-746` exists to provide.

**Fix:** `logger.warning` with `type(e).__name__` and `len(need_usability)`.

**Test:** `test_usability_batch_failure_is_warned_not_silent`.

---

### NUM-32 — `True` / `_retains = True` for operand-significance guards (polarity defended, level not)  [P2]

**Where:** `_mrmr_fit_impl/_assign_support.py:487-489`; `_mrmr_fit_impl/_assign_support_tail.py:286-288`.

**What:**
```python
                except Exception as e:
                    logger.debug("Marginal-MI significance probe failed (%s: %s) -- not silently dropping a possibly-genuine operand", type(e).__name__, e)
                    return True  # estimator error -> do not silently drop a possibly-genuine operand
```
```python
                            except Exception as exc:
                                logger.debug("mrmr: discriminator estimator failed; conservatively retaining (never drop genuine signal): %r", exc, exc_info=True)
                                _retains = True  # estimator error -> never drop genuine signal
```

**Why:** `True` for a support guard, but the fail-open polarity is explicitly reasoned and defensible
(keep a possibly-genuine operand). Not a correctness bug. The defect is visibility: a *persistent*
estimator fault silently re-attaches every noise operand of every composite with zero operator-visible
trace. Notably the **outer** handler for the second block (`_assign_support_tail.py:291-301`) was
already fixed to `logger.warning` with a long rationale about the two handlers *"disagreeing about
polarity"* and *"no way to tell from the logs which had fired"* — the inner one was left at `debug`, so
that fix is half-applied at this site.

**Fix:** keep the polarity; count the failures and emit one `logger.warning` summary per fit
("N/M operand significance probes failed; those operands were re-attached unverified"). Raise
`_assign_support_tail.py:287` to `warning` to match the intent already written at `:292-301`.

**Test:** `test_operand_probe_failures_are_summarised_at_warning`.

---

### NUM-33 — four sites substitute `0.0` for a relevance score being MAXIMISED  [P2]

**Where:** `_mrmr_fit_impl/_assign_support.py:168-170` (`_rel_ne = 0.0`);
`_assign_support_tail.py:455-457` (`_rel = 0.0`);
`_friend_graph_and_redundancy/_group4.py:202-204` (`_rel = 0.0`);
`_mrmr_fe_step/_step_score.py:307-310` (`pass`, leaving `_name_marg.get(_nm, 0.0)`).

**Why:** `0.0` is the minimum of MI's range and the comparison is `if _rel > _best_rel`, so a failing
candidate can never win — the fallback silently **reorders the argmax**. Conservative-ish (never
promotes a bad feature), hence P2; but if the estimator fails for ALL candidates every one ties at
`0.0` and the winner becomes an arbitrary iteration-order artefact, with nothing above `debug`.
`_step_score.py:308` is the worst: `logger.debug("suppressed: %s", e)` names nothing — not the feature
name `_nm0`, not the operation.

**Fix:** use `float("-inf")` so a failed candidate is provably excluded rather than tied at the floor,
and include the candidate name/index in every message.

**Test:** `test_relevance_probe_failure_excludes_candidate_rather_than_tying_at_zero`.

---

### NUM-34 — `_eligible_floor = False` can silently empty the whole raw-representative floor pool  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group4.py:194-196`.

**What:**
```python
                                except Exception as exc:
                                    logger.debug("mrmr: floor-eligibility check failed for this candidate; treating as ineligible (conservative): %r", exc, exc_info=True)
                                    _eligible_floor = False
```

**Why:** consumed two lines later (`if not _eligible_floor: continue`). Conservative per-candidate, but
a **systematic** fault (e.g. `X` not a DataFrame on a polars path, hitting `:186-189` every iteration)
silently empties the entire raw-representative floor pool with only `debug` records.

**Fix:** count failures and emit one `warning` when the failure count equals the candidate count — i.e.
when the check is systematically broken rather than per-candidate flaky.

**Test:** `test_floor_eligibility_systemic_failure_is_warned`.

---

### NUM-35 — `mrmr_gains_ = np.array([])` re-breaks the exact bug the attribute was restored to fix  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_assign_support.py:672-674` and `:691-693`.

**What:**
```python
    except Exception as exc:
        logger.debug("mrmr: mrmr_gains_ computation failed; using an empty array: %r", exc, exc_info=True)
        self.mrmr_gains_ = np.array([], dtype=np.float64)
```

**Why:** the comment at `:660-666` describes the consequence directly: an empty `mrmr_gains_` makes
`gains.size >= 3` False and the UAED auto-size block *"guaranteed dead code"* — so
`MRMR(uaed_auto_size=True)` silently returns the full screen output. The attribute was restored
specifically to fix that; this handler quietly re-creates it, at `debug`. `:691-693`
(`self._predictors_log_ = ()`) is the same shape with a smaller blast radius.

**Fix:** `logger.warning` naming that `uaed_auto_size` will silently no-op for this fit.

**Test:** `test_uaed_auto_size_not_silently_disabled_by_gains_failure`.

---

### NUM-36 — `except Exception → pass` on the recipe `cat_code_maps` attachment (train/serve skew)  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:1671-1673`.

**What:**
```python
                    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                        logger.debug("mrmr: attaching cat_code_maps to recipe %r failed: %r", getattr(r, "name", "?"), e, exc_info=True)
                        pass
```

**Why:** not neutral. Failing to attach `cat_code_maps` leaves the recipe without its train-time code
mapping, which the surrounding comment at `:1640-1643` ties to *"the same silent train/serve skew"* —
a serve-time correctness risk, not a perf one, yet it is at `debug`. (The sibling at `:1395-1397`,
dtype narrowing, IS genuinely neutral — a memory optimisation only.)

**Fix:** `logger.warning` naming the recipe; drop the dead `pass` in both.

**Test:** `test_cat_code_map_attach_failure_is_warned`.

---

### NUM-37 — `-np.inf` from OLS/QR probe closures conflated with a measured rejection  [P3]

**Where:** `_mrmr_fit_impl/_friend_graph_and_redundancy/_group2.py:231, 234, 483, 486, 492`;
`_group3.py:107`; and the conversion at `_group2.py:239`.

**Why:** the score is maximised, so `-inf` rejects the candidate — the conservative direction, hence
P3. Logging at `:230/:233/:106/:491` is good (exception type AND message). Two genuine nits:
`_group2.py:483,486` return `-np.inf` from `_rp_qr_ok == False` with **no log at that point** (the only
log was the one-time `:472`), so a whole candidate sweep silently returns `-inf` with a single upstream
debug line; and `:239` (`if not (np.isfinite(r2_base) and np.isfinite(r2_full)): return 0.0`) converts
the `-inf` into a neutral-looking `0.0` uplift, erasing "measured zero uplift" vs "the solve failed".

**Fix:** adopt the `_fit_impl_core.py:616-622` convention — `None` for "could not measure", `-inf` only
for "measured and rejected".

---

### NUM-38 — `False` for "is GPU available" from broad-except capability probes (perf-only)  [P3]

**Where:** `_mrmr_fe_step/_step_core.py:751-753` (`_gpu_fe_active = False`);
`_mrmr_fe_step/_step_pairs_rank.py:37-39` (`return False`), `:50-52` (`return codes`);
`_mrmr_fe_step/_step_score.py:86-88` (`_gate_resident = False`);
`_orthogonal_univariate_fe/_orth_mi_backends.py:64-66` (`return False`);
`_fe_gpu_vram.py:164-166`, `:218-220`.

**Why:** the named "False for is-GPU-available" pattern, from broad `except Exception` around a
capability probe. All are P3 rather than the `_select_mi_backend` catastrophe because **none is cached
at module level** — each is re-evaluated per call — and each substituted value routes to a correct
(merely slower) path. `_orth_mi_backends.py:64-66` in particular routes to the njit dispatcher, which
IS the fast production default. `_step_core.py:751` is the most notable because the adjacent `:742-744`
correctly uses `logger.warning` for the analogous *"~33x slower"* fallback.

**Fix:** split each into `except ImportError` (debug, genuinely optional) and `except Exception`
(warning naming `type(exc).__name__` — a real device fault), mirroring the `_select_mi_backend` remedy.

---

### NUM-39 — three consecutive broad `except Exception` around trial imports in the VRAM-teardown path  [P3]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core.py:70-88`.

**Why:** the comments call these *"optional dependency import guard"* while catching everything, so a
genuine `cudaErrorLaunchFailure` inside `free_all_blocks()` is indistinguishable from "cupy not
installed". Teardown-only and the `return False` at `:88` is a truthful "did not reclaim", not cached —
hence P3. But the documented consequence at `:60` (*"11.2s → 31.8s → 32.3s"* degradation) is invisible
at `debug`.

**Fix:** split `ImportError` from everything else; warn on the latter.

---

### NUM-40 — content-free `logger.debug("suppressed: %s", e)` + redundant `pass`, 13 sites  [P3]

**Where:** `_mrmr_sis_screen.py:112-117` (twice, nested);
`_mrmr_fingerprints.py:269-271, 408-410, 479-481, 571-573`;
`_mrmr_fe_provenance.py:298-300, 348-350, 384-386, 409-411`;
`_mrmr_fit_impl/_friend_graph_and_redundancy/_group3.py:334-336`;
`_mrmr_fe_step/_step_score.py:307-309`; `_adaptive_nbins.py:976-978`;
`info_theory/_cmi_cuda.py:807-809, 846-848`.

**Why:** correct-but-UNDIAGNOSABLE. The substituted values are genuinely neutral at every one of these
sites (an analytic fallback width, a cache put, a KTC update, a provenance row), so behaviour is fine.
The defect is that `"suppressed: %s"` names neither the function, the key, the column, nor which of
several nested handlers fired — three structurally different failures in `_cmi_cuda.py` are
indistinguishable in a log. The `pass` after each log is dead code.
`_mrmr_fe_provenance.py:348-351` additionally returns the sentinel `-1` rank after swallowing, so a
provenance row reads "not greedily ranked" when the lookup merely errored.

**Fix:** give each a message naming the operation and its inputs (e.g.
`"per_feature_edges: cache put failed col=%d (%s: %s)"`); drop every trailing `pass`. Note the
*surrounding* code in `_mrmr_fingerprints.py` is exemplary (`:134-140`, `:233-239` both escalate to
`logger.warning` and reason explicitly about non-neutrality) — these are leftovers that never got the
same treatment.

---

### NUM-41 — `except Exception → return ""` / `= {}` for diagnostic-only artifacts  [P3]

**Where:** `_mrmr_explain.py:81-83` (`return ""`); `mrmr/_mrmr_class.py:3589-3591` and
`mrmr/_mrmr_class_fit_helpers.py:591-593` (`self.degenerate_columns_ = {}`);
`_mrmr_fingerprints.py:272-276` (`items.append((k, id(v)))`).

**Why:** no selection decision is corrupted — `audit_degenerate_columns`' own docstring says it *"never
influences selection"*, and `_fmt_margin_band`'s contract is never-raise. Two residual nits:
`degenerate_columns_ == {}` is indistinguishable from "audited, found nothing"; and falling back to
`id(v)` inside a **cache-key** signature is the address-reuse hazard `_mrmr_fingerprints.py:234-239`
elsewhere explicitly refuses (here it can only cause spurious cache *misses*, never a wrong hit).

**Fix:** set a companion `degenerate_audit_failed_ = True` so `{}` is unambiguous;
`_fmt_margin_band` should embed `type(exc).__name__` in its returned text the way its siblings at
`:240-262` already do; substitute a never-matching `uuid4().hex` token rather than `id(v)`, as
`_content_array_signature` does at `:305`.

---

### NUM-42 — narrow but unlogged `except ValueError: pass`  [P3]

**Where:** `src/mlframe/feature_selection/filters/_neural_mi.py:281-282` and `:393-394`.

**Why:** narrow exception type on a best-effort cache/coercion path, low blast radius — but zero
logging at all.

**Fix:** a one-line `logger.debug` naming the input shape.

---

### Verified-clean (bug class 2)

Handlers checked and found genuinely correct — the models to copy, and not to be re-flagged.

**The 2026-08-02 MI-backend fixes still hold in current source:**
- `_orthogonal_univariate_fe/_orth_mi_backends.py:228-257` (`_select_mi_backend`) — `except ImportError`
  → `"sklearn"` at debug (`:231-235`); `except Exception` → `logger.warning` naming the exception
  type/message → `return "numba"` (`:236-257`). The module-level cache at `:260` is unchanged but now
  safe, since only a genuine `ImportError` can latch the downgrade.
- `_orth_mi_backends.py:166-186` — the per-call sibling fix also holds: `logger.warning` naming the
  exception type, message and slice width before falling back to sklearn. (Its *other* call site,
  `:204-206`, was missed — that is NUM-17.)

**Correctly narrowed probes:**
- `_mrmr_fe_step/_step_pairmi.py:177-179` (`except ImportError → _CUDA_AVAIL = False`, not module-cached);
  `batch_pair_mi_gpu.py:66-68` (`except ImportError` for the module-level `_CUPY_AVAIL`);
  `hermite_fe/__init__.py:258-262` (`except ImportError: pass` around the `import cupy` probe — the very
  module CLAUDE.md cites as the source of the transient fault, and it is correctly narrow);
  `_mrmr_partial_fit.py:70-75`; `_mrmr_validate_transform.py:221-222, 254-257, 450-451, 535-536, 702-703`;
  `mrmr/_mrmr_class_fit_helpers.py:346-350`; `_fit_impl_core.py:1176-1177`;
  `_resident_candidate_mi.py:149-151, 216-218`; `_ksg.py:456-458`; `_neural_mi.py:67-68, 426-427`
  (re-raised with an actionable install message).
- `_ksg.py:515-522` — correctly **split**: `except ImportError: pass` (genuinely absent) vs
  `except Exception` → trips a documented circuit breaker. Best-in-repo shape.
- `_fe_gpu_strict.py:207-231` — a broad except around `numba.cuda.is_available()` that logs THREE
  warnings (initial fault, per-retry, final), retries a bounded number of times, and only then returns
  `False` "so routing stays consistent". Exemplary handling of the "is GPU available" shape.
- `info_theory/_cmi_cuda.py:64-65, 83-84`; `_gpu_resident_pair_mi.py:117-118, 262-263` —
  `except (TypeError, ValueError)` on env-var parsing → documented default constants.
- `_mrmr_sis_apply.py:62-63` — `except (TypeError, ValueError) → None`, feeding an already-`Optional` floor.
- `_mrmr_tree_rescue.py:118-124` — narrow `except (ValueError, TypeError)`, and the slow path emits a
  `logger.warning` counting coerced cells and naming the columns.
- `_mrmr_fe_step/_step_core.py:205-208` — `except NotImplementedError: pass` on a documented Phase-0 stub.
- `_mi_aggregator.py:124-126` — `except np.linalg.LinAlgError` → uniform weights (the genuinely neutral
  substitution for a singular system).

**Tri-state / out-of-band sentinels (the pattern NUM-12, NUM-22 and NUM-37 should adopt):**
- `_mrmr_fit_impl/_fit_impl_core.py:616-622` — *"Only a genuine MEASURED sub-threshold uplift evicts."*
  `None ≠ 0.0`, explicit fail-open.
- `_mrmr_fe_step_helpers.py:688-695` — `_n_pairs_considered = -1` / `_n_pairs_with_additions = -1`: an
  out-of-band sentinel visibly impossible in the summary log, rather than a plausible-looking `0`.
- `_mrmr_fingerprints.py:134-140, 233-239` — escalate to `logger.warning` and substitute a
  **never-matching** token that disables the cache rather than risking a wrong hit; the comments reason
  explicitly about CPython address reuse.
- `mrmr/_mrmr_class.py:3624-3639` — `_ycorr_ok = False` when the identity-cache y-correlation cannot be
  confirmed: refuse the shortcut, run the full fit. Correct polarity.

**WARNING-level fallbacks naming the exception and the consequence:**
- `_mrmr_sis_screen.py:305-306, 310-311` — `log_throttle(logger, ..., logging.WARNING, "...failed (%s); scored 0", j0, j1, exc)`.
  Substitutes 0 for a failed MI block but at WARNING, naming the exception AND the exact slice
  `[j0:j1]`. The reference use of `log_throttle` on this surface.
- `_mrmr_sis_screen.py:359-360`; `_mrmr_degenerate.py:171-173`; `_mrmr_artifacts.py:220-230`;
  `_mrmr_tree_rescue.py:162-163` (`warnings.warn`, user-visible);
  `_mrmr_fe_provenance.py:511-517`; `mrmr/_mrmr_class.py:3464-3466, 3508-3509, 3541-3545, 3712-3716,
  3760-3764, 3841-3845, 4129-4130`; `mrmr/_mrmr_class_fit_helpers.py:200-210`;
  `_mrmr_validate_transform.py:507-513`; `_mrmr_fe_step/_step_pairmi.py:214-226`;
  `_mrmr_fe_step/_step_core.py:550-553, 742-745`; `_assign_support_tail.py:291-301`;
  `_mrmr_fe_step_helpers.py:89-93, 155-159, 321, 372, 581, 660`; all per-FE-family isolation handlers in
  `_fe_stage_cascade_early_a.py:245, 399, 485, 597, 678`, `_fe_stage_cascade_early_b.py`,
  `_fe_stage_cascade_mid_a.py`, `_fe_stage_cascade_mid_b.py`, `_fe_stage_temporal_agg.py`,
  `_hybrid_orth_family_variants/_group1-4.py`, `_friend_graph_and_redundancy/_group1.py:88, 150, 233`,
  `_group3.py:277, 419`, `_group4.py:231, 319`, `_finalise.py:342`, `_fit_impl_core.py:643, 1034, 2231`,
  `_helpers.py:453`; `_orthogonal_univariate_fe/_orth_dedup.py:201-202`;
  `batch_pair_mi_gpu.py:438-439, 450-451, 461-462, 482-483, 509-510`;
  `_gpu_resident_pair_mi.py:102-105, 777-780`; `_fastmi.py:222-230`;
  `_mi_aggregator.py:75-76, 163-164, 205-206` (per-estimator isolation via `log_throttle`, the failing
  estimator is **skipped**, not substituted); `stability.py:186-193` (names the seed, excludes the draw,
  and recomputes frequencies over the successful `B` — a model of "fallback that adjusts the
  denominator instead of faking a value").
- `_mrmr_fe_step/_step_pairs_rank.py:259-261` (returns the bar unchanged — genuinely neutral),
  `:689-691` (`_admit_via_perm = False` restores the base gate exactly, since it is an OR-ed *extra*
  admission route).
- `mrmr/_mrmr_class.py:171-182` (`_safe_restore`) — the swallow is a `finally`-restore step whose
  purpose is to not mask the fit's real outcome; debug is correct here.
- `mrmr/_mrmr_class_fit_helpers.py:107-124` — five GPU circuit-breaker **re-arm** calls; failing to
  re-arm leaves the breaker in its prior state (fail-safe), and no guard depends on them.
- `_fit_impl_core.py:58-67` — best-effort, idempotent, leaves the prior value intact.
- `_mrmr_explain.py:240-262` — five section handlers that both log AND embed `type(exc).__name__` into
  the user-visible narrative.
- `info_theory/_cmi_cuda.py:145-147` (`_CUPY_OK = False` from a direct `import cupy` + device query,
  where any failure genuinely means "no usable GPU" and the fallback is the correct-answer CPU path),
  `:788-789`, `:951-954` (launch fault trips the documented `_CMI_GPU_FAILED` breaker);
  `info_theory/_batch_kernels.py:913-916, 928-931`; `_cmi_cuda_ktc.py:71-74, 150-153` (KTC selection
  fallbacks — debug, but each names the exception with `exc_info=True` and the substituted value is a
  heuristic affecting only speed, never the MI value).
- `_mrmr_fe_step/_helpers.py:39-43` — returns `set()` but the comment reasons explicitly about the
  consequence and it fails loudly downstream by design. Acceptable; `warning` would still be better.

**No bare `except:` (no exception class) exists anywhere on the MRMR surface.**

---

## Proposed tests (beyond the per-finding ones)

- `test_no_raw_power_sum_variance_in_mrmr_surface` — a meta-gate in the spirit of the repo's existing
  shared meta-tests: AST-walk every `.py` under the MRMR surface plus the correlation/moment primitives
  and fail on a binary `Sub` whose left operand is a name matching `s?xx|s?yy|sq|sum_?sq|second|m2` and
  whose right operand is a `Mult`/`Pow` of a name matching `mean|mu|m1|s[xy]`. Would have caught NUM-1
  and NUM-2 mechanically, and is the only defence against the class recurring a fifth time.
- `test_no_additive_epsilon_in_a_variance_denominator` — the sibling meta-gate: flag
  `BinOp(Div, _, BinOp(Add, <name matching std|var|sd|norm|ptp|scale>, <float constant < 1e-3>))`.
  Would have caught NUM-5, NUM-6, NUM-7, NUM-9 in one pass. Allow-list the deliberate
  feature-definition ratios of NUM-11 by an explicit path/name list so the gate stays honest.
- `test_shift_invariance_of_every_correlation_primitive` — parametrise over
  `_abs_corr_finite_njit`, `_abs_corr_zerofill_njit`, `_pairwise_complete_abs_corr` (all 3 backends),
  `_eng_dedup_batch_corr`'s kernel and the `_gpu_resident_basis` RawKernel; assert
  `f(x, y) == f(x + 1e8, y)` to 1e-9. Pearson `r` is translation-invariant by definition, so this one
  invariant catches the entire class-1 surface at once and needs no regime guessing.
- `test_every_gate_fallback_logs_above_debug` — collect the MRMR surface's `except` handlers by AST,
  and for each whose body assigns a literal `True`/`False`/`0.0`/`{}`/`set()`/`[]`/`-inf` to a name
  matching `.*(floor|gate|ok|eligible|enabled|allow|valid|drop|retain).*`, require a
  `logger.warning`/`log_throttle(..., WARNING, ...)` in the same handler. Would have caught
  NUM-12, NUM-14, NUM-18, NUM-19, NUM-21, NUM-31 mechanically.
- `test_mrmr_fit_emits_no_debug_only_gate_disable` — behavioural companion: run a small fit with
  `caplog` at DEBUG and assert no record whose message matches `disable\w*|skipping the guard|continuing
  without|treating as` sits at DEBUG level. Mutation-resistant because it asserts on emitted records,
  not on source text.
- `test_uplift_primitive_is_shared` — once NUM-5's `_relative_uplift` helper exists, grep-gate that no
  file re-implements `/ (baseline + 1e-12)` inline. Prevents the 20-copy drift from re-forming.

## Prior-wave findings touching this lane

The `-07-25` tracker's items are outside this lane's two bug classes except where noted; what this
sweep can state from the CURRENT source:

- The 2026-08-02 `_select_mi_backend` narrow-`ImportError` fix **still holds** verbatim
  (`_orth_mi_backends.py:228-257`), and its module-level cache at `:260` is now safe. Confirmed, not
  regressed.
- The 2026-08-02 per-call numba/GPU dispatch `logger.warning` fix **still holds** at `:166-186` but was
  **never applied to the sibling call site at `:204-206`** — that is NUM-17, a genuine gap left by that
  wave rather than a regression.
- The 2026-08-04 "grep at any k >= 2, not just skew/kurt" mandate has been carried out on the
  `_binned_numeric_agg` / `_target_encoding` / `_derive_cell_stats` / GPU-twin family and on
  `_pairs_core` / `_eng_dedup_batch_corr` / `_usability_njit_pool` (all verified fixed above), but
  **`_orth_dedup.py` and the `_gpu_resident_basis.py` RawKernel were not reached by it** — NUM-1 and
  NUM-2. Both are live in the current source at `57f649fb6`.
- The 2026-08-05 `clear_fe_deadline` wiring is unrelated to these two classes; not re-checked here.

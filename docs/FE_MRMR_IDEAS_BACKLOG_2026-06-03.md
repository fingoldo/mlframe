# MRMR Feature-Engineering — improvement & synergy backlog (2026-06-03)

Synthesis of a 4-agent ideation sweep over mlframe's MRMR FE, briefed on what
**shipped** and what was **benchmarked-and-rejected** this session. Each idea is
grounded in the actual gate/operator/estimator code (file:function cited).

> **Guiding lesson from this session:** across five rejected ideas the binding
> constraint was always the **admission / selection machinery**, NOT the richness
> of feature construction. Features were constructible but died at a gate, or were
> already implemented / subsumed. So the highest-confidence wins are the ones that
> let a *genuinely-good feature pass a gate it currently fails*. Anything that
> lowers a bar must keep the **order-2 maxT permutation-null floor** as the outer
> guard, and must debias **both** sides of any ratio consistently.

---

## Session baseline (so we don't re-tread)

**Shipped this session (default-on):**
- Order-2 Westfall-Young maxT permutation-null floor on prospective-pair JOINT MI (`_permutation_null.pooled_pair_permutation_null_joint_mi_floor`).
- Adaptive-frequency Fourier + adaptive **chirp** (quadratic-arg warp `u=sign(z)·z²`).
- Signal-adaptive orthogonal-poly basis routing (Hermite/Legendre/Chebyshev/Laguerre by best-degree |corr|).
- `fe_strict` engineered-MI-prevalence tightening; per-operand prewarp ALS.
- `verbose=0` now silences tqdm at the source.

**Benchmarked + REJECTED (documented in-code as `bench-rejected (2026-06-03)`):**
- Poly-vs-Fourier competition gate — co-occurrence is complementarity, not redundancy (`generate_extra_basis_features`).
- Poly-feature synergy re-entry GAP-3 — pool entry is NOT the blocker; 4 downstream gates are (`_mrmr_fe_step.py` synergy bootstrap).
- Product-signal pair-basis routing — prod path already moment-routes + sweeps degrees (`generate_pair_cross_basis_features`).
- Rank/quantile operator — already exists as RankGauss, subsumed by spline (`mrmr.py` fe_rankgauss note).
- Target-supervised spline knots — no spline column survives the MI-uplift gate (`_fit_spline_for_col`).

**The four gates that blocked item-4's genuine poly×raw regression synergy** (the recurring villains): (1) the 0.97 engineered-MI-prevalence bar vs a finite-sample-inflated 2-D joint MI; (2) marginal-MI basis routing (blind to pure synergy); (3) cross-pair seed pool excludes linear operands; (4) the noise-aware abs_floor self-inflated by a lone large signal's MAD.

---

## TIER 1 — cheap, high-confidence, unblock features that died THIS session

### 1. Miller-Madow debiasing of the prevalence RATIO ⭐
- **Mechanism.** The prevalence gate computes `best_mi / pair_mi > 0.97`. Numerator = 1-D engineered MI over `nbins` bins (`compute_mi_from_classes`); denominator = 2-D joint MI over `nbins²` bins (`batch_pair_mi_prange`). Both raw plug-in. Plug-in MI bias ≈ `(Kx−1)(Ky−1)/2n`, so the denominator carries ~`nbins`× more positive bias → ratio structurally depressed below 1.0 even when the 1-D feature captures all the joint info. Fix: subtract the MM term from each side before the ratio (`mi_mm = mi_plugin − (Kx−1)(Ky−1)/2n`), reusing `entropy_miller_madow` (info_theory.py:119). 1-line per term, no new estimator.
- **Why not a dead-end.** Directly attacks item-4's named blocker #1 (1-D-vs-inflated-2-D MI). Pure estimator-scale fix, distinct from the rejected construction/pool ideas. The maxT null already proves the team trusts debiasing the selection statistic.
- **Win regime.** Moderate-large `nbins`, small-moderate `n` (bias term `~nbins²/2n` non-negligible) — exactly the poly×raw regression miss. →0 at large n, so large-n recovery byte-untouched.
- **Cheap validation (<1hr).** `y=He2(a)·b+noise`, n∈{500,2000,8000}, nbins=10. Log best_mi, pair_mi, raw ratio, MM ratio. Confirm: (a) corrected ratio crosses 0.97 for genuine synergy where raw doesn't at small n; (b) pure-noise frame stays <0.97 (no FP); (c) n=8000 raw≈corrected. Check engineered count on `test_biz_val_mrmr_default_filtering` noise fixture.
- **Feasibility / leak-safety.** Trivial, closed-form, leak-safe. **RISK:** lowers the denominator → could admit more pairs → must apply MM to the **order-2 maxT floor comparison too** (floor uses the same uncorrected `batch_pair_mi_prange` scale at `_mrmr_fe_step.py:441,469`), else the floor weakens.
- **Integration.** `_feature_engineering_pairs.py:721` (`_passes_joint_gate`); add `_mm_correct(mi, k_x, k_y, n)` near `entropy_miller_madow`.

### 2. Leave-candidate-out / trimmed abs_floor ⭐
- **Mechanism.** `noise_floor = med + 3.5·1.4826·MAD` pools `raw_baselines`/`pool` **including the candidate under test** (`_orthogonal_cluster_basis_fe.py:529-543`, `_mi_greedy_fe.py:468-483`). A lone strong signal drags up the median + MAD, raising the floor above the very signal it should admit. Fix: compute med/MAD from the pool **excluding the candidate** (leave-one-out), or from the lower ~85-90% order-statistic (trimmed reference the top signals can't lift).
- **Why not a dead-end.** Item-4 blocker #4, the documented "self-inflated by a LONE LARGE signal's MAD." The current comment claims median-MAD is robust "to a few outliers" — true for k-many signals, false for the lone (k=1) case.
- **Win regime.** Few strong signals among many noise columns. All-noise frames → LOO floor ~unchanged (FP control preserved).
- **Cheap validation (<1hr).** Pool of 1 strong-signal MI + 15 noise MIs → confirm current floor rejects the signal (bug reproduced); LOO/trimmed floor sits below signal, above all noise. Then 16-noise pool → both reject all (no regression). Seconds to run.
- **Feasibility / leak-safety.** Easy, pure NumPy, leak-safe. **RISK:** trimming slightly lowers the floor — verify a 2-signal+14-noise pool doesn't admit a borderline noise neighbor; cap trim at lower 85-90%.
- **Integration.** BOTH MAD-floor sites in one pass (grep-all rule): `_orthogonal_cluster_basis_fe.py:529-543` + `_mi_greedy_fe.py:468-483`.

### 3. Per-gate rejection ledger (binding-constraint diagnostics) ⭐
- **Mechanism.** Every gate (prevalence, maxT floor, abs MAD floor, prewarp uplift, 2nd-pass CMI, basis routing) silently drops candidates. Add a per-candidate ledger `(candidate, source_pair, first_gate_that_killed_it, margin_to_threshold)` → aggregate into `fe_rejection_ledger_` next to the survivor-only `fe_provenance_`. One-line summary: "of 412 candidates, 380 died at the 0.97 prevalence gate (median margin −0.04)."
- **Why orthogonal / not present.** Pure observability, touches no decision logic. `_mrmr_fe_provenance.py` (Layer 54) records survivors only — no rejection-reason field exists anywhere. Turns "I suspect gate A or B" into a measured verdict (serves the no-handwave-hypotheses + think-ahead-diagnostics rules).
- **Win regime.** Debuggability; makes every future FE tuning session self-diagnosing.
- **Cheap validation (<1hr).** Run on the poly×raw-regression case; confirm the ledger fingers the prevalence + abs-floor gates as the binding pair the session diagnosed by hand.
- **Feasibility / leak-safety.** Highest (lowest-risk item); metadata, fit-only, no leak surface.
- **Integration.** Counter increments at each gate site in `_mrmr_fe_step.py` / `_unified_fe_gate.py`, assembled like `fe_provenance_`.

### 4. Effective-bins (occupancy-aware) bias correction
- **Mechanism.** Refinement of #1: use the **occupied** bin count `K_eff = #{bins with count>0}` (what `entropy_miller_madow` already counts internally, info_theory.py:137) instead of nominal `nbins²` in the MM term. Heavy-tailed engineered columns frequently collapse to 3 occupied bins; nominal `K` over-corrects the numerator and understates the denominator inflation.
- **Why not a dead-end.** Makes #1 robust to the binning-collapse pathology the code already documents (`_feature_engineering_pairs.py:954-959`, MI 0.14 vs true 0.32). Reuses existing tooling.
- **Cheap validation.** Heavy-tailed `a²/b` fixture: print nominal-K vs occupied-K MM correction + ratio; cross-check vs KSG (#19) as ground truth.
- **Integration.** Same site as #1; apply occupied-K symmetrically to both terms.

---

## TIER 2 — the real frontier: scalable + higher-order synergy discovery

### 5. Permutation-null-calibrated prevalence bar
- **Mechanism.** Replace the hardcoded 0.97 with the **null** ratio: in the same K shuffles the order-2 maxT floor already runs, also compute the best 1-D engineered MI per pair under each shuffle, record `best_1d / pair_mi`; the q-quantile of that null-ratio distribution is the chance ceiling. Admit the real ratio only above it. Converts an absolute threshold into a self-gating, per-pool, finite-sample-aware one.
- **Why not a dead-end.** Reuses the trusted, shipped maxT machinery; composes with #1 (#1 fixes the MI *scale*, this fixes the *threshold*).
- **Win regime.** Small-n high-cardinality where the chance ratio is naturally well below 0.97. →1 at large n (won't loosen large-n behavior).
- **Cheap validation.** Noise fixture: empirical null-ratio q95 ≈ engineered ratios noise produces (FPs stay rejected). `y=He2(a)·b`: genuine ratio > null q95 while raw 0.97 rejects it.
- **Feasibility / leak-safety.** Moderate; adds one `discretize_array`+`compute_mi_from_classes` per shuffle per top-M pair (bounded). It IS a permutation test → leak-safe. **RISK:** null-ratio must use the best 1-D MI under shuffle (mirror the real max-over-transforms search), else it understates chance.
- **Integration.** Extend `pooled_pair_permutation_null_joint_mi_floor` to also return `null_ratio_quantile`; consume at `_feature_engineering_pairs.py:721`.

### 6. Surrogate-GBM split-co-occurrence seeder ⭐
- **Mechanism.** Fit one fast shallow GBM (LightGBM / HistGBM, ~100 depth-3 trees) on the discretised matrix per fold. Walk root-to-leaf paths; tally depth-discounted split-gain co-occurrence for every pair/triple that co-occur on a path. Emit top-K weighted pairs into the prospective pool and top-K triples directly into `triplets=` for `hybrid_orth_mi_triplet_fe`, **bypassing the univariate-MI `seed_count` gate** that's blind to pure synergy.
- **Why it beats all-pairs bootstrap.** O(n·trees·depth) — independent of p²/p³. A zero-marginal synergy operand still appears as a split partner conditioned on its co-splitter → reaches the pool where `seed_count` (univariate top-N) never would. Reaches 3-way for free via path co-occurrence. (Existing triplet/quadruplet modules seed by univariate top-N — exactly the blind seeder this replaces.)
- **Win regime.** Large p (100s–1000s cols) where all-pairs is infeasible; tree-discoverable multiplicative/threshold interactions.
- **Cheap validation (<1hr).** n=4000, p=200 iid, `y=sign(x7·x42·x113)+0.3·noise` (3-way needle, all marginals ≈0). Current: seed_count excludes all three → triplet never enumerated → miss. New: co-occurrence ranks {7,42,113} top → enumerated + recovered. Measure needle recall in proposed candidates + OOS uplift.
- **Noise control.** Proposer only *generates*; the maxT floors gate. Compute the floor over the **proposed** pool size (smaller family → less punishing, still bounds chance-max). Self-gate the GBM: only seed if its OOF score beats a permuted-y GBM baseline.
- **Feasibility / cost.** O(n·trees·depth) fit + O(trees·depth²) tally. LightGBM = 1 dep (extra-deps-OK rule). Opt-in seeder feeding the pool before the pair sweep.
- **Integration.** New `_surrogate_interaction_seeder.py`; called at top of `_run_fe_step` to populate `_seeded_pairs`/`_seeded_triplets`.

### 7. Order-3 Westfall-Young maxT floor (mandatory rail for 3-way)
- **Mechanism.** Generalise `pooled_pair_permutation_null_joint_mi_floor` to triples: shuffle y K times, per-shuffle MAX 3-D joint MI over the triple pool via a new `batch_triple_mi_prange` njit kernel (extend `batch_pair_mi_prange` + dense-renumber for the 3-way joint cardinality), floor at the q-quantile. Gate every 3-way candidate from #6/#9/#10.
- **Why.** Load-bearing safety rail — the triplet/quadruplet modules currently lack an order-matched floor; opening 3-way generation WILL surface chance-max noise triples without it. Ships ON, in the same change as any 3-way proposer.
- **Cheap validation.** n=2000, p=80 pure noise, all C(80,3) triples → current path engineers a spurious 3-way; with floor → 0 survive; genuine 3-way XOR needle still clears.
- **Feasibility.** K shuffles × batched triple-MI; dense-renumber keeps cardinality ≤ n; cost scales with proposed (small) triple count. `fe_triple_maxt_*` knobs mirror order-2 (self-gating, `=0` disables).
- **Integration.** `pooled_triple_permutation_null_joint_mi_floor` + `batch_triple_mi_prange` in `_permutation_null.py`/`info_theory.py`.

### 8. Interaction-information (co-information) ranking
- **Mechanism.** For each candidate tuple, compute signed `II(a;b;y) = I((a,b);y) − I(a;y) − I(b;y)` (and order-3 co-information for triples). Positive II = genuine synergy; negative = redundancy. Rank by positive II (not raw joint MI), and ROUTE: positive→product/cross-basis FE, ≈0→drop (additive, already screened), negative→cluster-aggregate/denoise.
- **Why it beats the ratio gate.** The current `pair_mi > sum·prevalence` ratio conflates synergy with finite-sample joint-MI inflation; II is the signed difference that directly measures synergy and tells you which FE family to apply.
- **Cheap validation.** n=2000, three pair types: synergistic (y=a·b, II>0), redundant (a≈b∝y, II<0), additive (y=a+b, II≈0). Current ratio gate admits the redundant pair + wastes search; II keeps only synergistic. Report II values + spurious-column counts.
- **Noise control.** MM-correct all three terms (#1); floor positive II via permutation null. Near-free: marginal MIs `cached_MIs[(a,)],[(b,)]` and joint `pair_mi` are all already computed.
- **Integration.** The uplift-decision block in `_mrmr_fe_step.py` (the `pair_mi > ind_elems_mi_sum * _prev_thresh` site); thread a routing tag.

### 9. RFF / random-projection interaction pre-screen (detect-without-enumerate)
- **Mechanism.** Draw R random **sparse** (support 2-4 cols) hyperplane projections, form low-degree random features `φ_r(x)=cos(w_rᵀx+b_r)`; score each `φ_r`'s MI vs y. A `φ_r` over support {a,b} whose MI exceeds the additive baseline flags an interaction in its support without scoring that algebraic pair. Promote high-evidence supports to the pool. O(R·n), R≪p².
- **Why it beats all-pairs.** Linear in R; birthday-paradox coverage of high-signal supports. Detects product/XOR mixing without enumerating p²/p³. Congenial to the existing adaptive-Fourier/RFF infra.
- **Win regime.** Very large p (where the all-pairs bootstrap is gated off by `fe_synergy_screen_max_features`); smooth/oscillatory interactions.
- **Cheap validation.** n=2000, p=500, `y=x3·x400` (zero-marginal needle). Current: bootstrap skipped (p>cap). RFF R=4000 sparse-2 → {3,400} support promoted → pair recovered.
- **Noise control.** Own maxT floor on the per-projection MI-uplift max (shuffle y, q-quantile). Promoted supports then face the algebraic order-2/3 floors — double-gated.
- **Feasibility.** O(R·n) memory-light; numba/cupy kernel via the HW dispatcher + `kernel_tuning_cache`.
- **Integration.** New `_rff_interaction_prescreen.py`; replaces the all-pairs `combinations(...)` enumeration when p exceeds the bootstrap cap.

### 10. Conditional-MI complementarity growth (Apriori lattice)
- **Mechanism.** Use the already-computed prospective-pair joint MIs as the order-2 frontier. Grow Apriori-style: for each surviving pair (a,b), test only third columns c maximising conditional uplift `I((a,b,c);y) − I((a,b);y)` (reuse `conditional_mi`/`merge_vars` njit kernels). Keep a triple only above the order-3 maxT floor (#7); grow to order 4 from survivors. Never enumerates all triples.
- **Why it beats all-pairs.** Reaches 3+ way without O(p³): candidate set per level = `|surviving_(k-1)| × |shortlist_c|`, maxT-pruned. Conditional MI surfaces a c that only matters given (a,b) — pure higher-order synergy.
- **Cheap validation.** n=3000, p=40, `y=sign(x1·x2)·x3` ((x1,x2) detectable 2-way; x3 ~0 marginal + ~0 pairwise). Current triplet seeds by univariate top-N → x3 excluded → miss. Lattice: (x1,x2) survives, conditioning shows uplift → triple kept.
- **Noise control.** Order-3 maxT floor on the conditional uplift over the small Apriori-pruned candidate count; anti-monotone pruning (a triple whose constituent pairs failed the order-2 floor isn't grown).
- **Integration.** Replace the univariate-MI `seed_count` in `_orthogonal_triplet_fe.py`/`_orthogonal_quadruplet_fe.py` with "grow from order-2 survivors + CMI shortlist"; frontier state in `_mrmr_fe_step.py`.

---

## TIER 3 — genuinely new operators (dedup-checked against the catalog)

### 11. Hinge / piecewise-linear change-point basis ⭐ — ✅ SHIPPED (opt-in, 2026-06-09)
- **Status.** SHIPPED behind `MRMR(fe_hinge_enable=True)` (default OFF -- niche operator, ~2.2 ms/col quantile-cut SSE scan). `_hinge_basis_fe.py`; recipe kind `hinge_basis` `{tau, side}` registered in `apply_recipe`. **Decisive gate correction:** a single relu leg is MONOTONE in x → MI-INVARIANT (DPI), so the MI-uplift gate the backlog premised is WRONG for it (it drops the leg exactly like isotonic/RankGauss); admission is the HELD-OUT INCREMENTAL LINEAR-R² over raw x (the hinge's value is the second slope handed to a downstream linear model, not MI). Detector fed RAW y (qcut binning destroys the kink). Benchmark (n=4000): slope-change `[x,hinge]` held-out Ridge R² 0.9888 > raw 0.9071 / best Chebyshev 0.9857 / best B-spline 0.9883 (tau_hat 0.688 ~ true 0.7); smooth y=x² hinge 0.9412 LOSES to degree-2 poly 0.9737 (complementarity); noise 0/20 admit; monotone target emits 0 hinge (dedup-clean). Triad in `test_hinge_basis_fe.py`; numbers in `D:/Temp/hinge_results.md`.
- **Mechanism.** Detect 1-2 breakpoints τ per column by scanning candidate quantile cuts for the max drop in 2-segment linear-fit SSE (a slope-aware stump). Emit `relu(x−τ)=max(x−τ,0)`, `relu(τ−x)`, optionally `1[x>τ]`. Stack τ for adaptive-knot piecewise-linear.
- **Signal shape nothing captures.** A **slope change** at a data-dependent threshold: `y=a·x+b·max(x−τ,0)` (pricing tiers, dose-response, saturation). Closest: `numeric_rounding`→flat steps (piecewise-constant, wrong form); cubic B-spline→smooth + quantile (non-adaptive) knots that round off a sharp kink; orth-poly needs high degree + rings (Gibbs).
- **Cheap validation (<1hr).** `y=2x+5·max(x−0.7,0)+noise`, x~U(0,1), n=4000: hinge `relu(x−τ̂)` |corr|≈0.95+ and Ridge R² beats raw/Chebyshev/spline; on smooth control `y=x²` hinge must NOT beat degree-2 poly.
- **Leak-safe replay.** Store `{τ_list, side}` (no y) → `np.maximum(x−τ,0)`. New recipe kind `hinge_basis`. Mirror the spline recipe builder.
- **Gates.** MI-gateable (different linear shape clears uplift over raw x); held-out τ-validation (reuse the `%3` stride split) kills chance breakpoints. RISK: on a monotone target a hinge can be near-collinear with raw x → existing Spearman dedup drops it.
- **Integration.** New `_hinge_basis_fe.py` (`generate_hinge_features`/`hybrid_hinge_fe_with_recipes`) behind `fe_hinge_enable`; register `hinge_basis` in `apply_recipe`.

### 12. Conditional-dispersion (2nd-moment) features ⭐ — ✅ SHIPPED (DEFAULT-ON, 2026-06-09)
- **Status.** SHIPPED **default-on** via `MRMR(fe_conditional_dispersion_enable=True)` (Family D). New module `_extra_fe_families_dispersion.py` (re-exported from `_extra_fe_families.py` to stay under the module-size limit); recipe kind `conditional_dispersion` `{x_i, x_j, edges, bin_mean, bin_std, disp_kind}` registered in `apply_recipe`. Default-on is the right call: it is **MI-gateable** (the `|z|` fold is NON-monotone → genuine MI on heteroscedastic targets, UNLIKE the MI-invariant hinge/isotonic) and **self-limiting** (a dual-uplift gate admits a column only when its MI beats BOTH raw xᵢ AND the |mean-residual| Family-B sibling; on homoscedastic / canonical fixtures it admits 0). Decisive numbers below; full report `D:/Temp/dispersion_results.md`.
- **Benchmark (n=4000).** HETEROSCEDASTIC two-bin (bin A `xᵢ~N(0,1)`, bin B `xᵢ~N(0,5)`, `y=1[|xᵢ|>2σ_bin]`): **MI(y;|z|)=0.168 > MI(y;raw xᵢ)=0.140 > MI(y;mean-resid)=0.126** — the dispersion |z| is the TOP-MI feature, capturing heteroscedastic signal the mean-residual misses; downstream LogReg AUC lift over raw ≥ +0.03. HOMOSCEDASTIC control (constant spread): the dual-uplift gate admits **0** genuine dispersion cols (|z| is rank-identical to |mean-resid|, Spearman 0.997 → dedup/gate drop). NOISE control (pure noise, random y): admits **0**. CANONICAL `y=a²/b+log(c)·sin(d)`: dispersion admits **0** and the engineered set is BYTE-IDENTICAL to dispersion-OFF → it does NOT perturb pair-FE recovery (canonical biz regression test green; mirrors the hinge-regression catch but via the MI-gate self-limit, no deferred-materialisation needed). End-to-end on a heteroscedastic REGRESSION target the dispersion col is selected into support AND fed into a downstream pair-FE composite.
- **Mechanism.** Bin xⱼ; per bin store conditional **std** of xᵢ (not just mean). Emit conditional z-score `z=(xᵢ−μ̂_bin)/σ̂_bin` and dispersion-anomaly `|z|`/`z²` (the load-bearing NON-monotone folds; the signed `z` is a near-duplicate of the mean-residual under dedup, so the default emissions are `(absz, z2)`). A bin with too few rows / (near-)constant xᵢ falls back to the global std (no /0).
- **Signal shape nothing captures.** Conditional **variance/volatility regime**: `y=1[xᵢ unusually far from its xⱼ-conditional spread]` (fraud amount anomalous for that merchant, volatility regimes). Family B `conditional_residual` emits the **mean** deviation only; a small mean-residual can be a huge outlier vs a tight conditional spread. Every other op models conditional *location*, none conditional *scale*.
- **Leak-safe replay.** Stores xⱼ edges + per-bin (μ̂,σ̂) (extends `conditional_residual`'s `bin_mean` with `bin_std`); replay digitises xⱼ + computes `z=(xᵢ−μ̂)/σ̂` closed-form, reads only X (verified byte-exact, 0.0 replay error). New kind `conditional_dispersion`.
- **Gates.** Carries genuine MI on heteroscedastic targets → clears the normal MI floor AND a dual-uplift gate vs raw xᵢ + vs the |mean-residual| Family-B sibling. Homoscedastic data → σ̂≈const → |z|≈scaled |residual| → the dual-uplift gate (and the cross-stage Spearman dedup) drop it (self-limiting). Pure noise → nothing clears the floor.
- **Integration.** `_extra_fe_families_dispersion.py` (`generate_conditional_dispersion_features` / `hybrid_conditional_dispersion_fe` with the dual-uplift gate) re-exported from `_extra_fe_families.py`; MRMR Family-D stage in `_mrmr_fit_impl.py` (Family-B append pattern, routed through `local_mi_gate` + the dual-uplift gate). Triad in `tests/feature_selection/test_conditional_dispersion_fe.py` (16 tests: unit + biz_value + cProfile).

### 13. Haar wavelet / localized multiresolution basis
- **Mechanism.** On x's support, emit a small dyadic set of Haar wavelet indicators `ψ_{j,k}(x)` (+1 left half / −1 right half of dyadic interval, 0 outside), scales j=0..3. Localized + multiresolution step/contrast detectors; keep top scales by held-out power.
- **Signal shape nothing captures.** Localized bump / multiscale piecewise structure: `y` jumps only in `[0.4,0.5]` of x. Fourier is **global** (Gibbs ringing on a bump); spline knots are **fixed quantile** (bump smoothed away); rounding is global. Wavelets = simultaneously localized in x AND multiscale.
- **Cheap validation.** `y=1[0.45<x<0.55]+noise`, x~U(0,1), n=4000: best Haar `ψ_{3,k}` sharply beats best Fourier/spline; on smooth control `y=sin(2πx)` Fourier wins, Haar does NOT (complementarity).
- **Leak-safe replay.** Store `(lo,span)` + dyadic `(j,k)`; closed-form indicator. New kind `orth_wavelet` (structurally like `orth_spline`).
- **Gates.** A Haar leg in the right window has high marginal MI (unlike a single Fourier phase-leg) → clears uplift directly. RISK: O(2ʲ) candidates → the noise-aware MAD floor + held-out scale-selection required.
- **Integration.** Add `"wavelet"` to `_EXTRA_BASIS_KINDS` + `build_orth_wavelet_recipe` behind `fe_wavelet_enable`.

### 14. Isotonic (monotone-constrained) reshaping
- **Mechanism.** Fit `sklearn.isotonic.IsotonicRegression` of y on each column (both directions, keep better by held-out fit); emit the fitted monotone step function; store breakpoint/level table for `np.interp` replay.
- **Signal shape nothing captures.** Monotone-but-arbitrarily-nonlinear link `y=g(x)`, g monotone, shape unknown (saturating/log-utility). Spline + orth-poly are **unconstrained** (fit spurious non-monotone wiggle at noise); RankGauss is monotone but **shape-fixed to Gaussian**. Isotonic imposes the monotonicity *prior* with free shape — a bias reduction the catalog lacks.
- **Cheap validation.** `y=sigmoid(3x)+noise`, n=1500: isotonic beats spline at small n/higher noise (lower variance), ties at large n. Control: must NOT beat spline on non-monotone `y=x²`.
- **⚠️ Gates (critical).** Isotonic is monotone → **MI-invariant by DPI** (like RankGauss); a naive MI-uplift gate DROPS it. Value is downstream linear/NN usability, NOT MI. Must admit via the **RankGauss-style pool (rank by raw marginal MI, NOT gate on engineered MI)** + a downstream-linear-lift test — exactly why it survives where supervised-knots died. Lower-ranked: regime overlaps RankGauss.
- **Integration.** Extend `_extra_fe_families.py` (Family C owns the non-MI-gated monotone path) with a Family D; `fe_isotonic_enable`.

---

## TIER 4 — process / robustness (each gated on a safe predicate)

### 15. Cross-fold recipe stability voting (folded into the default decision)
- **Mechanism.** The expensive search runs once on full data; add a cheap K-fold *confirmation*: replay each surviving recipe (leak-safe) on K held-out folds, recompute uplift, admit only if it clears the gate in ≥⌈qK⌉ folds. A consensus layer over the existing gates.
- **Why orthogonal / not present.** `_stability_fe.py` does bootstrap voting but as a separate opt-in estimator that *refits MRMR N times* (expensive, not default-wired). This reuses the single full-fit's recipes and only *replays* on folds → near-free, default-on candidate. Complements maxT (maxT kills chance-max within a fold; this kills fold-specific winners).
- **Cheap validation.** 3-4 synthetic frames, survivor count + OOS uplift with vs without a 5-fold q=0.6 quorum; expect noise-frame survivors →0, signal preserved.
- **Integration.** Post-gate filter in `_mrmr_fe_step.py` before recipes are copied into `self._engineered_recipes_`.

### 16. Successive-halving / bandit FE-search budget
- **Mechanism.** Replace the flat top-K pair sweep (`fe_max_pair_features=10`, `fe_synergy_max_pairs=16`) with a rung schedule: cheap low-fidelity score (coarse nbins/subsample/few optimizer iters) over ALL pairs → keep top fraction → spend the expensive operator search only on survivors → halve again. Generalise the existing UCB1 allocator (`_cat_confirm_bandit.py`, cat-only) to route operator-search budget.
- **Why orthogonal.** Changes *where compute goes*, gates untouched. Lets users raise effective search depth at equal wall-time (attacks the "richness costs wall-time" tension).
- **Cheap validation.** p≈40, n≈5000: current top-K vs 2-rung halving (low-fi all pairs → full top 25%). Expect 2-4× at identical survivors.
- **Feasibility.** Low-fi score must be a monotone-ish proxy (subsample joint-MI). `kernel_tuning_cache` for rung fractions per (n,p). Integration: pair-enumeration block `_mrmr_fe_step.py` (~239-470).

### 17. Robust (heavy-tail-aware) warp FITTING (not just robust floors)
- **Mechanism.** Gate *thresholds* use median+MAD already, but the warp/basis *fits* (CMA-ES poly coeffs, ALS prewarp, orth-poly projection) are ordinary least-error → a few extreme rows dominate. Add Huber/Tukey loss in CMA-ES or fit on winsorized/rank-trimmed x, gated to fire only above a kurtosis/skew threshold (heavy-tail predicate already at `_orthogonal_univariate_fe.py:349,465`).
- **Why orthogonal.** Robustness of the *estimator that builds the feature*. MAD appears only in floor computation, never in the optimizer loss.
- **Cheap validation.** Inject 1-2% outliers into a monotone-signal column; robust fit recovers the true warp + better OOS vs plain (which chases outliers).
- **Feasibility.** Gate on the heavy-tail predicate (common case untouched); keep the non-robust fitter under a distinct name (keep-all-kernels rule); winsor bounds stored in the recipe. Integration: `polynom_pair_fe.py` objective + prewarp ALS.

### 18. Confidence-weighted MI under class imbalance
- **Mechanism.** Plug-in MI(candidate;y) is majority-dominated → a warp that only separates the rare class scores low + gets rejected. Add a class-balanced MI (inverse-prior reweight of the joint histogram, or stratified-balanced subsample) for the FE relevance score when class prior < threshold.
- **Why orthogonal.** Re-weights the *evidence* the gates consume, not the gate logic. FE scoring is currently unweighted plug-in MI (no class_weight/sample_weight reaches it).
- **Cheap validation.** Synthetic frame where a warp separates only a 1% rare class: balanced-MI admits it while plain MI rejects; clean-noise control still engineers ~0.
- **Feasibility.** Gate on `n_rare ≳ few hundred` (rare-imbalance-needs-large-n). Integration: `_mi_classif_batch` scoring helpers.

### 19. KSG / k-NN continuous-MI tie-breaker
- **Mechanism.** All FE-gate MIs are binned plug-in; a heavy-tailed engineered column can collapse to 2-3 bins and lose most of its MI (documented at `_feature_engineering_pairs.py:954-959`, 0.14 vs true 0.32). When the binned ratio narrowly fails (0.90-0.97 band), recompute numerator+denominator with KSG on **raw continuous** values and re-test. `sklearn.feature_selection.mutual_info_*` ships KSG (install-before-reimplement).
- **Cheap validation.** `mul(log(c),sin(d))`: binned MI ~0.14 vs KSG ~0.32; ratio passes under KSG where binned fails.
- **Feasibility.** O(n log n)/pair with a k-d tree → restrict to the margin band + top pairs. RISK: KSG noisy at small n → require agreement with binned on noise (don't *raise* the ratio for noise pairs).

### 20. Cross-fit recipe warm-start prior
- **Mechanism.** Within CV/`partial_fit` refits (folds 80%+ overlap), MRMR re-searches FE from scratch. Maintain a recipe-prior store keyed on the X-fingerprint: prior-surviving recipes seed the pool first + seed CMA-ES coefficients → fewer iters, stabler selection. Distinct from the identity-skip cache (no-FE case only) and `fe_warm_start` (within-pair CMA only).
- **Cheap validation.** 5 bootstrap subsamples; CMA iters-to-converge + wall-time cold vs warm.
- **Leak control.** A prior recipe must still pass *that fold's* gates on *that fold's* data — warm-start changes search order/init, never admission. Integration: pool-construction in `_mrmr_fe_step.py`, keyed on `_mrmr_compute_x_fingerprint`.

### 21. Gradient-interaction (mixed second partials) detector
- **Mechanism.** Fit one smooth differentiable surrogate (small MLP / kernel-ridge / RFF). Estimate `E[(∂²f/∂xₐ∂x_b)²]` per pair via finite differences/autograd on a row sample. Large mixed partial = non-additive interaction (the calculus definition). Catches **smooth/rotated** interactions (saddles, `sin(a)·b`) that axis-aligned trees represent poorly — complementary to #6.
- **Cheap validation.** n=2000, p=60, `y=sin(x5)·x31+noise`: surrogate `∂²f/∂x5∂x31` large → pair proposed; compare ranking vs the GBM seeder on the same data (complementarity).
- **Noise control.** Permutation null on the max mixed-partial; require surrogate OOF score > permuted-y baseline. Heaviest/most niche → last. Integration: new `_gradient_interaction_seeder.py`, same plug point as #6.

---

## Already implemented — do NOT re-build (agents verified)
- RankGauss / qrank (`_extra_fe_families.py` Family C, opt-in); target-mean/WoE/kfold-TE; cyclical/calendar (`fe_modular_enable` + `_periodic_fe`); distance-to-centroid/RBF/group-anomaly (`_group_distance_fe`, + cdist/local_lift/RFF/RSD-kNN shortlist); log-ratio (`fe_pairwise_log_ratio_enable`); sign×magnitude/digit (`_numeric_decompose_fe`); unconstrained spline + numeric_rounding.
- **Meta FE-flag recommender** (`_meta_fe_recommender.py` Layer 99 — fingerprint→fe_* flags via rule cold-start + ParamOracle).
- **UCB1 permutation-budget bandit** (`_cat_confirm_bandit.py` — but cat-pairs only; #16 extends to numeric).
- **Bootstrap stability selector** (`_stability_fe.py` Layer 36 — exists but opt-in, NOT default-wired; #15 wires a cheap replay-based version).
- **Provenance surface** (`_mrmr_fe_provenance.py` Layer 54 — survivors only; #3 adds the rejection side).
- **Higher-order cross-basis** (`_orthogonal_triplet_fe.py`/`_orthogonal_quadruplet_fe.py`/`_orthogonal_total_correlation_fe.py` — seeded by the blind univariate-MI top-N that #6/#10 replace).
- **Half-subsumed (bench before building):** sign-concordance pair (the `zᵢ·zⱼ` cross-basis term already covers the centered-product half).

---

## Architectural through-line & iron rules
- **Pipeline shape for the synergy ideas:** *proposer* (GBM #6 / RFF #9 / CMI-lattice #10 / gradient #21) → **II ranking #8** → existing FE search → **order-matched maxT floor (#7 for 3-way)**. Proposers shrink the candidate family from p²/p³ to K, which makes the maxT floor *less* conservative (tighter multiple-comparison correction) — so smart proposal both scales AND improves power, provided each proposer carries its own permuted-y self-gate.
- **Iron rule:** anything that LOWERS a bar (#1, #4, #5, #8) must keep the order-2 maxT floor as the outer guard and debias **both** sides of any ratio/comparison consistently (the floor at `_mrmr_fe_step.py:441,469` is on the uncorrected `batch_pair_mi_prange` scale).
- **MI-invariant operators (#14 isotonic; RankGauss):** must route through the non-MI-gated pool + downstream-lift test, never an MI-uplift gate (DPI makes them invariant — this is why supervised-knots died).
- **Dispatcher/defaults:** the fastest proposer for (n, p, HW, dtype) should be the routed default behind a dispatcher with thresholds in `kernel_tuning_cache`, not an opt-in flag.

---

## Recommended execution order (benchmark-first; ship what wins, document what doesn't)
1. **Tier 1: #1 → #4 → #2 → #3.** Cheap, direct session-evidence that they unblock real features; #3 makes every later experiment self-diagnosing. (#1 must co-update the maxT floor.)
2. **Synergy frontier: #8 → #6 → #7** (II ranking is near-free; GBM seeder is the scaling lever; order-3 floor is its mandatory rail). Then #5, #9, #10 as needed.
3. **Operators: #11 (hinge) → #12 (dispersion) → #13 (wavelet) → #14 (isotonic, non-MI path).**
4. **Process/robustness: #15 → #16,** then #17-#21 as targeted wins.

Each idea ships with the unit + biz_value + cProfile triad; every reverted attempt gets an in-code `bench-rejected (date): numbers, reason` note. Findings/benchmark scripts live in `D:/Temp` during exploration.

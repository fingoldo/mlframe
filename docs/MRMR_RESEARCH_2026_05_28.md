# MRMR research synthesis — 2026-05-28

Three parallel research agents produced findings for the next sprint of MRMR / feature-selection improvements. Below: clean markdown extracts of each agent's final report, with priorities + cross-references.

**Status update:** several of the items below have since shipped and are no longer "next sprint" — JMIM, RelaxMRMR (3-way redundancy), Chao-Shen and KSG MI estimators, CPT (conditional permutation test), PID (partial information decomposition), the per-feature adaptive bin-count chooser, and cluster-stability selection are all implemented. The remaining items stay as forward research.

This file consolidates the final reports for easier reading.

---

# Agent A — MRMR simple wins (14 findings, 2020-2025)

> Context: `mrmr.py` is 961 LOC, at split threshold from MEMORY.md — every new item must land in a new sibling file, never inside mrmr.py.

## A. Better scoring criteria (alternatives to Fleuret CMI)

**1. JMIM — Bennasar, Hua, Setchi 2015** ("Feature selection using Joint Mutual Information Maximisation", *Expert Systems with Applications* 42(22), ~1500+ citations). Replaces JMI's *sum* of pairwise joint MI with the *minimum*: `J_JMIM(X_k) = min_{X_j in S} I(X_k, X_j ; Y)`. Brown 2012's framework shows JMI/CMIM/DISR are all special cases of conditional likelihood maximisation; JMIM reduces relative classification error by ~6% vs the next best in the original paper.
- **Why us:** plugs directly alongside existing `fleuret.py`; the `min` aggregator is more robust than Fleuret's sum when one already-selected feature is strongly correlated.
- **LoC:** ~80, sibling `_jmim_scorer.py`. **Risk:** changes selected order → repro break behind opt-in `redundancy_criterion="jmim"`. **Priority: HIGH.**

**2. RelaxMRMR / FJMI — Vinh, Zhou, Chan, Bailey 2016** (*Pattern Recognition* 53, ~300 citations). Adds a 3-D MI redundancy term `I(X_k;X_j;X_i|Y)` — relaxes the conditional-independence assumption Fleuret uses.
- **Why us:** mlframe already computes 2-D joint histograms via `batch_pair_mi_gpu.py`; extending to a 3-D batched joint hist on GPU is incremental and would catch higher-order redundancy that single-pair Fleuret misses.
- **LoC:** ~250 (sibling `_relaxmrmr_3d.py` + a 3-D CUDA kernel under the existing dispatcher). **Risk:** memory blowup for 3-D hist; needs guard by `min(n_bins**3, threshold)`. **Priority: MEDIUM** (gain only when 3-way redundancy exists).

**3. MRwMR-BUR — Gao et al. 2022** (arXiv:2212.06143; cs.LG; cited ~30, preprint). Adds an explicit "unique relevance" term boosting features whose contribution *cannot* be explained by any already-selected feature. Reports +2-3% accuracy with 25-30% fewer features selected.
- **Why us:** complements SU normalisation directly; the BUR term is computed from quantities mlframe already has (per-feature `I(X;Y)` and joint `I(X,S;Y)` from Fleuret).
- **LoC:** ~120 (`_bur_term.py`). **Risk:** repro-break behind flag; preprint citation count borderline — verify after final submit. **Priority: MEDIUM-HIGH.**

**4. Quadratic MI / Cauchy-Schwarz (QMIFS) — Sluga & Lotrič 2017** (*Entropy* 19(4):157, ~150 citations). Plugin-free MI estimator via Gaussian kernels + Cauchy-Schwarz divergence; handles discrete and continuous uniformly without discretisation.
- **Why us:** mlframe currently *must* discretise via `discretization.py` even for continuous targets; QMIFS would let regression problems skip binning entirely.
- **LoC:** ~150 sibling `_qmi_cs.py`. **Risk:** new dep (none — pure numpy/numba), but O(n²) without Nyström approx; gate by n<20k. **Priority: MEDIUM.**

**5. Matrix-based Rényi α-entropy CMI — Yu, Giraldo, Jenssen, Príncipe 2020** (*IEEE TPAMI*, ~250 citations). Estimates joint/conditional entropy from eigen-spectrum of a kernel Gram matrix; no histogram, no plug-in bias, scales to multi-variate naturally.
- **Why us:** mlframe's current MI estimation chain (`info_theory.py` + Miller-Madow) is plug-in; α-entropy beats plug-in on small samples (n<500) which is where Besag-Clifford early-stop is least reliable.
- **LoC:** ~180 sibling `_renyi_alpha.py`. **Risk:** O(n²) Gram matrix; new dep on scipy.linalg.eigh (already in deps). **Priority: MEDIUM** (paired with finding #7 below).

## B. Better MI estimation for small samples

**6. KSG / Kraskov estimator (NeurIPS 2023 evaluation)** — Czyż et al., "Beyond Normal: On the Evaluation of MI Estimators". Confirms KSG has the *lowest sample requirements* among non-neural estimators on smooth continuous distributions.
- **Why us:** mlframe's MI is histogram-only (quantile/uniform binning); KSG removes binning bias on continuous features and is already shipped in sklearn (`feature_selection.mutual_info_regression`).
- **LoC:** ~60 (wrapper in `_ksg_estimator.py` calling sklearn). **Risk:** zero new dep (sklearn already in deps); ~5-20× slower per pair so only use for screened shortlist, not the full O(p²) pass. **Priority: HIGH** (cheap and orthogonal to the SU win).

**7. Bayesian-block / Knuth adaptive binning** — Scargle Bayesian Blocks (cited 1000+), Knuth 2019 rule. Used in astropy. Optimises bin edges to maximise a marginal likelihood, handles skewed/multimodal data where Freedman-Diaconis fails.
- **Why us:** mlframe's `discretization.py` ships only quantile + uniform; for heavy-tailed regression targets both produce singleton-mass bins that inflate MI estimates.
- **Status (2026-05-28):** ✅ Native Knuth + Bayesian Blocks shipped at `discretization.py:histogram()`.

## C. Better stopping criteria

**8. CMI-permutation + CMI-heuristic — Yu & Príncipe 2019** ("Simple stopping criteria for information theoretic feature selection", *Entropy* 21(1):99, ~80 citations). Stops when `I(X_candidate; Y | S_selected)` — estimated via matrix-based Rényi — is not significantly larger than a permutation null.
- **Why us:** mlframe already runs Besag-Clifford; this work fuses CMI estimation + permutation into one step, giving an automatic stop *without* a `threshold * H(y)` knob. Eliminates one hyperparameter.
- **LoC:** ~90 in `permutation.py` (sibling exists, just add `cmi_stop_mode='renyi'`). **Risk:** none if Rényi is opt-in. **Priority: HIGH.**

**9. Spectral Information Criterion / UAED elbow detection — Llorente, Martino et al. 2023** (arXiv:2308.09102 / *Signal Processing* 2024). Universal automatic elbow detector that generalises AIC/BIC for arbitrary error curves.
- **Why us:** mlframe's `min_features_fallback` is a hard floor; UAED would auto-pick subset size from the CMI-gain curve when the user provides `n_features=None`.
- **LoC:** ~70 in `stability.py`. **Risk:** none (pure numpy). **Priority: MEDIUM.**

## D. Better permutation test methodology

**10. Conditional Permutation Test (CPT) — Berrett, Wang, Barber, Samworth 2020** (*JRSS-B* 82(1), ~400 citations). Permutes X conditional on Z, preserving the X|Z distribution; gives valid p-values under arbitrary confounding.
- **Why us:** mlframe's Besag-Clifford permutes the candidate column unconditionally, which inflates Type-I error when the candidate is correlated with already-selected features. CPT is the principled fix — directly relevant to the redundancy-controlled selection MRMR is designed for.
- **LoC:** ~160 sibling `_conditional_permutation.py`. **Risk:** needs an MCMC-style permutation; compute cost +3-5× per test; gate to last-stage confirmation only. **Priority: HIGH** (methodologically *correct* test for MRMR's redundancy claim).

## E. Group / structured feature selection

**11. Cluster Stability Selection — Faletto & Bien 2022** (arXiv:2201.00494, *JMLR* 2024). Pre-clusters highly-correlated features then applies stability selection at the cluster level; gives selection-frequency error bounds (Shah-Samworth 2013).
- **Why us:** mlframe has `_cluster_aggregate.py` and `friend_graph.py` already — the infrastructure is there; cluster-level stability bounds would make the existing cluster scoring statistically rigorous.
- **LoC:** ~150 wiring in `stability.py`. **Risk:** none. **Priority: HIGH** — leverages existing mlframe primitives.

**12. Complementary Pairs Stability Selection — Shah & Samworth 2013** (*JRSS-B* 75(1), ~1500 citations). Run selection on B random half-splits + complements; derive a tight error bound on falsely selected features without exchangeability assumptions.
- **Why us:** mlframe's Besag-Clifford controls per-feature Type-I but not family-wise error across all candidates; Shah-Samworth gives a defensible FWER bound.
- **LoC:** ~100 in `stability.py`. **Risk:** doubles permutation budget. **Priority: MEDIUM.**

## F. Estimator-choice and unique-vs-synergistic decomposition

**13. Mutual Information Estimator empirical study — Pawluszek-Filipiak et al. 2025** (*MDPI Information* 16(9):724). Benchmarks Miller-Madow, Chao-Shen, Shrinkage, Jackknife on mRMR; concludes corrected estimators (esp. **Chao-Shen** and **Shrinkage / James-Stein**) outperform Miller-Madow on small/sparse contingency tables.
- **Why us:** mlframe currently exposes only Miller-Madow as the opt-in correction; Chao-Shen handles unseen-symbol bias that MM doesn't.
- **LoC:** ~80 added to `info_theory.py` (sibling `_chao_shen_entropy.py`). **Risk:** none, gated by `entropy_correction='chao_shen'`. **Priority: HIGH** (direct extension of the just-shipped SU work, same shape of win).

**14. PID-based redundancy/relevance — Wollstadt, Schmitt, Wibral 2023** ("A rigorous information-theoretic definition of redundancy and relevancy in feature selection based on (partial) information decomposition", *JMLR* 24(131):1-44, ~50 citations). Decomposes `I(X_set;Y)` into **unique** / **redundant** / **synergistic** components via Williams-Beer PID.
- **Why us:** mlframe's `cat_interactions` step finds synergistic pairs *after* the fact; PID provides a principled criterion to detect synergy *during* selection so synergistic features aren't filtered out.
- **LoC:** ~250 sibling `_pid_decomposition.py`. **Risk:** PID definition not unique (BROJA, Iccs, MMI variants); pick one and document. **Priority: MEDIUM** (high theoretical value, larger effort).

## Recommended next steps (cheap-now ordering)

1. **Finding #13 (Chao-Shen entropy)** — same impact shape as the SU win just shipped; ~80 LOC.
2. **Finding #1 (JMIM)** — drop-in alternative aggregator to Fleuret; ~80 LOC.
3. **Finding #6 (KSG wrapper)** — zero new deps, sklearn already present; ~60 LOC.
4. **Finding #8 (CMI-permutation stop)** — eliminates one hyperparameter; ~90 LOC.
5. **Finding #11 (Cluster Stability Selection)** — leverages existing mlframe cluster infra.

All can land as sibling files under `feature_selection/filters/` per the monolith-split rule.

## Sources

- [JMIM — Bennasar, Hua, Setchi 2015](https://www.sciencedirect.com/science/article/pii/S0957417415004674)
- [Brown et al. 2012 framework / praznik](https://rdrr.io/cran/praznik/man/JMI.html)
- [MRwMR-BUR (arXiv 2212.06143)](https://arxiv.org/abs/2212.06143)
- [mRMR estimator choice study (MDPI 2025)](https://www.mdpi.com/2078-2489/16/9/724)
- [Kraskov KSG evaluation NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/36b80eae70ff629d667f210e13497edf-Paper-Conference.pdf)
- [Copula entropy feature selection (Ma)](https://arxiv.org/abs/0808.0845)
- [MINE neural MI FS](https://arxiv.org/abs/2510.02610)
- [Simple stopping criteria — Yu & Príncipe 2019](https://www.mdpi.com/1099-4300/21/1/99)
- [Spectral information criterion elbow](https://arxiv.org/pdf/2308.09108)
- [Conditional Permutation Test — Berrett 2020](https://academic.oup.com/jrsssb/article/82/1/175/7056014)
- [Cluster Stability Selection — Faletto & Bien](https://arxiv.org/pdf/2201.00494)
- [Shah & Samworth Complementary Pairs SS](https://www.statslab.cam.ac.uk/~rjs57/rssb_1034.pdf)
- [RelaxMRMR — Vinh et al. 2016](https://shuozhou.github.io/papers/vinh15pr.pdf)
- [Matrix-based Rényi α-entropy — Yu et al. 2020](https://arxiv.org/pdf/1808.07912)
- [Quadratic MI Feature Selection — Sluga & Lotrič 2017](https://www.mdpi.com/1099-4300/19/4/157)
- [PID feature selection — Wollstadt et al. 2023](https://arxiv.org/pdf/2105.04187)
- [Uber MRMR variants (Zhao et al. 2019)](https://arxiv.org/pdf/1908.05376)

---

# Agent B — Auto-clustering critique

## Current state

The MRMR redundancy logic is split across **four independent layers** with **non-overlapping cluster semantics**, which is the headline structural finding:

1. **Greedy Fleuret per-Z redundancy** (`filters/evaluation.py:230-339`, `filters/fleuret.py:184-257`). For each candidate `X`, gain is `min over Z in selected_vars` of `I(X; Y | Z)`. This is the canonical Fleuret 2004 / **Brown 2012 CMIM** criterion. There is **no clustering here at all** — the redundancy lookup is one-feature-at-a-time. A genuine collinear cluster `{Z1, Z2, Z3}` triggers three identical vetoes (no extra suppression beyond the min).

2. **Friend graph post-hoc diagnosis** (`filters/friend_graph.py`, 480 LOC). Built **after** the greedy loop on the already-selected set. Edge gate at `pairwise_mi_edge:165-188`: pairwise MI must clear `max(mi_eps=1e-6, edge_significance=3.0 * (na-1)(nb-1)/(2n))` (a G-test bias floor). Node classification at lines 332-353. **No connected-component analysis, no transitive closure, no graph partitioning**. `prune_by_friend_graph:368-409` is worst-first node removal protecting the single "justifier" neighbor — it is not a cluster operation.

3. **Cluster-aggregate FE step** (`filters/_cluster_aggregate.py`, 315 LOC). The **only place** that actually clusters: `_discover_clusters:143-222` builds an edge list gated by `|Pearson corr| >= corr_threshold=0.6` AND `pairwise_mi_edge`, then runs naive union-find connected components (`_connected_components:123-140`). A component is accepted only if PC1 explains >= `homogeneity_tau=0.6` of standardized variance. Members are aggregated (`mean_z` / PCA-PC1 / Bartlett) and the aggregate is **augmented onto the selected set**, not used to deduplicate it.

4. **Operator-supplied `feature_groups` in RFECV** (`wrappers/_rfecv.py:237-499`). All-or-nothing post-fit support expansion over user-declared **disjoint** groups. Completely orthogonal to MRMR; there is **no data-driven equivalent inside MRMR**.

The friend graph and cluster-aggregate paths share the `pairwise_mi_edge` primitive but **use different thresholds and different decision criteria** for the same underlying question "are these two features redundant?".

## Critical gaps

- **The Fleuret per-Z `min I(X;Y|Z)` criterion is provably brittle on multi-collinear groups.** When `{Z1, Z2, Z3}` are noisy reflections of one latent `z`, each gives `I(X;Y|Z_k) ≈ I(X;Y|z)`, so `min_k` underestimates the true conditional MI exactly when there is **most** redundancy to clean up. Brown et al. 2012 ([JMLR](https://jmlr.org/papers/v13/brown12a.html)) show CMIM (`min_k`) is dominated by **JMI** (`sum_k I(X, Z_k; Y)`) on small samples. The Fleuret docstring (`filters/fleuret.py:4-7`) even acknowledges this as TODO; still unfixed.

- **Two incompatible redundancy thresholds for the same edge.** Friend graph uses an MI **significance** floor only. Cluster-aggregate requires **both** `|corr| >= 0.6` and the same MI floor. A non-linear functional dependency `Y=f(X)` with `corr ~ 0` passes friend-graph but is **rejected** by cluster-aggregate. The "shared primitive" claim hides a semantic split.

- **No transitive / hierarchical cluster structure anywhere.** Cluster-aggregate uses single-linkage connected components on a hard-thresholded graph. Single-linkage on continuous similarity is known to "chain" — one weak edge merges two unrelated cliques. Sotoca & Pla 2010 and FCBF (Yu & Liu 2003) both use **rank-then-prune** or **agglomerative complete-linkage** on a CMI-based distance precisely to avoid this.

- **Threshold values are dev-machine constants.** `corr_threshold=0.6`, `homogeneity_tau=0.6`, `garbage_min_degree=3`, `unique_ratio=1.0` are hardcoded defaults with no data-driven calibration. Per the project memory's `kernel_tuning_cache` rule, these should route through a calibration cache.

- **No protection against synergy destruction.** The Fleuret `min` criterion rejects synergistic pairs (acknowledged in `fleuret.py:4-7`). Brown 2012 reports JMI / JMIM keeps synergies because they sum the joint `I(X,Z;Y)` rather than minimize a conditional.

- **Friend-graph "sink" detection scales O(k²) and skips when k > 200**. For typical MRMR runs that select 100-300 features this is just barely on — it silently degrades to "node stats only" with no clustering output. There is no fallback to a sparse k-NN graph (Roffo 2017 ICCV recommends keeping only top-k edges per node).

- **No graph-Laplacian / eigenvector-centrality global ranking.** Roffo's **Inf-FS** ranks features by their **infinite-path importance** on the feature-feature affinity graph — exactly the global structure the current four-layer pipeline never computes.

## Recommendations

- **Add JMI / JMIM as a `mrmr_redundancy_algo` option** (Brown 2012; Bennasar et al. 2015). Replace `min_k I(X;Y|Z_k)` with `min_k I(X;Y|Z_k) + sum_k I(X;Z_k;Y)` (interaction-information term). ~80 LOC. **Bench:** synthetic XOR + 10 collinear noise reflections of one informative feature; current Fleuret will reject the second XOR component; JMI keeps it.

- **Replace single-linkage components with FCBF-style ordered pruning** (Yu & Liu 2003). Sort by relevance then drop any feature whose `SU(Xi, Xj) > SU(Xj, Y)` — this is **chaining-immune** without requiring the `homogeneity_tau` PC1 gate. ~40 LOC swap of `_connected_components`.

- **Add Sotoca-Pla hierarchical CMI clustering as an alternative discovery backend** (Sotoca & Pla 2010 *Pattern Recognition* 43(6):2068-2081). Agglomerative complete-linkage on `d(X_i, X_j) = 2*H(X_i,X_j) - I(X_i;X_j) - I(X_i;Y) - I(X_j;Y)`; cut the dendrogram at a CV-tuned height. ~150 LOC.

- **Add Inf-FS eigenvector-centrality re-ranking pass on the friend graph** (Roffo et al. 2017 ICCV). After `build_friend_graph`, compute top eigenvector of the MI-weighted adjacency; features with centrality below percentile 5 are de-prioritized. ~60 LOC.

- **Route all four thresholds (`corr_threshold`, `homogeneity_tau`, `edge_significance`, `garbage_min_degree`) through `pyutilz.system.kernel_tuning_cache`** keyed on (n_samples, n_features, mean abs corr of pool). ~50 LOC.

## Sources

- [Brown et al. 2012 Conditional Likelihood Maximisation (JMLR v13)](https://jmlr.org/papers/v13/brown12a.html)
- [Bennasar et al. 2015 JMIM feature selection (Cardiff)](https://orca.cardiff.ac.uk/id/eprint/76215/7/1-s2.0-S0957417415004674-main.pdf)
- [Roffo et al. Infinite Feature Selection graph-based (arXiv 2006.08184)](https://arxiv.org/pdf/2006.08184)
- [Yu & Liu 2003 FCBF predominant-correlation pruning](http://bioconductor.jp/packages/3.16/bioc/vignettes/FCBF/inst/doc/FCBF-Vignette.html)
- [Sotoca & Pla 2010 Pattern Recognition 43(6):2068-2081](https://www.sciencedirect.com/science/article/abs/pii/S0031320309004828)
- [Review of grouping-based FS approaches (NCBI PMC10358338)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10358338/)
- [FEAST FS toolbox (Brown 2012 reference implementation)](https://github.com/Craigacp/FEAST)

---

# Agent C — FS competition map (6 methods × 15 relationship types)

## Per-method mathematical view + strengths/weaknesses

### A. MRMR (Fleuret variant) — `filters/mrmr.py`, `filters/fleuret.py`, `filters/evaluation.py`
**Math.** Greedy forward selector on discrete (binned) `X`, `y`. Per candidate computes `gain = I(X;Y) - max_{k in S} I(X;Y | S_k)` (Fleuret/CMIM formulation), validated via permutation null. Assumes informative interactions are at most pairwise with already-selected variables (`max_veteranes_interactions_order` defaults to 1) and that MI estimated on binned data approximates continuous MI well.

**Strengths.**
1. Captures monotone-or-not nonlinear marginals (MI is invariant to monotone transforms)
2. Removes pure redundancy (a perfect duplicate gets `gain≈0`)
3. Handles multiclass `y` natively
4. Permutation-confidence step gives p-value-like stopping
5. SU-normalized variant balances entropy of high-cardinality predictors

**Weaknesses.**
1. Explicitly rejects pure synergy / XOR — file docstring already names this
2. Binning destroys very-fine threshold signal at default `nbins`
3. `max_k` is sensitive — one already-selected near-duplicate kills the candidate (CMIM pessimism, see Brown 2012)
4. Group-additive cluster effects (need majority vote across cluster) get suppressed once one cluster member is in
5. Pure linear-Gaussian signal in high-D gets out-ranked by lower-MI but discrete signals because of binning bias

### B. RFECV wrapper — `wrappers/_rfecv.py`
**Math.** Model-in-the-loop backward (or MBH heuristic-search) elimination, ranks features by an importance signal from the trained estimator (gain / coef / permutation), CV-votes across folds, picks subset size by averaged CV score. Assumes model importance is monotone in true relevance.

**Strengths.**
1. Native to whatever model will ship (tree gain ≈ what CatBoost actually uses)
2. Picks up multivariate signals the model can fit (tree-friendly thresholds, sigmoidal, additive)
3. CV votes give noise robustness
4. MBH supports non-greedy steps so it can escape one-feature-at-a-time traps
5. `must_include` lets domain priors carry through

**Weaknesses.**
1. Splits importance across correlated copies — both look unimportant individually, both eliminated (NCBI RF-RFE study)
2. Unstable across folds when many correlated features (sklearn docs)
3. Inherits the base model's blind spots (linear model can't catch XOR; tree can't catch smooth high-order interactions efficiently)
4. Cost = `O(p · CV · model_fit)`
5. No FDR guarantee

### C. ShapProxiedFS — `shap_proxied_fs.py`
**Math.** Fit one big model on all features, compute OOF SHAP `phi_ij`. Approximate the score of subset `S` by `base + sum_{j∈S} phi_ij` (Shapley additive coalition proxy à la Mazzanti). Optimize subset via brute force / beam / GA / annealing, then honestly re-validate the top-N on a held-out split.

**Strengths.**
1. Searches in subset space, not feature space — natively handles compensation (drop A, keep B that captures same signal)
2. Cheap because no retrain per subset
3. Disjoint re-validation guards against proxy over-fit
4. Supports interaction-aware mode via `treeshap_interactions`
5. Active learning / uncertainty penalty for proxy quality

**Weaknesses.**
1. Additive coalition assumption breaks under strong interactions — SHAP linearity axiom is exactly the failure mode (Kumar 2020)
2. Proxy mis-credits features whose correlated survivors would compensate (`<50% coverage wall`, called out in docstring)
3. Requires a tree model that supports tree-SHAP
4. High-dim `n_features > 22` falls back to heuristic search — no exactness guarantee
5. Useless on data where SHAP itself misranks (e.g. categorical with many rare levels under TreeSHAP path-dependent attribution)

### D. BorutaShap — `boruta_shap.py`
**Math.** Permute each feature to make "shadow" features, fit RF/GBM, retain originals whose importance is statistically greater (binomial-test over rounds) than the max shadow. Importance can be Gini or mean-|SHAP|. Assumes "max-shadow" is a valid null for a relevant feature.

**Strengths.**
1. "All-relevant" set, not "minimal" — keeps backups for downstream stability
2. Captures nonlinear + tree-fittable interactions because the underlying RF/GBM does
3. FWER-style control via binomial p-values + BH/BY
4. Robust to monotone transforms
5. SHAP mode reduces bias toward high-cardinality features that plagues Gini

**Weaknesses.**
1. Keeps both members of a redundant pair — kxy.ai: "considered both highly correlated variables important"
2. Blind to XOR if RF depth/n_trees insufficient
3. Shadow-max null is loose at high `p` → low power
4. Cost = many RF refits
5. No subset-quality metric — only per-feature accept/reject

### E. Knockoffs (Barber-Candès) — `wrappers/_knockoffs.py`
**Math.** Build `X̃` with same `corr(X̃_j, X_{-j}) = corr(X_j, X_{-j})` but `X̃ ⫫ y | X`. For any importance `Z_j, Z̃_j`, statistic `W_j = Z_j - Z̃_j` has IID-sign null. Threshold gives finite-sample FDR ≤ q. mlframe uses equicorrelated fixed-design Gaussian knockoffs.

**Strengths.**
1. Provable finite-sample FDR — unique in this lineup
2. Couples with any importance W (gain, coef, SHAP)
3. No multiple-testing pessimism vs BY-FDR
4. Detects features whose marginal MI is near zero but joint signal is strong
5. Works for any `p < n` (fixed-design path)

**Weaknesses.**
1. Equicorrelated construction only valid for ~Gaussian X — RANK/JASA 2020 explicitly addresses this
2. Degrades severely under heavy multicollinearity (`lambda_min(Σ) → 0 → s → 0 → W → 0`, no power; module raises)
3. Requires numeric X
4. Power depends entirely on the importance statistic — bad W kills FDR power
5. `n < p` requires model-X variant (not yet wired here)

### F. Univariate HT prescreen — `wrappers/_univariate_ht.py`
**Math.** Per-feature test: Mann-Whitney U (binary y, numeric x), Kruskal-Wallis (multiclass y, numeric x), Kendall τ (continuous y, numeric x), χ² (categorical x, discrete y). BY-FDR across `p` tests. Assumes the marginal `(X_j, y)` distribution carries the signal.

**Strengths.**
1. Cheap — `O(n p log n)`
2. Distribution-free (rank-based)
3. Correct FDR control under arbitrary dependency (BY)
4. Strong on linear-monotone and unimodal-class-separable signals
5. Great as a *prescreen* upstream of expensive methods

**Weaknesses.**
1. Completely blind to XOR / pure interaction (univariate)
2. Blind to V-shaped / non-monotone signals (Kendall = 0)
3. Cannot rank redundant features against each other
4. Misses sigmoid/threshold where median split doesn't separate
5. No coverage of group-additive effects where only the SUM is informative

## Competition matrix

W = wins (rank-1) / T = ties (detects but doesn't dominate) / L = misses

| # | Relationship type | MRMR | RFECV | ShapProxied | Boruta | Knockoffs | UnivHT |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | Linear additive marginal, continuous y | T | W | W | W | W | W |
| 2 | Nonlinear monotone marginal (log/sqrt) | W | W | W | W | T | W |
| 3 | Non-monotone unimodal (V/U-shape) | W | W | T | W | T | L |
| 4 | Threshold step / piecewise constant | W | W | T | W | T | T |
| 5 | Sigmoid / tanh marginal | T | W | W | W | W | W |
| 6 | 2-way XOR (pure synergy, no marginal) | L | T | T | T | L | L |
| 7 | 3-way XOR / parity | L | L | L | L | L | L |
| 8 | Multiplicative interaction (`x1·x2`) | T | W | T | W | T | T |
| 9 | Two perfect duplicates (redundancy) | W | L | W | L | L | L |
| 10 | Block of 5 correlated copies, one signal | W | L | W | L | L | L |
| 11 | Group-additive (sum of 10 weak features) | L | W | W | W | W | T |
| 12 | High-cardinality categorical, target-mean signal | T | W | W | T | L | T |
| 13 | Heavy-tail/skew predictor, linear y | T | T | T | T | L | W |
| 14 | Many noise + 1 strong signal (n_inf=1, p=1000) | W | T | W | W | W | W |
| 15 | Mixed-relevance under known FDR budget | T | L | T | T | W | T |

## Synthetic benchmark scenarios

Each discriminates ≥ 2 methods orthogonally. Pseudocode in original report; condensed here:

| # | Name | n / p / n_inf | y formula | Discriminates |
|---|---|---|---|---|
| 1 | `bench_dataset_xor_2way` | 8000 / 50 / 2 | `y = x1 XOR x2` | RFECV/Boruta/Shap vs MRMR/Univ/Knockoffs |
| 2 | `bench_dataset_xor_3way_parity` | 20000 / 100 / 3 | `y = x1⊕x2⊕x3` | All fail except RFECV+deep |
| 3 | `bench_dataset_threshold_step` | 5000 / 200 / 1 | `y = 1[x_5 > 1.3]` | UnivHT loses (weak τ) |
| 4 | `bench_dataset_v_shape` | 5000 / 200 / 1 | `y = x_5²` | UnivHT loses (τ=0) |
| 5 | `bench_dataset_redundant_duplicates` | 4000 / 50 / 1+5dup | `y = x_1; x_2..x_6 = x_1` | MRMR/Shap win, Boruta/RFECV lose |
| 6 | `bench_dataset_correlated_block` | 4000 / 50 / 1+10corr | `y = x_1; block corr=0.95` | Knockoffs fails (Σ near-singular) |
| 7 | `bench_dataset_group_additive` | 8000 / 200 / 10 | `y = sum(x_1..x_10)` | MRMR loses (per-marginal too weak) |
| 8 | `bench_dataset_sigmoid_marginal` | 5000 / 100 / 3 | `y ~ Bernoulli(σ(...))` | Calibration check |
| 9 | `bench_dataset_high_card_cat` | 10000 / 30 / 1 | `cat 200 levels; y = target_mean[cat]` | Knockoffs fails (numeric only) |
| 10 | `bench_dataset_heavy_tail` | 4000 / 100 / 3 | `x ~ t(2); y = sum(x)` | Knockoffs fails (Gaussian assumed) |
| 11 | `bench_dataset_fdr_under_budget` | 2000 / 500 / 20 | Linear with FDR target | Only Knockoffs guarantees |
| 12 | `bench_dataset_p_gg_n_sparse` | 300 / 5000 / 5 | Linear, sparse | Knockoffs (model-X) wins |
| 13 | `bench_dataset_compensable_pair` | 5000 / 100 / 2 | `y=x_1+x_2; corr(x_1,x_3)=0.9` | ShapProxied uniquely best |
| 14 | `bench_dataset_pure_noise` | 2000 / 200 / 0 | Random y | Type-I check |
| 15 | `bench_dataset_mixed_realistic` | 10000 / 300 / 20 mixed | Combined types | Profile-level |

## Sources

- [Brown 2012 — "A Unifying Framework for Information Theoretic Feature Selection"](https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf)
- [Fleuret 2004 — "Fast Binary Feature Selection with Conditional Mutual Information"](https://www.researchgate.net/publication/220320537)
- [Candès et al. 2018 — "Panning for Gold: Model-X Knockoffs"](https://lucasjanson.fas.harvard.edu/papers/Panning_For_Gold_Model_X_Knockoffs_For_High_Dimensional_Controlled_Variable_Selection-Candes_ea-2017.pdf)
- [Fan et al. 2020 JASA — RANK](http://faculty.marshall.usc.edu/yingying-fan/publications/JASA-FDLL20.pdf)
- [Kumar et al. 2020 — "Shapley values for feature selection: the good, the bad, and the axioms"](https://arxiv.org/pdf/2102.10936)
- [Darbellay 2023 (kxy.ai) — "BorutaShap Does Not Work For The Reason You Think"](https://blog.kxy.ai/boruta-shap-is-as-bad-as-rfe/index.html)
- [Gregorutti et al. 2018 NCBI — "Using RFE in random forest with correlated variables"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6157185/)
- ["Feature Selection: A perspective on inter-attribute cooperation" (XOR cooperating features)](https://arxiv.org/pdf/2306.16559)

---

# Cross-agent prioritised backlog

Items ranked by cost/benefit, combining all 3 agents.

| Pri | Item | Source | LOC | Expected biz value |
|---|---|---|---|---|
| 🔴 P0 | **JMIM redundancy** as `mrmr_redundancy_algo='jmim'` | Agent A #1 + Agent B critic | ~80 | Fixes Fleuret-CMIM brittleness on multi-collinear (acknowledged TODO) |
| 🔴 P0 | **Chao-Shen entropy** correction | Agent A #13 | ~80 | Same impact-shape as SU; sparse-contingency bias fix |
| 🔴 P0 | **FCBF ordered pruning** in `_cluster_aggregate.py` | Agent B #2 | ~40 | Replaces chaining-prone single-linkage |
| 🟡 P1 | **Adaptive nbins** (Sturges / Freedman-Diaconis / Knuth / Bayesian Blocks / Fayyad-Irani / OptimalJoint) | User Q1 + Agent A #7 (partial done) | ~250 | Forces SU-on as default; eliminates fixed-10-bin compromise |
| 🟡 P1 | **15 synthetic bench scenarios** from Agent C | Agent C | ~600 | Foundation for auto-tune (DataFingerprint → best method) |
| 🟡 P1 | **kernel_tuning_cache for cluster thresholds** | Agent B #5 | ~50 | Per memory rule; eliminates hardcoded magic |
| 🟢 P2 | **Conditional Permutation Test** | Agent A #10 (Berrett 2020) | ~160 | Methodologically correct test for MRMR's redundancy |
| 🟢 P2 | **Inf-FS centrality re-rank** | Agent B #4 | ~60 | Global structural score replaces local degree heuristic |
| 🟢 P2 | **KSG estimator wrapper** | Agent A #6 | ~60 | Bypasses binning for continuous-y |
| 🟢 P2 | **Cluster Stability Selection** | Agent A #11 | ~150 | Leverages existing mlframe cluster infra |
| 🟢 P2 | **Sotoca-Pla hierarchical CMI clustering** | Agent B #3 | ~150 | Alternative discovery backend, complete-linkage |
| 🟢 P3 | **RelaxMRMR (3D MI)** | Agent A #2 | ~250 | 3-way redundancy detection on GPU |
| 🟢 P3 | **PID-based redundancy/relevance** | Agent A #14 | ~250 | Explicit synergy detection |
| 🟢 P3 | **MRwMR-BUR** | Agent A #3 | ~120 | Boost features with unique signal |

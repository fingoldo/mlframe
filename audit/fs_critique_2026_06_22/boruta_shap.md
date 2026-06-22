# BorutaShap audit + optimization (2026-06-22)

Scope: `src/mlframe/feature_selection/boruta_shap/` (`_auto_dispatch.py`, `_fit_explain.py`, `_io_plot.py`, `_shadow_stats.py`, `__init__.py`) + the `BorutaShap` registration in `registry.py`.

Interpreter: Windows-store CPython 3.14.3 (no anaconda on this host, per memory). shap 0.52.0.

## 1. cProfile baseline

Harness: gini driver, n=1500, p=40, n_trials=40, 3 fits. Total 43.9s.

| frame | cumtime | share | owner |
|---|---|---|---|
| `sklearn.tree._classes._fit` | 35.9s | 82% | third-party (intrinsic per-trial tree fit) |
| `_forest.fit` (33 calls = 11/fit) | 42.3s cum | - | RF refit every trial |
| mlframe per-trial helpers (shadow build, hits, binomial, history) | <0.3s total | <1% | ours |

Verdict: for the gini driver the wall is ~82% the RandomForest tree fit and almost all of the rest is sklearn forest plumbing (clone/get_params/check_random_state). The mlframe-side per-trial code is already thin (prior optimization waves: single-call drop, homogeneous-numeric shadow fast path, cached binomtest). The one actionable speedup is that the **default** RandomForest was constructed with `n_jobs=None` (sequential) — see Opt-1.

## 2. Findings

| # | file:line | severity | issue | status |
|---|---|---|---|---|
| B1 | `_shadow_stats.py:251-259` (`get_5_percent` / `get_5_percent_splits`) | High (crash) | `sample=True` with `n<=18`: `round(0.05*n)==0` -> `np.arange(start, n, 0)` raises `ZeroDivisionError`, killing the whole fit before any trial. | FIXED |
| B2 | `__init__.py:323,325` | Med (perf) | Default `RandomForestClassifier()/Regressor()` left `n_jobs=None` -> the dominant per-trial tree fit (82% of wall) runs single-threaded. The default model is ours to configure. | FIXED (Opt-1) |
| B3 | `_shadow_stats.py:343,345` (`test_features`) | Med (correctness) | Bonferroni base `n_tests=len(self.columns)` uses the *shrinking* current column count; as rejected features are dropped the correction weakens trial-over-trial (leniency drift). `self.hits`/`all_columns` are full-length and the shipped margin-gate already uses `len(self.all_columns)` as "the validated Bonferroni base" -> `test_features` is inconsistent with it. | FIXED |
| B4 | `_shadow_stats.py:338-360` | Low | `test_features` keeps re-testing already-removed features (`self.hits` is full length, indexed by `all_columns`). Harmless to the final partition (removed cols aren't in X) but wastes a few binomtests/trial and is the same root as B3. | DOC (subsumed by B3 base fix; cost is negligible after the binomtest cache) |
| B5 | `__init__.py:321-325` + `_shadow_stats.feature_importance` | Low | Single-class target: RF fits, importances ~0, gate accepts nothing -> empty `accepted` with no signal to the caller. Defensible, but silent. | DOC |
| B6 | `__init__.py:668-669` (`TentativeRoughFix`) | Low | Uses `print(...)` of a numpy array (not `logger`); a non-ASCII feature name would crash under cp1251 stdout, and it bypasses the verbose flag. `TentativeRoughFix` is not called from `fit` (on-demand only), so blast radius is small. | FIXED (logger) |
| B7 | `_shadow_stats.create_shadow_features` | Low | `fit` docstring step 1 claims the system is "always extended by at least 5 shadow attributes even if the number of attributes is lower than 5"; the implementation makes exactly one shadow per real column (no >=5 pad). On 1-2 feature inputs the shadow null is thin. | DOC (changing the shadow count is selection-altering; needs a bench before any default flip, out of scope here) |

## 3. Optimizations

| id | change | measurement | identity |
|---|---|---|---|
| Opt-1 | default RF `n_jobs=-1` (B2) | tree fit is 82% of wall and was sequential; `n_jobs=-1` parallelizes it across cores. Speedup is hardware-core-count proportional and applies to every default-model fit. Caller-supplied models are untouched. | bit-identical selection (RF with fixed `random_state` is deterministic regardless of `n_jobs`; trees are independent) |

No mlframe-side hot path exceeded the >=0.5% bar after the prior waves (per-trial helpers <1% combined), so no kernel rewrites were warranted. cProfile harness retained at `audit/fs_critique_2026_06_22/` notes.

## 4. Strengths / weaknesses

Strengths
- All-relevant selection: keeps features whose importance beats the shadow null, so it retains interaction/redundant partners that a marginal filter (plain MI) would drop. Good when downstream model exploits redundancy.
- Wrapper fidelity: importance comes from the actual surrogate (or caller's model), so it reflects the model family that will consume the features.
- Robust to many irrelevant features: the shadow gate + binomial accumulation over trials suppresses pure noise reasonably well, and the held-out `permutation` driver drives accepted-noise to ~0.
- Good engineering already present: nanpercentile gate, RNG isolation, scoped warnings, polars view, early-decided stop, margin-gated tentative stop, cross-subsample stability mode.

Weaknesses
- Runtime: O(n_trials x model-fit); a SHAP driver recomputes TreeSHAP per trial (~137x gini). Scales poorly with n_trials and wide frames. Mitigations shipped (early stop, gini default).
- Correlated features: independent-permutation shadows destroy joint structure, so a real noise column retains finite-sample covariance and can clear the gate (documented single-draw false-positive limit). `premerge_clusters` + INTERSECTION stability are the levers; the registry already wraps it in cluster-medoid reduction.
- Stability across seeds: single-fit acceptance of borderline columns is seed-sensitive; only INTERSECTION-mode stability (>=8-10 subsamples) is reliable, at a recall cost.
- Thin shadow null on <=2 features (B7); no >=5 pad despite the docstring.

Improvement ideas (future, not applied)
- Implement the canonical >=5 shadow pad (recycle real columns) so the null is well-populated on narrow frames; bench selection delta first (B7).
- Optionally emit a warning when `accepted` is empty AND the target is single-class / all-importances-zero (B5).
- Consider routing the default-model trial fits through warm-start / smaller per-trial forests when n_trials is large.

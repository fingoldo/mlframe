# RFECV / SFFS / Stability-Selection wrapper family — critique (2026-06-22)

Scope: `src/mlframe/feature_selection/wrappers/rfecv/*` + registry RFECV registration.
MRMR / worktrees / other selector families explicitly excluded.

## 1. Findings table

| # | File:line | Severity | Issue | Fix | Status |
|---|-----------|----------|-------|-----|--------|
| F1 | `_stability_select.py:200-208` | P2 | Per-bootstrap top-K always takes exactly `top_k` indices via `lexsort`, including features whose summed FI is 0. When fewer than `top_k` features have positive importance, pure-noise (zero-FI) columns accrue selection counts every bootstrap and can cross `stability_threshold`, admitting noise into `support_`. | Restrict the per-bootstrap top-K to features with strictly positive score before counting. | FIXED |
| F2 | `_finalize.py:97-101` | P2 | After the SFFS swap pass, `new_best_nf = max(evaluated_scores_mean, key=...get)` is unreliable when any N has a NaN mean (`max` with NaN comparands returns an arbitrary key). A genuinely-better swapped subset can be masked by a NaN entry, silently discarding the swap improvement. | Pick the argmax over finite means only. | FIXED |
| F3 | `_diagnostics.py:301-302` | Low | `n_features_bootstrap_ci_` returns `(low, n_features_, high)` where the middle is `self.n_features_`, which can fall OUTSIDE `[low, high]` (the picker used a different rule than bootstrap-argmax). Misleading "CI" where the point estimate is not bracketed. | Clamp the reported point estimate into `[low, high]`. | FIXED |
| F4 | `_fit.py:533-535` | Low | Verbose-only "gain vs dummy / gain vs all features" uses `base_perf[0]` / `base_perf[-1]` as if they were N=0 / N=full, but they are the smallest / largest *evaluated* N (positional). When N=0 was not evaluated or full set not evaluated, the printed gain is wrong. Cosmetic (log only). | Documented as Low; no behaviour change. | DOC |
| F5 | `_nan_policy.py:107` | Low | ndarray NaN detection `_arr.dtype.kind in "fc"` skips integer arrays — correct (ints can't hold NaN) but object-dtype ndarrays with embedded `float('nan')` are not scanned. Edge case; object ndarray X is already discouraged. | Documented; no change (object X path is unsupported by design). | DOC |
| F6 | `_multioutput.py:42-46` | Low | Per-column sub-fit clones `self` and recurses with `multioutput_strategy=None`; `sample_weight` / `groups` forwarded correctly. No bug, but a single all-constant output column would raise inside the sub-fit and abort ALL columns (no per-column resilience). | Documented as a known limitation (fail-fast is acceptable; a constant target is a real config error). | DOC |
| F7 | `_finalize.py:179` | Low | `n_features_` recomputed via `np.sum(support_)` for bool masks and `len(support_)` for index arrays — correct, but the index-array branch counts duplicates if `must_include` glue produced repeats. Glue dedups via `i not in idx_must`, so safe. | Verified safe; no change. | VERIFIED |

## 2. Strengths / weaknesses verdict

RFECV (MBH-driven backward/curve search)
- Strong: rich CV-curve with selection rules (auto=one_se_max, argmax, plateau, knee/pareto diagnostics); honest val/OOF separation; deep edge-case hardening (single-class, minority<cv, NaN/Inf y, dup columns, leakage scan, ID-like/near-dup column drops); group-aware + time-series CV auto-detect; checkpoint/resume; numpy fast-path for all-numeric frames.
- Weak: cost is O(p · iters · cv · estimator-fit); on p≫n the search collapses to {0, full} and leans on the FP-control cap (a heuristic, documented). Permutation-FI is O(p·n_repeats)/fold — mitigated by the wide-data fallback but that trades debiasing for a usable curve.
- Best when: moderate p (≤ few hundred), real signal, redundancy already clustered upstream. Poor when: p≫n pure-noise (only the cap protects you) or extreme p without `max_nfeatures`.

SFFS swap pass
- Strong: cheap final-mile local improvement (K paired swaps); deterministic tiebreak; correctly skipped when val-CV early-stopping would make swap scores incomparable.
- Weak: uses bare `cross_val_score` (ignores fit_params / val_cv / ES); F2 masked legit improvements under NaN folds (now fixed).
- Best when: small K, no ES; marginal but safe.

Stability selection (Meinshausen–Bühlmann)
- Strong: robust on small-n / high-p where CV voting is noise-dominated; provable error control; deterministic tiebreak; n≥20 floor.
- Weak: F1 admitted zero-FI noise into top-K (now fixed); top_k default = p//4 is generous; single-shot threshold with no per-N curve.
- Best when: small-n/high-p screening for a stable feature core. Poor when: you need the score-vs-N tradeoff (use the RFECV curve instead).

Improvement ideas: (a) complementarity-aware stopping in RFECV outer loop to distinguish pure-noise from noise-diluted-recoverable signal (the documented bench-rejected reject lives here); (b) randomized stability selection with feature reweighting; (c) expose a combined stability×score default rule behind a flag.

## 3. Optimizations

cProfile baseline taken on a representative RFECV fit (Ridge core, n=600, p=40, cv=3).
No hotspot exceeded the project's ship threshold inside the in-scope wrapper code — wall is
dominated by sklearn estimator fits and `cross_val_score` (external). The two fixes (F1/F2)
remove wasted work (F1 fewer noise counts; F2 a redundant unreliable scan) and are strict
correctness changes with no measurable slowdown. No speculative kernels added.

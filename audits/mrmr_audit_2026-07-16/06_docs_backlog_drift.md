## MRMR docs/backlog consistency audit

Covered: code vs `docs/MRMR_RESEARCH.md`, `docs/FE_MRMR_IDEAS_BACKLOG.md`, and
`src/mlframe/feature_selection/filters/_benchmarks/mrmr_critique_2026_07/*.md`.

### Real doc-drift bugs (docs overclaim vs. actual code)
1. **Chao-Shen entropy correction** — `docs/MRMR_RESEARCH.md` lists it as "DONE," but `_mrmr_class.py:3364-3371` emits a warning that `mi_correction='chao_shen'` isn't wired into the relevance/null path and silently falls back to plug-in MI. Drifted. (The tracker's own `_TRACKER.md` row N-F6 already states this correctly, so it's just the top-level doc banner that's wrong.)
2. **KSG estimator "shipped"** — doc cites `mi_estimator="ksg"`, but no such param exists on `MRMR`; code comments (`_mrmr_class.py:215-219`) explicitly say KSG isn't wired into `.fit()`. Drifted.
3. **SU normalization claimed as forced-on default** — doc's cross-agent table says SU-forced-on; code default is `mi_normalization: str = "none"` (`_mrmr_class.py:277-287`, explicit comment preserving the regression sentry). Drifted.
4. **JMIM redundancy param name** — backlog says `mrmr_redundancy_algo='jmim'`; actual knob is `redundancy_aggregator` (`_mrmr_class.py:243, 299-300`) — `mrmr_redundancy_algo` is a different, always-`"fleuret"` parameter. Drifted (functionality shipped, but under an undocumented name).

### Doc-vs-doc staleness
5. `_VERIFICATION.md` still marks S-F2, S-F4, N-F1, N-F2, P-1..P-4, P-11, ST-1, ST-4 as REMAINING/TODO, while `_TRACKER.md`'s progress log marks the same IDs DONE with commit hashes. `_TRACKER.md`'s own "COMPLETE" banner also doesn't match its per-section tables, which it flags itself. Needs a refresh pass to reconcile.

### Consistent (no drift)
JMIM aggregator logic itself, RelaxMRMR/PID-synergy/BUR-lambda (shipped opt-in, default 0.0/off, doc doesn't claim default-on), Inf-FS eigenvector centrality / matrix-Rényi CMI / QMIFS / permutation-null prevalence bar / Apriori CMI lattice / KSG tie-breaker in FE-gate band (all accurately marked not-started in backlog).

### Rejected-and-documented (do not resurface)
S-F3 (JMIM exponent), N-F3 (perm_pvalue full-budget extrapolation) — both have dedicated bench files and regression tests pinning the rejected alternative.

### Good, still-unimplemented ideas worth resurfacing
- **FCBF ordered pruning** for cluster aggregation — `_cluster_aggregate.py` still uses hard-threshold connected components (`corr_threshold`, `homogeneity_tau`); not started, not rejected.
- **kernel_tuning_cache for cluster thresholds** — `_cluster_aggregate.py` hardcodes `corr_threshold=0.6, homogeneity_tau=0.6` instead of using the repo's standard `pyutilz.system.kernel_tuning_cache` pattern for adaptive thresholds.

### Net
The module's ~120 constructor params mostly do trace cleanly to backlog/critique finding IDs via inline comments — the drift is concentrated in a few "DONE" overclaims in `MRMR_RESEARCH.md`'s top banner (Chao-Shen, KSG, SU-default) and one wrong param name (JMIM), plus one doc-hygiene gap (`_VERIFICATION.md` vs `_TRACKER.md`).

# Game-theory research plans for mlframe

Nine self-contained implementation plans applying cooperative/non-cooperative game theory to
mlframe's feature selection, data valuation, ensembling, and feature-engineering layers. Each
document is written to be executed by an implementing model WITHOUT access to the originating
conversation: all file paths, signatures, seam line references, formulas, synthetic-data specs and
acceptance thresholds are stated inline.

## Index

| Doc | Concept | Integration verdict |
|---|---|---|
| [gt_01](gt_01_shapley_interaction_indices.md) | Faith-Shap / Shapley-Taylor interaction indices | ShapProxiedFS config extension (`proxy_mode="faith_interaction"`) |
| [gt_02](gt_02_core_nucleolus_stability.md) | Least-core / nucleolus subset stability | ShapProxiedFS config extension (`refine_mode="core"`) |
| [gt_03](gt_03_banzhaf_value_attribution.md) | Banzhaf value feature ranking | ShapProxiedFS config first; optional full `BanzhafFS` selector as phase 2 |
| [gt_04](gt_04_data_shapley_valuation.md) | Data Shapley / KNN-Shapley row valuation | NEW component `src/mlframe/data_valuation/` (not a feature selector) |
| [gt_05](gt_05_shapley_ensemble_weighting.md) | Shapley model weighting + ensemble pruning | votenrank blend + composite/ensemble gate variant |
| [gt_06](gt_06_adversarial_reweighting_game.md) | Non-cooperative games: DRO reweighting as a minimax game | research-exploratory; concrete part reuses gt_04 plumbing |
| [gt_07](gt_07_fe_generator_shapley_budgeting.md) | Shapley budgeting of FE generator families | MRMR FE-layer integration (existing cost ledger) |
| [gt_08](gt_08_interaction_levers_auto_default.md) | Auto-activated interaction levers (`proxy_mode="auto"` default) | ShapProxiedFS config + default flip behind a data-driven gate |
| [gt_09](gt_09_two_phase_residual_fs.md) | Two-phase (residual) attribution for weak-signal recall | ShapProxiedFS config extension (`residual_passes`) |

Recommended execution order: **gt_08 → gt_09** (direct ShapProxiedFS improvements, highest
immediate value; both motivated by an empirical wide-dataframe investigation whose full trace lives
in `src/mlframe/feature_selection/shap_proxied_fs/_benchmarks/PLAN_wide_dataframe_improvements.md`)
→ gt_01 → gt_02 → gt_03 (FS family) → gt_04 → gt_05 → gt_06 → gt_07.

## Shared conventions (read before implementing ANY document)

These are repo-level rules from `CLAUDE.md` and hard-won session findings. Every plan below assumes
them; violations will be rejected in review.

### Testing
- **Every feature ships three artifacts in the same change**: unit tests, a quantitative
  `biz_val` test, and a cProfile harness (saved in-package under a `_benchmarks/` dir, not /tmp).
  Skip clauses: pure refactors, trivial helpers, docs-only changes.
- biz_val naming: file `tests/<pkg>/test_biz_val_<class>.py`, functions
  `test_biz_val_<class>_<param>_<scenario>`. Assertions are quantitative
  (`assert res.recall >= 0.55`), never existence checks. Threshold set 5-15% below the measured
  value. Each test <5s where possible; mark `@pytest.mark.slow` + `@pytest.mark.timeout(...)`
  otherwise.
- **pytest invocation on the dev box REQUIRES** `CUDA_VISIBLE_DEVICES=""` (collection segfaults
  otherwise — see the pyarrow/GPU note in `tests/conftest.py` near the `pyutilz.system` import),
  plus `--no-cov -p no:anyio`. Windows: never rely on non-ASCII in printed output (cp1251 console).
- Every bug fix ships a regression test verified to fail pre-fix and pass post-fix, same commit.

### Defaults & benchmarking
- A default flip requires a **majority-of-scenarios win replicated across seeds** on the honest
  holdout metric, with no regression on any covered scenario. One-bed or one-seed wins stay opt-in.
- REJECTED ≠ DELETED: a rejected option keeps its committed bench, a tunable flag in prod, and a
  `# bench-attempt-rejected` note at the call site.
- Gate a big win on its safe condition: when a lever only wins on a detectable regime, ship a
  data-driven gate (fires on that regime, exact-legacy path elsewhere), never an unconditional flip.
- A/B validation: warm first, best-of-N/median never one-shot, isolated AND end-to-end, identity
  gate alongside speed (bit-identical or documented ~1e-9 FP-reorder delta proven decision-neutral).

### Code
- New non-trivial functionality goes in a NEW focused sibling module re-exported from the parent
  facade; never grow an existing file toward the 1k-LOC ceiling. After any module split, AST-audit
  the sibling for unresolved names (lazy NameError trap).
- mypy-clean from the start (no implicit Optional, concrete types over `object`, return
  annotations matching actual returns).
- Comments: up to 160 chars/line, minimalist, WHY-only; no process/audit metadata (phase markers,
  dates, finding IDs).
- sklearn estimator params: store constructor values VERBATIM (`clone()` compares post-init
  identity; normalize at use sites, never in `__init__`).
- Memory: frames can be 100+ GB — never `.copy()` a frame to work around anything;
  mutate-and-restore or views.

### Git (shared working tree — multiple concurrent agent sessions)
- NEVER `git stash`, `git reset --hard`, `checkout --`, force-push. To sync: `fetch` + `merge`.
- Before every commit: `git diff --cached --stat` and unstage (`git restore --staged`) anything not
  yours — concurrent sessions' files WILL leak into your index.
- Pre-commit's `mlframe-meta-tests` hook is flaky on this box (native crash); `--no-verify` is the
  session-sanctioned workaround, but then run the relevant tests yourself before committing.

### Adding a full feature selector (the 6-step contract)
Only gt_03 phase 2 needs this; quoted from `src/mlframe/feature_selection/registry.py` docstring:
1. Selector module under `src/mlframe/feature_selection/`.
2. `register(_SimpleSpec(name=..., instantiate=..., report_extract=...))` in
   `feature_selection/registry.py` (registrations at bottom, `:235-254`).
3. `use_<sel>_fs` flag + `<sel>_kwargs` + validator in `training/_feature_selection_config.py`.
4. Branch in `_build_pre_pipelines` (`training/core/_setup_helpers_pre_pipelines.py`) + kind string
   in `_selector_kind`.
5. Factory added to the shared contract parametrization in
   `tests/feature_selection/contracts/test_fs_selector_contract.py` (`:31-68`).
6. `tests/feature_selection/test_biz_val_<sel>.py`.

### Key shared file paths
- ShapProxiedFS package: `src/mlframe/feature_selection/shap_proxied_fs/` (facade `__init__.py`,
  fit orchestration `_shap_proxied_fit.py`, attribution `_shap_proxy_explain.py`, proxy objective
  `_shap_proxy_objective.py`, interactions `_shap_proxy_interactions.py`, refine/revalidate
  `_shap_proxy_revalidate/`).
- Empirical wide-dataframe investigation (fixtures, measured recall tables, root-cause analysis
  referenced by gt_02/08/09): `src/mlframe/feature_selection/shap_proxied_fs/_benchmarks/PLAN_wide_dataframe_improvements.md`.
- Interaction benches: `src/mlframe/feature_selection/_benchmarks/bench_shap_interaction_proxy.py`;
  regime data helper: `src/mlframe/feature_selection/_benchmarks/_shap_proxy_regime_data.py`
  (`make_regime_dataset`).

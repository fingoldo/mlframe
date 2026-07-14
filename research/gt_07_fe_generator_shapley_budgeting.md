# gt_07: Shapley budgeting of feature-engineering generator families

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

MRMR's feature-engineering step fans out over generator FAMILIES (pairwise products, polynomial
bases, categorical pair/triple encodings, dispersion families, etc.) that differ wildly in cost
and yield. Today cost is TRACKED but not ACTED on: every family gets its budget every fit
regardless of whether its output ever survives selection. A family that burns 40% of FE wall to
produce columns that are always discarded is pure waste — repeatedly, on every fit.

Game framing: generator families are players; the value of a coalition is the quality of the
selected feature set built from their combined outputs. Family credit = Shapley-style attribution
of realized feature importance to the family that produced each surviving column. ROI =
credit / wall-cost. Budget reallocation: next round's per-family budget ∝ ROI — a
mechanism-design-flavored compute market where families "earn" future compute by producing
surviving, important features.

Honest simplification for v1: full Shapley over families (retraining selection per family
coalition) is unnecessary — each generated column already receives an importance/SHAP score from
the selector, and columns map 1:1 to families by name. Family credit = Σ importance of its
surviving columns is the additive approximation; it inherits Shapley's meaning to the extent
selector importances do (MRMR relevance or ShapProxied mean|φ| both qualify). State this
approximation explicitly in code docstrings. A `credit="loo"` upgrade (leave-one-family-out
re-selection deltas) is specced as opt-in for when families' outputs are strongly redundant.

## 2. Existing machinery (verified paths — the cost ledger already exists)

- **Cost ledger**: `src/mlframe/feature_selection/filters/_fe_family_timing.py` — process-global
  `_FE_FAMILY_WALL: dict[family -> [wall_seconds, n_invocations]]`; API:
  `fe_family_timer(name)` ctx-manager (`:33`), `fe_timed(name)` decorator (`:58`),
  `record_fe_family_wall` (`:46`), `reset_fe_family_wall` (`:71`), `get_fe_family_wall()` (`:77`).
- Memory-budget sibling (pattern reference): `filters/_feature_engineering_mem_budget.py`
  (budget ratios, `times_spent` accumulator, lock discipline).
- FE orchestration seams: `filters/_mrmr_fe_step/` (core: `_step_core.py`),
  `filters/_feature_engineering_pairs/` (`_pairs_core.py`, `_pairs_score.py`),
  `filters/_extra_fe_families.py`, `filters/_extra_fe_families_dispersion.py`,
  `filters/feature_engineering.py`, `filters/polynom_pair_fe.py`, `_cat_pair_fe.py`,
  `_cat_triple_fe.py`.
- Column→family mapping: FE output column names encode their generator (inspect the naming in
  `_pairs_core.py` / `polynom_pair_fe.py` first; build the name→family parser from the ACTUAL
  conventions found, with a unit test pinning each family's pattern — do not guess).
- Per-hardware persistence pattern: `pyutilz.performance.kernel_tuning.cache` usage precedent
  throughout `shap_proxied_fs/_shap_proxied_resolvers.py` — reuse the same pattern for persisting
  learned budgets across fits/processes.

## 3. Design

### 3.1 New module `src/mlframe/feature_selection/filters/_fe_family_budget.py`
```python
def family_credit(
    selected_importances: dict[str, float],    # surviving column name -> importance score
    column_family_fn,                          # name -> family str (the parser from §2)
    *, credit: str = "additive",               # "additive" | "loo" (v2)
) -> dict[str, float]:
    """Aggregate per-family credit. Unmapped columns -> family '_original' (raw input features,
    excluded from budgeting). Returns family -> credit >= 0."""

def family_roi(credit: dict[str, float], wall: dict[str, list]) -> dict[str, float]:
    """ROI = credit / max(wall_seconds, eps). Families with wall but zero credit get ROI 0;
    families never yet run get ROI = None (must not be starved before first trial)."""

def reallocate_budgets(
    roi: dict[str, float | None], *,
    base_budget: dict[str, float],             # current per-family budget (fraction or seconds)
    floor: float = 0.1,                        # min fraction of base budget any family keeps
    smoothing: float = 0.5,                    # EMA vs previous budgets
    exploration: float = 0.1,                  # fraction reserved for never-run/zero-ROI families
) -> dict[str, float]:
    """Proportional-to-ROI with floor + exploration reserve. The floor is MANDATORY: a family
    zeroed forever can never redeem itself, and target regimes change between datasets --
    this is the classic explore/exploit tension; the floor+exploration terms are the epsilon."""

def persist_budgets(budgets, *, cache_key="mlframe.fe_family_budget") -> None
def load_budgets(*, cache_key="mlframe.fe_family_budget") -> dict | None
    # kernel_tuning_cache-backed, per the shap_proxied resolver precedent
```

### 3.2 Consumption seam
The FE step must actually HONOR a per-family budget. Read `_mrmr_fe_step/_step_core.py` and
`_feature_engineering_pairs/_pairs_core.py` to find the per-family invocation loop; the minimal
viable enforcement is per-family CANDIDATE COUNT scaling (families propose k·budget_fraction
candidates instead of k), which every family loop already parameterizes; wall-clock-based
enforcement (stop family when its time slice is spent, via `fe_family_timer` readings) is the v2
refinement. Wire: (1) at FE-step start, `load_budgets()` (None → all-equal), scale each family's
candidate quota; (2) after selection completes, compute credit from the selector's importance
output (MRMR relevance scores of survivors — locate where MRMR exposes per-selected-feature
relevance; ShapProxied `mean|φ|` if invoked via that selector), `reallocate_budgets`,
`persist_budgets`. Gate the whole loop behind a new MRMR param `fe_budget_learning: bool = False`
(opt-in until benched) + `fe_budget_kwargs: dict | None`.

### 3.3 Reporting
`report["fe_family_budget"] = dict(wall=get_fe_family_wall() snapshot, credit, roi,
budgets_before, budgets_after)` — visibility is half the feature's value even before enforcement.

## 4. biz_val tests

File: `tests/feature_selection/test_biz_val_fe_family_budget.py`.

Synthetic bed engineered so family value is KNOWN: dataset where pairwise PRODUCT features carry
real signal (y depends on x1*x2) but, say, polynomial-basis features of single columns do not —
so the product family is "cheap useful", and (by config) give the useless family an inflated
candidate quota to make it "expensive useless".

1. `test_biz_val_fe_budget_shifts_toward_useful_family` — run 2 sequential fits with
   `fe_budget_learning=True`: after fit 1, useless family's budget fraction drops ≥50% from
   equal-split; useful family's rises.
2. `test_biz_val_fe_budget_recall_preserved_and_wall_drops` — fit 2 (with learned budgets) vs
   fit 1: selected-feature recall of the true signal features unchanged (the useful family was
   never the one cut), FE-step wall of fit 2 ≤ 0.8× fit 1 (measure first; floor per convention).
3. `test_biz_val_fe_budget_floor_prevents_starvation` — 3 sequential fits: useless family's
   budget never falls below `floor` fraction; then flip the bed (make the previously-useless
   family the signal carrier) and verify recovery within 2 fits (the explore/exploit guarantee).

Unit tests: name→family parser pins one real naming pattern per family (against actual generated
names, not hardcoded guesses — generate a tiny FE output in the test); reallocation invariants
(sum preserved, floor respected, exploration reserve distributed, never-run family not starved);
persist/load round-trip with the kernel_tuning_cache backend isolated to a tmp dir (the test
conftest already isolates `PYUTILZ_KERNEL_CACHE_DIR` — see `tests/conftest.py`).

## 5. Acceptance criteria
- Module + parser + seam wiring behind `fe_budget_learning=False` default; report block always
  populated when the flag is on; mypy-clean.
- All tests green locally (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`).
- cProfile evidence that the budgeting bookkeeping itself is negligible (<0.5% of FE wall).
- MRMR docstring documents the additive-credit approximation and the `credit="loo"` upgrade path.
- Default-flip decision deferred to a multi-dataset bench (real datasets, not just the synthetic
  bed — family usefulness is dataset-dependent by nature); bench file committed with verdict.

## 6. Known risks / rejected alternatives
- True family-Shapley (re-run selection per family coalition): rejected v1 — each v(C) is a full
  FE+selection run; the additive credit is the practical approximation, LOO the middle ground.
- Credit double-counting under redundant families (two families producing near-identical
  survivors): additive credit rewards both; LOO fixes it; note in docstring, do not solve in v1.
- Cross-dataset persistence poisoning: budgets learned on dataset A applied to dataset B may be
  wrong — key the persisted budgets by a dataset fingerprint (n_features + column-name hash;
  reuse any existing content-hash helper found in the codebase) so unrelated datasets start
  fresh. This is REQUIRED, not optional: silent cross-dataset carryover is a correctness bug.
- Parallel-fit races on the process-global `_FE_FAMILY_WALL`: reads are snapshots; follow the
  existing lock discipline of `_feature_engineering_mem_budget.py`.

# gt_02: Least-core / nucleolus stability as a principled alternative to parsimony-greedy refine

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

ShapProxiedFS's final pruning stage, `within_cluster_refine`
(`src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_refine.py:535`),
greedily drops any member whose removal keeps the honest holdout loss within `parsimony_tol`
(default 0.02 = 2%) of the best seen. This is a THRESHOLD decision: a genuinely-predictive weak
feature whose marginal honest-loss contribution is under 2% is always dropped, regardless of how
consistently it contributes.

Empirical trace (full table in
`src/mlframe/feature_selection/shap_proxied_fs/_benchmarks/PLAN_wide_dataframe_improvements.md`):
on a p=3000 fixture (6 strong w=1.0 + 6 weak w=0.25), weak features survive the entire pipeline
into the search winner (17 members) and refine prunes them all (17 → 5). Sweeping
parsimony_tol {0.02 → 0.005 → 0.0} recovers features monotonically (5 → 7 → 8 selected) but even
tol=0.0 keeps only 1/6 weak — a greedy sequential drop cannot distinguish "redundant given the
rest" from "individually small but jointly meaningful".

Cooperative game theory offers the principled alternative. Model the selected members as players,
the (negated) proxy loss as the characteristic function v(C). The **core** is the set of credit
allocations x where no coalition is "unhappy": x(C) ≥ v(C) for all C, x(N) = v(N). The **least
core** relaxes to the smallest uniform slack ε*: minimize ε s.t. x(C) ≥ v(C) − ε ∀C. A feature
whose removal-coalition N∖{j} can "block" (v(N∖{j}) ≈ v(N), i.e. j adds nothing even jointly) gets
x_j ≈ 0 and is safely droppable; a weak-but-real feature keeps x_j > 0 because SOME coalitions
need it. The drop rule becomes principled — grounded in coalition stability, not a scalar
tolerance. The **nucleolus** (lexicographic minimization of sorted excesses) is the canonical
unique point in the least core; for our purposes the least-core allocation + per-feature x_j is
sufficient (nucleolus refinement is an optional second LP stage, include as a flag).

## 2. Integration verdict

**Config extension of ShapProxiedFS** — new constructor param `refine_mode: str = "greedy"`
(`"greedy"` = legacy `parsimony_tol` path, byte-identical; `"core"` = this plan). Not a new
selector: only the refine stage's member-drop decision changes; honest revalidation and all other
stages are untouched. This also depends on / composes with gt_09's `protected_cols` refine
extension (either order of implementation works; if gt_09 landed first, `refine_mode="core"`
simply ignores `protected_cols` because core allocation subsumes protection).

## 3. Existing machinery to reuse (verified paths)

- Game oracle: `subset_loss(phi, base, y, idx, metric)` in
  `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_objective.py`, or better the memoised
  `_Evaluator` (`_shap_proxy_heuristics.py`) — v(C) = −loss(C). NOTE: refine operates on ORIGINAL
  member columns post-unit-expansion, but the proxy phi lives in proxy/unit space. Map members →
  units via `unit_to_members` (available at the refine call site in `_shap_proxied_fit.py`,
  search `member_groups = [[int(c) for c in unit_to_members[int(u)]] for u in best_idx]`).
  Coalitions are formed over UNITS (proxy columns of the winning candidate) — the same granularity
  the proxy is faithful at; expansion to member columns happens after, exactly as today.
- Honest-loss oracle for the FINAL verification: `_honest_loss` via the existing
  `HonestLossCache` (`_shap_proxy_revalidate/`), same plumbing refine already uses.
- LP solver: `scipy.optimize.linprog(method="highs")` — scipy is a core dependency
  (`pyproject.toml` `scipy>=1.10`), HiGHS is bundled. No new dependency.
- Refine call site: `_shap_proxied_fit.py`, block `with _stage("within_cluster_refine"):` —
  the branch point for `refine_mode`.

## 4. Design & implementation steps

1. **New module**
   `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_core_stability.py`:

   ```python
   def least_core_allocation(
       evaluator, players: tuple[int, ...], *,
       n_coalitions: int = 512, rng: np.random.Generator,
       nucleolus_refine: bool = False,
   ) -> tuple[np.ndarray, float, dict]:
       """Sampled least-core LP over the proxy game restricted to `players` (unit indices).

       v(C) = evaluator.loss(players) - evaluator.loss(C)   # improvement over... see NOTE below
       Actually use the standard normalization: v(C) = loss(∅ over players' complement is
       ill-defined for a loss game) -> define v(C) = L_max - loss(C) where
       L_max = loss(()) = evaluator's empty-subset convention (+inf) is unusable; use
       L_ref = loss of the WORST singleton among players (finite, cheap) as the zero point:
       v(C) = max(0, L_ref - loss(C)). Monotone transform of loss; document the choice — core
       membership is invariant to the additive shift, only the zero point of x_j moves.

       LP variables: x_1..x_k, eps.  Objective: min eps.
       Constraints: sum(x) == v(N);  for each sampled coalition C: sum_{j in C} x_j >= v(C) - eps;
       x_j >= 0 (loss game is monotone-ish; nonnegativity keeps the interpretation "credit").
       Coalition sample: all singletons + all leave-one-out (k + k, exact for the constraints that
       matter most) + uniform random subsets to n_coalitions total. Returns (x (k,), eps*, info)
       with info = dict(n_constraints, binding_coalitions, lp_status).
       nucleolus_refine=True: second LP fixing eps*, minimizing the second-largest excess
       (iterative standard scheme, cap at 3 iterations — document that full nucleolus is O(2^k)
       LPs in the worst case and we deliberately truncate).
       """
   ```

   ```python
   def core_refine(
       members, unit_players, evaluator, honest_loss_fn, *,
       drop_threshold: float = 0.02,   # fraction of total allocation, NOT loss units
       n_coalitions: int = 512, rng, nucleolus_refine: bool = False,
   ) -> tuple[list[int], dict]:
       """Drop units with x_j / sum(x) < drop_threshold, expand survivors to member columns,
       verify ONCE against the honest holdout (accept iff honest loss within the caller's
       parsimony_tol of the pre-refine loss — the honest gate stays, it just arbitrates ONE
       principled proposal instead of steering a greedy walk). On honest-gate failure fall back
       to the legacy greedy path (never return a worse-than-greedy outcome silently) and record
       fallback=True in info."""
   ```

2. **Constructor** (facade `__init__.py`): `refine_mode: str = "greedy"` (validate in
   `("greedy", "core")` without mutating), `core_n_coalitions: int = 512`,
   `core_drop_threshold: float = 0.02`, `core_nucleolus: bool = False`. Clone-safe storage.
3. **Branch at the refine call site**: `refine_mode=="core"` → `core_refine`, else legacy
   `within_cluster_refine` untouched. Report:
   `report["within_cluster_refine"]["mode"] = "core"|"greedy"`, plus for core:
   `allocation` (name→x_j), `eps_star`, `dropped_by_core`, `fallback`.
4. **cProfile harness** `_benchmarks/profile_core_refine.py`: k∈{10, 20, 30} players ×
   n_coalitions∈{256, 512, 1024}; the cost driver is n_coalitions × O(n) proxy-loss evals (memoised)
   + one HiGHS LP (milliseconds at this size). Target: core refine wall ≤ legacy greedy refine wall
   at k=17 (greedy pays k² honest FITS; core pays sampled proxy evals + ONE honest fit — should be
   strictly cheaper; verify and report).

## 5. biz_val tests

File: `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_core_refine.py`.
Fixture: the session mixed-strength generator (see
`test_biz_val_shap_proxied_parsimony_tol_recall.py::_make_mixed_strength_fixture`).

1. `test_biz_val_core_refine_recovers_weak_over_greedy` — `refine_mode="core"` vs `"greedy"`
   (both at default tol/threshold): weak recall(core) > weak recall(greedy) (measured greedy
   baseline: 0/6; require core ≥ 2/6), downstream-AUC ≥ greedy − 0.005.
   `@pytest.mark.slow @pytest.mark.timeout(900)`.
2. `test_biz_val_core_refine_drops_true_redundancy` — bed with 4 informative + 4 EXACT duplicates
   (rho=0.98 copies): core must still prune duplicates (n_selected ≤ 6), proving it doesn't just
   "keep everything" — duplicates' leave-one-out coalitions block them (v(N∖dup) ≈ v(N) → x_dup≈0).
3. `test_biz_val_core_refine_honest_fallback` — adversarial: force `core_drop_threshold=0.9`
   (absurd, drops nearly all); assert fallback fired and the result equals the greedy outcome
   (the safety net works).

Unit tests: `least_core_allocation` on textbook 3-player games with KNOWN cores (e.g. the classic
gloves game L={1}, R={2,3}: core = all allocation to the scarce side; assert x recovers it within
1e-6 using EXHAUSTIVE coalitions), eps*=0 for convex games, dummy player gets x=0; LP infeasibility
handling; determinism under fixed rng.

## 6. Acceptance criteria
- `refine_mode="greedy"` remains byte-identical (seeded A/B on one fixture: identical
  selected_features_ and report, modulo the new "mode" key).
- Textbook-game unit tests prove LP correctness (the scientific core — do not skip).
- biz_val 1-3 green locally; cProfile comparison committed showing core ≤ greedy wall at k=17.
- Bench (mixed-strength + duplicate + pure-strong beds × 3 seeds) committed; default stays
  `"greedy"` unless core majority-wins recall at non-inferior AUC AND non-superior wall — record
  the verdict either way.

## 7. Known risks / rejected alternatives
- Sampled-coalition core is approximate: a missed blocking coalition inflates x_j. Mitigation:
  always include all singletons + leave-one-outs exactly (these dominate the binding set for
  near-additive games); report `binding_coalitions` so under-sampling is visible.
- Exact nucleolus: rejected as default (iterated LP complexity); truncated refinement behind
  `core_nucleolus=True` only.
- Defining v from HONEST losses instead of proxy: rejected — k+n_coalitions honest fits is the
  exact retraining cost the whole selector exists to avoid; the proxy game is the point. The one
  honest verification of the final proposal keeps the honesty contract.

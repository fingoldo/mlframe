# DX dimension dispositions — Composite Targets audit (2026-06-10)

Triage of findings in `findings_dx.md`. ALREADY_RESOLVED applies only to DX1 (per the
integrator's fixed-id list). Every other finding (DX2–DX22) was re-opened against current
code before deciding. Verified against current source — no speculation.

| id | severity | disposition | file:line | reason | patch-spec (RESOLVE_NOW only) |
|---|---|---|---|---|---|
| DX1 | P1 | ALREADY_RESOLVED | discovery/__init__.py:146-170 | In the integrator's fixed-id list (iter_transform multi-base). | |
| DX2 | P2 | REJECTED | estimator/_predict.py:283-322 | Current code already has the `if transform.requires_base: ... else: base_arr=None` branch + zeros placeholder (lines 283-308, 321-322). Unary predict_quantile no longer crashes. Claim describes pre-fix code. | |
| DX3 | P2 | RESOLVE_NOW | estimator/_predict.py:332,338,347 | The three `transform.inverse(...)` calls omit `groups`; a `linear_residual_grouped` wrapper raises "groups kwarg is required". `_predict_unclipped` threads it (64-72); predict_quantile does not. Safe, mirrors existing path, no numerics change. | See patch below (PATCH-DX3). |
| DX4 | P2 | RESOLVE_NOW | estimator/_predict.py:253-263,289-306 | `_reciprocal_residual_inverse` = `1/(t+1/base)` is strictly DECREASING in t (verified); reciprocal_residual IS in the default transforms list (config line 108). predict_quantile guards only ratio/logratio → silent q-flip. Minimal safe fix: raise NotImplementedError for reciprocal_residual (mirror ratio guard). Full monotone-metadata reversal is FUTURE. | See patch below (PATCH-DX4). |
| DX5 | P2 | RESOLVE_NOW | composite/__init__.py:20-27,51-56 | Package docstring "Out of scope: Discovery … future PR / Cross-target ensembling: future PR" but both ARE re-exported (lines 180,258). Pure docstring correction; no code/behaviour. | See patch below (PATCH-DX5). |
| DX6 | P2 | FUTURE | composite/__init__.py:99-258; transforms/__init__.py:295 | No `__all__`, 96 underscore re-exports, read-only `TRANSFORMS_REGISTRY` proxy not re-exported. Adding `__all__` + re-export touches the public star-import surface across many callers — needs a deliberate surface decision + test, not a quick safe pass. (DX21 is the cheap subset and is RESOLVE_NOW.) | |
| DX7 | P2 | DOC | transforms/naming.py:5-7; discovery/__init__.py:455 | Docstrings promise legacy import paths (`composite_transforms` "resolves transparently"; "prefer … composite_auto_detect") that `find_spec` confirms do not exist → ModuleNotFoundError if followed. Docstring-only caveat fix across composite/** module headers. | |
| DX8 | P2 | DOC | README.md:125-145 | README composite section shows only the manual wrapper; discovery/ensemble/cache/kill-switch/doc-links absent. Doc expansion only (add Tier-1 snippet + links). | |
| DX9 | P2 | DOC | docs/examples/composite_targets.md:67,92-94 | Doc says "all four core transforms are tried" + `transforms=["diff","ratio","logratio","linear_residual"]`; real default is the 24-entry list (`_composite_target_discovery_config.py:82-110`). Doc inaccuracy only. | |
| DX10 | P2 | RESOLVE_NOW | README.md:68 | Line reads `pre-commit installгресс` (stray Cyrillic "гресс" glued on); copy-paste of the dev-setup block fails. Trivial text fix. | See patch below (PATCH-DX10). |
| DX11 | P2 | RESOLVE_NOW | estimator/_estimator.py:59-111,406 | Class docstring Parameters omits `runtime_stats_callback`, `auto_variance_stabilise`, `base_columns`, `group_column` (only inline `__init__` comments). `fit` (406) has no docstring. Docstring-only additions hoisting existing inline text; no code change. | See patch below (PATCH-DX11). |
| DX12 | P2 | FUTURE | ensemble/stacking.py:84-137 | `residual_dedup_indices` is production-wired (CT-ensemble build) with zero tests. Writing the 4 proposed tests is real work (new test file + fixtures), not a quick safe code edit. Queue as test-debt. | |
| DX13 | LOW | DOC | composite/__init__.py:47; estimator/__init__.py:45,57 | 3 docstrings describe the y-clip as multiplicative `[Q001/10, Q999*10]`; impl is span-based `q_low-0.9*span` / `q_high+9*span` (estimator/__init__.py:78-79). Code is intended-correct; docstrings wrong → DOC. | |
| DX14 | LOW | DOC | transforms/__init__.py:1,96 | Module docstring "11 transforms" but registry has 32; `TAG_EXTENDED` comment "placeholder; future presets may add more" stale (tag is load-bearing). Doc/count fix; prefer deferring to `list_transforms()` to avoid drift. | |
| DX15 | LOW | FUTURE | estimator/_estimator.py:692-717 | `update`/`get_buffer_state`/`predict_pre_clip`/`get_booster` + 5 delegated properties still runtime-bound (701-717), invisible to mypy/help(). Adding in-body stubs + class-body property wrappers is a broader surface change with re-binding-order risk; out of scope for a quick safe pass. | |
| DX16 | LOW | RESOLVE_NOW | composite/__init__.py:59-72 | Monolith-era imports (contextlib, math, re, warnings, dataclass, field, timer, most typing names, numpy, pandas, BaseEstimator, RegressorMixin, clone) are unused — file is pure re-exports below. Safe deletion (keep `logging`/`logger`, keep `from __future__`). | See patch below (PATCH-DX16). |
| DX17 | LOW | DOC | discovery/_eval.py:288; transforms/registry.py:389-391 | Unary specs are named via `compose_target_name(target_col, transform_name, base)` unconditionally → `y-cbrtY-<base>` with a spurious base segment, contradicting the registry comment ("no base segment, e.g. y-cbrtY"). Stripping the base changes spec.name/base_column → cache-key/provenance/dedup ripple → NOT a safe quick edit. Cheapest correct action: fix the registry comment to match current behaviour. | |
| DX18 | LOW | FUTURE | tests/test_docs_examples_smoke.py:24-25 | Docs smoke test pins only `CompositeTargetEstimator`; composite_targets.md/tutorial symbols (`CompositeTargetDiscoveryConfig`, `CompositeSpec`, `report_to_markdown`, MLFRAME_DISABLE_COMPOSITE) unpinned despite the test's own claim. New test additions = test work, queue it. | |
| DX19 | LOW | FUTURE | _composite_target_discovery_config.py:17 | 94-field config with no generated knob reference; dict-config acceptance undocumented. Needs a `scripts/` doc generator + doc-drift test (architectural/tooling) — out of scope for a quick pass. (The 1-line dict-config docstring note could be DOC but the headline deliverable is the generator.) | |
| DX20 | LOW | RESOLVE_NOW | estimator/__init__.py:87,114-117 | `_extract_base` docstring + TypeError advertise structured-ndarray support that has no branch; a structured ndarray hits the same TypeError telling you to pass one. Drop the false claim from docstring + message (cheaper + safer than implementing a new branch). | See patch below (PATCH-DX20). |
| DX21 | LOW | RESOLVE_NOW | composite/__init__.py:258 | `CompositeTargetDiscovery` is exported but its required `CompositeTargetDiscoveryConfig` is not — every example needs two import locations. Add a lazy re-export at the package bottom. Safe, additive, no behaviour change. | See patch below (PATCH-DX21). |
| DX22 | LOW | DOC | tests/training/ (layout) | Composite tests split across `tests/training/composite/` (38) and `tests/training/test_composite_*` (28); `pytest tests/training/composite` runs ~60%. `git mv` is a churny relocation; the safe, low-risk action is to DOCUMENT the canonical selector (`pytest -k composite`) — flag the layout split as a note. | |

## RESOLVE_NOW patch specs

### PATCH-DX3 — thread `groups` through predict_quantile inverse calls
File: `src/mlframe/training/composite/estimator/_predict.py`, in `predict_quantile`.
After the base/guard block (~line 308) and before computing `t_raw`, build inverse kwargs
mirroring `_predict_unclipped` (lines 64-72):

```python
    inverse_kwargs: dict[str, Any] = {}
    if transform.requires_groups:
        if not self.group_column:
            raise ValueError(
                f"CompositeTargetEstimator.predict_quantile: transform "
                f"'{self.transform_name}' requires groups but "
                f"``group_column`` is not configured."
            )
        inverse_kwargs["groups"] = _extract_groups(X, self.group_column)
```
Then pass `**inverse_kwargs` to all three `transform.inverse(...)` calls (lines 332, 338, 347).
`_extract_groups` is already imported at module top (line 12).

### PATCH-DX4 — guard reciprocal_residual against silent quantile flip
File: `src/mlframe/training/composite/estimator/_predict.py`, inside the
`if transform.requires_base:` block (after the logratio guard, ~line 306):

```python
        # reciprocal_residual inverse y = 1/(T + 1/base) is strictly
        # DECREASING in T: a high T-quantile maps to a LOW y-quantile.
        # Raise rather than silently swap (same policy as the ratio guard).
        if self.transform_name == "reciprocal_residual":
            raise NotImplementedError(
                "predict_quantile is not supported for transform "
                "'reciprocal_residual': y = 1/(T + 1/base) is decreasing in T, "
                "so the inverse swaps the quantile ordering. Use predict() for "
                "point predictions or switch transform."
            )
```
Also add a row to the docstring quantile-preservation table (lines 255-260) noting
reciprocal_residual flips / raises. (Full monotone-metadata-driven `alpha->1-alpha`
reversal is the FUTURE follow-up.)

### PATCH-DX5 — fix package docstring out-of-scope / public-surface claims
File: `src/mlframe/training/composite/__init__.py`, lines 51-56. Delete the two stale
"future PR" lines (Discovery, Cross-target ensembling) since both are re-exported here;
keep only the genuinely-out-of-scope item:
```
Out of scope for this module
----------------------------
- ``base_margin`` / classification residuals: regression only here.
```
And in the "Public surface" section (20-27) add the real exported families
(CompositeTargetDiscovery, CompositeCrossTargetEnsemble, DiscoveryCache, CompositeProvenance/
report_to_markdown, streaming refit, bayesian alpha, forward stepwise, feature stacking).

### PATCH-DX10 — fix corrupted pre-commit command in README
File: `README.md`, line 68. Replace `pre-commit installгресс` with `pre-commit install`.

### PATCH-DX11 — add missing constructor params + fit docstring
File: `src/mlframe/training/composite/estimator/_estimator.py`. In the class Parameters
section (59-111) add entries for `runtime_stats_callback`, `auto_variance_stabilise`,
`base_columns`, `group_column` (hoist the existing inline `__init__` comments at 132-175,
145-152, 137-144, 165-175). Add a short `fit` docstring (line 406+): documents
`sample_weight` pass-through, `**fit_kwargs` forwarded to the inner estimator, domain-row
dropping, and "returns self". Docstring-only; no code edit.

### PATCH-DX16 — delete dead monolith-era imports
File: `src/mlframe/training/composite/__init__.py`, lines 59-72. Remove the unused imports:
`contextlib`, `math`, `re`, `warnings`, `from dataclasses import dataclass, field`,
`from timeit import default_timer as timer`, the `typing` import block (66-68), `import numpy
as np`, `import pandas as pd`, `from sklearn.base import BaseEstimator, RegressorMixin, clone`.
Keep `from __future__ import annotations` and `import logging` + the `logger = ...` line (74).
Confirm none are referenced anywhere below (file is pure re-exports) before removing.

### PATCH-DX20 — drop false structured-ndarray claim from _extract_base
File: `src/mlframe/training/composite/estimator/__init__.py`. Line 87 docstring: change
"Pull base values from X (pandas / polars / structured ndarray)." to
"Pull base values from X (pandas / polars DataFrame)." Lines 114-117 TypeError message:
change "pass pandas / polars DataFrame or a structured ndarray with named columns." to
"pass a pandas or polars DataFrame." (Optionally update the `_extract_base_matrix` fallback
comment at line 189 which references "ndarray-with-names".)

### PATCH-DX21 — re-export CompositeTargetDiscoveryConfig from the composite package
File: `src/mlframe/training/composite/__init__.py`, at the bottom (after line 258). Add:
```python
# Convenience re-export so callers get discovery + its config from one place.
from ..configs import CompositeTargetDiscoveryConfig  # noqa: E402,F401
```
(Use whichever module path `CompositeTargetDiscovery.__init__`'s lazy dict-config import
already uses — `mlframe.training.configs` per the audit — to avoid a new circular import.)

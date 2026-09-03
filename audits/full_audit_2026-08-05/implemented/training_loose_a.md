# Audit: training_loose_a

**Cluster**: `training_loose_a` (part of the loose `*.py` files directly inside `src/mlframe/training/`,
alphabetical first-third split, files 1-28 of 83)

**Scope**: `__init__.py`, `_aggregate_cv_early_stopping.py`, `_calib_oof_outputs.py`, `_calibration_models.py`,
`_classif_helpers.py`, `_composite_target_discovery_config.py`, `_composite_target_discovery_config_base.py`,
`_confidence_analysis.py`, `_configs_base.py`, `_conformal_finalize.py`, `_conformal_split.py`,
`_cv_aggregation.py`, `_data_helpers.py`, `_dataset_cache_fingerprint.py`, `_direct_horizon_bucket_forecaster.py`,
`_easy_ensemble.py`, `_eval_helpers.py`, `_feature_importances.py`, `_feature_name_sanitize.py`,
`_feature_selection_config.py`, `_format.py`, `_gpu_probe.py`, `_helpers_training_configs.py`, `_io_save.py`,
`_iterative_stratification_njit.py`, `_mc_dropout.py`, `_model_configs.py`, `_model_configs_behavior.py`.

Out of scope (per instructions): `feature_selection/filters/**` (MRMR engine) and
`feature_selection/shap_proxied_fs/**` — already covered by a dedicated closed audit cycle.

**Files reviewed**: 28 (all read in full)
**LOC reviewed**: 11,172

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| TRAINING_LOOSE_A-1 | P1 | `_data_helpers.py:342-355` | `_validate_target_values`'s single-class diagnostic for an **empty** classification target raises `IndexError` (from `arr_np.flat[0]` on a 0-length array) while formatting the `ValueError` message; the surrounding `except ValueError: raise` / `except Exception: log.debug(...)` pair catches the `IndexError` in the generic branch and **silently swallows it** — the function returns normally instead of raising anything, defeating its entire purpose for the exact "upstream filter eliminated every row" scenario its own docstring describes. | Format the message from a value that's guaranteed to exist (guard `arr_np.size == 0` before indexing, or build the message without indexing into `arr_np`), and/or narrow the `except Exception` to something that can't accidentally absorb a bug in the error-construction code itself. | Static AST/grep scanner: flag any `raise ValueError(f"...{arr[i]!r}...")` (or similar array-indexing inside an f-string passed to `raise`) that sits inside a `try` block whose `except` clause does not also catch `IndexError`/`KeyError` explicitly — the message-construction code can itself fail and get silently reclassified as "no error" by a broad sibling `except Exception`. |
| TRAINING_LOOSE_A-2 | P2 | `_eval_helpers.py:39-65` vs `:151-158` | `_align_xgb_cat_categories`'s docstring says the fix "compute[s] the UNION of category levels across **all three splits**" so "XGBoost now sees val/test as a subset of the train cat universe" — but the actual (correct, leak-safe) implementation unions only **train + val**, explicitly excluding test with an in-line comment explaining why (test categories must never feed back into the union). The top-level docstring is stale/wrong and could mislead a future editor into "fixing" the code to match it, reintroducing the exact leak the current code guards against. | Rewrite the top docstring to describe the actual train+val-only union and the NaN-for-unseen-test-category behavior (matching the inline comment at line ~152), so the two descriptions agree. | Doc-consistency lint: grep function docstrings for "all three splits" / "train, val, test" / "train/val/test union" phrasing and cross-check that the function body actually touches all three named variables in the described way (or flag for human review when a splits-related docstring claim can't be corroborated in the same function body). |
| TRAINING_LOOSE_A-3 | P2 | `_io_save.py:187-268`, `:392-412` | `save_mlframe_model`'s pre-pickle walk (`_collect_pre_dump_swaps`) mutates the **shared, not deep-copied** nested object graph in place (torch.compile unwrap, Lightning-bloat nulling) and restores it only in a `finally` block. Even under `lean=True` only the top-level `SimpleNamespace` is shallow-copied — every nested sub-object (fitted estimators, wrapped models, etc.) is the *same* object the caller's live `model` still references. Two concurrent `save_mlframe_model` calls on bundles that share a nested sub-object (e.g. parallel per-model saves from a joblib-threaded ensemble step) can interleave their strip/restore windows, so one thread's dill/pickle serialization can observe attributes nulled by the other thread, or the caller's live in-memory object can be left with a bloat attribute *not* restored if the two `finally` blocks race. | Deep-copy (or `copy.copy` at every node being mutated, not just the top-level namespace) before stripping, or hold a per-object lock / do the strip on a private clone reachable only from `_payload`. | Concurrency regression test: build two `SimpleNamespace` bundles sharing one nested fitted-model object, call `save_mlframe_model` on both from two threads simultaneously (with an artificial delay injected into the strip step), and assert the shared nested object's attributes are restored to their original values after both calls return, and that both saved files deserialize with the correct (non-null) attributes. |
| TRAINING_LOOSE_A-4 | P3 | `_confidence_analysis.py:104-107` | `run_confidence_analysis` mutates the caller-supplied `confidence_model_kwargs` dict in place (`confidence_model_kwargs["iterations"] = 200`, `.setdefault("early_stopping_rounds", 30)`) instead of copying it first. Currently harmless because the sole in-repo call site (`_calib_oof_outputs.maybe_run_confidence_analysis`) always passes a fresh `dict(confidence.model_kwargs)`, but any other/future caller reusing a shared kwargs dict across multiple models/calls would see it silently polluted after the first call. | `confidence_model_kwargs = dict(confidence_model_kwargs or {})` at function entry before mutating. | Generic scanner: flag any function that receives a `dict`-typed parameter and later does `param[key] = value` or `param.setdefault(...)` without a preceding `dict(param)` / `param.copy()` on that same parameter (classic silent-caller-mutation footgun). |
| TRAINING_LOOSE_A-5 | P3 | `_aggregate_cv_early_stopping.py:54-58` | `select_best_iteration_by_aggregate_cv` validates `curves.shape[1] == 0` (zero rounds) but never validates `curves.shape[0] == 0` (zero folds). A 0-fold input silently produces `NaN` from `np.mean`/`np.median`/`trim_mean` over an empty axis (with a `RuntimeWarning`), rather than raising the same kind of clear `ValueError` the function already raises for the 0-round case two lines above. | Add `if curves.shape[0] == 0: raise ValueError(...)` alongside the existing 0-round check. | Property test: for any aggregation-function accepting a 2-D array with a documented "at least 1 X" invariant on one axis, fuzz both axes independently at size 0 and assert a `ValueError` (not a silent `NaN`/warning) is raised in every degenerate-axis case, not just the one already covered. |
| TRAINING_LOOSE_A-6 | P3 | `_easy_ensemble.py:21,59-68,78,98` | `easy_ensemble_fit_predict` validates `bag_feature_subsample` but never validates `n_bags`. `n_bags <= 0` produces an empty `bag_preds` list, and `np.mean(bag_preds, axis=0)` at the end raises an opaque `ValueError: zero-size array to reduction operation` instead of a clear, function-specific message. | `if n_bags < 1: raise ValueError("easy_ensemble_fit_predict: n_bags must be >= 1")` near the top, alongside the existing `bag_feature_subsample` guard. | Grep/AST scanner: for any function with a `n_*: int` loop-count parameter that feeds a `for _ in range(n_*)` loop whose results are later reduced (`np.mean`/`np.concatenate`/etc.), check for an explicit `>= 1` (or documented `0`-is-valid) guard on that parameter near the top of the function. |
| TRAINING_LOOSE_A-7 | P3 | `_data_helpers.py:36,64` | `logger = logging.getLogger(__name__)` is defined twice at module scope (once right after the imports, once again right after `_validate_trusted_path`). Harmless (idempotent) but dead/confusing — a reader might assume the second one shadows a different configuration. | Delete the duplicate at line 64. | Lint rule (ruff/AST): flag any module where the same top-level `name = expr` binding (same `name`, same simple call expression) appears more than once at module scope with no intervening use. |
| TRAINING_LOOSE_A-8 | P3 | `_data_helpers.py:665,669,673` | In `_setup_eval_set`'s shard-fanout branch, the three list-extension expressions `shard.sample_weight if shard.sample_weight is not None else None` (and the `base_margin` / `group_ids` siblings) are logical no-ops — `X if X is not None else None` always evaluates to `X`. They read as a defensive None-guard but do nothing; a maintainer skimming this code could reasonably (and incorrectly) assume some `None`-coercion is happening here. | Simplify to `sw_list.extend(shard.sample_weight for shard in extra_eval_sets)` (and the two siblings), or, if a substantive guard was actually intended (e.g. defaulting to `0`/skip), implement it. | Grep/AST scanner: flag any ternary expression of the exact shape `X if X is not None else None` (or `X if X else X`) anywhere in the codebase — always a no-op by construction, always worth a human look. |

## Counts

- P0: 0
- P1: 1
- P2: 2
- P3: 5

## Narrative

### TRAINING_LOOSE_A-1 (P1) — `_validate_target_values` silently swallows its own empty-target diagnostic

`_validate_target_values` is a purpose-built early-diagnostic: its docstring explains it exists specifically to
turn "upstream filter aggression (outlier_detection + trainset_aging_limit + rare imbalance class) eliminated
the minority class entirely from train" into a clear, actionable `ValueError` instead of an opaque C++ crash
deep inside CatBoost/XGBoost. For the single-class case it does `raise ValueError(f"... only one unique value
({arr_np.flat[0]!r}); ...")`. When the target array is completely **empty** (0 rows — the natural extreme of
the exact failure mode the docstring describes), `np.unique(arr_np)` returns an empty array (`len(...) == 0 <
2`), so the code enters the `raise` branch — but `arr_np.flat[0]` on a 0-length array raises `IndexError`
*while the f-string is being formatted*, before `ValueError.__init__` ever runs. The enclosing
`except ValueError: raise` doesn't match `IndexError`, so it falls through to the sibling
`except Exception as e: logger.debug(...)`, which logs at DEBUG and returns normally. Net effect: for an
empty classification target, this function does not raise *anything* — it silently passes.

Verified live:
```
python -c "
import numpy as np
from mlframe.training._data_helpers import _validate_target_values
_validate_target_values(np.array([], dtype=np.float64), subset_name='train', is_classification=True)
print('NO CRASH')"
# -> NO CRASH
```
and isolated the exact mechanism:
```
python -c "
import numpy as np
arr_np = np.array([], dtype=np.float64)
try:
    raise ValueError(f'x has value {arr_np.flat[0]!r}')
except ValueError:
    print('caught as ValueError')
except Exception as e:
    print('caught as', type(e).__name__, e)"
# -> caught as IndexError index 0 is out of bounds for axis 0 with size 0
```
`_validate_target_values` is called from `_trainer_train_and_evaluate.py` on both `train_target` and
`val_target` with `is_classification=_is_clf` — exactly the call shape that would hit this on an
all-rows-filtered split. There is already a near-miss regression test,
`tests/training/test_validate_target.py::test_empty_target_does_not_crash`, but it only exercises
`is_classification=False` — it never covers the classification branch, which is the one that actually
contains the bug. This is a textbook instance of the "error-swallowing hides a real bug" class this project's
own conventions call out explicitly.

### TRAINING_LOOSE_A-2 (P2) — stale docstring on a leakage-safety-critical function

`_align_xgb_cat_categories`'s module-level docstring states the fix "compute[s] the UNION of category levels
across all three splits per column" so that "XGBoost now sees val/test as a subset of the train cat universe."
The actual loop (with its own, correct, in-line justification) unions only `train_cats` + `val_df`'s
categories, explicitly excluding `test_df` — "test categories must never feed back into train at fit-time,
that's the canonical leak." The behavior is right (a test-only category is deliberately left out of the union
so it casts to NaN, which XGBoost treats as ordinary missing data rather than "unseen category"), but the
top-level docstring describing *how* is simply wrong, and describes the leaky version of the fix. This is
exactly the kind of contradiction that invites a well-meaning future edit ("the docstring says all three
splits, let me fix the code to match") to silently reintroduce a train/test leak in a function whose entire
purpose is preventing one.

### TRAINING_LOOSE_A-3 (P2) — `save_mlframe_model`'s pre-pickle strip mutates a shared graph

`_collect_pre_dump_swaps` walks `_payload.__dict__` recursively and, for any `torch.compile`-wrapped module,
Lightning `Trainer`/`DataModule`/`DataLoader`-shaped attribute, or FS-internal `_mlframe_*` marker it finds,
mutates the object's `__dict__` in place (swap or null) so the heavy attribute never reaches `dill`/`pickle`;
it records `(obj, key, orig_value)` tuples and restores them in the `finally` block after the dump. This is
safe for a single-threaded, single-call use. But `lean=True` only shallow-copies the *top-level*
`SimpleNamespace` (`_lean = SimpleNamespace(**{k: v for k, v in vars(model).items() if ...})`) — every nested
attribute (a fitted estimator, a wrapped Lightning module, etc.) is the *same object* still reachable from the
caller's live `model`. If two `save_mlframe_model` calls that share such a nested object run concurrently
(plausible wherever mlframe already uses joblib-threading for per-model work — e.g. permutation FI, ensemble
member saves), the strip/restore windows can interleave: one thread could serialize a nested object that the
other thread has already nulled out (or is mid-restore for), and the caller's live in-memory object could end
up with the wrong final state depending on `finally`-block ordering. This was not proven to be hit by an
actual concurrent call site within this cluster's files, so it is reported as a real, demonstrable-by-code-
reading correctness risk rather than a confirmed production incident — but the mechanism (mutate a live shared
graph, restore in `finally`, no lock, no deep copy) is exactly the shape of bug that fires only under
concurrency and is very hard to reproduce after the fact.

### TRAINING_LOOSE_A-4 through -8 (P3)

These are smaller, lower-blast-radius findings — a latent caller-dict-mutation footgun in the confidence-model
kwargs path (currently masked by the sole call site always passing a fresh copy), two missing degenerate-input
guards (`select_best_iteration_by_aggregate_cv` on 0 folds, `easy_ensemble_fit_predict` on `n_bags<=0`) that
trade a clear `ValueError` for a `NaN`/opaque-numpy-error, a duplicate module-level `logger = ...` statement,
and three no-op ternary expressions (`X if X is not None else None`) in `_setup_eval_set`'s shard fan-out that
read as defensive coding but do nothing. None of these were observed to produce a wrong trained model or wrong
prediction; they are hygiene/robustness/dead-code items the fix-wave should still pick up per the "every P3
gets fixed" mandate.

### Dimension coverage notes

- **Data leakage**: reviewed carefully given the cluster's density of split/calib/conformal/composite-config
  code (`_calib_oof_outputs.py`, `_conformal_split.py`, `_composite_target_discovery_config*.py`,
  `_calibration_models.py`). No leakage bugs found in the code paths themselves; the one leakage-adjacent
  finding (TRAINING_LOOSE_A-2) is a documentation/consistency issue on an already-correct implementation.
- **Reproducibility / RNG seeding**: every RNG use found in this cluster (`_classif_helpers.py`'s chain-order
  builder, `_easy_ensemble.py`, `_iterative_stratification_njit.py`) is explicitly seeded via a caller-supplied
  or module-documented seed. No unseeded-RNG issues found.
- **Mutable default arguments**: grepped every file in scope for `def f(..., x=[] )` / `def f(..., x={})`
  shapes; zero matches. No mutable-default-argument bugs in this cluster.
- **GPU/CPU dispatch**: `_gpu_probe.py`, `_helpers_training_configs.py`'s device-gating logic, and
  `_feature_importances.py`'s CUDA-batched permutation path were reviewed; dispatch logic is consistent and
  defensively probed (build-info checks, opt-in env vars for LightGBM CUDA). No dispatch bugs found.
- **Computational efficiency**: this cluster is mostly config/orchestration/dispatch code rather than hot
  numerical kernels; the one hot kernel in scope (`_iterative_stratification_njit.py`) is already an
  `@njit`-ported, benchmarked replacement for the pure-Python reference with a documented validation sweep.
  No unnecessary `.copy()`/O(n^2) issues found; existing shallow-copy (`deep=False`) patterns in
  `_eval_helpers.py` / `_confidence_analysis.py` / `_io_save.py` are correctly scoped.
- **Test coverage gaps**: beyond TRAINING_LOOSE_A-1's near-miss test, the config-validator-heavy files
  (`_configs_base.py`, `_composite_target_discovery_config*.py`, `_feature_selection_config.py`,
  `_model_configs*.py`) are declarative Pydantic models with validators; they were not independently checked
  against the test suite's exact validator-coverage matrix (out of scope for a read-only code review), but no
  validator logic bug was found that would indicate a coverage gap is actively hiding a defect.

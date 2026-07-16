## MRMR sklearn/serialization compatibility audit

Confirmed: no `get_params`/`set_params`/`__getstate__`/`__reduce__` overrides anywhere in the module — MRMR relies entirely on `BaseEstimator`'s defaults plus the custom `__setstate__`/`__repr__`.

### Findings

**1. `n_jobs=-1` is resolved to a concrete CPU count *before* `store_params_in_object`, breaking the sklearn "store unmodified" contract — `_mrmr_class.py:2805-2806` vs `2849`**
```python
if n_jobs == -1:
    n_jobs = psutil.cpu_count(logical=False)   # line 2805-2806, BEFORE store
...
store_params_in_object(obj=self, params=get_parent_func_args())   # line 2849
```
`get_parent_func_args()` (`pyutilz.core.pythonlib`) reads the caller frame's **current local variable values** via `inspect.getargvalues`, not the original call arguments. Since `n_jobs` was reassigned before this call, `self.n_jobs` stores the *resolved* core count, never `-1`. This directly contradicts the adjacent comment block (lines 2825-2832) which explains at length that `random_state`/`skip_retraining_on_same_shape` resolution is deliberately deferred to fit-time specifically *to avoid this exact problem* — yet `n_jobs` (and `parallel_kwargs`, see #2) get the same treatment the comment says was rejected.
- Failure scenario: **cross-machine joblib** — an `MRMR(n_jobs=-1)` constructed on a 32-core CI/driver machine and shipped via `GridSearchCV(n_jobs=...)`/pickled to a worker pool on a different/smaller machine (or serialized once and reused later on different hardware) carries a hard-coded `n_jobs=32` forever; it never re-resolves to the worker's own core count. `get_params()['n_jobs']` also permanently shows `32`, not `-1`, so a user calling `set_params(**estimator.get_params())` on a fresh instance to "reproduce" the config gets a different effective value if run on different hardware, and `clone(mrmr)` propagates the frozen count instead of the auto-sentinel.
- Same root cause affects `parallel_kwargs`: `_mrmr_class.py:2808-2821` mutates `parallel_kwargs` from `None` to a concrete `dict(max_nbytes=MAX_JOBLIB_NBYTES, backend="threading")` before storage, so `get_params()['parallel_kwargs']` never reports the constructor's actual `None` default either.

**2. No schema-version stamp for pickle compatibility — purely getattr/key-existence inference (`_mrmr_setstate_defaults.py`, `_mrmr_class.py:2910-2952`)**
`__setstate__` infers "how old" a pickle is purely from *which keys are absent* from `state`, using a hand-maintained roster (`_SETSTATE_LEGACY_DEFAULTS`) plus a `_SETSTATE_LEGACY_OVERRIDES` frozenset of 10 keys, plus a final "any remaining ctor param not yet in state" catch-all sourced from a fresh instance (lines 2942-2951). There is no `__version__`/`_mrmr_schema_version` field written by `__getstate__` (there is no `__getstate__` override at all — state is the raw `__dict__`). This works today because the roster is exhaustively hand-maintained, but it is a maintenance trap, not a real compatibility contract:
  - Any future param that is *renamed* (not just added) needs a manual entry in both the roster and a getattr-fallback at every read site; nothing fails loudly if a contributor forgets — the new pickle just silently gets a wrong/legacy value with no version check to catch the mismatch.
  - Because inference is by absence-of-key rather than an explicit version number, a pickle produced by a **future/newer** code version being loaded by an **older** installed mlframe (downgrade scenario, e.g. rolling back a deploy) is not handled at all: `state` will contain keys the older `__init__`'s ctor-default introspection (`_ctor_defaults()` via `inspect.signature(cls.__init__)`) doesn't recognize, and `self.__dict__.update(state)` at line 2952 will happily set attributes the older code never reads and never validates — no error, just silently-ignored newer fields (silent misbehavior, not a crash).
  - `_ctor.items()` catch-all loop at 2947-2951 calls `type(self)()` (line 2943) — i.e., **runs a full MRMR construction** (with all its default-resolution side effects, including the `n_jobs`/`parallel_kwargs` mutation from finding #1) inside every legacy-pickle unpickle. This is a nontrivial, deopaque side effect for something that "just" restores state, and its `except Exception` fallback (2944-2946) silently downgrades to roster-only defaults on any construction failure — masking the real cause with only a debug-level log line.

**3. joblib/pickle: no non-picklable live resources found on instance state, but the `type(self)()` re-entrant construction in `__setstate__` is a hazard under joblib worker unpickling**
No thread locks, open file handles, GPU (cupy) resident arrays, or closures/lambdas are stored as instance attributes (`_FIT_CACHE` is a class-level `OrderedDict` — `_mrmr_class.py:164` — not instance state, so it is not pickled per-instance and is not a joblib cross-process concern). However:
- The fresh-instance construction inside `__setstate__` (`_fresh = type(self)()`, line 2943) means **every** unpickle (including the routine `GridSearchCV`/`cross_val_score` worker-side unpickle after `n_jobs>1` pickles the fitted/unfitted estimator to a subprocess) pays the cost of a full constructor call and is subject to whatever import-time or environment-dependent behavior that constructor has (e.g. `psutil.cpu_count()` reading the **worker's own** core count for `_fresh.n_jobs`, which then can get copied onto `state[k]` for any ctor param not already present — a second, independent place where a worker-local hardware value leaks into restored state, diverging from the value on the driver process that pickled the original instance).
- `cache_dir` (`_mrmr_class.py:2771`, a `str | None` path for the disk-cache) is stored as a plain string, so it round-trips through pickle fine, but if set, a worker process unpickling the estimator will attempt to read/write that path — a real cross-process/cross-machine correctness risk if the worker doesn't share the driver's filesystem (not a pickling *crash*, but a silent behavior divergence: cache misses or `PermissionError`/`FileNotFoundError` depending on OS, un-audited here).

**4. `__repr__` override is a textual patch to a version-fragile sklearn method — `_mrmr_class.py:2852-2862`**
```python
def __repr__(self, N_CHAR_MAX: int = 700) -> str:
    r = super().__repr__(N_CHAR_MAX=N_CHAR_MAX)
    if "n_workers=" not in r and r.endswith(")"):
        _inner = r[:-1]
        _sep = "" if _inner.endswith("(") else ", "
        r = f"{_inner}{_sep}n_workers={getattr(self, 'n_workers', 1)})"
    return str(r)
```
- It force-surfaces `n_workers` even when it equals its default, deliberately defeating sklearn's `set_config(print_changed_only=True)` filtering (the global default since sklearn ~0.23) for that one param. This is intentional per the inline comment and doesn't crash anything, but it means `repr(MRMR())` no longer round-trips through `print_changed_only` semantics the way every other param does — a minor but real inconsistency for anyone diffing reprs to infer "what did the user change from default."
- The override assumes `BaseEstimator.__repr__` (a) accepts an `N_CHAR_MAX` keyword and (b) always produces a string ending in literal `")"` with no trailing content after the closing paren. Both are true for the current `sklearn.utils._pprint._EstimatorPrettyPrinter`-based implementation, but neither is a documented/public contract — a future sklearn version that changes the trailing format (e.g., appends a `# doctest: ...` suffix, or truncates without a paren when `N_CHAR_MAX` is hit at an inner nesting level, which can happen with a ~300-param estimator) would silently stop appending `n_workers=` (the `endswith(")")` guard fails closed, not open — so worst case is losing the extra annotation, not corrupting output, but it's an untested assumption about upstream internals, not the public `BaseEstimator` API).
- Notebook HTML repr (`_repr_html_`/`estimator_html_repr`) is not affected — it does not delegate to this `__repr__` for its diagram, only for the "show params" fallback text in some sklearn versions, so no notebook-breaking risk found here.

**5. `get_params()`/`set_params()` round-trip mechanically fine, `clone()` fine, ~300-param constructor has no mutable-default-argument sharing bug**
Checked the full `__init__` signature (`_mrmr_class.py:194-2801`) for literal mutable defaults (`[]`, `{}`) — none found; every collection-typed param defaults to `None` or an immutable tuple, and the FE/tuple defaults (e.g. `("mean_z",)`, `(2, 3)`) are immutable so they are safe to share across instances/clones. `store_params_in_object` sets each param verbatim onto `self` (aside from the two exceptions in finding #1), and `_ctor_defaults()` (`_mrmr_class_config.py:268-280`) reads defaults straight from `inspect.signature`, so `BaseEstimator._get_param_names()`'s introspection-based `get_params()` will enumerate and round-trip all ~300 params correctly. No accidental extra mutable state is captured in `__init__` beyond `self.signature = None` (line 2850, a private non-ctor-param attribute, harmless for `clone()` since `clone()` only copies `get_params()` keys).

## Summary table

| # | File:Line | Risk | Failure scenario |
|---|---|---|---|
| 1 | `_mrmr_class.py:2805-2806, 2821, 2849` | `n_jobs=-1`/`parallel_kwargs=None` resolved before `get_parent_func_args()` snapshot; `get_params()` never reports the sentinel | cross-machine joblib/GridSearchCV, `clone()`, `set_params(**get_params())` reproducibility |
| 2 | `_mrmr_setstate_defaults.py` + `_mrmr_class.py:2910-2952` | No explicit pickle schema-version field; compatibility inferred purely from key absence; hand-maintained roster | old-pickle-in-new-code works today by discipline only; new-pickle-in-old-code (downgrade) silently sets unrecognized attrs with no validation |
| 3 | `_mrmr_class.py:2943` | `__setstate__` calls `type(self)()`, a full re-entrant construction, on every unpickle, subject to worker-local hardware/env values leaking into restored state | joblib worker-side unpickle in `cross_val_score`/`GridSearchCV(n_jobs>1)` on heterogeneous hardware |
| 4 | `_mrmr_class.py:2852-2862` | `__repr__` override textually patches `BaseEstimator.__repr__`'s output and bypasses `print_changed_only` for `n_workers` | future sklearn repr-format change (fails closed, not corrupting); repr-diffing tools misread "changed vs default" for this one param |
| 5 | `_mrmr_class.py:194-2801` | none found | n/a — no mutable-default-argument or `get_params`/`clone` bug in the ~300-param constructor |

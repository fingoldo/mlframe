# Core MRMR class & fit orchestration — audit (2026-07-25)

Cluster **CORE_CLASS** = the `mrmr/` subpackage: `_mrmr_class.py` (4173 LOC — the ~300-param ctor,
`fit`/`_fit_body` orchestration, `__getstate__`/`__setstate__`, `__repr__`, the thin `transform` delegator),
its config/fit/transform mixins (`_mrmr_class_config.py`, `_mrmr_class_fit_helpers.py`,
`_mrmr_class_transform.py`), the tiny shared leaf (`_mrmr_class_shared.py`), the nested-pydantic-config
additive migration + `set_params` override (`_mrmr_config_dataclasses.py`), the ctor-param validation
allow-lists (`_mrmr_param_constants.py`), the legacy-pickle default roster (`_mrmr_setstate_defaults.py`),
and the package facade (`__init__.py`). Cross-checked against the prior wave
`mrmr_audit_2026-07-22/core_class.md` (findings CORE_CLASS-1..5).

## Prior-audit (2026-07-22) CORE_CLASS findings — verified status against current source

| Prior ID | Prior claim | Status now | Evidence |
|----------|-------------|-----------|----------|
| CORE_CLASS-1 (P0) | `store_params_in_object` `postfix` regression stored every ctor param as `self.<name>_param_` | **FIXED** | `_mrmr_class.py:3085` now passes `postfix=""` explicitly, with a WHY comment at 3082-3084. |
| CORE_CLASS-2 (P1) | `clear_fit_cache()` clears `_FIT_CACHE` unlocked | **FIXED** | `_mrmr_class_config.py:67-72` now acquires `_MRMR_FIT_CACHE_LOCK` (lazy import) around `len()`+`clear()`. |
| CORE_CLASS-3 (P1) | nested-config kwarg + `set_params` on a covered flat attr → `clone()` `RuntimeError` | **FIXED** | `set_params` is overridden (`_mrmr_config_dataclasses.py:288-300`, bound at `__init__.py:216`) to run `invalidate_stale_mrmr_configs` after `BaseEstimator.set_params`; a config whose fields disagree with the now-updated flat attrs is nulled, keeping `get_params()` self-consistent so `clone()` no longer trips its identity check. |
| CORE_CLASS-4 (P2) | `_restore_toggles_snapshot_and_raise` unpacked 9 thread-locals but restored only 5 | **FIXED** | `_mrmr_class.py:3702-3705` now restores all 9 (relaxmrmr/pid/cmi-perm/cpt added) via `_safe_restore`, with a "currently dormant, defensive" comment. |
| CORE_CLASS-5 (P3) | 100+ leftover audit-metadata comments across the cluster | **STILL OPEN** | Re-confirmed below as CORE_CLASS-3 (this wave); cleanup only partial. |

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| CORE_CLASS-1 | P2 | correctness (config-default drift) | `_mrmr_config_dataclasses.py:124` vs `_mrmr_class.py:2304-2306` | `HybridOrthScorersConfig.ensemble_scorers` defaults to `()` (empty), but the flat ctor param `fe_hybrid_orth_ensemble_scorers` defaults to the non-empty 5-tuple `("plug_in","ksg","copula","dcor","hsic")`. `apply_mrmr_config_objects` (`:242-243`) copies EVERY scorer field onto the matching flat attr whenever a `hybrid_orth_config` is passed, so passing even an all-defaults config silently overrides this one attr. | `m = MRMR(hybrid_orth_config=HybridOrthConfig())` then `m.fe_hybrid_orth_ensemble_scorers` == `()` (not the 5-scorer default a bare `MRMR()` carries). A downstream fit with `fe_hybrid_orth_default_scorer="ensemble"` then runs the rank-fusion ensemble over an EMPTY scorer set. This is the ONLY drifting field — every other mapped scorer/DCD/synergy/group/stability default was verified to match its flat ctor default, so a targeted fix (set the config default to the same 5-tuple) closes it. |
| CORE_CLASS-2 | P3 | sklearn contract (latent) | `_mrmr_config_dataclasses.py:257-285` (`invalidate_stale_mrmr_configs`) + `:288-300` (`mrmr_set_params`) | Setting a nested config object via `set_params` is silently discarded. `set_params` (via `BaseEstimator.set_params`) only `setattr`s `self.dcd_config`; it never expands its fields onto the flat attrs (that only happens in `__init__` via `apply_mrmr_config_objects`). `invalidate_stale_mrmr_configs` then sees the new config disagree with the unchanged flat attrs and nulls it. Net: the config is dropped AND the flats are untouched. | `m = MRMR(); m.set_params(dcd_config=DCDConfig(dcd_enable=False))`. Afterwards `m.dcd_enable` is still `True` (flat untouched) and `m.dcd_config` is `None` (invalidated) — the requested change vanished. Reachable via `GridSearchCV(MRMR(), {"dcd_config": [...]})`, which silently no-ops every candidate. The current nulling is deliberately clone-safe (prevents the CORE_CLASS-3 `RuntimeError`), so this is a documented-behavior gap, not a crash; a fuller fix would apply a set-via-`set_params` config's fields onto the flats. |
| CORE_CLASS-3 | P3 | house-convention (comment style) | `_mrmr_class.py` (many), `_mrmr_class_fit_helpers.py:189,450`, `_mrmr_class_config.py:22,328`, `_mrmr_config_dataclasses.py:27,289`, `_mrmr_class_transform.py:244`, `_mrmr_param_constants.py:6,15,61`, `_mrmr_setstate_defaults.py:317,322,352` | Leftover audit/process-metadata in comments/docstrings, which `CLAUDE.md`'s "Comment style (CRITICAL — repeated complaints)" forbids: audit-report filenames (`09_error_messages_ux.md` at `_mrmr_class_fit_helpers.py:189`, `_mrmr_class_config.py:328`, `_mrmr_class_transform.py:244`), finding IDs (`fix audit row FS-P2-1` at `_mrmr_param_constants.py:6,15`; `S-F5` at `_mrmr_config_dataclasses.py:27`; `CORE_CLASS-3 fix` at `:289`; `Critic2/E fix` at `_mrmr_param_constants.py:61`; `perf audit findings #4/#7/#8, 2026-07-17` at `_mrmr_class_config.py:22`), dangling scrubbed-ID fragments (`# 1 fix (loop iter 35):` at `_mrmr_class_fit_helpers.py:450`; `# 1:` at `:471`), and bare date stamps (`2026-06-09`/`2026-06-10` in `_mrmr_setstate_defaults.py:317,322,352` and dozens in `_mrmr_class.py`, e.g. `:508,536,561,872`). This is the still-open CORE_CLASS-5 from 2026-07-22. | Cosmetic; no behavior impact. Concrete cleanup targets listed. (NB: the many `Layer NN` references in `_mrmr_class.py`/`_mrmr_param_constants.py` are the FE-stack's own domain vocabulary, not audit metadata — leave those.) |

## Non-findings / confirmed-clean angles

- **`__getstate__`/`__setstate__` pickle correctness** (`_mrmr_class.py:3182-3262`): `__getstate__` strips the
  non-picklable `_fit_reentrancy_lock_` and stamps `_mrmr_schema_version`; `__setstate__` warns on a newer-schema
  downgrade, injects the legacy roster + every remaining ctor default (fresh-instance-sourced, cached). No runtime
  cache / lock / thread-local is baked into state. `_effective_n_jobs`/`_effective_parallel_kwargs`/
  `_effective_random_seed` resolve sentinels lazily so a cross-host unpickle re-resolves. Clean.
- **Thread-local snapshot/restore on mid-block raise**: both raise paths (`_mrmr_class.py:3720`, `:3774`) route
  through `_restore_toggles_snapshot_and_raise`, which now restores all 9 MI-correction thread-locals; the `finally`
  block (`:4024-4150`) restores the same 9 plus group-MI/DCD/cluster-aggregate/fast-search/screen-subsample/
  ctor-alias overrides via `_safe_restore`. Snapshot is taken at fit entry so nested/worker fits are not clobbered.
  Clean.
- **`clone`/`get_params`/`set_params` round-trip**: ctor stores every param unmodified (`postfix=""`); the
  `set_params` + `invalidate_stale_mrmr_configs` pairing keeps `get_params()` self-consistent for `clone()`. The
  config-field↔flat-attr map defaults were audited field-by-field: only `ensemble_scorers` drifts (CORE_CLASS-1);
  all DCD/synergy/group-aware/stability/fast-search fields and every other hybrid-orth field match their flat ctor
  defaults, so an all-defaults config round-trips cleanly.
- **`_FIT_CACHE` concurrency**: `clear_fit_cache()` now shares `_MRMR_FIT_CACHE_LOCK`; the process-wide in-flight
  counter (`_mrmr_class_fit_helpers.py:39-40,69-89`) is lock-guarded and re-arms the GPU breakers only on the 0→1
  transition. Per-instance `_fit_reentrancy_lock` (non-blocking) enforces no-concurrent-fit-on-same-object.
- **Security / injection**: none. No SQL/HTTP/eval/exec/subprocess on untrusted input; only own-state pickle;
  the one `os.environ` write (`_apply_fast_search_profile`) is now a thread-local Fourier-detect cap, not env.
- **Monolith split (AST unresolved-name hazard)**: mixins declare needed cross-MRO names as `TYPE_CHECKING`/
  ClassVar stubs; `_mrmr_class_shared.py` breaks the `_mrmr_class`↔`_mrmr_class_fit_helpers` cycle for
  `_mrmr_y_columns`. No obvious `Load`-context name unresolved at runtime in the read paths.

## Proposals (perf / refactor / test — not bugs)

1. **CORE_CLASS-1 fix + regression guard**: set `HybridOrthScorersConfig.ensemble_scorers` default to the same
   `("plug_in","ksg","copula","dcor","hsic")` 5-tuple as the flat ctor param, and add a meta-test that asserts, for
   every `(config_field -> flat_attr)` entry in `_CONFIG_ATTR_FIELD_MAPS` + the two hybrid-orth maps,
   `config_default == MRMR()._ctor_defaults()[flat_attr]`. That single parametrized test would have caught this and
   prevents future drift as either surface grows.
2. **CORE_CLASS-2 fix (optional)**: have `mrmr_set_params` expand a set-via-`set_params` nested config's fields onto
   the flat attrs (mirroring `apply_mrmr_config_objects`) BEFORE `invalidate_stale_mrmr_configs`, so
   `set_params(dcd_config=...)` / `GridSearchCV` over a config param actually takes effect, with a test fitting
   through `GridSearchCV(MRMR(), {"dcd_config":[DCDConfig(dcd_enable=False), DCDConfig(dcd_enable=True)]})`.
3. **CORE_CLASS-3 cleanup**: a low-risk comment pass folding each audit-metadata reference's real WHY into plain
   prose and deleting the ID/date/filename tokens (keep `Layer NN` domain terms). Natural to batch with proposal 1's
   edit since both touch this cluster.
4. **mypy note**: I did not re-run mypy this wave (the prior audit reported all 9 files clean via the shared cache);
   the CORE_CLASS-1/2/3 fixes above are runtime/comment changes, none of which should perturb typing, but a
   `mypy --cache-dir=../.mlframe_mypy_cache_shared` over the 9 files after any edit is the cheap confirmation.

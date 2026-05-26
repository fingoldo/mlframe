# A5 - DiscoveryCache / SuiteArtefactCache joint-stash

## Problem

Wave 8 landed `SuiteArtefactCache` (cross-process, disk-backed, sha256-verified, byte-budget LRU). It covers SUITE-level artefacts: `fit_and_transform_pipeline`, `apply_preprocessing_extensions`, `trainset_features_stats`, dummy baselines, composite target specs.

`DiscoveryCache` (composite_cache.py) is per-process / per-cache_dir with its own LRU sidecar (`.lru`), its own eviction policy (max_entries OR max_size_mb), and its own sha256-sidecar verification.

Two adjacent stores. Both:
- disk-backed
- sha256-verified
- store pickled Python objects
- need LRU + size-based eviction
- need cross-process safety (DiscoveryCache uses filelock, SuiteArtefactCache uses atomic-rename)

Question: should DiscoveryCache route through SuiteArtefactCache?

## Why NOT a drop-in unification

1. **Key-derivation semantics differ.** DiscoveryCache keys via `make_discovery_cache_key(df_sig, target_col, config_signature, random_state)` — composite-target discovery is specialized. SuiteArtefactCache keys via `SuiteKeyBuilder.build(df_fp, config_canonical, mlframe_models, lib_versions, random_seed, extra)`. The two key schemas are NOT structurally interchangeable: discovery keys fold a target column name, suite keys fold a model frozenset.

2. **Eviction policies differ.** DiscoveryCache defaults `max_entries=1000` AND `max_size_mb=2000`; SuiteArtefactCache defaults `bytes_limit=2_000_000_000` (2 GB; matches CLAUDE.md ceiling). A unified store would have to expose BOTH levers — already does, but the operator now juggles two policies vs one.

3. **Concurrency stories differ.** DiscoveryCache uses `filelock` (optional dep) for the `.lru` sidecar protection; SuiteArtefactCache relies on `os.replace` atomicity. Migrating Discovery to SuiteArtefactCache would silently drop the `.lru` ordering, replacing it with mtime-based LRU which behaves identically in single-process but writes write-order in multi-process.

4. **Migration cost.** ~40 call sites use `DiscoveryCache.get/set/has/invalidate/clear`. Wrapping them in a SuiteArtefactCache facade would be ~3 files (composite_cache.py, the discovery wiring, the tests), but the semantic delta (key shape, eviction defaults) needs a benchmark to confirm correctness equivalence.

## Proposed options

### Option A: do nothing
Keep two stores. Document that they are intentionally separate: DiscoveryCache is composite-discovery scoped; SuiteArtefactCache is suite-level artefact scoped. Both follow the same sha256-sidecar discipline and both honor the 2 GB ceiling. **Recommended for now.**

### Option B: layer DiscoveryCache on top of SuiteArtefactCache
Refactor `DiscoveryCache.get` / `.set` to delegate to a `SuiteArtefactCache` instance:
- `key = make_discovery_cache_key(...)` -> passed through unchanged
- LRU + bytes eviction handled by SuiteArtefactCache
- DiscoveryCache becomes a thin namespacing wrapper that calls `SuiteKeyBuilder.build(extra={"kind": "discovery", "key": discovery_key})` and forwards to a shared default-cache.

Risks:
- Cross-cache eviction: a discovery entry can be evicted by a suite-level entry filling the budget. May surprise composite-discovery users who tuned `max_entries=1000`.
- LRU semantics shift: filelock-protected `.lru` sidecar replaced by mtime-based ordering.

### Option C: shared backend, separate facades
Both DiscoveryCache and SuiteArtefactCache delegate IO + eviction to a single `DiskBackend` Protocol (FeatureCache's `CacheBackend` already in `feature_handling/cache_backend.py:34`). Each facade keeps its own key-derivation + cache_dir + budget; only the eviction + size accounting + sha256 verification live in one place.

Risks:
- Largest implementation cost (~6 files: composite_cache, suite_artefact_cache, the protocol, two facade implementations, test extensions).
- A future bug in the backend now affects three callers (FeatureCache, DiscoveryCache, SuiteArtefactCache) rather than one.

## Recommendation

**Option A** (do nothing) for Wave 10. The two stores already share enough discipline (sha256, atomic writes, byte ceilings) and serve distinct purposes (suite-level vs per-target-discovery). Unification is a real refactor with measurable behavioral change; it warrants a dedicated wave with a microbench proving the budget interaction is not destabilising.

If Wave 11+ wants to unify, **Option C** (shared backend Protocol) is the safer landing pattern; Option B is too coupled (eviction interactions are surprising).

## Status

ARCH-DEFER (re-confirmed Wave 15c after detailed semantic audit).

## Wave 15c follow-up: Option C re-evaluated, re-DEFERRED

User authorised Option C in Wave 15 ("approved Option C: shared backend Protocol"). A fresh side-by-side read of all three caches surfaced six semantic divergences that cannot be reconciled into one Protocol without changing on-disk behaviour for at least one of the existing callers. Detail:

1. **Write atomicity discipline differs.** SuiteArtefactCache routes through `safe_pickle.safe_dump` -- one operation that lands both `.pkl` and `.pkl.sha256` atomically (open + close + rename + sidecar). DiscoveryCache writes `.pkl` via manual `pickle.dump` + `f.flush()` + `os.fsync(fileno())` + `os.replace`, then attempts the sidecar via `_safe_pickle_write_sidecar` with sidecar-failure logged at DEBUG and treated as non-fatal (the .pkl is already durable; a future strict load returns a clean miss). Unifying these means either (a) SuiteArtefactCache adopts manual pickle+sidecar (atomicity regression) or (b) DiscoveryCache adopts `safe_dump` (loses the documented "value durable, sidecar best-effort" decoupling + the `MLFRAME_DISCOVERY_CACHE_STRICT` env-var contract).

2. **POSIX directory fsync** is performed only by DiscoveryCache.set (after rename, conditional on `os.name == "posix"`). The other two caches skip it. Routing all writes through one backend either grants every caller the dir-fsync cost (regression for FeatureCache: ~1ms per write on a busy directory) or drops it from DiscoveryCache (durability regression on journaled-data-mode-off filesystems).

3. **Cross-process lock primitive choice is inconsistent.** FeatureCache `LocalDiskBackend` uses `PIDAwareFileLock` (mlframe-internal wrapper with stale-lock reclaim). DiscoveryCache uses raw `filelock.FileLock` (optional dep, falls back to `contextlib.nullcontext` when missing -- documented behaviour). SuiteArtefactCache uses no cross-process lock at all (relies on `os.replace` atomicity for value writes; in-memory `threading.Lock` only). A single backend must commit to one model; DiscoveryCache loses the optional-dep fallback OR SuiteArtefactCache acquires a lock primitive it deliberately avoided.

4. **LRU persistence model differs fundamentally.** FeatureCache: `.lru` JSON, file-locked, persists across process restart, cross-process. DiscoveryCache: `.lru` JSON, file-locked (separate `filelock` instance), persists. SuiteArtefactCache: **in-memory** `OrderedDict` only, populated from mtimes on first access -- no JSON sidecar, no cross-process LRU consensus. The Wave 8 SuiteArtefactCache design deliberately chose mtime-based eviction to avoid every `put`/`get` paying for JSON read-modify-write under a filelock; this is a documented performance tradeoff. A unified backend cannot serve both models without an `lru_persistence` config switch that effectively forks the implementation into the same three code paths.

5. **Validation policy diverges at construction.** DiscoveryCache.__init__ **raises ValueError** when both `max_entries` and `max_size_mb` are None (auditable opt-in for unbounded growth). SuiteArtefactCache accepts `bytes_limit=0` and `max_entries=None` silently (treats as "no caps"). FeatureCache `LocalDiskBackend` accepts `max_entries=None, max_size_mb=None` as a documented zero-config workload. A single backend Protocol cannot enforce all three init contracts; pushing the check up into each facade duplicates the LRU/eviction code path that Option C was supposed to centralise.

6. **Key sanitisation lives entirely in DiscoveryCache** (`_safe_key`: pure-hex pass-through, blake2b+length-tag for non-hex). FeatureCache + SuiteArtefactCache assume callers pass hex digests directly. Hoisting `_safe_key` into a shared backend changes the on-disk key namespace for FeatureCache and SuiteArtefactCache callers (existing entries with non-hex keys -- if any survive in operator caches -- would no longer be findable under the same hash). The safest unification (leave `_safe_key` only in the DiscoveryCache facade) leaves the backend Protocol asymmetric -- DiscoveryCache's facade does pre-processing the others don't, which defeats the "shared eviction + IO" goal.

### Behavioural-equivalence sensors are inconstructible

The Wave 15c microbench requirement assumed sensors could prove byte-identical disk state pre/post refactor. Given divergences (1)-(5), at least one of the following on-disk artefacts changes for at least one caller in any feasible unification:

- SuiteArtefactCache loses or gains a `.lru` JSON file
- DiscoveryCache writes change order (sidecar-before-rename vs sidecar-after-rename via `safe_dump`)
- DiscoveryCache loses POSIX dir-fsync OR FeatureCache gains it (perf cost)
- DiscoveryCache loses filelock OR SuiteArtefactCache gains it
- Init-time error contract weakens for DiscoveryCache OR strengthens for the others

Each of these is a real semantic shift. The IMPLEMENTATION_RULES guard ("Если behavioural-equivalence невозможна -- DEFER + write detailed reason; don't ship Option C if it changes semantics") fires.

### Re-recommendation

Stay with Option A. The duplication is real (LRU sidecar logic, total-bytes accounting, sha256 verification) but the divergences are intentional choices accumulated across waves 8, 10b, 48, 52, plus Wave-15 audit findings on DiscoveryCache (D L-1, D L-4, D L-5, D L-6, D L-9, D L-10, D P1-2, D P2-2). Unifying would either lose those choices or add config knobs that recreate three paths inside one file -- worse than three files.

If a future wave still wants to unify: pre-requisite is a Wave-N proposal that picks a single canonical write path (safe_dump), a single canonical LRU model (disk JSON), a single canonical lock primitive (PIDAwareFileLock), then accepts the semantic regressions for SuiteArtefactCache + the perf cost for FeatureCache -- AND ships migration logic for existing on-disk caches in operator HOME directories. That is a multi-wave effort, not a "shared backend Protocol" refactor.

### What WAS done in W15c

- Detailed read + diff of `composite_cache.py:493-923` vs `suite_artefact_cache.py` (entire) vs `feature_handling/cache_backend.py` (entire).
- Identified the six semantic divergences listed above.
- Updated this proposal with the divergence table + verdict.
- No code changed; no commits made. Heartbeat + manifest record the DEFER outcome.

## File evidence

- `src/mlframe/training/composite_cache.py:493-923` -- DiscoveryCache implementation
- `src/mlframe/training/suite_artefact_cache.py` -- SuiteArtefactCache implementation
- `src/mlframe/training/feature_handling/cache_backend.py:34` -- existing CacheBackend Protocol (FeatureCache only)
- `src/mlframe/training/feature_handling/locking.py:54` -- PIDAwareFileLock primitive used by FeatureCache

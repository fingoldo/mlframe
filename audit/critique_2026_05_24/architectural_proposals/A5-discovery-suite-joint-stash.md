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

ARCH-DEFER (Wave 11+).

## File evidence

- `src/mlframe/training/composite_cache.py:493-923` -- DiscoveryCache implementation
- `src/mlframe/training/suite_artefact_cache.py` -- SuiteArtefactCache implementation
- `src/mlframe/training/feature_handling/cache_backend.py:34` -- existing CacheBackend Protocol (FeatureCache only)

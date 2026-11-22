# Audit 07: MRMR concurrency of FE stages (wave 2)

Scope: `_fe_stage_concurrency.py`, `_edge_pool.py`, `_fe_device_scheduler.py`, `_fit_context.py`, `_fe_deadline.py`, `_hybrid_orth_family_variants/_concurrent.py`, `_fe_family_timing.py`, lock sites in `_adaptive_nbins.py` and `_family_maxt_gate.py`. Read-only, Haiku model, nothing executed. Paths under `filters/`.

Tracker rows: AUD23-04, 05, 06, 47, 48.

1. Merge loses an attribute created by two stages in different folds. Severity high. Mechanism confirmed; occurrence hypothesis.
   - `_fe_stage_concurrency.py:132-133`, called from `_hybrid_orth_family_variants/_concurrent.py:104-127`. The docstring promises concatenation of a new attribute created by several stages, but the `created` concatenation happens only inside one `merge_stage_state` call; `_concurrent.py` calls it once per group with the same `base_state`.
   - Fold 1 (slot 0) sets `X` to its filled list; fold 2 (slot 3) also sees `X` as new (its copy created it), so line 133 `setattr` replaces the filled list with slot 3's. Deterministic fold-order logic, not a thread race. Needs a shared new name, such as an empty `*_features_` list created in one group and filled in another.

2. Rebuilt list discards earlier appends. Severity high if reached. Confirmed in code; occurrence hypothesis.
   - `_fe_stage_concurrency.py:120-124`: stage A appends; stage B's copy has a changed prefix, so it takes `setattr(target, name, list(value))`, dropping A's items, since B's copy was built from `base`, not the target.

3. Deletions are not merged. Severity medium. Confirmed. `_fe_stage_concurrency.py:126-129`: a key or set member removed by a stage survives in the target; only additions and changed values fold.

4. Shallow `private_copy` shares nested state across sibling stages. Severity medium. Mechanism confirmed; occurrence hypothesis.
   - `_fe_stage_concurrency.py:53-59`, `_concurrent.py:66`: only top-level list/dict/set attributes are copied. A nested container mutated in place is shared with no lock; its identity makes `_same` (line 71) return True, so the merge cannot see the change. `kw` in `_concurrent.py:66` passes shared objects (e.g. `_fe_family_on`) to every stage.

5. Slot pools have no nesting scope. Severity medium. Hypothesis.
   - `_fe_device_scheduler.py:113-125`, `:129-140`: pools keyed `(device_id, i)`. `run_groups_concurrently` reuses keys 0-2 (`_concurrent.py:97`). If reached from inside a cascade stage holding keys 0-2 on the same device, the inner `free_all_blocks()` (line 140) and `set_limit` (line 124) act on the outer pool. The `set_limit` read-modify-write (line 124) is outside `_SLOT_POOLS_LOCK`, so two concurrent fits can race. Call graph not checked.

6. Edge-wave counter restore can drop decrements. Severity low. Hypothesis for nesting.
   - `_edge_pool.py:20-37`; unlocked read at `_adaptive_nbins.py:689-691`. Nested `concurrent_stages` exits restore a stale `prev`, discarding outer `stage_finished()` decrements: too many edge threads (under-share), not a wrong result. The unlocked read is benign under the GIL.

7. Lap timers leak per thread. Severity low. Growth confirmed; leak into gates is hypothesis.
   - `_fe_family_timing.py:111`: `timers.append(self)` never removes entries. Laps opened on edge-pool workers are never closed by `close_thread_laps()`, which only the stage worker calls (`_fe_stage_concurrency.py:208`). Applies only if gates run inside `map_on_edge_pool`, not verified.

8. Failure path skips the merge. Severity low. Confirmed behaviour.
   - `_fe_stage_concurrency.py:233-237`: when a kept stage fails, the code raises before `merge_stage_state`; successful sibling state is discarded, leaving the estimator at its pre-wave state. Sequential run would keep earlier stages. A divergence from sequential semantics, not a race.

## Checked and fine
- `stage_finished()` runs in the `finally` of `_run` (`_fe_stage_concurrency.py:217`); cleanup errors handled (`:208-216`).
- Deadline and fit context captured on the calling thread and published in each worker (`:193-202`, `_edge_pool.py:72-78`).
- `_EDGE_POOL` creation under `_EDGE_POOL_LOCK`, retired pools kept alive (`_adaptive_nbins.py:629-637`).
- `_family_maxt_gate.py:352-373` holds one lock across label-bit cache check and insert.
- `_fe_family_timing` counters updated under `_LOCK`.

## Not verified
- Whether `_fit_locals()` (`_fit_context.py:21`) misses any `threading.local` in `info_theory/shared`.
- Whether `rng` objects passed into `_label_bits_block` are shared across threads.

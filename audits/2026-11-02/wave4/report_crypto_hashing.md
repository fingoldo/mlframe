# Wave 4 audit 2026-11-02: weak cryptography, hashing and randomness

Scope: `src/mlframe`, read-only, Opus 5.5 low effort (one targeted grep plus three full reads). Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| CRY5-1 | Low | `utils/disk_cache.py:101-120` (xxh3 key of a persistent pickle cache; object dtype hashes only a summary) | DOC |
| CRY5-2 | Low | four in-process memo keys `hash(arr.tobytes())`: `info_theory/_cmi_resident_cache.py:52`, `_fe_cmi_redundancy_gate.py:239`, `_fe_accuracy_gate.py:258`, `_fe_resident_operands.py:107` | RESOLVED |
| CRY5-3 | Low | salted `hash()` in seeds and fingerprints: `evaluation.py:504,571`, `_confirm_predictor.py:226`, `_confirm_predictor_helpers.py:116`, `composite/ensemble/__init__.py:405`, `_dataset_cache_fingerprint.py:180` | REJECTED |
| CRY5-4 | Info | truncated digests `composite/provenance.py:103` (48 bits), `_fe_family_budget.py:229` (64 bits) | DOC |
| CRY5-5 | Info | `data/datasets/replay.py:96` compares non-secret content hashes with `==` | REJECTED |

Checked and safe: sha1 with `usedforsecurity=False` for numba cache invalidation; crc32 only derives seeds; memo keys use blake2b; temp and token names use `uuid4`; `random` and `numpy.random` appear only in statistical code. No secret comparison exists, so `hmac.compare_digest` has no site.
Open leads: whether `composite_id` gates a load; whether `_dataset_cache_fingerprint.py:180` is persisted.

Dispositions after reading the code (2026-11-02):
- CRY5-2: three of the four sites now use `hash_array_content` (128-bit, same digest as the disk cache): `tests/feature_selection/filters/test_memo_keys_use_content_digest.py`. The fourth (`_fe_resident_operands.py:107`) is the documented fallback of a deliberately fast non-copying hash; a collision there costs one stale device operand and its comment states the trade-off, so it is left.
- CRY5-3: REJECTED. The seed sites hash tuples of integers (variable indices and `random_seed`), which are deterministic across processes; `composite/ensemble/__init__.py:405` and `_dataset_cache_fingerprint.py:180` build in-process memo keys that are never persisted or compared across processes (`_hash_sampled_rows` says its key falls back to an `id()`-based one), so the per-process hash salt cannot matter.
- CRY5-1 and CRY5-4 stay DOC: the disk-cache key is non-adversarial by design; the object-dtype branch hashes a summary (its docstring says so), and `composite_id` is a label.

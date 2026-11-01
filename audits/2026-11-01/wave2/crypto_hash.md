# Wave 2 audit: cryptographic and hash primitive choice (CRY3)

Read-only pass over `src/mlframe`. Scope: md5/sha1/sha256/blake2b uses, uuid and os.urandom use, stdlib `random` use, and
seeds derived from hashes. Earlier tracker rows (2026-10-31 and 2026-11-01) were checked and not re-reported. Cross-references
are noted where a primitive-level point touches an already-recorded item.

## Summary

No security-relevant use of a weak hash, a predictable token, or a non-CSPRNG secret was found. Every blake2b, sha256, sha1
and md5 call either keys a cache or dedup over trusted data, seeds a reproducible RNG, or is a bench/checksum that is not a
security boundary. The one real gap is that integrity sidecars are unkeyed digests; that is recorded as DESER-02/06 and is
cross-referenced here, not counted as new.

Counts: high 0, medium 0, low 1 (CRY3-2), info 4 (CRY3-1, CRY3-3, CRY3-4, CRY3-5). Cross-reference only: 1.

## Findings

### CRY3-1 (info): docstring calls an unkeyed hash "keyed"
- Location: `src/mlframe/data/datasets/_rng.py:21`
- Primitive: `hashlib.blake2b` used without a key, to derive RNG entropy from path segments. The docstring says it is "a keyed,
  standard-library cryptographic digest".
- Protects: reproducible seeding of synthetic datasets. No secret is involved.
- Attacker controls input: no (segment names are code-defined).
- Proposed disposition: **DOC**. Correct the wording to "an unkeyed, standard-library digest with explicit output width". The
  code is correct; only the claim is wrong.

### CRY3-2 (low): salted unkeyed blake2b used as a blinding order
- Location: `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_blinding.py:59`
- Primitive: `blake2b(f"{salt}::{name}", digest_size=8)` picks the order of arm labels in a benchmark table.
- Protects: the author's own arm is not identified while reading a table (anti-bias mechanism). It is not a secrecy control:
  the salt is stored in the `.blinding.json` mapping file beside the output, so anyone with that file can recompute the mapping.
  With a small known arm set, the mapping can be recovered by brute force in any case.
- Attacker controls input: no. This is a benchmark reporting aid.
- Proposed disposition: **REJECTED** as a security finding (not a confidentiality boundary; the module docstring already
  states the goal is bias mitigation). Keep the low rating only because the docstring could be read as stronger than it is.

### CRY3-3 (info): unkeyed blake2b-128 as content address for pickled cache entries
- Location: `src/mlframe/utils/disk_cache.py:93-98` (`_hasher`), `:424` (`tmp_{uuid4}` temp name)
- Primitive: unkeyed blake2b, 16-byte digest, for keys and content addressing of cached pickles.
- Protects: cache identity over trusted computed values. A collision requires finding a 128-bit collision, which is not
  feasible; an attacker would also need write access to the cache directory, and that access already allows arbitrary pickle
  content to be planted (a deserialization issue, see the 2026-10-31 DESER rows).
- Attacker controls input: the data can be chosen by an attacker in some pipelines, but the key is a content hash, so a chosen
  input cannot make two different contents share a key without a collision attack.
- Proposed disposition: **REJECTED** for the hash itself (no collision risk at 128 bits; the real trust boundary is the pickle
  file, covered by DESER rows).

### CRY3-4 (info): non-security md5/sha1 uses are flagged correctly
- Locations: `src/mlframe/_numba_cache_deps.py:199` (`hashlib.sha1(usedforsecurity=False)`, dependency stamp for numba cache);
  `src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_gated_recall_bench.py:66`
  (`hashlib.md5(..., usedforsecurity=False)`, bench label); `src/mlframe/reporting/_benchmarks/profile_suite_render.py:52`
  and `bench_async_render_suite_ab.py:46` (sha1 for output-equality comparison in benchmarks).
- Primitive: md5/sha1 with `usedforsecurity=False`, so no FIPS-mode failure and no security claim.
- Protects: cache invalidation and benchmark equality only.
- Attacker controls input: not relevant (no trust boundary).
- Proposed disposition: **RESOLVED** (already safe; the `usedforsecurity=False` flag is present on every md5/sha1 call in
  `src`, verified by grep). A `FIPS` build would not break these sites.

### CRY3-5 (info): token-like identifiers come from CSPRNG or are non-security
- Locations: `src/mlframe/reporting/_shared_arrays.py:58` (shared-memory name prefix: pid + `uuid.uuid4().hex[:8]`);
  `src/mlframe/utils/rng_scope.py:39` (`os.urandom(8)` seed draw); `src/mlframe/training/feature_handling/fingerprint.py:179`
  (`session_id` uuid4); `src/mlframe/training/io.py:165`, `src/mlframe/utils/disk_cache.py:424`,
  `src/mlframe/feature_engineering/transformer/_key_bank.py:247`, `src/mlframe/utils/_param_oracle_store.py:82`,
  `src/mlframe/feature_selection/filters/_fe_family_budget.py:265` (uuid4 in temp file names).
- Primitive: `uuid.uuid4()` (os.urandom backed) and `os.urandom`.
- Protects: collision avoidance for temporary files and segments; no access token is derived from them.
- Attacker controls input: no.
- Proposed disposition: **REJECTED** (not a security token; uuid4 is cryptographically random anyway, so this is not a weak
  source even if a name were treated as a secret).

### CRY3-6 (info, REJECTED): stdlib `random` is used only for non-security sampling
- Locations: `src/mlframe/utils/_param_oracle.py:518` (`random.Random()` default for a sampler), `src/mlframe/feature_selection/wrappers/_helpers.py:226`
  (`random.choice` for jitter, annotated `nosec B311`), `src/mlframe/feature_selection/_benchmarks/bench_eng_base_set_cache.py`
  (benchmark generator), `src/mlframe/utils/misc.py:82-94` (global seed setter for reproducibility).
- Primitive: Mersenne Twister, non-cryptographic.
- Protects: sampling and reproducibility. None of these outputs is a token, a key, a nonce, or a password.
- Attacker controls input: no.
- Proposed disposition: **REJECTED** (not a risk; no secret is generated from stdlib `random` or numpy RNG anywhere in `src`).

## Cross-reference (not new)

- Unkeyed sha256 sidecars for model bundles and pipeline dumps (`src/mlframe/training/io.py:385`, `src/mlframe/estimators/pipelines.py:109-110`)
  give integrity against corruption only. An attacker with write access to both the file and its sidecar defeats them, and
  the code comment at `pipelines.py:110` says so. The primitive-level fix is an HMAC with a key outside the directory. This
  is already recorded as DESER-02 (FUTURE) and DESER-06 (FUTURE); no new row.

## Scope limits

- Grep-based. Every `hashlib` call site was listed, but only the security-relevant ones were read in full.
- Did not audit `pyutilz` (external) or the GPU/CuPy helper hashes beyond their call sites.
- Test files were not included in the sweep.

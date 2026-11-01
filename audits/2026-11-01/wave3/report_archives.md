# Wave 3 audit 2026-11-01: archives and decompression

Scope: `src/mlframe` (read-only). Class: archive extraction (tarfile, zipfile, shutil.unpack_archive), decompression without an output cap
(zstd, gzip, bz2, lzma, zlib), archive formats used for bundles and caches, entry names used in paths. Builds on
`audits/2026-10-31/_TRACKER.md`, `audits/2026-11-01/_TRACKER.md` and `audits/2026-11-01/wave2/_TRACKER.md`.

The auditor could not write this file itself; the findings below are its returned text, recorded here unchanged in substance.

## Summary

6 findings: 0 High, 0 Medium, 2 Low, 4 Info. The package has no archive extraction at all: no `tarfile`, `extractall`, `unpack_archive`,
`gzip`, `bz2` or `lzma`, and `zipfile` appears only to catch `BadZipFile`. Zip slip, tar filters, symlink or hardlink entries and entry-name
path joins cannot occur.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| ARC4-1 | Info | whole package (no extraction code; `zipfile` imported only at `training/feature_handling/cache.py:52`) | RESOLVED |
| ARC4-2 | Low | `training/feature_handling/cache.py:557, 591, 593` | DOC |
| ARC4-3 | Low | `feature_engineering/transformer/_key_bank.py:179-181` | REJECTED |
| ARC4-4 | Info | `training/_bounded_zstd.py:17-27` | DOC |
| ARC4-5 | Info | `training/io.py:613-616`, `training/core/_metadata_loader.py:29`, caps at `_bounded_zstd.py:46-47, 83-84` | RESOLVED |
| ARC4-6 | Info | `training/_io_save.py:255`, `training/core/_setup_helpers_metadata.py:211` | RESOLVED |

## Findings

### ARC4-1 No archive extraction exists (Info, RESOLVED)
No code extracts an archive or opens a member by name, so the zip slip, tar filter, link-entry and entry-name classes have no surface.

### ARC4-2 Feature-cache `.npz` members have no size cap (Low, DOC)
Source: a feature-cache `.npz` on local disk. `np.load(path, mmap_mode="r")` opens the zip, then `npz["value"]` or `npz[npz.files[0]]`
inflates a member and `np.array(...)` materialises it, with no cap on member size, member count or expanded size. Member names are dict
keys only, never paths. Scenario: someone with write access to the cache directory plants a small deflated `.npz` whose member expands to
many GB, and the read exhausts memory. `allow_pickle` defaults to False, so this is a memory denial of service, not code execution.
Disposition DOC: the cache directory is a trusted local store, the same trust level as the `safe_load` pickle fallback. A cheap hardening is to
check `ZipInfo.file_size` against a ceiling before inflating.

### ARC4-3 `.npy` loads from the key-bank cache (Low, REJECTED)
`np.load` with `allow_pickle` off and no size check; the header's declared shape drives the allocation. A truncated or lying file raises, a
genuinely huge file just loads. The fingerprint is a caller-built path component, not archive input. Not an archive, pickle is off, cache is
local and trusted.

### ARC4-4 zstd ceiling can exceed RAM on small hosts (Info, DOC)
`training/_bounded_zstd.py:17-27`: the default ceiling is 64 GiB, and `MLFRAME_MAX_DECOMPRESSED_BYTES` set to 0 or a negative value disables
the cap (documented in the docstring). On a host with less than 64 GiB RAM the default does not stop an out-of-memory failure from a
decompression bomb. Already tracked as SEC-06; operators on small hosts should lower the variable.

### ARC4-5 Every zstd read is bounded (Info, RESOLVED)
Every zstd read goes through `BoundedReader` or `decompress_bounded`, and the streaming cap counts bytes across concatenated frames, so
multi-frame and single-frame bombs are bounded the same way.

### ARC4-6 Compression-only write paths (Info, RESOLVED)
`training/_io_save.py:255` and `training/core/_setup_helpers_metadata.py:211` only use `ZstdCompressor`.

## Checked with no finding

- `tarfile`, `shutil.unpack_archive`, `extractall`, `gzip`, `bz2`, `lzma`, `zlib` and `ZstdDecompressor` outside `_bounded_zstd.py` and `io.py`: not used to extract or decompress.
- `zlib` is used only as `zlib.crc32` for RNG seeding (`feature_selection/filters/_binned_agg_linear_usability.py:161, 212`).
- zstd elsewhere is parquet compression (`training/_fixed_splits.py:229`, `training/preprocessing.py:499`, `training/utils.py:965`) and `_benchmarks` scripts (excluded).
- `joblib.load` and `dill.load` calls are guarded by `validate_trusted_path`; not audited further here, since they belong to another class.

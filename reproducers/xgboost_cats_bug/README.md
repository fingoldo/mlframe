# XGBoost silent process kill with sparse polars Categorical codes

**Affects:** XGBoost 3.2.0, `tree_method="hist"`, `enable_categorical=True`
**Platform:** Windows 10 Pro 10.0.19045, 128 GB RAM, 237 GB pagefile
**Symptom:** process exits silently with code `3221226505` (`0xC0000005`,
`STATUS_ACCESS_VIOLATION`). No Python traceback, no C++ exception, no stderr.
**Confirmed on Linux:** same over-allocation occurs, process does not crash
(glibc commits pages lazily), but RAM usage explodes and bin indexing is wrong.

---

## TL;DR

`xgboost::common::AddCategories` in `src/common/quantile.cc` allocates the
per-feature cut-values buffer with `max_category_code + 1` entries instead of
`n_unique` entries. A single `polars.Categorical` column where one value has a
physical code of `2_526_058` while `n_unique = 51` causes XGBoost to allocate
`~10 MB` for what should be a `~200 B` buffer. On Windows this over-allocation
corrupts the heap and kills the process.

**Workaround:** cast the column to `pl.Enum(sorted(unique_values))` before
fitting. `pl.Enum` enforces compact codes `[0..n_unique-1]` — XGBoost allocates
the correct buffer size.

**Fix:** compact physical codes to `[0..n_unique-1]` inside `AddCategories`
before building the cut-values buffer. See section 4 and
[`fix_add_categories_compact.cc`](./fix_add_categories_compact.cc).

---

## 1. How sparse codes arise in polars

This is not a contrived edge case. The exact production sequence:

1. `polars >= 1.19` makes the global `StringCache` permanently enabled —
   `disable_string_cache()` is a no-op in 1.33+. The cache assigns a physical
   code to every new unique string, incrementing a global counter.
2. Casting a high-cardinality string column (`skills_text`, ~2.5 M unique
   values) to `pl.Categorical` fills the cache with codes `0..2_525_991`.
3. A low-cardinality neighbour column `category` (50 distinct strings) is cast
   to `pl.Categorical` next. Its strings land at codes `2_525_992..2_526_041`.
4. `fill_null("__MISSING__")` on `category` registers one more string —
   `"__MISSING__"` — at code `2_526_058`.

End state for the `category` column:

| physical code | string |
|---|---|
| 2_525_992 | `Digital Marketing` |
| 2_525_993 | `Virtual Assistance` |
| ... | 48 other category strings |
| **2_526_058** | `__MISSING__` |

`n_unique = 51`. `max_physical_code = 2_526_058`.
XGBoost receives this column and calls `AddCategories` with a set of 51 floats
whose maximum value is `2_526_058`. The bug then fires.

---

## 2. C++ root cause — `AddCategories` in `src/common/quantile.cc`

```cpp
// xgboost/src/common/quantile.cc  (~lines 389-402, simplified)
auto AddCategories(std::set<float> const &categories, HistogramCuts *cuts) {
  if (categories.empty()) return InvalidCat();
  auto &cut_values = cuts->cut_values_.HostVector();
  auto max_cat = *std::max_element(categories.cbegin(), categories.cend());
  CheckMaxCat(max_cat, categories.size());
  for (bst_cat_t i = 0; i <= AsCat(max_cat); ++i) {
    cut_values.push_back(i);   // BUG: loop iterates max_cat+1 times, not categories.size()
  }
  return max_cat;
}
```

The loop runs `max_cat + 1` times regardless of how many distinct categories
were observed. With `max_cat = 2_526_058` and `n_unique = 51`:

| quantity | value |
|---|---|
| `categories.size()` | 51 |
| `max_cat` | 2 526 058 |
| loop iterations | 2 526 059 |
| bytes pushed | `2 526 059 × 4` ≈ **9.64 MB** |
| bytes needed | `51 × 4` = **204 B** |
| over-allocation factor | **~47 000×** |

### 2.1 Why the existing guard does not help

`src/common/categorical.h` rejects codes above `2^24 = 16_777_216`:

```cpp
constexpr inline bst_cat_t OutOfRangeCat() {
  return static_cast<bst_cat_t>(16777217) - 1;
}
```

`2_526_058 << 2^24`, so `CheckMaxCat` passes silently and the loop proceeds.
Lowering the threshold is not the fix — even a max_cat of `100_000` produces
a 400 KB buffer for 51 categories, still 8 000× waste.

### 2.2 Why Windows crashes and Linux does not

On Windows with `IterativeDMatrix` the ~10 MB allocation occurs repeatedly
across sketch-merge batches. The allocator places a cut-values buffer adjacent
to a page-guard region; a subsequent `push_back` overruns it and SEH fires
`0xC0000005`. No Python-level exception is raised — the OS kills the process
instantly.

On Linux glibc the over-allocated mapping is committed lazily. The process
survives but wastes ~10 MB of heap per categorical feature and may produce
wrong predictions if histogram bins are indexed past the legitimate range.

### 2.3 Why the crash is machine-dependent

The crash requires a specific heap layout at the moment of the `AddCategories`
allocation. On a 128 GB Windows machine that has loaded a large multi-column
parquet (~760 MB of Arrow buffers) the heap is fragmented enough that the
over-allocation hits a guard page. On machines with less RAM or a different
allocation history the same over-allocation succeeds silently. The underlying
bug (wrong buffer size) is present on all machines.

---

## 3. Reproducers

### 3.1 Pure-synthetic standalone reproducer (no real data needed)

[`repro_xgb_synthetic_v2.py`](./repro_xgb_synthetic_v2.py) — confirmed to
crash on Windows 10 / 128 GB RAM / polars 1.33.1 / xgboost 3.2.0.

```
python repro_xgb_synthetic_v2.py
```

What it does:

1. Generates 2 526 059 unique ASCII strings with heavy-tailed lengths
   (p50 ≈ 40 chars, p99 ≈ 4 800 chars) using parallel worker processes and
   saves them to a temp parquet. Loading via parquet — not from a Python list —
   is essential: it replicates the Arrow buffer allocation pattern of a real
   parquet load, which is what produces the heap state that makes Windows crash.
2. Casts the loaded strings to `pl.Categorical`, priming the global StringCache
   with 2 526 059 entries (keeps the series alive to prevent cache eviction in
   polars >= 1.40).
3. Builds a `category` column (51 unique strings) and casts it through the
   polluted cache → physical codes land above 2 526 000.
4. Calls `fill_null("__MISSING__")` → `__MISSING__` gets code 2 526 109.
5. Calls `XGBClassifier.fit()` → process is silently killed.

```
python repro_xgb_synthetic_v2.py --workaround
```

With `--workaround` the script casts `category` to `pl.Enum(sorted_uniques)`
after `fill_null`. Physical codes collapse to `[0..50]`, XGBoost allocates
204 bytes instead of 10 MB, fit completes normally.

Generation takes ~115 s on a 16-core machine. The temp parquet is deleted
afterwards; use `--keep-parquet` to reuse it on subsequent runs.

### 3.2 Real-data bundle (requires production parquet)

If the crash does not reproduce on your machine with the synthetic script (the
heap-layout sensitivity means it may not), a bundle built from real data can
be used. Run [`dump_raw_crash_bundle.py`](./dump_raw_crash_bundle.py) on the
machine that holds the parquet:

```
python dump_raw_crash_bundle.py --parquet path/to/jobs_details.parquet --out-dir D:\Temp\xgb_bundle
```

Then copy `D:\Temp\xgb_bundle\` and run the generated `reproduce.py`:

```
python reproduce.py          # expected: silent kill 0xC0000005
python reproduce.py --workaround  # expected: FIT_OK
```

The bundle contains two files:
- `skills_text_uniques.parquet` — all unique `skills_text` values from the
  full dataset (the cache-pollution source).
- `crash_slice.parquet` — 311 168 rows of `category` only (the XGB input).

### 3.3 Diagnostic output from a confirmed crash run

```
polars 1.33.1, xgboost 3.2.0, platform=win32
generating 2_526_059 synthetic primer strings...
  saved ...synthetic_skills.parquet (761.6 MB) in 115.8s
loading primer parquet and priming StringCache...
  StringCache primed (2_526_059 entries) in 3.1s
after category cast: n_unique=51, codes_max=2526108
after fill_null:     n_unique=51, codes_max=2526109

fitting XGB train=(211168, 1) val=(100000, 1) -- expect silent kill (0xC0000005) on Windows
[process exits here with rc=3221226505, no further output]
```

---

## 4. Proposed fix — compact codes before sketching

The root cause is that `AddCategories` treats the physical code value as a bin
index. The fix: build a compact `code → rank` mapping once per feature so the
cut-values buffer has exactly `categories.size()` entries.

```cpp
// BEFORE: buffer sized by max_cat+1
for (bst_cat_t i = 0; i <= AsCat(max_cat); ++i) {
  cut_values.push_back(i);
}

// AFTER: buffer sized by n_unique; downstream indexing uses rank, not code
bst_cat_t rank = 0;
for (float cat : categories) {           // std::set is sorted
  cut_values.push_back(cat);             // preserve physical code for serialisation
  code_to_rank[AsCat(cat)] = rank++;
}
```

Memory impact on this bug's trigger case: `9.64 MB → 204 B` per feature.

Impact on well-formed inputs (compact codes `[0..k-1]`): `code_to_rank[i] == i`
for all `i` — identity mapping, no behavioural change, saved models remain
byte-compatible.

Full patch with unit tests, serialisation notes, and a list of the ~6
downstream indexing sites that need a one-line update is in
[`fix_add_categories_compact.cc`](./fix_add_categories_compact.cc).

Three options were considered:

| option | verdict |
|---|---|
| **1. Compact codes (this patch)** | Preferred. Fixes root cause, O(k log k), zero regression on valid inputs. |
| 2. Size cuts by `n_unique`, keep physical-code indexing | Fixes allocation but leaves ~6 indexing sites inconsistent. |
| 3. Lower `OutOfRangeCat()` threshold | Breaking change; does not fix sub-threshold sparse cases. |

---

## 5. Workaround (confirmed in production)

Cast every `pl.Categorical` column to `pl.Enum(sorted(unique_values))` before
passing it to XGBoost. `pl.Enum` enforces compact physical codes by
construction — no over-allocation, no crash.

```python
for col in cat_columns:
    uniques = sorted(df[col].unique().drop_nulls().to_list())
    df = df.with_columns(pl.col(col).cast(pl.Enum(uniques)))
```

---

## 6. Files in this directory

| file | purpose |
|---|---|
| `repro_xgb_synthetic_v2.py` | Standalone synthetic reproducer — no real data needed |
| `dump_raw_crash_bundle.py` | Dumps a minimal real-data bundle from a production parquet |
| `fix_add_categories_compact.cc` | Proposed C++ patch with tests and migration notes |
| `repro_xgb_minimal_for_upstream.py` | Parquet-based reproducer (requires real data, 2 columns) |
| `repro_xgb_exact_prod_state.py` | Reconstructs exact prod cache state from extracted dict |
| `repro_xgb_synthetic_realistic.py` | Earlier synthetic attempt (variable-length natural strings) |
| `bisect_*.py` | Scripts used to bisect row threshold and column set |
| `dump_*.py` | Other dump/IPC experiments from the bisection campaign |
| `extract_crash_state.py` | Extracted physical code → string mapping from prod memory |
| `trace_category_codes.py` | Traces category code changes through the pipeline |

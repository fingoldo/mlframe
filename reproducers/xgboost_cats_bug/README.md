# XGBoost access violation on sparse categorical physical codes

**XGBoost:** 3.2.0
**Platform:** Windows 10 Pro (10.0.19045), 128 GB RAM
**Trigger:** `polars.Categorical` columns whose physical codes are sparse (max code >> n_unique)
**Symptom on Windows:** silent process termination with exit code `3221226505` (`0xC0000005`, `STATUS_ACCESS_VIOLATION`). No Python traceback, no C++ exception, no stderr output.
**Symptom on Linux:** same allocation happens, but the page just gets committed; no visible crash. The bug is still there (wasted memory + garbage bin values), it just does not fault.

## TL;DR

`xgboost::common::AddCategories` (in `src/common/quantile.cc`) allocates the per-feature cut-values buffer with size proportional to `max_category_code + 1` rather than `unique_count`. A `polars.Categorical` column whose dictionary contains a code like `2_526_058` for just one of 89 strings makes XGBoost reserve a `~10 MB` array to represent 89 categories. On Windows, this specific allocation pattern occurring inside `IterativeDMatrix`'s batched sketching triggers a heap / page-guard corruption and the process dies with `0xC0000005`.

Fix: compact the physical codes (`[0, 88, 2_526_058]` -> `[0, 1, 2]`) before sketching, so the cut-values buffer is sized by actual cardinality.

---

## 1. How a sparse dictionary occurs in polars

This is not a contrived edge case. Here is the canonical path that produced it in production:

1. `polars >= 1.19` makes the global `StringCache` permanently enabled — `disable_string_cache()` is a no-op in 1.33+.
2. A wide parquet with many string columns is cast to `Categorical` in a single `with_columns([...])`. One of those columns (`skills_text`) has ~2 million unique strings; casting it registers ~2M entries in the global cache.
3. A neighbour column `category` (89 distinct strings) is then resolved against the now-polluted cache. Any value added *after* cache pollution — for instance through `fill_null("__MISSING__")` — is assigned a physical code far above 88.

End-state dictionary for a 9 M-row production frame:

| physical code | string |
|---|---|
| 0 | `Digital Marketing` |
| 1 | `Virtual Assistance` |
| ... | 86 other real strings, codes 2..87 |
| 87 | `Grant & Proposal Writing` |
| **2 526 058** | `__MISSING__` |

`n_unique = 89`. `max_physical_code = 2_526_058`. That single sparse code is enough to destabilise XGBoost.

The reproducers in this directory (`repro_xgb_minimal_for_upstream.py`, `repro_xgb_exact_prod_state.py`, etc.) isolate the polars side of this.

---

## 2. Where XGBoost mishandles it — C++ root cause

File: [`src/common/quantile.cc`](https://github.com/dmlc/xgboost/blob/v3.2.0/src/common/quantile.cc), function `AddCategories`. The relevant slice:

```cpp
// xgboost/src/common/quantile.cc  (simplified, ~lines 389-402)
auto AddCategories(std::set<float> const &categories, HistogramCuts *cuts) {
  if (categories.empty()) {
    return InvalidCat();
  }
  auto &cut_values = cuts->cut_values_.HostVector();
  auto max_cat = *std::max_element(categories.cbegin(), categories.cend());
  CheckMaxCat(max_cat, categories.size());
  for (bst_cat_t i = 0; i <= AsCat(max_cat); ++i) {
    cut_values.push_back(i);       // <-- (A)  loop bound is max_cat, not categories.size()
  }
  return max_cat;
}
```

The single offending line is marked `(A)`. `cut_values` is grown up to `max_cat + 1` entries, regardless of how many *distinct* categories were actually observed.

With our dictionary above:

* `categories.size()` = 89
* `max_cat` = `2 526 058`
* iterations of the loop = `2 526 059`
* bytes appended: `2 526 059 * sizeof(float)` ≈ **9.64 MB**

The 87 real codes in `[0..88]` plus the sentinel at `2 526 058` produce an array of ~2.5 M floats where only 89 slots are ever indexed meaningfully. The remaining ~99.996 % of the buffer is left as uninitialised padding that the histogram code will later index into.

### 2.1 The guard that fails to catch this

`src/common/categorical.h` does have an upper bound:

```cpp
// xgboost/src/common/categorical.h:34
constexpr inline bst_cat_t OutOfRangeCat() {
  return static_cast<bst_cat_t>(16777217) - 1;   // 2^24 == 16_777_216
}
```

`CheckMaxCat` rejects `max_cat >= 2^24`. Our `2 526 058` is well below that threshold, so the check passes and the oversized `push_back` loop proceeds.

### 2.2 Why the threshold is not the real problem

`OutOfRangeCat() == 16_777_216` was chosen to reject genuinely insane category codes. Lowering it does **not** fix the root cause: even a max_cat of 1 M produces a 4 MB cut-values buffer for a column with 89 real uniques — 99 % waste, and on Windows the specific allocation size / pattern interacting with `IterativeDMatrix`'s batched sketching still corrupts the heap.

The real defect is that `AddCategories` conflates *physical code value* with *bin index*. Bin count should be `categories.size()`. Bin identity should be derived from a `code -> rank` mapping built once per column, not from the raw physical code.

### 2.3 Why Windows dies and Linux silently misbehaves

On Windows, with `IterativeDMatrix` streaming batches, the ~9.6 MB allocations happen repeatedly across sketch merges. The allocator ends up placing a cut-values buffer adjacent to a page-guard region; a subsequent `push_back` overruns it and SEH `0xC0000005` terminates the process. No Python-level exception is raised because the default abort handler never runs — the OS just kills the process.

On Linux glibc the same oversize buffer is committed lazily; writes stay inside the mapping; the fit completes but subsequent predictions may produce garbage, and RAM usage explodes silently.

---

## 3. Minimal reproducer

```
# On Windows (adjust to your python path as needed):
D:/ProgramData/anaconda3/python.exe repro_xgb_minimal_for_upstream.py --parquet path/to/your.parquet
# Expected: silent process exit with rc=3221226505 (0xC0000005)

D:/ProgramData/anaconda3/python.exe repro_xgb_minimal_for_upstream.py --parquet path/to/your.parquet --workaround
# Expected: fit completes normally.
```

The workaround simply casts the offending column through `pl.Enum(sorted(unique_values))`, which forces compact physical codes `[0..n_unique-1]` by construction. XGB then allocates the correctly-sized buffer and everything works.

A 100 %-synthetic reproducer turned out to be surprisingly hard to produce. Closest attempts in this directory:

* `repro_xgb_exact_prod_state.py` — uses the exact 89-string + sentinel dict extracted from prod, with uniform-width padding (`f"__pad_{i:08d}"`). Produces the correct sparse dictionary but has not triggered the Windows SEH kill, because the padding is bytewise regular.
* `repro_xgb_synthetic_realistic.py` — extends the above with variable string lengths (heavy-tailed, p50~50, p99~1500, max~4800) and natural-language English vocabulary for padding. This matches two properties of the real `skills_text` column that the uniform-padding version was missing:
  - **String length variability.** Arrow's string array stores an offset buffer; a mix of short and very long values produces a fragmented value buffer that changes the Windows heap layout around the 9 MB XGB allocation.
  - **Natural hash distribution.** Polars' StringCache is a Rust HashMap. `f"skill_X"` has artificial regularity that may bias bucket layout; English-like content scatters hashes the way real parquet content does.

Even the realistic synthetic does not always reproduce the silent kill deterministically — the arrow-buffer state produced by a real parquet load + sort appears to matter for the Windows allocator's specific failure mode, even though the logical dictionary state is reproducible.

The bug itself (oversized allocation) is easy to confirm from the C++ source regardless. The silent-kill surface behaviour requires a heap state that only the full prod parquet reliably produces.

---

## 4. Proposed fix — compact codes before sketching

The simplest and least invasive fix is to remap physical codes to a dense range inside `AddCategories` before building the cut-values buffer. Bin identity becomes a rank in the sorted set; the cut-values buffer has exactly `categories.size()` entries; downstream histogram and prediction code index the buffer by rank, not by physical code.

See [`fix_add_categories_compact.cc`](./fix_add_categories_compact.cc) in this directory for a full patch, and section 5 below for the diff and test notes.

### Why option 1 (compact) rather than options 2 or 3

Three candidate fixes were considered:

| option | change | verdict |
|---|---|---|
| **1. Compact codes (this patch)** | Remap `{code_1, code_2, ...}` to dense `[0..k-1]` ranks before sketching; index bins by rank. | **Preferred.** Fixes root cause: no over-allocation regardless of input. O(k log k) one-time remap, k = categories.size(). Memory saved scales with (max_cat - n_unique). |
| 2. Size cuts by `n_unique`, not `max_cat+1` | Only change the loop bound, keep physical-code indexing. | Fixes the allocation but leaves every downstream indexing site needing to translate physical code -> bin. Larger, riskier diff. |
| 3. Lower `OutOfRangeCat()` | e.g. drop from `2^24` to `2^16`. | Breaking change for anyone with legitimately high-code categoricals. Does not fix sparse-but-sub-threshold cases. |

---

## 5. The patch

The full diff is in `fix_add_categories_compact.cc` (drop-in replacement for the `AddCategories` body plus a small helper). Summary of the change:

```cpp
// BEFORE:  cut_values sized by max_cat; bin i corresponds to physical code i.
for (bst_cat_t i = 0; i <= AsCat(max_cat); ++i) {
  cut_values.push_back(i);
}

// AFTER:  cut_values sized by n_unique; bin k corresponds to the k-th
// smallest physical code. A compact map from physical code -> bin rank
// is returned so downstream indexing stays correct.
bst_cat_t rank = 0;
for (float cat : categories) {              // std::set -> sorted
  cut_values.push_back(cat);                // store actual code, not rank
  code_to_rank[AsCat(cat)] = rank++;
}
```

The feature metadata gains a `std::unordered_map<bst_cat_t, bst_cat_t>` (`code_to_rank`) per categorical column. Every site that used to do `bin = physical_code` now does `bin = code_to_rank.at(physical_code)`. There are ~6 such sites; each is a one-line change.

Memory impact on the bug-trigger case: `9.64 MB -> 356 B` per categorical column (89 floats instead of 2.5 M floats).

Behavioural impact on well-formed inputs (compact codes already starting at 0): `code_to_rank[i] == i` for all i, so the mapping is identity and nothing changes at the model level. Existing models remain byte-identical — the fix is observationally a no-op on correct inputs and a correctness fix on pathological inputs.

---

## 6. References

* C++ source: `src/common/quantile.cc::AddCategories`, `src/common/categorical.h::OutOfRangeCat`
* polars StringCache behaviour: `polars >= 1.19` keeps `StringCache` permanently enabled; see polars changelog for 1.19.
* Reproducers: all `.py` files in this directory.
* Workaround (confirmed in production): `df.with_columns(pl.col(col).cast(pl.Enum(sorted(df[col].unique().drop_nulls().to_list()))))`.

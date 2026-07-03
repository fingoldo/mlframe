# MRMR FE leak-safety — engineered_recipes replay
No P0/P1 train/serve leak. kfold_target_encoded, cat_pair/triple_cross(target), target_aware_group_bin verified OOF-at-fit / full-at-serve; key canonicalization consistent.
- EN-1 [P2] _encoding_recipes.py:110/331-332/375-377 — builders coerce keys asymmetrically (`{str(k):float(v)}` vs raw tuple, no canonical), safe today only because generators pre-canonicalize; a future caller with raw keys silently routes to global fallback. Fix: canonicalize keys inside builders.
- EN-2 [Low] _encoding_recipes.py:73-74/157/190/224 + _recipe_dispatch.py:152 — polars branch calls pd.DataFrame unconditionally; polars-present/pandas-absent → AttributeError. Fix: guard pd is not None.
- EN-3 [Low] _missingness_ratio_recipes.py:37 — _apply_mi_greedy_transform forces float64 on source cols; object/string source raises only at transform (fit/serve asymmetry). Fix: validate numeric at build.
- EN-4 [Low] _recipe_extract.py:193,198 — raw-integer numeric factorize source astype(int64) truncates float round-trip (2.9999→2) mismatching fit code. Fix: round-to-nearest for integral-valued float, or persist bin_edges.
- EN-5 [Low] target_aware_group_bin global-fallback OOF optimism (duplicate of FE-F3).
Not found: no transform-time y reference; no y-derived stat replayed as pure-X without OOF; no naming/ordering nondeterminism.

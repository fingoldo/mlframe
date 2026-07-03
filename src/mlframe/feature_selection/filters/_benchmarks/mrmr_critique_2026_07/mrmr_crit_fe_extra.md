# MRMR FE leak-safety — extra families (rare/conditional-residual/rankgauss/dispersion/additive/pure-form/raw-redundancy)
All serving-leak-safe (recipes store only X-derived state; y consulted for selection only = in-sample optimism, not skew).
- EX-1 [P2] _conditional_gate_fe.py:357-362 — `_is_argmax_eligible` returns bool(np.all(np.isfinite(a[np.isfinite(a)]))) — the pre-filter makes isfinite always True, so NaN columns are never excluded → NaN taus/argmax/gate constants reach screening. Fix: `return bool(a.size==0 or np.isfinite(a).all())`.
- EX-2 [Low] _conditional_gate_fe.py:147-175 — argmax/gate meaning shifts on serve-only NaN (argmax→first-NaN idx; gate NaN→b). Recommend explicit replay NaN policy.
- EX-3 [Low/P2] _conditional_gate_fe.py:718 — tau chosen to maximize MI over grid on full train, then permutation null computed on the argmax-selected column → null underestimates selection-inflated MI. DOC (consistent with framework in-sample-MI-gate design).
- EX-4 [Low/P2] _extra_fe_families.py:795,809-813 — RankGauss stores full sorted NON-unique fit column (~80MB/col at 10M rows) in the pickled recipe; docstring wrongly says "unique". Fix: store unique+tie-counts / fix docstring.
Minor: additive_fusion fit uses nan_to_num(ha±hb) for edges — verify nested-parent replay applies same NaN→0 before quantise (_recipe_unary_binary.py).

# MRMR FE leak-safety — dispatch + temporal replay
Dispatch table + grouped/composite/kfold-TE paths verified LEAK-SAFE (OOF-at-fit / full-lookup-at-serve, canonical keys, unknown-kind hard-raise).
- FE-F1 [P1] apply_temporal_expanding _seed L634-653 / apply_temporal_lag L768-770 — seed entire train history ignoring stored timestamps → test row inside train time range sees FUTURE train values (look-ahead leak/skew). Rolling (L738 mask all_t<t) is correct. Fix: per-row timestamp-filtered merge. [DONE d804cdf4]
- FE-F2 [P2] _entity_key_series L92-105 raw `.astype(str)` (fit `1`→"1", transform `1.0`→"1.0") vs grouped paths' canonical_group_token → int/float dtype drift routes every replay key to global_prior (dead feature at inference). Fix: route through canonical_group_token.
- FE-F3 [P2] _grouped_quantile_fe.py:413,442 — target_aware_group_bin small-group OOF fallback uses global_edges (fit on ALL rows incl. scored fold) → y-leak inflating MI(bin;y). Fix: per-fold train-only fallback edges.
- FE-F4 [P2] _cat_target_encoding_and_weighted.py:81-99 — naive path (n_oof_folds<=0) emits per-cell mean incl. current row (target leak in the fit-time feature). Fix: force K>=2 for emitted-as-feature paths / gate behind explicit flag.
- FE-F5 [Low] apply_temporal_rolling L731-735 casts ns int64 → float64 (2^53 exact-int overflow, ~hundreds-ns boundary error). Fix: keep int64. (expanding/lag fixed in d804cdf4.)
- FE-F6 [Low] grouped_quantile pct_rank self-inclusion fit vs train-only replay (≈1/m per-group offset). DOC.
- FE-F7 [Low] _fe_stage_temporal_agg.py:101-105 — winner-name collision with an existing X column yields duplicate-labeled frame. Fix: dedup on concat.

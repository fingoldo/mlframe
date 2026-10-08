# New py-ci-shared gates: tests and CI wiring

## Known-gap register (KG)

Each KG id is cited as `(KG-n)` in the xfail / known_gap reason of the tests listed. A gap that closes flips its known_gap to a failure, so the entry leaves the register.

| id | Gap | Tests |
|----|-----|-------|
| KG-1 | RFECV does not prune pure-noise columns under severe class imbalance (1% positives keeps 6-11 of 11 columns, 8 of them noise) | biz_val/test_biz_val_imbalanced_rare_class.py |
| KG-2 | Multicollinear pollution: the selectors keep a majority of the high-VIF cluster, collapse onto the rank-deficient surrogate, or leave a high-VIF subset | biz_val/test_biz_val_multicollinear_pollution.py (3 tests) |
| KG-3 | HybridSelector keeps the redundant bridge feature of a graded chain a~b~c because corr(a,b)=0.86 is below the default corr_thr=0.92 | biz_val/test_biz_val_weak_family_adversarial.py |
| KG-4 | Out-of-fold leak in a target-aware FE family | fe/provenance/test_fe_target_aware_leak_contract.py |
| KG-5 | End-to-end MRMR fit on anchor scenario B forms no member swap (swap_log empty); the member branch is pinned only by direct evaluate_swap_candidate tests | mrmr/biz_val/test_biz_value_mrmr_dcd/test_anchor_refinement.py |
| KG-6 | MRMR under-selection regression: Westfall-Young FWER-null candidate-pool inflation | mrmr/biz_val/test_biz_value_mrmr_underselection.py |
| KG-7 | MRMR sample_weight is a fixed-size MC resample rather than row duplication and target-encoding FE ignores sample_weight | mrmr/fe/test_biz_val_mrmr_sample_weight_fe.py |
| KG-8 | Selector parity gaps: fit rejects a bare ndarray, ndarray get_feature_names_out ignores user names, no set_output, transform is DataFrame-only | stability/test_selector_contract_protocol_extra.py, test_selector_contract_shared.py |
| KG-9 | Selectors do not validate transform-time width, nor reject duplicate column names at fit entry (silent positional indexing) | stability/test_selector_contract_shared.py |
| KG-10 | Selection depends on input column order (positional tie-break / shadow ordering) | stability/test_selector_contract_shared.py |

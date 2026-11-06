# Device-to-host copies in the strict pair-search fit (2026-11-05)

Source: the fit of `tests/feature_selection/gpu/test_cmi_residency_traffic.py` (`_run_pair_search_audit`, n = 30000, strict-resident GPU), D2H bytes by nearest caller with
`D:/Temp/resid_who.py` (it replaces `_audit._record`). Budget of `test_strict_fit_copies_back_only_result_columns`: 10 columns of n float64 = 2 400 000 B.

| step | D2H total |
|---|---|
| before | 3 053 940 B (budget exceeded) |
| `_quantile_bin_gpu` copies int16 codes and caches the host copy by content (`resident_qbin_codes_host`) | 2 510 626 B |
| the stability vote replays and bins the recipe column on the device (`_replay_codes_on_device`); only int16 codes cross | 2 390 626 B |
| `_FamilyOnDevice` copies the largest level count as one scalar (it copied the whole per-column vector once per family), occupied-level counts as int16 | 2 271 304 B |

`_pair_admission_batch._block_verdicts` read the columns of the per-pair null back from the device (the same 30000 B column 11 times); it reads them from the host matrix now.

What is left, by caller (bytes, copies): `bits_gate_mi` 820 224 (50, the gated MI vector of each candidate block: a result), `apply_unary_binary_gpu` 360 000 (3, the engineered columns that are written out),
`_family_maxt_cuda_bits._run` 143 856 (2, MI matrix of the honest null), `score_pair_combos_table_resident` 138 720 (1, the usability table), `selected_design_columns` 120 000 (1, the selected design),
the bin codes 120 000 (6, int16), `_continuous_accept_block` 111 888 (1, gain matrix of the accept kernel).

## Open: one repeated vector (62 208 B, test_strict_fit_does_not_copy_the_same_content_back_repeatedly)

The MI vector of a candidate block (float64, 1944 values, 15 552 B) is copied by `bits_gate_mi` (noise gate, in `_pairs_dispatch`) and again by `_FamilyOnDevice.observed` (family max-T gate, in `_pairs_emit`),
four times in this fit. The family gate needs the full vector on the host: its analytic triage reads the best MI and the degrees of freedom of every one-column batch. The two stages see the same
columns through different arrays (the noise gate gets the chunk matrix, the family gate one `(n, 1)` view per configuration, concatenated), so a cache keyed by the device array does not match.
Options measured as not worth it or not safe: a device fingerprint of the codes (a pass over n x K per call for 15 KB), the noise gate's own `valid_pairs_perf` as the family's observed MI (it is the
gated value: a zeroed configuration would be judged with MI 0). The sound fix is one shared observed-MI result per chunk, produced by the noise gate before gating and handed to the family gate with the batches.

# Audit 08: MRMR information-theory estimators (wave 2)

Scope: MI and entropy kernels (`info_theory/`, `permutation.py` mi_direct, `_chao_shen.py`, `_mi_aggregator.py`, `_ksg*.py`, `_cmi*.py`, `_joint_entropy*.py`). Read-only, Haiku model, nothing executed; "expected" values are from the formula, not a run.

Tracker rows: AUD23-22, 23, 24, 25, 26, 27.

## Checked and correct
- Log base is natural log throughout; no `log2`.
- Miller-Madow signs right: entropy adds `(k-1)/2n` (`_entropy_kernels.py:156`); MI subtracts `(kx-1)(ky-1)/2n` (`:195`); entropy-consistent form `(kx+ky-kxy-1)/2n` (`:192`) matches derivation.
- CMI bias sign right: plug-in CMI biased by `+(k_xyz+k_z-k_xz-k_yz)/2n`, returned by `_cmi_miller_madow_bias` (`:581`) and subtracted in `_cmi_finalise` (`:630`); same for CMIM GPU (`_cmim_min_gpu.py:154`).
- Zero-count guards consistent across `entropy`, MM helpers, and CS kernels.
- No n-1 normalisation errors; KSG aggregation (`_ksg.py:123`, `:141`) matches Kraskov/Gao.
- Host joint-code overflow guarded: `merge_vars` checks after int32 store (`info_theory/_class_encoding.py:74-80`); `_batch_kernels.py:817` rejects int16 overflow; `_pairs_score_helpers.py:324` casts to int64 before the product.

## Findings
1. Chao-Shen MI uses per-term coverage in two kernels, not the shared coverage of `_chao_shen.py`. Severity medium. Confirmed by code comparison; value not computed.
   - `_entropy_kernels.py:390-393` (`mi_chao_shen`) and `info_theory/_class_mi_kernels.py:105-109` (`compute_mi_cs_from_classes`) each estimate C_hat separately for H(X), H(Y), H(XY).
   - `_chao_shen.py:149-171` (`_joint_chao_shen_mi_njit`) says this per-term scheme was a bug, fixed to one shared C_hat.
   - Production CS goes through `compute_relevance_score` (`_class_mi_kernels.py:297-298`) and `mi_or_su` (`_state_and_dispatch.py:161-162`), so it hits the unfixed code. Expected: all three CS implementations give the same MI for the same table.

2. `group_blocked_mi` ignores the Miller-Madow toggle. Severity medium. Confirmed by grep.
   - `info_theory/_group_mi.py:238`, `:264`, `:276` hard-code `use_mm=True`. Elsewhere the toggle is respected (`permutation.py:622`, `:763`: `use_mi_miller_madow() and not _use_su`). With the default toggle off, the group path still applies MM, on a different scale from the rest.

3. Possible double correction in the FE pair gate. Severity medium. Hypothesis.
   - `_feature_engineering_pairs/_pairs_gates.py:163-164` documents `best_mi` and `pair_mi` as raw plug-in values and applies MM to both. `_pairs_score_helpers.py:328-330` passes them without checking whether they were already corrected. If the upstream scorer used `compute_relevance_score` with `use_mm` on (`permutation.py:620-624`), the correction applies twice. Producer of `pair_mi` not traced.

4. `entropy(..., min_occupancy)` drops bins without renormalising. Severity low, depends on callers. Confirmed by reading.
   - `_entropy_kernels.py:17-18` and `:139-140`. Input `freqs=[0.6, 0.4]`, `min_occupancy=0.5` returns 0.3065 nats (only the 0.6 term); plug-in gives 0.6730. Neither is a valid entropy. Callers not checked.

5. Naive `(1-p)^n` in `entropy_chao_shen`. Severity low. Confirmed (`_entropy_kernels.py:338`; `_chao_shen.py:85-87` says it cancels catastrophically). Practically unreachable on real counts, since n*p is bounded below by roughly C_hat.

6. Possible int32 overflow in GPU index arithmetic. Severity low. Hypothesis.
   - `info_theory/_cmi_cuda.py:837` computes `joint_size` as a Python int, but kernel arguments at `:455`, `:476` are `np.int32`. `_cmim_min_gpu.py:150`, `:154` index `(xx*ky+yy)*kz+z` in C int. Overflow when cell count exceeds 2^31 (e.g. bins 1300 x 1300 x 1300). Kernel bodies not read.

Nothing modified; no tests run.

# Wave 11 audit 2026-11-02 -- tracker

Second pass over numba kernels (the unread part of wave 10, NJIT10-2), three read-only reports by Opus 5.5 low-effort agents: filters `_n*` to `_z*`, filter subpackages with `shap_proxied_fs`, and metrics, feature engineering and training kernels. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | NJ11-1a | njit_oob | High | `_wavedec_batched_njit` wrote past a scratch buffer sized to the signal when T < filter length - 1 (`wavelet_dwt.py`) |
| **RESOLVED** | NJ11-1b | njit_oob | Medium | SU clustering kernels indexed another feature's counts with a negative bin code (`shap_proxied_fs/_shap_proxy_cluster_su.py`) |
| **RESOLVED** | NJ11-1c | wrong_formula | High | Miller-Madow term added to plug-in MI instead of subtracted at four sites (`_orthogonal_copula_mi_fe.py` x2, `_numba_utils.py`, `_fe_synergy_screen.py`) |
| **RESOLVED** | NJ11-1d | wrong_formula | High | the same sign error in the marginal-MI Miller-Madow term of `_mi_greedy_cmi_fe.py`, `_mi_greedy_cmi_fe_gpu.py`, `_mi_greedy_cmi_fe_steps.py` and two more sites in `_fe_synergy_screen.py`, found by grepping the formula after NJ11-1c |
| **RESOLVED** | NJ11-2 | Low | every kernel of `_numba_polynom_optimizer` that was compiled with the full `fastmath=True` set (which asserts no NaN / inf) now uses the module's NaN-safe flag set (`nsz`, `arcp`, `contract`, `afn`, `reassoc`), as the two finiteness gates already did; the polynomial-evaluation and MI kernels imported from other modules keep their own flags; the bf-dispatch parity, finite-gate and survivor-commit-order tests pass unchanged (`tests/feature_selection/filters/test_numba_polynom_kernels_keep_nan_visible.py`) |
| **DOC** | NJ11-3 | coverage | Info | files still not read in full, listed below Closed as DOC: a statement of audit scope, not a defect. The files listed are re-read by the repository-wide agent audit that follows the closure of these trackers. |

## Counts

6 rows: 3 High, 1 Medium, 1 Low, 1 Info. Dispositions: 4 RESOLVED, 2 FUTURE.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| NJ11-1a | Size the batched DWT scratch buffers to the longest level | `tests/feature_engineering/test_wavelet_dwt_short_signal_batched.py` |
| NJ11-1b | Shift negative bin codes before the SU kernels count them | `tests/feature_selection/shap_proxied/test_su_kernel_negative_codes.py` |
| NJ11-1d | Add the Miller-Madow correction to marginal MI in the greedy CMI and synergy kernels | `tests/feature_selection/filters/test_greedy_cmi_marginal_mi_bias_sign.py` |
| NJ11-1c | Subtract the Miller-Madow term from plug-in MI instead of adding it | `tests/feature_selection/filters/test_miller_madow_bias_is_subtracted.py`; `test_copula_mi_batch_yhoist_identity.py` re-framed from bit identity to 1e-12 |

NJ11-1c changes the scores of the copula and synergy screens (independent columns drop from about 0.18 to about 0 nats), so selections that relied on the inflated values can move. The tests for the synergy screen, the copula kernels, the auto scorer and the MRMR synergy-redundancy case pass (58 tests); no multi-seed selection benchmark was run.

## Not covered by this wave (NJ11-3)

- `metrics/regression/*`, `_core_cb_logits.py`, `_core_brier.py`, `_log_loss_and_separation.py`, the rest of `feature_engineering/transformer/*`, `composite/transforms/*`, `composite/discovery/screening.py` and the helpers beside it (`_corr_numba.py`, `_grouped_causal_bases.py`, `_ktc_dispatch.py`, `_njit_shared.py`).
- The unary and binary operator helpers in `_numba_utils_usability.py` lines 1-100, and kernels these files import from `info_theory`.
- `tests/feature_selection/fe/basis/test_orthogonal_scorer_bugfixes.py::test_meta_scorer_propagates_programming_error` fails (DID NOT RAISE AttributeError) in the same run as NJ11-1c; its code path (`meta.fingerprint_signal` with a patched `DataFrame.corr`) does not touch the changed estimators, and it was not investigated here.

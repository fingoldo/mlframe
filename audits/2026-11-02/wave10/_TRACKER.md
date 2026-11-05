# Wave 10 audit 2026-11-02 -- tracker

Three read-only reports on `src/mlframe` by Opus 5.5 low-effort agents (environment-variable parsing, index and column alignment, correctness of numba kernels). The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | ENV10-1 | env_parsing | High | `MLFRAME_PERM_BLOCK_CACHE_MB` parsed with raw `int()` at import (`_family_maxt_gate.py`) |
| **RESOLVED** | ENV10-2 | env_parsing | High | `MLFRAME_PAIR_NULL_MAX_ROWS`, two raw `int()` sites (`_step_pairs_rank.py`) |
| **RESOLVED** | ENV10-3 | env_parsing | High | `MLFRAME_FE_GATE_MAX_ROWS` (`_step_score.py`) |
| **RESOLVED** | ENV10-4 | env_parsing | High | `MLFRAME_FE_FUSION_MAX_ROWS` (`_fe_additive_fusion_gpu_resident.py`) |
| **RESOLVED** | ENV10-5 | env_parsing | Medium | `MLFRAME_MRMR_FALLBACK_REDUNDANCY_FRAC` crash and no range check (`_finalise.py`) |
| **RESOLVED** | ENV10-6 | env_parsing | Medium | `MLFRAME_DISCOVERY_LAZY_PREBIN_MIN_N` parsed even when unused (`discovery/_fit.py`) |
| **RESOLVED** | ENV10-7 | env_parsing | Medium | `MLFRAME_PYSR_INPUT_BYTES_LIMIT` rejects `2e8`, accepts non-positive (`bruteforce.py`) |
| **RESOLVED** | ENV10-8 | env_parsing | Medium | `MLFRAME_MRMR_PICKLE_REPLAY_STATE_MAX_MB` crash inside pickling (`_mrmr_stability_report.py`) |
| **RESOLVED** | ENV10-9 | env_parsing | Medium | two names for the same free-VRAM floor, the documented one ignored by the gpu-resident gate |
| **RESOLVED** | ENV10-10 | env_parsing | Low | `_USAB_FORCE_FULL_REFIT=0` read as true |
| **RESOLVED** | ALIGN10-1 | alignment | High | `ShortlistTransformerAdapter.transform` accepted a different column order |
| **RESOLVED** | ALIGN10-2 | alignment | Medium | the same adapter dropped the caller's index |
| **RESOLVED** | ALIGN10-3 | alignment | High | `PdKBinsDiscretizer` with default onehot on a DataFrame raised a shape error |
| **RESOLVED** | ALIGN10-4 | alignment | Medium | `PdOrdinalEncoder` and `PdKBinsDiscretizer` dropped the row index |
| **RESOLVED** | ALIGN10-5 | alignment | Medium | `RegimeSplitEnsemble` mixed polars and ndarray inputs across submodels |
| **RESOLVED** | ALIGN10-6 | alignment | Low | `visualize_prediction_vs_truth` looked rows up by label |
| **REJECTED** | NJIT10-0 | njit_kernels | Info | no race, out-of-bounds or uninitialised-read finding in the kernels read |
| **RESOLVED** | NJIT10-1 | njit_kernels | Low | `eval_candidates` read `sub[-1]` of an empty sample at n=0 (agent's dropped candidate, fixed anyway) |
| **DOC** | NJIT10-2 | njit_kernels | Info | kernels the agents did not read line by line (list below) Closed as DOC: a statement of audit scope, not a defect. The kernels listed were not read line by line in that wave; the repository-wide agent audit that follows the closure of these trackers re-reads them. |

## Counts

19 rows: 5 High, 8 Medium, 3 Low, 3 Info. Dispositions: 17 RESOLVED, 1 FUTURE, 1 REJECTED.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| ENV10-1 to ENV10-10 | Parse MLFRAME numeric and boolean variables through the shared env helpers | `tests/utils/test_env_parse_malformed_values.py` |
| ALIGN10-1, ALIGN10-2 | Align the shortlist adapter's query columns to fit order and keep the input index | `tests/feature_engineering/transformer/test_suite_adapter_column_order_and_index.py` |
| ALIGN10-3, ALIGN10-4 | Name the one-hot columns and keep the row index in the pandas encoders | `tests/estimators/test_pd_encoders_keep_index_and_onehot_columns.py` |
| ALIGN10-5 | Subset RegimeSplitEnsemble inputs with the flavour-preserving helper | `tests/training/composite/test_regime_split_ensemble_polars_input.py` |
| ALIGN10-6 | Pick rows by position in visualize_prediction_vs_truth | `tests/estimators/test_visualize_prediction_vs_truth_positional.py` |
| NJIT10-1 | Return zero scores from eval_candidates on zero rows | `tests/feature_selection/test_offset_eval_candidates_empty_rows.py` |

## Not covered by this wave

NJIT10-2: the njit pass did not read line by line the feature-selection filters `_n*` to `_z*` (most likely: `_numba_polynom_optimizer.py`, `_pairwise_modular_fe.py`, `_parity_fe.py`, `_wavelet_basis_fe.py`, `_orth_dedup.py`), the subdirectories `_dynamic_cluster_discovery`, `_orthogonal_univariate_fe`, `_mrmr_fit_impl`, `_mrmr_fe_step`, `_feature_engineering_pairs`, `engineered_recipes`, `shap_proxied_fs`, the regression and remaining classification metric kernels, `feature_engineering/transformer/*`, `composite/transforms/*` and `composite/discovery/*`. A second pass split across several agents is the next step.

Not verified on this machine: ENV10-2 to ENV10-5 and ENV10-8 have no dedicated test (they sit deep in fit paths) and rely on the shared `env_int` / `env_float` helpers, whose range handling the tests above cover through the sites that can be reached directly.

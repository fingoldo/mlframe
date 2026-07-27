# info_theory_b — MI estimator zoo (KSG/Rényi/Chao-Shen/BUR/fastMI/neural) + dispatcher/aggregator/PID/JMIM/interaction-information/synergy-detector

This cluster is the standalone MI-estimator "zoo" that sits beside MRMR's core plug-in relevance/redundancy loop: k-NN estimators
(`_ksg.py`: Mixed-KSG + KSG-LNC), a matrix-based Rényi alpha-entropy estimator (`_renyi_alpha.py`), two small-sample entropy/MI
correction kernels used inside MRMR proper (`_chao_shen.py`, and the Miller-Madow-corrected redundancy terms in `_bur_term.py` /
`_jmim_scorer.py`), a copula-FFT-KDE estimator (`_fastmi.py`), five PyTorch neural estimators (`_neural_mi.py`: MINE, InfoNet, MIST,
MINDE, DPMINE), a unified ad-hoc dispatcher (`_mi_dispatch.py`) and multi-estimator aggregators (`_mi_aggregator.py`: median /
GENIE / calibration-picker), an O(n)-per-call analytic replacement for MRMR's permutation null at large n (`_analytic_mi_null.py`),
a four-term Partial Information Decomposition (`_pid_decomposition.py`), a signed interaction-information router for the FE
prospective-pair gate (`_interaction_information.py`), and a cheap pre-fit synergy probe that gates the JMIM aggregator
(`_synergy_detector.py`). Only `_chao_shen.py`, `_bur_term.py`, `_jmim_scorer.py`, `_analytic_mi_null.py`, `_pid_decomposition.py`,
`_interaction_information.py`, and `_synergy_detector.py` are wired into MRMR's actual fit path today; `_ksg.py`, `_renyi_alpha.py`,
`_fastmi.py`, `_neural_mi.py`, and `_mi_aggregator.py` are reachable only via the standalone `score_pair_mi` dispatcher for ad-hoc
benchmarking (per the modules' own docstrings — Family-2 estimators are intentionally kept out of MRMR's hot loop). mypy is clean
across all 13 files (`python -m mypy --cache-dir=.mlframe_mypy_cache_shared <files>` → "Success: no issues found in 13 source
files"). No file in this cluster exceeds the ~800-900 LOC guideline (`_neural_mi.py` at 754 is the largest, see finding below). No
SQL/HTTP/UI surface exists in any of these 13 files except two model-checkpoint fetches (InfoNet: manual `gdown` + a vendored
`torch.load(..., weights_only=True)` loader that already has a dedicated SEC1 regression test; MIST: automatic first-call download
via the external `mist_statinf` package's `MISTForHF.from_pretrained`, outside this repo's direct control) — confirmed, no other
network/DB/generated-HTML surface in this cluster.

Prior-audit cross-reference: `audits/mrmr_audit_2026-07-20/c3_info_theory.md` (10 findings) covers `_renyi_alpha.py` and `_ksg.py`
directly and the `info_theory/` subpackage (a different, sibling directory not in this assignment). B-8 (alpha=1.0 singularity,
`_renyi_alpha.py:109`) and B-9 (renyi_alpha bits-vs-nats unit mismatch) and B-10 (`_ksg.py:490` GPU bare-`except ImportError`) are
all **verified fixed** in the current tree via commit `741926f8c` — B-8 now raises `ValueError` on `alpha==1.0`; B-9 is fixed at the
`score_pair_mi` dispatcher boundary (`_mi_dispatch.py:149-150` multiplies by `ln(2)` before returning); B-10 now has a proper
`_KSG_GPU_FAILED` circuit breaker with a logged `except Exception` fallback. **B-10's own writeup explicitly named the identical
gap in `_fastmi.py:209` as "lower exposure" detail — that half was never fixed; see INFO_THEORY_B-2 below.** The `mrmr_critique_2026_07/mrmr_crit_numstat.md` N-F6 Chao-Shen/null-mismatch finding is DOC-resolved (MRMR now warns on the
`chao_shen` no-op fallback) and doesn't implicate `_chao_shen.py`'s own internals, which the 07-20 audit already spot-checked and
found directionally correct (Miller-Madow / Chao-Shen / PID / BUR / JMIM sign and floor conventions all confirmed correct).

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| INFO_THEORY_B-1 | P1 | bug | `_mi_aggregator.py:88-104` (`genie_weights`) + `_mi_dispatch.py:187-197` (`_score_aggregator`) | The GENIE bias-cancelling weighted ensemble silently degrades to a **plain unweighted mean** on every single production call — see reproduction below. | NEW, not in any prior report |
| INFO_THEORY_B-2 | P1 | cpu_gpu_parity | `_fastmi.py:201-210` | `fastmi(..., prefer_gpu=True)`'s cupy FFT path still only catches `ImportError`; a real runtime GPU fault (OOM, driver hiccup) propagates uncaught instead of falling back to CPU. | STILL OPEN — prior report `c3_info_theory.md`'s B-10 named this exact file:line as part of the same finding ("lower exposure"); commit `741926f8c` fixed only the `_ksg.py` half |
| INFO_THEORY_B-3 | P1 | bug | `_neural_mi.py:204,672,748` (`mine_mi`/`minde_mi`/`dpmine_mi` return statements) | `max(0.0, converged)` silently collapses a NaN/Inf training divergence (e.g. from a NaN-contaminated input column, or Adam/EMA numerical blow-up) to `0.0` ("no signal") instead of surfacing the failure. | NEW — same footgun class the prior audit's B-8 fixed in `_renyi_alpha.py`, unaddressed here |
| INFO_THEORY_B-4 | P2 | bug | `_synergy_detector.py:169,197` | Two bare `except Exception: pass  # nosec B110` swallows around the `kernel_tuning_cache` lookup, with zero logging — violates the repo's "no silent except-Exception swallowing without logging" convention; a real registry bug would silently and permanently pin `detect_synergy`'s threshold to the hardcoded default forever. | NEW — same bug class the prior audit flagged (P2) for `_cmi_cuda_ktc.py`/`_batch_kernels.py`, a different pair of files |
| INFO_THEORY_B-5 | P2 | bug | `_renyi_alpha.py:152-181` (`renyi_alpha_cmi`) | `renyi_alpha_cmi` still returns bits (log2-based), unconverted — the B-9 fix only converts `renyi_alpha_mi`'s output at the `score_pair_mi` boundary; `renyi_alpha_cmi` is public API (`__all__`) and the module's own docstring anticipates future MRMR conditional-MI wiring, which would silently reintroduce the exact ~1.44x mismatch B-9 just fixed for the marginal case. | NEW (partial-coverage gap in the B-9 fix) |
| INFO_THEORY_B-6 | P2 | doc/design | `_neural_mi.py:544` (`mist_mi` docstring) | Docstring claims `"seed: ignored; feed-forward inference is deterministic"`, but `seed` genuinely drives which rows get sub-sampled whenever `x.size > max_input_n` (default 2000) — the common case for real screening columns (`_neural_mi.py:561-565`). | NEW |
| INFO_THEORY_B-7 | P2 | edge_case | `_pid_decomposition.py:241`, `_bur_term.py:37`, `_jmim_scorer.py:47` | `pid_decomposition`/`bur_term`/`jmim_score` allocate dense joint-count arrays sized directly from caller-supplied cardinalities (`K_x1*K_x2*K_y`, `K_x*K_y`, `K_x*K_z*K_y`) with no upper-bound sanity check, unlike every k-NN/neural/Rényi estimator in this same cluster (`max_input_n`/`max_n`). In practice MRMR's own adaptive-binning pipeline caps `nbins`, so the exposure is caller-discipline only, not a reachable-by-default bug. | NEW |
| INFO_THEORY_B-8 | P2 | test_gap / dead_code | `_neural_mi.py:590-748` (`minde_mi`, `dpmine_mi`) | Both are exported in `__all__` but (a) never wired into `_mi_dispatch.py`'s `score_pair_mi` (`estimator` only recognizes `"mine"`/`"infonet"`/`"mist"` for neural, `_mi_dispatch.py:137`) and (b) have **zero test coverage anywhere** in `tests/` (confirmed via repo-wide grep). Both are self-documented "EXPERIMENTAL/SKELETON... not production-competitive," so the gap is expected but total. | NEW |
| INFO_THEORY_B-9 | P2 | API design | `_fastmi.py:146,194` (`fastmi`) | `fastmi()` exposes no `seed`/`random_state` parameter; the MISE-bandwidth grid search sub-samples via a hardcoded `np.random.default_rng(0)` when `n > 1000`. Every other estimator in this cluster (`mixed_ksg_mi`, `ksg_lnc_mi`, `renyi_alpha_mi`/`_cmi`, `mine_mi`, `infonet_mi`, `mist_mi`, `minde_mi`, `dpmine_mi`) exposes a caller-controllable seed. | NEW |
| INFO_THEORY_B-10 | P2 | bug (defensive masking) | `_interaction_information.py:266-267` (`route_prospective_pairs`) | `cached_MIs.get((va,), 0.0)` silently substitutes `0.0` for a missing marginal-MI cache entry rather than asserting/logging; if an upstream caching bug ever drops an entry, this reads as a spuriously HIGH interaction-information (since `pair_mi - 0 - mi_b` inflates `II`), routing a pair to `synergy` instead of surfacing the real cache defect. | NEW |
| INFO_THEORY_B-11 | P2 | architecture | `_neural_mi.py` (754 LOC) | Five independent estimator families (MINE/InfoNet/MIST/MINDE/DPMINE) in one file, approaching the repo's ~800-900 LOC carve-before-it-grows guideline. Not yet a violation. | NEW (housekeeping) |
| INFO_THEORY_B-12 | P2 | edge_case | `_synergy_detector.py:57-69` (`_quantize`) | NaN rows are handled inconsistently between the two branches: the low-cardinality direct-factorization path maps NaN to bin 0 (`lut.get(v, 0)` default, since `nan == nan` is False so the lookup always misses), while the continuous quantile path places NaN in the LAST bin (`np.searchsorted` sorts NaN to the end). Cosmetic for this opt-in heuristic gate (`detect_synergy` only feeds the `redundancy_aggregator='auto'` decision), not a correctness-critical path. | NEW |
| INFO_THEORY_B-13 | P2 | security (hygiene) | `_neural_mi.py:380-388` (`_get_mist_hf_model`) | MIST's HuggingFace checkpoint (`mist_statinf.MISTForHF.from_pretrained('grgera/MIST')`) auto-downloads and loads on first call with no equivalent `weights_only=True`/safetensors verification or regression test the way the vendored InfoNet loader got after its SEC1 fix (see `tests/feature_selection/test_infonet_weights_only_load.py`). The loading mechanism lives in the external `mist_statinf` package, outside this repo's direct control, so this is a hygiene note (verify/pin the dependency's loading mechanism) rather than a fixable bug here. | NEW |

### INFO_THEORY_B-1 reproduction (empirically confirmed, not a hand-waved hypothesis)

`genie_mi_panel`'s docstring (and `_mi_dispatch.py`'s module docstring) claim GENIE "solves a small linear system ... for
bias-cancelling weights" and "strictly beats any single member in expectation." In production, `_mi_dispatch.py:187-197`'s
`_score_aggregator` calls `genie_mi_panel(x, y_arr, kind="genie", ...)` with **no** `bias_rates`/`variances` argument — every
`score_pair_mi(..., estimator='genie')` call hits `genie_mi_panel`'s "not provided" fallback: `bias_rate = 1/sqrt(N)` for
**every** estimator, `variance = 1.0` for every estimator (`_mi_aggregator.py:140-141`). Because the fallback bias vector is a
*constant* vector (identical value for all K estimators), the `(K+2)x(K+2)` constraint matrix `genie_weights` builds is
**exactly singular by construction** (the `b^T` constraint row is a scalar multiple of the `1^T` row) — confirmed directly:

```
rank: 4  shape: (5, 5)
det: 0.0
LinAlgError raised: Singular matrix
```

`genie_weights` catches this `LinAlgError` and falls back to `w = ones(K)/K` (plain uniform averaging) — verified end-to-end on a
real signal pair with the actual `fd`/`qs`/`mixed_ksg` estimator panel:

```
weights: [0.33333333 0.33333333 0.33333333]
genie result:                       0.48789968286513474
plain unweighted mean of the three: 0.4878996828651348      # identical to float precision
```

So `estimator='genie'` never actually runs GENIE's bias-cancelling weighting in production — it is, today, an expensive
(~23x plug_in wall-time per the module's own docstring) way to compute a plain unweighted mean of `fd`/`qs`/`mixed_ksg`. No
existing test (`test_genie_panel_signal` in `test_biz_val_mi_estimators.py`, `test_score_pair_mi_signal_exceeds_noise_fast` in
`test_mi_dispatch_contract.py`) can detect this, because both only assert `mi_signal > mi_noise (+floor)`, a bar an unweighted
mean clears just as easily as a correctly-weighted GENIE — and `genie_weights` itself has **zero direct unit tests anywhere**
(confirmed via grep), so nothing pins the weighting math independent of this degenerate default path.

## Proposals

- **(bug-fix idea)** Give `genie_mi_panel`'s default bias-rate fallback actual per-estimator differentiation (e.g. plug-in-family
  estimators ~ `M/N`, KSG-family ~ `k/N`, matching each estimator's real known asymptotic bias order) instead of the same
  `1/sqrt(N)` constant for every name — this is the one-line root cause of INFO_THEORY_B-1's singular system. Add a direct unit
  test for `genie_weights` with genuinely different bias-rate vectors so the LP math is pinned independent of the panel-level
  default path.
- **(edge_case)** Give `_fastmi.py`'s GPU cupy branch (`fastmi(..., prefer_gpu=True)`) the same `except Exception:` circuit-breaker
  fallback `_ksg.py`'s `ksg_mi_dispatch` now has (closing the still-open half of the prior audit's B-10).
- **(edge_case)** Add an explicit `np.isfinite` gate (or NaN-safe median) around `mine_mi`/`minde_mi`/`dpmine_mi`'s trained MI
  trace before the final `max(0.0, ...)`, raising or logging on a genuinely diverged run instead of silently reporting "no
  signal."
- **(other)** Add logging (`logger.debug`) to `_synergy_detector.py`'s two bare `except Exception: pass` sites around the
  `kernel_tuning_cache` lookup, mirroring the prior audit's proposal for `_batch_kernels.py`/`_cmi_cuda_ktc.py`.
- **(other)** Either convert `renyi_alpha_cmi` to nats now (preempting a second bits/nats bug the day it gets wired into
  anything), or add an explicit bits-scale docstring caveat + guard mirroring `renyi_alpha_mi`'s current state.
- **(other)** Fix `mist_mi`'s docstring to state that `seed` affects the sub-sample selection above `max_input_n`, rather than
  claiming it is unconditionally ignored.
- **(edge_case)** Add an explicit cardinality-product cap (mirroring `_ksg.py`/`_neural_mi.py`/`_renyi_alpha.py`'s
  `max_input_n`/`max_n` pattern) to `pid_decomposition`/`bur_term`/`jmim_score`'s dense joint-histogram builders, so a
  caller bypassing MRMR's normal binning cap gets a clear error instead of an OOM.
- **(coverage_gap)** Either wire `minde_mi`/`dpmine_mi` into `_mi_dispatch.py`'s estimator id list (documenting them as
  experimental) or add at minimum a smoke test importing and running each once on CPU, so a hard crash regression doesn't
  ship silently in either.
- **(other)** Add a `seed`/`random_state` parameter to `fastmi()` for consistency with every other estimator in the cluster.
- **(other)** Preemptively split `_neural_mi.py` into a subpackage (one submodule per estimator family) before it crosses the
  800-900 LOC guideline, per the repo's "carve before a file nears ~800-900 LOC" convention.

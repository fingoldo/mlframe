# x_ml_correctness_meta

## Scope

Cross-cutting, pattern-level review (per the assignment brief, explicitly "not exhaustive per-file") across:

- `src/mlframe/training/` (~57.6k LOC, 671 files)
- `src/mlframe/feature_engineering/` (~50.5k LOC, 308 files)
- `src/mlframe/feature_selection/` excluding `filters/` and `shap_proxied_fs/` (~57.6k LOC, 363 files)
- `src/mlframe/calibration/` (~7.2k LOC, 45 files)
- `src/mlframe/metrics/` (~18.4k LOC, 85 files)

Total assigned surface: ~191k LOC / ~1472 files. Given the scope's own framing as a pattern-level review, this pass combined (a) full, close reads of the entire `calibration/` package plus the calibration-adjacent core of `metrics/calibration/`, and (b) full reads of a risk-weighted sample of the highest-leakage/highest-reproducibility-risk files elsewhere in scope (OOF-encoding, cross-target-ensemble stacking/calibration, RFECV fold evaluation, honest-diagnostics aggregation, Venn-Abers calibration, forward/backward CV-driven feature selection, k-fold row-attention OOF loop), with (c) repo-wide grep sweeps across the full assigned scope for known bug-class signatures relevant to the five focus dimensions (global-RNG mutation, mutable default args, bare `except: pass`, `test_target`-as-OOF-proxy fallbacks, un-seeded `default_rng()`).

**Files closely/fully read: 35** (see narrative below for the list).
**LOC closely/fully read: ~9,700** (plus grep-pattern coverage of the full ~191k LOC assigned scope for the specific signatures listed above).

## Summary by dimension

- **Data leakage (fit-time info crossing into held-out data):** Reviewed extensively. The codebase is unusually disciplined here — `LeakageSafeEncoder`, `ordered_target_encode`, `train_postcalibrators`, `post_calibrate_model`, `calibrate_venn_abers`, `_carve_inner_eval_split`, and RFECV's fold evaluator all carry explicit, enforced (not just documented) leakage guards, several with dedicated overlap-detection assertions (`train_postcalibrators`'s `calib==test` overlap guard is a good example of defense-in-depth). One confirmed regression found: **honest_diagnostics.py's calibration block silently pairs OOF predictions with mismatched-row labels** when `oof_target` is absent (finding #1) — this is exactly the bug class the sibling `post_calibrate_model` code explicitly fixed and now hard-raises on, but `honest_diagnostics.py` was not updated to match.
- **Reproducibility (RNG seed threading, hidden global mutable state):** No unguarded global-state mutation of numpy's own RNG was found (`np.random.seed()` appears in only one place in scope, inside an `@njit` function where it seeds numba's own internal generator, not numpy's — documented and intentional). One un-seeded `default_rng()` fallback found in `swap_noise_augment` (data-augmentation-only, not training-decision-affecting) — judged too low-impact to write up as a standalone finding given the explicit "not manufacture trivial findings" instruction, but noted here for completeness; not scored.
- **Calibration correctness (in-sample vs OOF/held-out):** `calibration/policy.py::pick_best_calibrator` and `calibration/post.py::compare_postcalibrators` both ship a documented, fixed "same_oof" optimism bug (Isotonic self-selecting via interpolation) with an `inner_cv` honest default and the legacy path kept opt-in only for replay. No new occurrence of that bug class found. The one exception is finding #1 above (a different failure mode: wrong-row pairing, not same-row optimism).
- **sample_weight threading:** Consistently threaded end-to-end everywhere checked — `BinaryPostCalibrator.fit`, `compare_postcalibrators`, RFECV's `_eval_fold_body` (including the early-stopping val re-split slicing, which correctly re-slices `sample_weight` alongside `true_train_index`), and all four `fit_*_meta_stacker` functions in `_stackers.py`. No threading gaps found.
- **train/val/test/OOF terminology:** Docstrings and code consistently distinguish val (early-stopping, biased) from test/OOS (honest) from OOF (the CV analog of test) — e.g. `_carve_inner_eval_split`'s docstring explicitly documents the group-blind-carve-under-early-stopping failure mode by production incident numbers. One naming/API-contract issue found (finding #2, `target_james_stein` not being real James-Stein shrinkage) is a naming-fidelity issue, not a val/test conflation.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|---|---|---|---|---|---|
| X_ML_CORRECTNESS_META-1 | P1 | `src/mlframe/training/honest_diagnostics.py:326-338` | `_calibration_block`'s OOF-target fallback pairs `oof_probs` (train-row-order) with `test_target` (a disjoint, differently-ordered row set) by positional truncation, silently computing a meaningless calibration/ECE read-out reported as `"status": "ok"` | Remove the `test_target` fallback; when `oof_target` is absent, return `{"status": "skipped", "reason": "oof_target absent; cannot align oof_probs to labels"}`, mirroring `training/evaluation.py`'s hard-raise for the identical situation | AST/grep scanner: flag any function that reads `getattr(x, "oof_*", ...)` and, on a `None`/missing fallback, substitutes a differently-named `*_target`/`*_probs` attribute without an explicit row-count/index-identity check between the two sources |
| X_ML_CORRECTNESS_META-2 | P2 | `src/mlframe/training/feature_handling/target_encoders.py:31-33,575-580` | `LeakageSafeEncoder(method="target_james_stein")` is documented as "shrinkage toward prior with variance-aware shrinkage factor (closed-form for Gaussian targets)" but its actual implementation (`shrink = smoothing / (n_c + smoothing)`) is algebraically identical to `target_mean`/`target_m_estimate` — not a real (variance-ratio-based) James-Stein estimator, and produces bit-identical output to `target_mean` given equal `smoothing` | Either implement true variance-aware James-Stein shrinkage (`shrink` a function of within/between-category variance, not just count), or rename the method / fix the docstring to state it is a count-based smoothing alias, not James-Stein shrinkage | Property test: fit `target_mean` and `target_james_stein` with identical `smoothing` on the same `(X, y)` and assert the outputs are NOT bit-identical (currently they are) — a general "differently-named methods must diverge on at least one non-trivial input" checker over any encoder/estimator module exposing multiple named `method=` variants |

## Counts

- P0: 0
- P1: 1
- P2: 1
- P3: 0

## Narrative

### X_ML_CORRECTNESS_META-1 (P1) — honest_diagnostics calibration block pairs OOF predictions with unrelated test-split labels

`training/honest_diagnostics.py::_calibration_block` (lines 313-368) builds the reliability-plot / auto-pick-calibrator honest-diagnostics artefact from `(oof_probs, oof_target)`. When `oof_target` is not present on the model entry, it falls back:

```python
y = getattr(model_entry, "oof_target", None)
if y is None:
    y = getattr(model_entry, "test_target", None)
...
n = min(oof_arr.shape[0], y_arr.shape[0])
oof_arr = oof_arr[:n]
y_arr = y_arr[:n]
```

`oof_probs` is stamped in train-row (cross-validated) order; `test_target` is a *disjoint* set of rows from the honest test split. Truncating both to `min(len)` and pairing positionally means row *i* of the OOF predictions (a prediction for train row *i*) is scored against row *i* of `test_target` (the label of an unrelated test row). This is not a coarse approximation — the two arrays have no row correspondence at all, so the resulting ECE / auto-picked-calibrator verdict is close to random noise, yet the function returns `{"status": "ok", ...}` with no indication anything is wrong. This block feeds `metadata["honest_diagnostics"]["calibration"]`, which is explicitly the artefact meant to let a reviewer trust the suite's calibration story.

I found this by cross-referencing the exact same failure mode already fixed and hard-guarded elsewhere in the same package: `training/evaluation.py::post_calibrate_model` (both the multi-output path, lines 289-305, and the binary path, lines 354-377) explicitly documents this bug class — "the old `target_series.iloc[:len(oof)]` positional slice is only correct when train is the leading contiguous block... under a shuffled / group-aware split it fit the calibrator on mismatched (prob, label) pairs" — and now raises `ValueError` rather than falling back to any proxy when `oof_target` is missing. `honest_diagnostics.py`'s comment ("fall back to test_target as poor-but-consistent proxy") suggests the author believed this fallback was merely lossy, not structurally wrong; it is the latter. `training/_calib_oof_outputs.py` confirms `oof_target` is normally stamped alongside `oof_probs` by the standard trainer path, so the fallback is a defensive branch for model entries that reach `honest_diagnostics` without going through that exact path (e.g. externally-constructed or composite-ensemble entries) — exactly the case most likely to be under-tested.

### X_ML_CORRECTNESS_META-2 (P2) — `target_james_stein` encoder method is not James-Stein shrinkage

`training/feature_handling/target_encoders.py`'s module docstring (lines 31-33) describes `target_james_stein` as "shrinkage toward prior with variance-aware shrinkage factor (closed-form for Gaussian targets)" — i.e. a real James-Stein estimator, whose shrinkage factor depends on the ratio of within-category to between-category variance. The actual implementation, in both the full-train path (`_encode_per_row`, lines 575-580) and the OOF path (`_kfold_encode`, line 470-478, whose comment states "target_mean / target_m_estimate / target_james_stein share this OOF shape"), computes:

```python
shrink = self.smoothing / (n_c + self.smoothing)
out[i] = (1 - shrink) * m_c + shrink * prior
```

which is algebraically identical to `(n_c*m_c + smoothing*prior)/(n_c+smoothing)` — the exact same formula `target_mean`/`target_m_estimate` use. Given the same `smoothing` value, `method="target_james_stein"` and `method="target_mean"` produce bit-identical encodings; the "james_stein" label promises variance-aware shrinkage a caller does not actually get. The in-code comment even self-documents the gap ("Simplified: shrink toward prior with factor that depends on per-category sample size" — count-based, not variance-based). This is a misleading-API/naming defect: a caller choosing `target_james_stein` specifically for its variance-aware properties (e.g. on a category with low within-group variance where a real JS estimator would shrink less aggressively than count alone implies) silently gets plain count-based smoothing instead.

### Dimensions explicitly checked with no findings

- **Global RNG mutation:** grepped the full assigned scope for `np.random.seed(` and `random.seed(` — the only hit is inside an `@njit` kernel (`training/_iterative_stratification_njit.py:35`) where it seeds numba's own internal (non-numpy) RNG stream, which is documented in the module docstring as an accepted, non-bit-identical-but-quality-equivalent tie-break source; not a numpy global-state bug.
- **Mutable default arguments:** grepped for `def f(...=[]` / `def f(...={}` across the full scope — zero hits.
- **Bare/silent `except: pass`:** grepped for `except ...:` immediately followed by a bare `pass` across the full scope — zero hits; every caught exception in the sampled files is logged (via `logger.warning`/`logger.debug`/`log_throttle`) before falling through.
- **sample_weight threading in composite/ensemble sub-fits:** `_stackers.py`'s four meta-stacker fitters, `_calibration.py`'s `OutputCalibrator`, RFECV's `_eval_fold_body` (train-side AND scorer-side, plus the val-re-split re-slicing), and `calibration/post.py::compare_postcalibrators`'s inner-CV fold loop all thread `sample_weight` correctly through every sub-`.fit()` call, gated on `inspect.signature` where the wrapped estimator's support is uncertain.
- **Calibration in-sample vs OOF:** `pick_best_calibrator`'s and `compare_postcalibrators`'s `selection="same_oof"` optimism bug (Isotonic self-selecting to near-zero in-sample ECE) is already fixed and defaulted away (`selection="inner_cv"`), with the legacy path kept opt-in-only and loudly logged when used.

## Report path

`C:/Users/Admin/Machine learning/mlframe/audits/full_audit_2026-08-05/x_ml_correctness_meta.md`

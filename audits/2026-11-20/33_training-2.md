# Cluster 33 "training-2" read-only audit (partial)

Scope: 100 files from `audits/2026-11-20/manifests/33_training-2.txt` (master dd0284ce9). About 12.9k lines of non-benchmark source and about 5k lines of `composite/_benchmarks/*`. **Coverage is partial.** Only the regions cited below were read; no file was read end to end. The remaining files were checked only by grep for swallowed exceptions, fit/predict calls, quantile and rank logic, and clip/epsilon handling. The full audit is not complete.

| ID | Severity | file:line | Class | Finding | Failure scenario | Suggested fix |
|---|---|---|---|---|---|---|
| 33-F1 | P1 | src/mlframe/training/cb/_cb_pool.py:436-438 | silent exception swallowing | `_fill_cb_cat_nan_sentinel` wraps the whole NaN-to-`__MISSING__` fill in `except Exception` and logs only at DEBUG, then returns `X` unchanged or partly filled. The predict path (line 524) then calls CatBoost on a frame that still has NaN cat cells. | A predict frame with NaN in a category column hits a fill exception (for example an unexpected dtype at line 432). The default log level hides it, and CatBoost raises "Invalid type for cat_feature ... NaN" with no hint about the root cause. | Narrow the except to the expected error types and log at WARNING with the column name, or re-raise. |
| 33-F2 | P1 | src/mlframe/training/cb/_cb_pool.py:68-69 | silent exception swallowing | Label coercion (`_coerce_label_for_cb_pool`) catches `Exception` and logs at DEBUG. The label is then passed through without the float32/float64 narrowing. | A label cast fails for an unexpected type. The label reaches CatBoost in an unintended dtype and nothing is logged at default level. | Same as 33-F1: narrow the except and log at WARNING. |
| 33-F3 | P2 | src/mlframe/training/composite/conformal.py:65, 307; src/mlframe/training/composite/conformal_online.py:55, 79 | dead code and duplication | The conformal rank `math.ceil((n+1)*(1-alpha))` plus its `rank > n -> inf` guard is copied four times across two modules. The four copies can drift apart. | None today; a future change to one copy (for example a different tie or rank rule) leaves the others silently inconsistent. | Factor into one helper, for example `_conformal_rank_quantile(values, alpha, sorted_signed)`. |
| 33-F4 | P3 | src/mlframe/training/composite/classification.py:165 vs 171 | API/contract consistency | Binary `_margin_from_estimator` clips probabilities to `[1e-6, 1-1e-6]`. The multiclass branch clips to `[1e-12, 1]`. The two margin scales differ by about 2x in log space, so a base-margin estimate depends on class count. | Not a wrong result on its own, since both are valid clips. Binary margins are capped at about +/-13.8 and multiclass at about -27.6, which changes boosting init for saturated probabilities. | Use one epsilon constant and document the margin cap. |

## Checked, no finding

- `composite/conformal.py:55-70` and `:290-310`: rank is `ceil((n+1)(1-alpha))`, and `rank > n` returns `inf`. I compared the float64 computation against an exact decimal reference for n up to 3000 and alpha in {0.01 ... 0.5}. The two agree everywhere, so there is no float rank error.
- `composite/conformal_glm.py:45-123`: `_variance_function` floors V(mu) at a tiny epsilon before `sqrt`, so `residuals / std` cannot divide by zero for Poisson, Gamma or Tweedie at mu = 0.
- `composite/autoconfig.py:95-110`: skew and kurtosis return `(0, 0)` for fewer than 20 finite values or std <= 1e-12. The guard is correct.
- `composite/compare.py:140-156`: the bootstrap is row-chunked and consumes the RNG in the same order as a monolithic draw. The quantile CI logic is correct.
- `composite/conformal.py:116-135`: the sigma-bin model handles an empty finite set, collapses duplicate quantile edges with `np.unique`, and floors sigma. The edge-case branches are correct.
- `cb/_cb_pool.py:466-489`: `_cb_val_pool_predict` catches only the narrow exception tuple and falls through to the normal path, so the cache probe is not a silent failure.
- `cb/_cb_pool.py:555-570`: the predict NaN guard checks the whole output, not a sample, and logs a warning before the one-shot retry.

## Not verified (open questions, not findings)

- `cb/_cb_pool.py:432`: the NaN fill casts category columns via `astype("string")`. For a float-coded category with NaN, a non-NaN value like `1.0` becomes the string `"1.0"`. Whether this matches the training-time representation was not verified, because I did not read the fit-side cat handling. This needs a follow-up read before it can be a finding.
- `cb/_cb_pool.py:130, 137, 741, 792, 826, 842` contain more `except Exception` blocks that I did not read.
- `composite/cache_store.py:598`: `except OSError: pass` around the directory fsync. This is probably intentional for platforms that do not support directory fsync. Not read in context.
- `composite/compare.py:75`: accuracy thresholds soft predictions at 0.5 only when they are non-integer. Not a bug in the cases I traced.

## Files read fully

None. Partial reads: `cb/_cb_pool.py` (about 60 lines), `composite/conformal.py` (about 90 lines), `composite/conformal_glm.py` (about 80 lines), `composite/classification.py` (about 30 lines), `composite/autoconfig.py` (about 20 lines), `composite/compare.py` (about 40 lines), `composite/conformal_online.py` (about 20 lines).

## Coverage gaps

- Not read at all: 92 of 100 manifest files, including `composite/diagnostics.py`, `composite/cache_store.py`, `composite/cache.py`, `composite/_moe_gate.py`, `composite/_canonical_hash.py`, and `composite/cv.py`. These are the largest and riskiest remaining targets.
- Grep only: the 40 `composite/_benchmarks/*` files. They are not runtime code, so this is low risk.
- No public function in scope was checked for test coverage. The report does not list coverage gaps.


## Dispositions

- 33-F1: FIXED. the CatBoost categorical NaN fill failure is logged at WARNING with the error type; the except is kept broad (not narrowed) so the fill never raises into predict; no separate test
- 33-F2: FIXED. label-stacking failure logged at WARNING; no separate test
- 33-F3: FIXED. The conformal rank rule lives in one helper (conformal_rank), used by the split, signed and online calibrations. Two further copies in conformal_classification.py and conformal_glm.py are outside the finding and are not yet changed. Tests: tests/training/composite/test_biz_val_conformal_ood.py (16 pass).

- 33-F4: DEFERRED. Re-read: both clips are valid; unifying the epsilon changes the saturated-probability margins that seed boosting (binary capped near +/-13.8 vs multiclass near -27.6) for existing runs, so picking the single constant is a modelling choice for the composite-classification owner rather than a bug fix.

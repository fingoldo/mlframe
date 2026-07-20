# Clustering / stability selection / Dynamic Cluster Discovery (DCD)

10 findings, 5 proposals.

## Findings

### [P1] bug -- src/mlframe/feature_selection/filters/stability.py:170

**StabilityMRMR's per-bootstrap fit has no try/except, so ONE degenerate subsample crashes the whole .fit() call instead of being excluded like its sibling _stability_cluster.py implementations.**

MRMR(dcd_enable=..., ...) wrapped in StabilityMRMR(n_bootstraps=20, sample_fraction=0.5). `_one_bootstrap` calls `est.fit(X_sub, y_sub)` with no try/except (line ~179-181); if any one of the 20 subsamples happens to be degenerate for the inner MRMR (e.g. a rare class dropped below the stratify-safety floor, a permutation-test numerical edge case, or any transient inner exception), the exception propagates uncaught out of `StabilityMRMR.fit`, aborting the ENTIRE stability-selection run instead of excluding that one bootstrap and continuing (as `cluster_stability_selection`/`complementary_pairs_stability` in `_stability_cluster.py` explicitly do, with n_success/n_failed tracked and surfaced -- see the hierarchy-stability-9 audit fix in that sibling file). With `n_jobs>1` the same uncaught exception propagates through joblib's `Parallel(...)`.

### [P1] bug -- src/mlframe/feature_selection/filters/_stability_fe.py:138

**stability_select_fe's per-bootstrap MRMR fit (_run_bootstraps) has no try/except, so one degenerate bootstrap subsample crashes the whole stability sweep, the same gap as StabilityMRMR.**

stability_select_fe(X, y, n_bootstraps=10, sample_fraction=0.75) subsamples n=10 times and calls `m.fit(Xb, yb)` (line 148) with no exception guard. Any one bootstrap draw that trips an inner MRMR/FE edge case (e.g. a near-constant subsampled column, a degenerate permutation null) raises out of `_run_bootstraps` and aborts `stability_select_fe`/`StabilityFESelector.fit` entirely, unlike the explicitly-hardened bootstrap loops in `_stability_cluster.py` that tolerate a fraction of failures and report n_effective/n_failed.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_cluster_aggregate.py:398

**The FCBF-style ordered-relevance pruning added to _discover_clusters to reject single-linkage chain artifacts (A-B-C merged via bridge member B) has no dedicated behavioral test exercising the exact scenario its docstring describes.**

A dataset with three features A, B, C where corr(A,B) and corr(B,C) both clear corr_threshold but corr(A,C) does not (a genuine bridge/chain, not a shared-latent reflection group) should be split so the FCBF prune keeps only {rep, direct-correlates-of-rep} and drops the transitively-bridged member. Grepping tests/feature_selection/clustering and tests/feature_selection/filters for FCBF/chain/bridge/transitive terminology found no test constructing this exact A-B-C bridge fixture and asserting the chain member is excluded from the final cluster (test_cluster_aggregate_knobs.py covers the seven discovery knobs on a single-latent 6-member reflection fixture only, not the chain-rejection code path itself). A regression here (e.g. an accidental revert to using `abs(corr[a,b])>=corr_threshold` transitively instead of vs-representative) would silently re-admit spurious multi-factor clusters and nothing would fail.

### [P2] bug -- src/mlframe/feature_selection/filters/_ks_stability.py:92

**ks_stability_filter's multi-split mode has no validation on split_frac; a caller-supplied split_frac > 1.0 crashes with an opaque numpy ValueError instead of a clear input-validation error.**

ks_stability_filter(train_df, test_df, n_splits=5, split_frac=1.5) computes `train_size = max(1, round(train_vals.size * split_frac))` which exceeds `train_vals.size`, then `rng.choice(train_vals, size=train_size, replace=False)` raises `ValueError: Cannot take a larger sample than population when 'replace=False'` from inside numpy with no context tying it back to the bad `split_frac` argument. Sibling stability helpers in this cluster (`_stability_fe.py`'s `_bootstrap_indices`, `stability.py`'s `StabilityMRMR.fit`) explicitly validate their analogous fraction parameters up front with a clear ValueError; this one does not. No test in tests/feature_selection/biz_val/test_biz_val_filters_ks_stability*.py exercises split_frac outside (0,1].

### [P2] bug -- src/mlframe/feature_selection/filters/_stability_cluster.py:224

**Per-bootstrap selector_fn failures are caught with a bare except Exception that discards the exception entirely (not even logged per-occurrence); only an aggregate count is reported at the end, so the actual failure cause is unrecoverable from logs when bootstraps fail.**

If a user's selector_fn systematically raises on every bootstrap in `cluster_stability_selection`/`complementary_pairs_stability` (e.g. a subtle shape/dtype bug in their own selector callable, or an inner MRMR fit consistently failing on the half-sample size), the only diagnostic is a final `logger.warning('%d/%d bootstraps failed ...')` with no exception type/message/traceback for any individual failure -- a user cannot tell if failures are all the SAME root cause or N different ones without re-running under a debugger. Every other except-Exception in this cluster (_dcd_swap.py, _dynamic_cluster_discovery/__init__.py, _screen_dcd_swap.py) logs `%r` of the caught exception at least once; these two loops (lines ~224 and ~314) do not.

### [P2] test_gap -- tests/feature_selection/clustering/test_dcd_swap_no_debug_prints.py:90

**The existing debug-print regression guard only asserts the substring "DEBUG" is absent from stdout, which would NOT catch a leftover print using a different prefix such as "DBGX".**

The task brief for this audit states a concurrent session had a live `print(f"DBGX ...")` mid-edit in `_dcd_swap.py` as of 2026-07-20. As read for this audit, no such print statement is present anywhere under src/ (grepped the whole tree) -- so it was either not yet inserted or already removed by the time of this read; I cannot confirm its live presence, only flag the tracked risk per the task brief. Separately and regardless of that specific instance: `test_evaluate_swap_candidate_prints_nothing_to_stdout` / `test_evaluate_swap_candidate_below_threshold_prints_nothing` assert `"DEBUG" not in captured` (line 90) and `buf.getvalue() == ""` (line 125) respectively -- the second (stricter) test would catch ANY stdout output including a DBGX-prefixed print, but only exercises the below-threshold early-return path; the first test (the swap-eligible path most likely to carry an in-progress debug print) only checks for the literal substring "DEBUG" and would silently pass with a live "DBGX..." print still in place.

### [P2] bug -- src/mlframe/feature_selection/filters/_dynamic_cluster_discovery/_dcd_swap.py:721

**commit_swap's is_member_swap detection uses getattr(decision, "branch", "aggregate") with a default value that can never actually be reached, since decision is always a fully-constructed SwapDecision dataclass instance whose branch field always exists (default "none", never absent).**

No runtime failure -- this is dead/misleading code: the `"aggregate"` fallback in `getattr(decision, "branch", "aggregate")` implies branch could be missing on some SwapDecision instances, but the dataclass declares `branch: str = "none"` (init/__init__.py line ~103) so the attribute is always present with at worst the value "none", never absent. A future reader could be misled into thinking this getattr is a real defensive path against an object built without a branch field, when it structurally cannot be.

### [P2] design -- src/mlframe/feature_selection/filters/_dynamic_cluster_discovery/_dcd_metrics.py:171

**state._joint_entropy_batch_cache is set/read via getattr/direct attribute assignment (in _dcd_metrics.py and _dcd_pair_su_batch.py) without being declared as a DCDState dataclass field or annotated with a type: ignore[attr-defined], unlike the sibling dynamic attributes _auto_method_cache and _warned_sotoca_membership which ARE marked.**

mypy-hostile pattern only (per audit instructions: note, do not fix). `state._joint_entropy_batch_cache = _joint_cache` (in _dcd_pair_su_batch.py) and `getattr(state, "_joint_entropy_batch_cache", None)` (in _dcd_metrics.py) both operate on an attribute that mypy cannot see declared anywhere on DCDState -- unlike `state._auto_method_cache` in _dcd_swap.py which carries an explicit `# type: ignore[attr-defined]` marker documenting the same pattern. A strict mypy run on this file would (or should) flag attr-defined here with no suppressing comment to explain it's intentional runtime-only bookkeeping.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_dynamic_cluster_discovery/_dcd_swap.py:38

**_select_swap_method_auto's K-fold OOF bake-off cache (state._auto_method_cache, keyed by tuple(member_names)) has no direct unit test verifying the cache actually short-circuits re-evaluation of the SAME cluster across repeated calls, nor that a different member set produces a cache miss.**

A future edit that changes the cache key derivation (e.g. accidentally keying on `anchor` alone instead of the full member-name tuple) would silently let two DIFFERENT clusters share a stale K-fold winner/method, and no existing test (grepped for _auto_method_cache / cache_key across tests/) would catch it -- the swap tests exercise end-to-end MRMR.fit outcomes (selection-equivalence), which would only fail if the wrong method happened to change the accept/reject outcome, not merely the recorded method choice.

### [P2] design -- src/mlframe/feature_selection/filters

**CPU/GPU parity and GPU-residency angles: this entire cluster (clustering / stability selection / DCD) contains zero GPU code paths -- confirmed clean, not merely unexamined.**

_(no issues found in this cluster for this angle)_

## Proposals

### (coverage_gap) Add a bootstrap-failure-resilience test + fix for StabilityMRMR and stability_select_fe

Mirror the _stability_cluster.py pattern (try/except around the inner .fit() call, count n_success/n_failed, divide by n_success, log a warning naming the exception) in stability.py's StabilityMRMR._one_bootstrap and _stability_fe.py's _run_bootstraps. Add a regression test analogous to test_stability_cluster_effective_bootstraps.py that injects a failing estimator/MRMR on every Nth bootstrap and asserts the run completes with a reduced effective count rather than raising.

### (coverage_gap) FCBF chain-artifact rejection test for _discover_clusters

Build a 3-feature A-B-C bridge fixture (corr(A,B) and corr(B,C) both >= corr_threshold, corr(A,C) well below it -- e.g. B = 0.5*A + 0.5*C + noise) and assert the discovered cluster excludes the non-directly-correlated member per the FCBF-style ordered-relevance pruning in _cluster_aggregate.py's _discover_clusters, distinguishing it from a genuine single-latent reflection group where all pairwise correlations clear the threshold.

### (edge_case) Validate split_frac in ks_stability_filter

Raise a clear ValueError when split_frac is not in (0, 1], mirroring _bootstrap_indices' validation in _stability_fe.py, instead of letting numpy's rng.choice raise an opaque 'Cannot take a larger sample than population' error.

### (other) Log the actual exception in cluster_stability_selection / complementary_pairs_stability bootstrap loops

Change `except Exception: n_failed += 1; continue` to at least log the first occurrence's %r (matching the pattern used everywhere else in this cluster, e.g. commit_swap's build-recipe except block), so a systematic selector_fn bug is diagnosable from logs instead of only visible as a raw failure count.

### (coverage_gap) Strengthen test_dcd_swap_no_debug_prints.py to assert on full stdout emptiness, not a DEBUG substring

Change the swap-eligible-path test's assertion from `"DEBUG" not in captured` to `captured == ""` (matching the stricter below-threshold test already in the same file), so any future leftover debug print regardless of its prefix (DBGX, TRACE, etc.) is caught.

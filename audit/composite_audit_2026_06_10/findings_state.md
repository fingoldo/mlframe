# Composite Targets audit — dimension: Cache / streaming / provenance / diagnostics

Auditor scope: src/mlframe/training/composite/{cache.py, streaming.py, provenance.py, diagnostics.py} (+ direct call sites: estimator/_update.py, core/_phase_composite_discovery.py).
Date: 2026-06-10. All line numbers from current working tree. Empirical claims verified on this machine (Python 3.14.3, polars 1.41.2) — verification noted inline.

---

## S1 — P1 / bug — src/mlframe/training/composite/cache.py:362-363, 419-423, 164-175

**Title: data_signature hashes raw object-array POINTER bytes for string/categorical/datetime columns — signature is nondeterministic, discovery cache can never hit on realistic frames.**

Three sites fold np.ascontiguousarray(<object-dtype ndarray>).tobytes() into the blake2b digest:

1. cache.py:362-363 — polars non-numeric sample: df.get_column(c).gather(sample_idx).cast(pl.Utf8).to_numpy() returns OBJECT dtype (verified, polars 1.41.2); .tobytes() serialises the PyObject* addresses, not the string content.
2. cache.py:423 — pandas per-column sample: sampled = full[sample_idx] for an object/string column → same pointer bytes.
3. cache.py:164-175 — pandas _row_order_fingerprint: df.head(n_take).to_numpy() on any mixed-dtype frame coerces to object → pointer bytes. The except (TypeError, ValueError) repr-fallback (lines 166-168, 172-173) NEVER fires — .tobytes() on object arrays does not raise (verified).

Empirical proof (run on this machine):
- polars frame with one string column: data_signature(df, ...) != data_signature(df, ...) WITHIN the same process (gather materialises fresh str objects per call → new addresses).
- pandas frame with one string column: stable within a process but DIFFERENT digest on every new process (run 1: 2f72e04c…, run 2: a31f4a57…).

Impact: the production caller passes ALL frame columns (_phase_composite_discovery.py:162: _disc_feature_cols = list(filtered_train_df.columns)), so any frame containing a string / categorical / datetime column (nearly every real dataset) gets a fresh signature per run → the disk-backed discovery cache silently never hits and the "minutes on multi-million-row frames" MI/Wilcoxon/rerank cost is re-paid every run. Fail-open (results stay correct), hence P1 not P0 — but the feature is dead in realistic use and nothing logs it.

Why tests missed it: every signature-stability test uses numeric-only frames inside one process (see S16).

Suggested fix: never .tobytes() an object array. For sampled string content use a content encoding, e.g. ("\x1f".join("" if v is None else str(v) for v in sampled)).encode("utf-8") or np.asarray(sampled, dtype="U").tobytes() (fixed-width unicode = content bytes). For the polars sample, df.get_column(c).gather(sample_idx).hash() (vectorised content hash) avoids Python-str materialisation entirely. For the pandas head/tail fingerprint use pd.util.hash_pandas_object(df.head(n_take), index=False).to_numpy().tobytes() (deterministic content hash, also fixes the dead except). Add the cross-process regression test of S16 in the same change.

---

## S2 — P1 / perf — src/mlframe/training/composite/cache.py:156

**Title: _row_order_fingerprint polars path runs hash_rows() over the ENTIRE frame to fingerprint only the first 256 rows.**

row_hashes = df.hash_rows().slice(0, n_take).to_numpy() — DataFrame.hash_rows() computes a u64 hash for EVERY row of EVERY column (full O(N*C) frame scan + n-row u64 allocation), then all but 256 values are discarded. On the multi-million/100GB-class frames this module explicitly optimises for (CACHE-P0-1/2 comment at lines 320-326 celebrates eliminating exactly this class of whole-frame work for ~100x), this re-introduces a full extra frame pass per signature call.

Verified: df.slice(0, 256).hash_rows() is BIT-IDENTICAL to df.hash_rows().slice(0, 256) on polars 1.41.2 (per-row content hash is row-independent) — the fix is free.

Suggested fix: row_hashes = df.slice(0, n_take).hash_rows().to_numpy() (one line). Optionally also hash a tail slice (see S3).

---

## S3 — P2 / bug — src/mlframe/training/composite/cache.py:159-175 (pandas), 150-158 (polars)

**Title: Residual row-order blind spots: pandas fingerprint still head/tail-8 only; polars fingerprint lost tail coverage.**

Post-audit-D-P1-2 coverage is asymmetric and incomplete:
- pandas: only the first 8 and last 8 rows are hashed (n_edge=8). A reorder confined to rows 8..n-9 that doesn't move a sampled position (1000 of N) produces an IDENTICAL signature → stale spec replay — the exact bug class the fix targeted, still open for pandas. A two-row inner swap on a 10M-row frame has ~0.02% chance of touching a sampled position.
- polars: the new prefix covers rows 0-255 but the tail fold was dropped — a reorder/edit confined to the tail region (beyond all sampled positions) is invisible.

Suggested fix: pandas — hash first/last 256 rows via pd.util.hash_pandas_object (content-based; simultaneously fixes S1 at this site). polars — add df.slice(max(0, h-256), 256).hash_rows() next to the S2 fix. Document that positions outside (prefix + tail + sample) remain a known sampling blind spot.

---

## S4 — P2 / bug — src/mlframe/training/composite/cache.py:816-829

**Title: LRU eviction deletes the .sha256 sidecar even when deleting the value file FAILED — surviving entry becomes permanently unverifiable.**

The os.remove(path + ".sha256") block runs unconditionally after "except OSError: pass" on the value removal. On Windows, os.remove(path) raises PermissionError whenever a concurrent process holds the file open (e.g. a sibling get() mid-read — a scenario the module's own filelock machinery treats as realistic). The entry then survives WITHOUT its sidecar: under MLFRAME_DISCOVERY_CACHE_STRICT=1 it is refused forever; non-strict degrades to unverified-WARN.

Suggested fix: move the sidecar removal inside the success branch (after removed += 1).

---

## S5 — P2 / usability — src/mlframe/training/composite/cache.py:708-721

**Title: get() swallows corrupt / tampered / unloadable entries with zero logging — operator silently re-pays full discovery every run.**

Both the typed except (FileNotFoundError, OSError, EOFError, UnpicklingError, AttributeError, PickleVerificationError) and the blanket "except Exception" return default silently. FileNotFoundError is a legitimate silent miss, but a PERSISTENT corrupt/tampered entry (digest mismatch, truncated file, renamed class) produces an unbounded sequence of silent multi-minute recomputes with no operator signal — in contrast, the oversize path right above (689-699) correctly emits logger.warning. Matches the project's known "silent error swallowing" bug class.

Suggested fix: split the except: FileNotFoundError → silent miss; everything else → logger.warning("DiscoveryCache: unreadable entry %s (%s: %s); treating as miss", ...) (optionally once-per-path).

---

## S6 — P2 / perf — src/mlframe/training/composite/cache.py:262-267

**Title: pandas _col_stats int branch runs np.unique(arr) — full O(n log n) sort per integer column — the exact cost the polars branch deliberately removed.**

The polars numeric branch dropped nuniq in favour of null= precisely because "nuniq previously required a full column scan" (comment 384-389), accepting a digest-shape change. The pandas branch (line 266) still pays a full sort of the whole column per int column per signature call — seconds on multi-million-row int-heavy pandas frames, on top of the unavoidable per-column to_numpy().

Suggested fix: mirror the polars digest shape: intmin=...;intmax=...;null=0 (numpy int dtypes cannot hold NaN). One-time cache invalidation, same as already accepted for polars.

---

## S7 — P2 / bug — src/mlframe/training/composite/streaming.py:84-110

**Title: Drift detector tests ONLY alpha — a pure intercept (beta) drift never triggers a refit, yet it biases every prediction.**

z = |alpha_buf - current_alpha| / SE(alpha) (line 99) is the sole trigger. A regime change y = alpha*base + (beta + D) (level shift — the most common production drift for residual targets) keeps alpha stable, so reason="no_drift" is returned forever while predictions carry constant bias D. The fresh beta_buf is computed (line 80) but only deployed when the ALPHA z-test fires. Naming nit: docstring says "Chow-style" but a Chow test is an F-test on pooled-vs-split SSE over both coefficients; this is a single-coefficient Wald z.

Suggested fix: add the intercept test — SE(beta) = sigma_resid * sqrt(1/n + mean(base)^2/(n*var(base))), fire on max(z_alpha, z_beta) > threshold (or implement the actual 2-coefficient Chow F). Ship a biz_value test: a synthetic pure-level-shift stream must trigger a refit.

---

## S8 — P2 / extension — src/mlframe/training/composite/streaming.py:76-105

**Title: Refit-on-detection fits on the FULL rolling buffer (pre-drift + post-drift rows mixed) — the deployed alpha is a lagged blend biased toward the dead regime.**

When the z-test fires, alpha_buf/beta_buf were estimated over the whole buffer (default 10k rows). If the change-point sits at 80% of the buffer, the refit coefficients are an ~80/20 blend of old and new regimes — systematically wrong exactly when correction was requested, needing several buffer turnovers to converge. Standard streaming practice fits the post-change segment.

Suggested fix: after detection, locate the change-point (two-segment SSE split maximisation, O(n) with prefix sums) — or cheaply refit on the trailing half — and return coefficients fitted on post-change rows only (subject to min_buffer_n). Report the change-point index in info.

---

## S9 — P2 / perf — src/mlframe/training/composite/estimator/_update.py:54-64

**Title: Every update() round-trips the whole 10k-row buffer through boxed Python floats.**

self._buffer_y_.extend(y_arr.tolist()) boxes each incoming value; np.asarray(self._buffer_y_, dtype=np.float64) then unboxes the ENTIRE deque (default online_refit_buffer_n=10_000) on EVERY call — O(buffer_n) Python-level work per update regardless of batch size. For the intended high-frequency streaming harness the boxing dominates the OLS by orders of magnitude.

Suggested fix: replace the two deques with a preallocated (buffer_n,) float64 ring buffer (head index + count); update() writes via vectorised slice assignment; the check reads at most 2 contiguous views. No semantic change; FIFO eviction stays exact.

---

## S10 — P2 / usability — src/mlframe/training/composite/provenance.py:168-179 (+ fields 50-78)

**Title: Audit trail prints the LEGACY target__transform__base name — which no longer matches spec.name anywhere else — and multi-base specs are misrepresented as single-base.**

1. to_audit_trail reconstructs "{target_col}__{transform_name}__{base_column}" (line 169). Per spec.py:33-39 the canonical name is the hyphen short form (e.g. y-linres-lag1); the double-underscore format "is no longer produced by discovery". In report_to_markdown the section header prints the real spec.name (line 308) and the next paragraph names the same object y__linear_residual__lag1 — two identifiers for one spec in one report; stakeholders cannot cross-reference metadata/model keys.
2. CompositeProvenance has no extra_base_columns field; for a multi-base spec (spec.py:56) provenance reports only the first base — the "reproduce the inverse at serving time" payload is incomplete exactly where the inverse needs the extra columns.

Suggested fix: add name: str (from spec.name) and extra_base_columns: tuple[str, ...] to the dataclass + to_dict + the trail text; use self.name in to_audit_trail.

---

## S11 — P2 / docs — src/mlframe/training/composite/provenance.py:219-226

**Title: logratio inverse formula hardcodes 10* instead of the fitted soft_cap_k, and says "softcap" where the implementation is a hard clip.**

Rendered: y_hat = base * exp(softcap(T_hat, median_t +/- 10*mad_eff)). Actual inverse (transforms/linear.py:120-130): np.clip(t_hat, median_t - k*mad_eff, median_t + k*mad_eff) with k = fitted_params["soft_cap_k"] — stored per-fit precisely so it can vary. If soft_cap_k ever differs from 10 the stakeholder formula is wrong; and "softcap" implies smooth saturation when it is a hard clip — a reader implementing the inverse from the formula alone (the class's stated purpose) reproduces different predictions near the cap.

Suggested fix: interpolate k = fitted_params.get("soft_cap_k", 10.0); render clip(T_hat, median_t +/- k*mad_eff).

---

## S12 — P2 / extension — src/mlframe/training/composite/provenance.py:183-240

**Title: Formula/description table covers 4 of ~36 registered transforms; all others render a generic stub — provenance degrades to an opaque key for ~90% of the catalogue.**

_TRANSFORM_DESCRIPTIONS + _format_transform_formulas know diff, ratio, logratio, linear_residual only. The registry (transforms/registry.py:196-659) registers ~36 transforms (additive_residual, median_residual, y_quantile_clip, linear_residual_robust/multi/grouped, quantile_residual, monotonic_residual, ewma_residual, rolling_quantile_ratio, frac_diff, cbrt_y, log_y, yeo_johnson_y, quantile_normal_y, 5 chains, asinh_residual, centered_ratio, polynomial_residual_deg2, rank_residual, smoothing_spline_residual, reciprocal_residual, geometric_mean_residual, pairwise_interaction_residual). All fall through to "T = forward(...) [name]" — contradicting the class docstring promise "(b) reproduce the inverse at serving time without consulting source code". Low-hanging: the robust/multi/grouped/additive family stores plain alpha/beta(-vectors) and could reuse the linres formatter today.

Suggested fix: make the human formula a field of the Transform registry entry (template + param names) so every registered transform carries its own forward/inverse text and a new transform CANNOT silently fall through; keep only the generic fallback in _format_transform_formulas. Until then, extend the dict to the alpha/beta family + chains (compose the two stage formulas).

---

## S13 — P2 / bug — src/mlframe/training/composite/provenance.py:249, 303-306

**Title: report_to_markdown(random_state=42) default fabricates provenance — the audit paragraph asserts "was discovered using random_state=42" even when the real discovery seed differed.**

to_audit_trail prints discovery_random_state as a load-bearing reproducibility claim. report_to_markdown silently injects 42 when the caller omits the kwarg (docs/examples/composite_targets.md callers do exactly this). A wrong-by-default seed in a stakeholder audit artefact is the same trust-failure class as a hallucinated citation.

Suggested fix: make random_state keyword-required without default, or accept None and render "random_state=unspecified".

---

## S14 — P2 / bug — src/mlframe/training/composite/diagnostics.py:40-52

**Title: _lazy_pyplot's Agg-selection branch is dead code — matplotlib.get_backend() never returns falsy — so the documented headless-safety doesn't exist.**

get_backend() resolves the _auto_backend_sentinel and always returns a non-empty backend string, so "if not matplotlib.get_backend():" can never be True and matplotlib.use("Agg") is unreachable. Consequence: on a workstation with a display, the first diagnostics call resolves a full interactive backend (Tk/Qt) inside library code — slower figure creation and thread-unsafe (Tk in worker threads). The comment also misquotes its reference: tests/training/conftest.py:10 does an UNCONDITIONAL matplotlib.use('Agg') before any pyplot import — the opposite of this code.

Suggested fix: check the unresolved rcParam sentinel (dict.__getitem__(matplotlib.rcParams, "backend")) instead, or document that the caller owns backend selection and delete the dead branch + misleading comment.

---

## S15 — P2 / bug — src/mlframe/training/composite/diagnostics.py:383 (vs pyproject.toml:64)

**Title: ax.boxplot(..., tick_labels=names) requires matplotlib >= 3.9, but the project pins matplotlib>=3.7 — plot_per_fold_tiny_rmse crashes with TypeError on 3.7/3.8.**

The labels → tick_labels rename landed in matplotlib 3.9; on 3.7/3.8 the kwarg is unknown. Line 393 already calls ax.set_xticklabels(names, rotation=30, ...) making the boxplot kwarg redundant anyway.

Suggested fix: drop tick_labels=names from the boxplot call (set_xticklabels at :393 already does the job version-independently), or bump the pin to matplotlib>=3.9 in both dependency blocks (pyproject.toml:64, 174).

---

## S16 — P2 / test-gap — tests/training/composite/test_composite_cache_row_order.py, tests/training/composite/test_composite_discovery_cache.py

**Title: All data_signature determinism tests use numeric-only frames within a single process — the exact blind spot that let S1 ship.**

test_data_signature_still_stable_for_identical_frame (row_order:94-99) and test_same_df_same_signature (discovery_cache:21) build float/int-only frames and compare digests inside one interpreter. Neither string/categorical/datetime columns nor cross-process stability (the actual disk-cache contract: "re-run discovery" = new process) is asserted.

Suggested fix: add (a) a same-process double-call equality assert on a frame with str + datetime columns for BOTH pandas and polars (the polars one fails today); (b) a subprocess test that computes the signature of a canned frame in a child process and asserts equality with the parent's digest. Verify (b) fails pre-S1-fix per the test-every-bug-fix rule.

---

## S17 — LOW / docs — src/mlframe/training/composite/cache.py:498, 562-577

**Title: Stale concurrency docstring + promised-but-missing WARN on absent filelock.**

Class docstring still says "Thread-safe for single-process use only; concurrent writers from multiple processes will race on the same key (caller's responsibility)" — contradicted by the cross-process machinery the class now ships (filelock-guarded LRU/eviction at 556-577/766-783, atomic os.replace value writes, idempotent invalidate). Separately, the DISC-LRU-RACE comment promises filelock absence "falls back ... with a one-time WARN", but "except ImportError: return contextlib.nullcontext()" is silent — no warning exists.

Suggested fix: rewrite the docstring to state actual guarantees; add the one-time logger.warning (module-level flag) or delete the WARN claim.

---

## S18 — LOW / perf — src/mlframe/training/composite/cache.py:788-806 vs 956-972; 936-953

**Title: Byte-budget accounting inconsistency + orphan tmp/lock files never cleaned.**

(1) Eviction's total_bytes sums only *.pkl sizes (788-797) while the documented companion _discovery_cache_bytes_total counts .pkl PLUS .pkl.sha256 (963-965) — the enforced cap and the reported footprint disagree by the sidecar overhead. (2) A crash between mkstemp and os.replace (in set() or _save_lru) leaves *.tmp / .lru.*.tmp orphans that neither eviction (globs *.pkl only) nor clear() ever deletes; .lru.lock also survives clear().

Suggested fix: include sidecar sizes in the eviction total (or exclude them from the byte-total helper — pick one, document); have clear() (and optionally eviction) sweep *.tmp older than ~1h and the .lru.lock.

---

## S19 — LOW / bug — src/mlframe/training/composite/provenance.py:97-106

**Title: composite_id hashes json.dumps(..., default=str) — numpy arrays in fitted_params stringify via str(ndarray), which truncates and obeys global printoptions.**

For array-valued params (median_residual bin_edges/bin_medians, monotonic_residual knots, spline coefficients), default=str yields the numpy repr: elements beyond np.printoptions threshold (default 1000) collapse to "..." (two specs differing only in the truncated region share an id), and user code that changes printoptions (precision/threshold) changes the digest — breaking "same spec recurring in future runs is recognisable".

Suggested fix: canonicalise before dumping: default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o) (or hash o.tobytes() for large arrays).

---

## S20 — LOW / bug — src/mlframe/training/composite/provenance.py:292-301

**Title: Ensemble weight lookup reports only the FIRST component matching spec.name + "#" — multi-model specs show one arbitrary model's weight as "the" spec weight.**

Component names are f"{spec_name}#{i}" with i over multiple trained models per spec (_phase_composite_post_xt_ensemble/__init__.py:206). The break at first match means the audit paragraph claims "received weight w0" when the spec's actual ensemble mass is the SUM over its components. (Edge: a base column containing '#' can prefix-collide across specs since column names are user-controlled.)

Suggested fix: sum weights over all components matching the prefix and report "weight {sum:.3f} across {k} model(s)"; match on exact nm.rsplit("#", 1)[0] == spec.name to kill the collision edge.

---

## S21 — LOW / bug — src/mlframe/training/composite/diagnostics.py:606-615, 1-29

**Title: plot_mi_gain_with_jitter missing from __all__ (only the deprecated alias is exported); module docstring says "Four plot helpers" but there are eight.**

__all__ lists plot_mi_gain_with_ci (the DeprecationWarning alias) but not the canonical plot_mi_gain_with_jitter — star-import users only reach the function through the deprecated name. The header docstring enumerates 4 plots; the module has 8 public helpers.

Suggested fix: add plot_mi_gain_with_jitter to __all__; refresh the docstring list.

---

## S22 — LOW / bug — src/mlframe/training/composite/diagnostics.py:437-451

**Title: Spearman via double-argsort assigns arbitrary distinct ranks to TIED scores — biased rank-correlation in the family-disagreement heatmap.**

np.argsort(np.argsort(...)) gives ties consecutive integer ranks in stability order rather than average (fractional) ranks, so equal per-family RMSEs (plausible when two specs are mathematically redundant — cf. is_redundant_with_linres) bias the displayed correlation. scipy is already imported elsewhere in the module.

Suggested fix: corr[i, j] = scipy.stats.spearmanr(score_matrix[i], score_matrix[j]).statistic (keep the std==0 NaN guard).

---

## S23 — LOW / usability — src/mlframe/training/composite/diagnostics.py:500-505

**Title: plot_alpha_stability with len(window_indices) != len(alpha_per_window) produces a cryptic matplotlib shape error.**

x = np.arange(len(window_indices)) then ax.plot(x, alpha_arr) — mismatched lengths raise "ValueError: x and y must have same first dimension" deep in matplotlib with no hint which argument is wrong.

Suggested fix: validate up front with a message naming both lengths.

---

## S24 — LOW / docs — src/mlframe/training/composite/provenance.py:17

**Title: _TRANSFORMS_REGISTRY imported and never used.**

Sole reference is the import line itself. Dead import; mildly misleading (suggests formulas are registry-driven — they are not, see S12). If S12's registry-driven formula field lands, the import becomes real; otherwise delete it.

---

## S25 — LOW / extension — src/mlframe/training/composite/cache.py:429-466, 100-110

**Title: compute_config_signature_v1 makes version-folding opt-in — direct API users following the module docstring build keys with no code-version component.**

library_versions defaults to None and the module-header recipe (100-110) for "callers manage cache lookup at their orchestration level" never mentions it. The suite caller does it right (_phase_composite_discovery.py:85-115 folds mlframe + 8 libs + python), but an R&D user wiring make_discovery_cache_key(data_signature(...), col, compute_config_signature_v1(cfg)) per the docstring gets keys that survive mlframe upgrades → stale spec replay across versions (screening-default changes like the knn→bin / mi→hybrid flips would silently replay old-default results).

Suggested fix: fold a module-level _DISCOVERY_CACHE_SCHEMA_VERSION + mlframe.__version__ into compute_config_signature_v1 unconditionally (keep library_versions as the richer override); bump the constant whenever payload shape or discovery semantics change.

---

## Verified-clean (inspected, no finding — do not re-flag without new evidence)

- Long-path \\?\ prefix + glob.glob("*.pkl"): empirically verified working on Python 3.14 (the ? in the prefix is handled by glob) — NOT a bug on the supported interpreter.
- rng.choice(n, size=1000, replace=False): O(k), no O(n) permutation alloc (timed 2e7 vs 2e9 populations: ~0.1ms both).
- streaming SE formula: sigma_resid/(sqrt(n)*std0(base)) is the correct OLS slope SE with ddof=0 std; SSE/(n-2) is the right residual df.
- set() atomic write: fsync-before-replace + fd-adoption tracking + POSIX dir-fsync + Windows skip — correct.
- New-entry LRU participation in eviction (touch before evict) — correct; legacy ts=0 evict-first is documented intent.
- _safe_key hex passthrough + blake2b-with-bytelen for non-hex — collision-safe.
- plot_qq Filliben positions + linregress fit — matches scipy.probplot math; tail-preserving decimation sound.
- Welford / MI-LCB / transform round-trip items from tests/composite_discovery_audit_notes.md — not re-reported.
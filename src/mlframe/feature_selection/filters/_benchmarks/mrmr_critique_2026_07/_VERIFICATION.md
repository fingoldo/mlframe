# MRMR critique — completeness verification (READ-ONLY audit)

Cross-checked every finding ID in all 7 per-agent reports against `_TRACKER.md` (PROGRESS LOG + disposition tables) and against the actual code (grep/read + `git log`/`git show`).

## Finding-by-finding table

| ID | Report disposition | Tracker live status | VERIFIED in code? | Still TODO? |
|----|--------------------|---------------------|-------------------|-------------|
| S-F1 | FIX (GPU null budget) | DONE (84a0f6a8) | YES — gpu.py:922 `npermutations = max(int(npermutations), _NULL_MEAN_MIN_PERMS)`; :1016 `full_budget=npermutations` | no |
| S-F2 | FIX (thread use_jmim) | REMAINING/TODO | n/a (not attempted) | YES |
| S-F3 | FIX | REVERTED→FUTURE | YES reverted — exponent `** (nexisting + 1)` present in jmim branch evaluation.py:364 (original behavior); concrete bench path recorded | FUTURE |
| S-F4 | FIX | REMAINING/TODO | n/a | YES |
| S-F5 | FIX (validate) | DONE (33343a58) | YES — `_VALID_REDUNDANCY_AGGREGATORS=(None,'jmim','auto')` wired into _mrmr_validate_transform.py:45 | no |
| N-F1 | FIX (thread use_mm) | REMAINING/TODO | n/a | YES |
| N-F2 | FIX (validate wide) | REMAINING/TODO | n/a | YES |
| N-F3 | FIX | REVERTED→FUTURE | YES reverted — permutation.py:71 full_budget denom logic intact; concrete calibration-bench path recorded | FUTURE |
| N-F4 | FIX (BH/BY) or DOC | Low/doc (implicit TODO) | n/a | YES |
| N-F5 | DOC/FUTURE | Low/doc (implicit TODO) | n/a | YES |
| N-F6 | FIX-with-N-F1 | Low/doc (implicit TODO) | n/a | YES |
| N-F7 | DOC | Low/doc (implicit TODO) | n/a | YES |
| N-F8 | DOC/FIX | Low/doc (implicit TODO) | n/a | YES |
| FE-F1 | FIX (per-row time merge) | DONE (d804cdf4) | YES — _temporal_agg_fe.py per-row strictly-past merge (L671-696, 749) | no |
| FE-F2 | FIX (canonical token) | DONE (bf7023e3) | YES — _temporal_agg_fe.py:97 routes entity col through canonical group token | no |
| FE-F3 | FIX (per-fold edges) | DONE (bf7023e3) | YES — _grouped_quantile_fe.py:439 `fold_global_edges = _fit_group_edges(x[_tr_finite],...)` | no |
| FE-F4 | FIX (guard K>=2) | DONE (bf7023e3) | YES — _cat_target_encoding_and_weighted.py:82 `if n_oof_folds<=0 and not allow_naive_leak: ...n_oof_folds=2` | no |
| **FE-F5** | FIX (keep int64) rolling | **table says TODO (rolling); PROGRESS LOG omits it** | **YES DONE in code** — commit 34e831f5 keeps native int64-ns; `h_t=np.asarray(h.get("t",[]))` (no float64 cast) | **MIS-RECORD** |
| FE-F6 | DOC | Low/doc (implicit TODO) | n/a | YES |
| FE-F7 | FIX (dedup) | DONE (33343a58) | YES — _fe_stage_temporal_agg.py:108-109 `X_ta.loc[:, ~X_ta.columns.duplicated()]` | no |
| EN-1 | FIX (canonicalize keys) | TODO | n/a | YES |
| EN-2 | FIX (guard pd) | DONE (33343a58) | YES — _encoding_recipes.py:71/73 `if pd is not None`, `elif pl is not None and pd is not None` | no |
| EN-3 | FIX | Low (implicit TODO) | n/a | YES |
| EN-4 | FIX | Low (implicit TODO) | n/a | YES |
| EN-5 | dedup of FE-F3 | see FE-F3 | RESOLVED via FE-F3 (DONE) | no (covered) |
| EX-1 | FIX (isfinite.all) | DONE (33343a58) | YES — _conditional_gate_fe.py:365 `return bool(a.size==0 or np.isfinite(a).all())` | no |
| EX-2 | FIX/DOC | Low (implicit TODO) | n/a | YES |
| EX-3 | DOC | Low (implicit TODO) | n/a | YES |
| EX-4 | FIX (unique / doc) | Low (implicit TODO) | n/a | YES |
| P-1 | FIX (mutate-restore/gate) | REMAINING/TODO | n/a | YES |
| P-2 | FIX (branch dtype) | REMAINING/TODO | n/a | YES |
| P-3 | FIX (shared _y_discrete) | REMAINING/TODO | n/a | YES |
| P-4 | FIX | REMAINING/TODO | n/a | YES |
| P-5..P-10 | FUTURE (bench-gated) | grouped FUTURE | n/a (gated/opt-in STRICT) | FUTURE (grouped) |
| P-11 | FIX | REMAINING/TODO | n/a | YES |
| ST-1 | FIX | REMAINING/TODO | n/a | YES |
| ST-2 | DOC/FUTURE | Low/doc (implicit TODO) | n/a | YES |
| ST-3 | FIX (dtype int64) | DONE (33343a58) | YES — _finalise.py:281 `self.support_ = np.array(_topk, dtype=np.int64)` | no |
| ST-4 | FIX | REMAINING/TODO | n/a | YES |

## Summary

- **Total distinct findings:** 38 (S-F1–5, N-F1–8, FE-F1–7, EN-1–5, EX-1–4, P-1–4/P-5..10/P-11, ST-1–4).
- **DONE and code-verified: 11** — FE-F1, FE-F2, FE-F3, FE-F4, FE-F5, FE-F7, S-F1, S-F5, EN-2, EX-1, ST-3. All 11 confirmed physically present in the cited code (not superficial); commits 84a0f6a8 / 33343a58 / bf7023e3 / d804cdf4 / 34e831f5.
- **FUTURE with concrete path: 2** — N-F3 (perm_pvalue full-budget; pinned by bench_perm_pvalue_addone.py + test; add calibration bench) and S-F3 (jmim `**(nexisting+1)` exponent; jmim greedy biz-value bench). Both confirmed reverted to original behavior in code (fix NOT half-applied), each with a recorded next step. P-5..P-10 grouped FUTURE (bench-gated, opt-in STRICT paths — not on default wall).
- **Resolved by dedup: 1** — EN-5 ≡ FE-F3 (FE-F3 DONE).
- **Still TODO / unaddressed: 24** — S-F2, S-F4, N-F1, N-F2, N-F4, N-F5, N-F6, N-F7, N-F8, FE-F6, EN-1, EN-3, EN-4, EX-2, EX-3, EX-4, P-1, P-2, P-3, P-4, P-11, ST-1, ST-2, ST-4 (S-F2/S-F4/N-F1/N-F2/P-1..4/P-11/ST-1/ST-4 named in REMAINING; the rest fall under the vague "Low/doc items" catch-all).

## Integrity flags (skeptical pass)

- **(a) Missing from tracker:** NONE. Every report ID appears in a tracker disposition table.
- **(b) Marked DONE but not in code:** NONE. All 11 DONE items verified present.
- **(c) Silently dropped:** NONE outright, but two RECORDING defects:

  1. **FE-F5 mis-recorded (the notable one).** It is DONE in code — committed in `34e831f5` ("rolling replay keeps int64-ns time axis") — yet its disposition-table row still reads `TODO (rolling)`, and the PROGRESS LOG DONE line (added by that very same commit) omits FE-F5 and says "10 fixed+pushed" when 11 are. The live status undercounts its own work.

  2. **Per-section disposition tables are STALE across the board.** The Status column of the tables still says `TODO` for 10 of the 11 done items (S-F1, S-F5, FE-F2, FE-F3, FE-F4, FE-F7, EN-2, EX-1, ST-3 — only FE-F1's row was flipped to DONE). Only the top PROGRESS LOG reflects reality; anyone reading the tables would conclude these are unaddressed. S-F3/N-F3 table rows also still say `FIX … TODO` rather than FUTURE.

- **"Low/doc items" catch-all (no-premature-closure concern):** 14 findings (N-F4–8, FE-F6, EN-1/3/4, EX-2/3/4, ST-2) have no individual terminal disposition — they are lumped under "and the Low/doc items" in the REMAINING line. Per the project's own "every ID gets an explicit disposition row" rule, these have NOT reached terminal status; they remain open TODO/DOC and should each get an explicit row before closure.

## Bottom line
No finding is lost or fabricated, and no DONE claim is hollow — all 11 done fixes and both FUTURE reverts are code-accurate. The gaps are (1) FE-F5's status is wrong (done-but-recorded-TODO), (2) the per-section tables are stale (10 done items still show TODO), and (3) 14 Low/doc findings still lack individual terminal dispositions. The critique is NOT yet at 100% terminal status: 24 findings remain open.

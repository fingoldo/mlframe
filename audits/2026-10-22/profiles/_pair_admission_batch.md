# _pair_admission_batch (batched pair admission / asymmetric relaxation), 2026-11-03

Utilization table of the 100k all-families GPU fit: `score_prospective_pairs` 3.6 s at 15 % GPU and 29 % CPU; its sampled leaves are the per-block host
gathers `data[::stride][:, cand_cols]` -> int64 (lines 73 / 78) feeding `batched_cmi_gpu`, about 1 s.
Tried and reverted: one upload of the strided code matrix per call and the block / anchor columns gathered on the device. The per-pair
`_conditional_perm_null` takes the candidate and anchor columns as host arrays (`np.ascontiguousarray(x[:, j])` raised on the cupy column, and the
asymmetric path's broad except returned no verdicts at all), so the columns would have to come back per block: a device-to-host copy of the same size
as the upload it saved. What remains of the gain is the host fancy-index gather itself; not worth the second code path.
Lever if this stage grows: make `_conditional_perm_null`'s analytic branch take the cards only (it needs the column just for its size), then the block
never has to exist on the host.

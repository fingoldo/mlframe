# Device interaction pair-table TreeSHAP kernel profile (GTX 1050 Ti, sm_61), 2026-11-17

Module `_shap_proxy_treeshap_interactions_table_gpu`: one thread per sample walks every tree with the bit mask of the satisfied path features and adds the per-leaf main-table and pair-table entries; the pair tensor is accumulated per row block and symmetrised on the device. Host twin: the numba interaction table kernels of `_shap_proxy_treeshap_interactions_table` (record `_shap_proxy_treeshap_numba.md`, 65x over the conditioned-scan reference at 1000 rows).

Workload: XGBoost 200 trees depth 4, 40 features. Whole call, float32 default (float64 reference mode `MLFRAME_SHAP_TREESHAP_REAL=float64`), 7 interleaved rounds, GPU 73-77 C, no throttle.
- 3000 rows: numba conditioned-scan reference 7088 ms, old GPU conditioned-scan kernel 16648 ms (contended run, approximate), device pair-table kernel 55 ms; float32 difference about 3e-6, float64 1e-15.
- 20000 rows: frame (local-memory) DFS stack 251.7 ms best / 277.8 median against the shared-memory stack 280.2 (shared 11 percent slower, kept as the exempt local frame in the register-pressure registry).
nvprof: occupancy 0.24 at 3000 rows (too few threads for the card), the stack frame is the register-pressure exemption with its interleaved A/B evidence.
Launches per call: 1 kernel per row block (+ transpose, zeros, symmetrise); the block is sized so the (P, P, rows) pair tensor of a block stays within 256 MB of device memory (`_PHI_CHUNK_BYTES`).
Host side (cProfile of the whole call, 3000 rows): the pair-table build is about 25 ms of 110 ms; the rest waits on the kernel and the download. GPU utilization is low at small row counts for that reason (occupancy 0.24, see above); bandwidth is not the limit.
Tuner: `shap_proxy_treeshap_interactions_backend` (numba against device over cells = rows x features^2 and trees, 30 features depth 4): the host kernel wins at 900 000 cells (1000 rows), the device kernel at 3 600 000 (4000 rows) for 50 and 200 trees; the measured floor 2 000 000 replaces the hand floor 4 800 000.

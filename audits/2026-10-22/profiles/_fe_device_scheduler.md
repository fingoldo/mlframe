# _fe_device_scheduler: the FE stage cascades never ran concurrently in the 100k fit

Stage concurrency (`run_stages_concurrently`, on by default since 2026-11-03) asked `plan_stage_slots` for one device slot per cascade
(early_b, mid_a, mid_b). In every warm 100k x 100 fit it got none ("no device plan (a stage would get too little VRAM); running the cascades
one by one"), so the three cascades ran sequentially. Two causes:

1. `_device_facts` read the driver's free memory only; by the time the stages are planned the earlier families have left freed blocks in
   the default pool, which the driver counts as used. It now adds `get_default_memory_pool().free_bytes()` (as `_stages_fit_device` does).
2. The usable memory (free - 1 GB cushion) was split by the VRAM weights alone (461 / 274 / 110 MB peaks): mid_b's 13 % share stayed under
   the 256 MB minimum until about 3 GB were free, which a 4 GB card never has mid-fit. Each stage now gets the minimum first and the rest by
   weight (still refused when the usable memory does not cover the minimum of every stage).

Same process, back to back, warm 100k x 100 every family (GTX 1050 Ti): concurrent 75.8 s vs sequential 84.8 s, the same six selected
features, no CUDA errors in either run. `MLFRAME_FE_STAGE_CONCURRENCY=0` keeps the cascades sequential.

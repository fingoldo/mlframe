# _fe_device_scheduler: stage concurrency never engages, and enabling it can abort the process

## Finding

Stage concurrency (`run_stages_concurrently`, on by default since 2026-11-03) asks `plan_stage_slots` for one device slot per cascade
(early_b, mid_a, mid_b). In every warm 100k x 100 fit it gets none ("no device plan (a stage would get too little VRAM); running the cascades
one by one"), so the cascades run sequentially. Two reasons: `_device_facts` reads the driver's free memory only (the default pool's cached
free blocks count as used there), and the usable memory is split by the VRAM weights alone (461 / 274 / 110), which keeps mid_b's 13 % share
under the 256 MB minimum until about 3 GB are free.

## Attempt (bc092dd66, reverted in 42a38547a)

Counting the pool's cached free blocks and giving every stage the minimum before the weighted split made the cascades overlap: warm
100k x 100, same process back to back, 75.8 s concurrent vs 84.8 s sequential, same six features, no CUDA errors in that fit.

Reverted because it can abort the process. In the wider MRMR FE tests an xdist worker died with `Fatal Python error: Aborted` inside
`cupy.argsort` (`_gpu_resident_rank_bin.rank_bin_codes_batch_gpu_resident:158`, mid_a row-argmax stage, concurrent-stage thread): cupy sorts
through thrust, and a thrust temporary allocation that exceeds the slot pool's hard limit raises a C++ `bad_alloc` that terminates instead of
reaching the Python out-of-memory fallbacks every stage has. Two standalone 100k fits earlier the same day stopped silently the same way. Slot
pool OOM (cupy `OutOfMemoryError` at 253 MB against 627-683 MB limits) also showed up repeatedly in the canonical tests.

## Fixed and re-enabled

The abort is a property of every cupy sort under a pool with a hard limit, not of concurrency alone (the default pool has one under
`ensure_fe_gpu_pool_limit` too). `_device_sort_safe.device_argsort` / `device_sort` check the room of the pool behind the current allocator
(slot pool or default pool) against the sort's temporaries (2 x data + 3 x elements x 8 bytes), release its cached free blocks if short, and
raise cupy's `OutOfMemoryError` in Python when still short, which every GPU path already catches to take its host twin
(`tests/feature_selection/gpu/test_device_sort_safe.py`). All 33 `cp.argsort` / `cp.sort` calls in the library go through them. The planner
change is re-applied on top.

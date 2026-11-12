# Profile record: `_wavelet_basis_fe` (wavelet leg selection)

Recorded 2026-11-12 on the GTX 1050 Ti (Pascal, sm_61, 6 SMs), CUDA 12.8. Harness: `select_wavelet_legs_batched` on 100000 rows, max_scale 6 (127 dyadic legs), max_legs 40, one warm call then min of 3. Both routes admit the same 14 legs. The harness runs one route at a time, with nothing else on the card.

## Wall time per call

- Host route (numpy legs, `binned_mi_from_codes_gpu` on the stacked codes): 251 ms.
- Device-born route (legs built on the card from the resident column): 12 ms.

## Host side (cProfile, host route, 4 calls)

Ordered by internal time, the function itself is 0.28 s over the calls. `numpy.stack` of the leg columns is 0.34 s over 8 calls, the cost of the code stacks the streaming change bounds. The rest of the profile is import and module loading, not the leg selection.

## GPU side (nvprof, device route, 4 calls)

Kernel launches per call: `leg_codes` 1 (builds the codes of every leg in one launch), `mi_from_codes_tile` 2 (train and held-out), `leg_supports` 1, and the reductions 4 (the range guard of `binned_mi_from_codes_gpu`, run because the device route does not trust its codes). Launches are 4 kernels, 2 reductions and small copies per call, not one per leg.

Time share of device activity over the 4 calls, 27.9 ms in total:
- `leg_codes` 57.3% (3.99 ms per call): the largest item.
- `mi_from_codes_tile` 15.0% (1.04 ms per call, two launches).
- Reductions (`cub::DeviceReduce`) 21.2% (the range guard, four per call, about 1.5 ms per call).
- `leg_supports` 4.2%.

Counters for one warm call (sm_efficiency, achieved occupancy, DRAM write throughput):
- `leg_codes`: sm efficiency 98.4%, achieved occupancy 0.73, DRAM write throughput 26.6 GB/s against about 112 GB/s peak. The kernel writes int64 codes whose values are only 0, 1 and 2, so its traffic is eight times what the values need; it is not at bandwidth, so the write pattern and the code width are the levers.
- `mi_from_codes_tile`: sm efficiency 97.4%, achieved occupancy 0.45, DRAM write throughput 0.69 GB/s (the joint histograms are shared-memory bound, not memory bound).
- `leg_supports`: sm efficiency 97.4%, achieved occupancy 0.85, DRAM write throughput 0.28 GB/s.

## Reading

The device route is about 20 times faster than the host route at this size, and the device time is dominated by building the codes (`leg_codes`), with the range guard a close second. Two follow-ups are recorded in the tracker: the range guard can be skipped for codes built as leg+1 in {0, 1, 2} (the device route passes `codes_trusted` False only to mirror the host, which has the same guarantee), and the code width of `leg_codes` could be int8 or int16 if the MI path accepts it without a widening copy.

## Streaming

The leg blocks are bounded by `_LEG_BLOCK_BYTES` (host) and `_DEVICE_LEG_BLOCK_BYTES` (device route), both 256 MB of int64 codes per split; see GPU11-67 and GPU11-68 in the tracker.

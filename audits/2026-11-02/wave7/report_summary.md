# Wave 7 audit 2026-11-02: encoding, pickle hygiene, integer overflow

Opus 5.5 low effort, read-only. Findings and dispositions are in `_TRACKER.md` next to this file. Highlights:

- Encoding: `text=True` subprocess calls decoded with the machine code page could raise `UnicodeDecodeError` outside the probes' except clauses; both probes now decode UTF-8 with replacement.
- Integer overflow: `merge_vars` computed class ids in the caller's `dtype`, so `np.int8` wrapped at 128 cells; ids are now int64 and a failed store raises.
- Pickle: every GBM carrying the mlframe early-stopping callback failed plain pickle (lambda stop flag, live widget); fixed with a module-level default and `__getstate__`.
- Memory: the MRMR permutation-null closure pinned a float64 copy of the frame after fit; now released.

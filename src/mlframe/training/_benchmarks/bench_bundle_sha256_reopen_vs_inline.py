"""Bench: io.py ``_bundle_sha256`` reopen-and-hash vs a single-pass inline hash of the compressed bytes.

The save path writes the zstd-compressed bundle to disk via ``atomic_write_bytes`` then, when stamping the
``.meta.json`` sidecar, ``_bundle_sha256`` REOPENS the just-written file and streams it back through SHA-256.
The lead: hash the compressed bytes inline (as the zstd writer produces them) and skip the reopen-read.

What this bench measures, warm + multi-iter, across bundle sizes (incl. a 30MB+ fat bundle):
  A) reopen_and_hash  -- the shipped path: open(bundle, 'rb') + chunked f.read -> sha256 (cold + page-cached).
  B) inline_hash      -- sha256.update over the SAME compressed-bytes buffer already in RAM (no reopen).

Both produce the identical digest BY CONSTRUCTION (same bytes), so this is a pure wall comparison -- the
question is only whether skipping the reopen-read is a measurable win that justifies entangling the
writer/durability path (the writer would have to tee bytes through the hash as it streams to the fd).

Run:
    python -m mlframe.training._benchmarks.bench_bundle_sha256_reopen_vs_inline
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time

import numpy as np
import zstandard as zstd


def _make_compressed_bundle_bytes(n_floats: int, level: int = 4) -> bytes:
    """Produce a realistic zstd-compressed bundle byte blob of roughly model-bundle shape."""
    import pickle
    payload = {
        "arr": np.random.default_rng(0).standard_normal(n_floats).astype(np.float32),
        "meta": {"k": list(range(2000))},
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    c = zstd.ZstdCompressor(level=level, write_checksum=True, write_content_size=True, threads=-1)
    return c.compress(raw)


def _reopen_and_hash(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _inline_hash(blob: bytes) -> str:
    h = hashlib.sha256()
    h.update(blob)
    return h.hexdigest()


def _time(fn, *args, iters: int) -> float:
    fn(*args)  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def _stream_writer_bytes(raw: bytes, level: int = 4) -> bytes:
    """The EXACT on-disk byte production path used by ``save_mlframe_model._writer`` (zstd stream_writer)."""
    import io as _io
    kw = dict(level=level, write_checksum=True, write_content_size=True, threads=-1)
    buf = _io.BytesIO()
    with zstd.ZstdCompressor(**kw).stream_writer(buf, closefd=False) as zf:
        zf.write(raw)
    return buf.getvalue()


def _frame_equivalence_probe() -> None:
    """Two ways one could capture bytes for inline hashing, vs the shipped stream_writer on-disk bytes.

    A) one-shot ``compressor.compress(raw)`` -> DIFFERENT bytes (different frame framing / content-size header),
       so swapping the writer to one-shot would change the on-disk bundle AND its digest -- NOT bit-identical.
    B) a HashTee fileobj forwarding ``.write`` to the real fd + updating sha -> IDENTICAL bytes + identical digest,
       but it must wrap the fd inside ``atomic_write_bytes`` where ``threads=-1`` hands the descriptor to a
       background flush thread and the post-write ``f.flush()/os.fsync(fileno())`` durability invariant lives.
    """
    import pickle
    payload = {"arr": np.random.default_rng(0).standard_normal(500_000).astype(np.float32), "meta": list(range(3000))}
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ship = _stream_writer_bytes(raw)
    one_shot = zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=-1).compress(raw)

    import io as _io

    class _HashTee:
        def __init__(self, f):
            self.f = f
            self.h = hashlib.sha256()

        def write(self, b):
            self.h.update(b)
            return self.f.write(b)

        def flush(self):
            return self.f.flush()

    buf = _io.BytesIO()
    tee = _HashTee(buf)
    with zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=-1).stream_writer(tee, closefd=False) as zf:
        zf.write(raw)
    teed = buf.getvalue()
    print("\nframe-equivalence probe (why inline is not a free swap):")
    print(f"  one-shot compress() bytes == shipped stream_writer bytes : {one_shot == ship}  (len {len(one_shot)} vs {len(ship)})")
    print(f"  HashTee stream_writer bytes == shipped bytes             : {teed == ship}")
    print(f"  HashTee inline digest      == reopen digest              : {tee.h.hexdigest() == hashlib.sha256(teed).hexdigest()}")


def main() -> None:
    sizes = [
        ("tabular ~0.5MB", 100_000),
        ("medium ~4MB", 1_000_000),
        ("fat ~32MB", 8_000_000),
    ]
    iters = 50
    print(f"{'shape':<18} {'on-disk MB':>10} {'reopen ms':>10} {'inline ms':>10} {'speedup':>8} {'identical':>9}")
    for label, n in sizes:
        blob = _make_compressed_bundle_bytes(n)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bundle.dump")
            with open(path, "wb") as f:
                f.write(blob)
            disk_mb = os.path.getsize(path) / (1024 * 1024)
            t_reopen = _time(_reopen_and_hash, path, iters=iters)
            t_inline = _time(_inline_hash, blob, iters=iters)
            same = _reopen_and_hash(path) == _inline_hash(blob)
            sp = t_reopen / t_inline if t_inline > 0 else float("inf")
            print(f"{label:<18} {disk_mb:>10.2f} {t_reopen:>10.3f} {t_inline:>10.3f} {sp:>7.2f}x {str(same):>9}")
    _frame_equivalence_probe()
    print(
        "\nVerdict (REJECT, keep reopen): inline hashing of the in-RAM compressed bytes saves the reopen-read only "
        "-- measured ~0.83 ms @0.5MB / ~1.7 ms @4MB / ~9.9 ms @32MB. But there is no FREE bit-identical capture: "
        "one-shot compress() produces DIFFERENT on-disk bytes (changes the bundle + its digest); the only "
        "bit-identical capture is a HashTee around the stream_writer fd, which entangles the hash with the "
        "threads=-1 background-flush thread + the atomic-write fsync(fileno) durability invariant. The save wall is "
        "dominated by pickle.dumps + zstd compress (hundreds of ms to seconds on multi-model suites); the reopen "
        "delta is sub-1% of save wall and the durability-path entanglement is not worth it. Revisit only if 30MB+ "
        "fat-bundle hashing surfaces as a save-side hotspot in a full-pipeline profile."
    )


if __name__ == "__main__":
    main()

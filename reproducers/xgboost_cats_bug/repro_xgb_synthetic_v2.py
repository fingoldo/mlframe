r"""Pure synthetic reproducer v2 for XGBoost 3.2.0 + polars Categorical crash.

Key insight vs v1:
  v1 created pl.Series directly from a Python list -> Arrow buffers allocated
  via Python memory path, different heap layout -> XGB allocation never hits
  a page guard -> FIT_OK.

  v2 saves the primer strings to a temp parquet and loads them back.
  The parquet deserialization path allocates Arrow buffers the same way as
  loading a real production parquet -> heap fragmentation matches -> crash.

No real data required. Strings are synthetic but go through the same
parquet round-trip that makes the real bundle work.

Usage
-----
    D:/ProgramData/anaconda3/python.exe repro_xgb_synthetic_v2.py
    D:/ProgramData/anaconda3/python.exe repro_xgb_synthetic_v2.py --workaround
    D:/ProgramData/anaconda3/python.exe repro_xgb_synthetic_v2.py --keep-parquet
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

# Note: we used to call SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX) on Windows
# to suppress the "application has crashed" dialog on the expected access violation. That flag
# also disables Windows Error Reporting, so no crash dump is written when the process dies —
# exactly when we want one. Leave WER alone so ProcDump / WER LocalDumps can capture a dump.
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ---------------------------------------------------------------------------
# Real prod category dictionary — 50 strings, codes will land at 0..49 after
# cache is primed, then __MISSING__ lands at ~2_526_059.
# ---------------------------------------------------------------------------
CATEGORY_STRINGS = [
    "Digital Marketing", "Virtual Assistance", "Marketing, PR & Brand Strategy",
    "Graphic, Editorial & Presentation Design", "Lead Generation & Telemarketing",
    "Content Writing", "Management Consulting & Analysis", "ERP/CRM Software",
    "Web Development", "Mobile Development", "Video & Animation",
    "Ecommerce Development", "Web & Mobile Design", "3D Modeling & CAD",
    "DevOps & Solution Architecture", "Electrical & Electronic Engineering",
    "Desktop Application Development", "Branding & Logo Design",
    "Network & System Administration", "Building & Landscape Architecture",
    "AI Apps & Integration", "Editing & Proofreading Services",
    "Performing Arts", "Data Entry & Transcription Services",
    "Scripts & Utilities", "Photography", "AI & Machine Learning",
    "Customer Service & Tech Support", "Data Analysis & Testing",
    "Recruiting & Human Resources", "Finance & Tax Law", "Art & Illustration",
    "Energy & Mechanical Engineering", "QA Testing", "Product Design",
    "Accounting & Bookkeeping", "Civil & Structural Engineering",
    "Game Design & Development", "Financial Planning",
    "Market Research & Product Reviews", "Project Management",
    "Data Extraction/ETL", "Other - Accounting & Consulting",
    "Personal & Professional Coaching", "Physical Sciences",
    "Information Security & Compliance", "Interior & Trade Show Design",
    "Professional & Business Writing", "Translation & Localization Services",
    "Contract Manufacturing",
]

# ASCII chars used for random string content — printable, ASCII-safe, no cp1251 issues.
_CHARS = np.frombuffer(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-_/:",
    dtype=np.uint8,
)
_N_CHARS = len(_CHARS)


def _gen_chunk(args: tuple) -> list[str]:
    """Generate a chunk of unique variable-length strings (worker function).

    Each string is prefixed with its global index as hex, guaranteeing
    global uniqueness across chunks without coordination.
    Content after the prefix is random ASCII chars generated via numpy
    (vectorised — much faster than Python rng.choice per character).
    """
    start, end, seed = args
    rng = np.random.default_rng(seed)
    n = end - start

    # Sample lengths: 50% short (4-40), 40% medium (40-500), 10% long (500-4799).
    x = rng.random(n)
    lengths = np.where(
        x < 0.50, rng.integers(4, 41, n),
        np.where(x < 0.90, rng.integers(40, 501, n),
                 rng.integers(500, 4800, n))
    ).astype(np.int32)

    # Generate all random content bytes at once (vectorised).
    prefix_len = 10  # f"{idx:08x} " = 10 chars
    content_lens = np.maximum(0, lengths - prefix_len)
    total_content = int(content_lens.sum())
    rand_bytes = _CHARS[rng.integers(0, _N_CHARS, total_content)]
    content_str = rand_bytes.tobytes().decode("ascii")

    result: list[str] = []
    pos = 0
    for i in range(n):
        prefix = f"{start + i:08x} "
        cl = int(content_lens[i])
        result.append(prefix + content_str[pos: pos + cl])
        pos += cl
    return result


def generate_primer_parquet(path: Path, n: int, seed: int,
                             n_jobs: int = 0) -> None:
    """Generate n unique variable-length strings and save to parquet.

    Uses multiprocessing to parallelise string generation.
    n_jobs=0 means use all CPU cores.
    """
    import multiprocessing as mp

    if n_jobs <= 0:
        n_jobs = mp.cpu_count()
    n_jobs = min(n_jobs, n)

    chunk_size = n // n_jobs
    chunks = [
        (i * chunk_size,
         (i + 1) * chunk_size if i < n_jobs - 1 else n,
         seed + i)
        for i in range(n_jobs)
    ]

    with mp.Pool(n_jobs) as pool:
        parts = pool.map(_gen_chunk, chunks)

    strings = [s for part in parts for s in part]
    pl.DataFrame(
        {"skills_text": pl.Series("skills_text", strings, dtype=pl.String)}
    ).write_parquet(path, compression="zstd", compression_level=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-primer", type=int, default=2_526_059,
                    help="Number of unique primer strings (default matches prod cache size)")
    ap.add_argument("--n-train", type=int, default=211_168)
    ap.add_argument("--n-val",   type=int, default=100_000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--n-jobs",  type=int, default=0,
                    help="Worker processes for string generation (0=all cores)")
    ap.add_argument("--workaround", action="store_true")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="Keep the generated parquet after run (for reuse)")
    args = ap.parse_args()

    import xgboost
    print(f"polars {pl.__version__}, xgboost {xgboost.__version__}, "
          f"platform={sys.platform}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        primer_path = Path(tmpdir) / "synthetic_skills.parquet"
        if args.keep_parquet:
            primer_path = Path("synthetic_skills.parquet")

        # ---- Step 1: generate primer strings and save to parquet ----------
        if not primer_path.exists():
            print(f"generating {args.n_primer:_} synthetic primer strings...",
                  flush=True)
            t0 = time.perf_counter()
            generate_primer_parquet(primer_path, args.n_primer, args.seed,
                                    n_jobs=args.n_jobs)
            sz = primer_path.stat().st_size / 1e6
            print(f"  saved {primer_path} ({sz:.1f} MB) in "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
        else:
            print(f"reusing {primer_path}", flush=True)

        # ---- Step 2: load parquet and cast to Categorical -----------------
        # Loading through parquet gives Arrow the same allocation path as
        # a real production parquet load -> same heap layout -> crash.
        print("loading primer parquet and priming StringCache...", flush=True)
        t0 = time.perf_counter()
        skills = pl.read_parquet(primer_path)
        # Keep primer_cat alive until after XGB fit — polars 1.33+ permanently
        # retains StringCache entries as long as at least one Categorical series
        # holds a reference (in 1.40+ entries are evicted when last ref is dropped).
        primer_cat = skills["skills_text"].cast(pl.Categorical)
        del skills
        print(f"  StringCache primed ({primer_cat.n_unique():_} entries) "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)

        # ---- Step 3: build category column through polluted cache ----------
        rng_np = np.random.default_rng(args.seed)
        n_total = args.n_train + args.n_val
        cat_values = rng_np.choice(CATEGORY_STRINGS, size=n_total).tolist()
        # Inject ~5% nulls to exercise fill_null path.
        null_mask = rng_np.random(n_total) < 0.05
        cat_values_with_nulls = [None if null_mask[i] else cat_values[i]
                                  for i in range(n_total)]

        df = pl.DataFrame({"category": pl.Series("category",
                                                   cat_values_with_nulls,
                                                   dtype=pl.String)})

        df = df.with_columns(pl.col("category").cast(pl.Categorical))
        print(f"after category cast: n_unique={df['category'].n_unique()}, "
              f"codes_max={df['category'].to_physical().drop_nulls().max()}",
              flush=True)

        df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
        print(f"after fill_null:     n_unique={df['category'].n_unique()}, "
              f"codes_max={df['category'].to_physical().max()}", flush=True)

        if args.workaround:
            print("WORKAROUND: cast to pl.Enum(sorted_uniques)", flush=True)
            uniques = sorted(df["category"].unique().drop_nulls().to_list())
            df = df.with_columns(pl.col("category").cast(pl.Enum(uniques)))
            print(f"codes_max after Enum: {df['category'].to_physical().max()}",
                  flush=True)

        train = df[:args.n_train]
        val   = df[args.n_train:]
        y_tr = rng_np.integers(0, 2, args.n_train, dtype=np.int8)
        y_v  = rng_np.integers(0, 2, args.n_val,   dtype=np.int8)

        print(f"\nfitting XGB train={train.shape} val={val.shape} -- "
              f"expect silent kill (0xC0000005) on Windows", flush=True)
        m = XGBClassifier(
            n_estimators=5, enable_categorical=True, tree_method="hist",
            device="cpu", n_jobs=-1, verbosity=1,
            max_cat_to_onehot=1, max_cat_threshold=100,
            early_stopping_rounds=3,
            objective="binary:logistic", eval_metric="logloss",
        )
        t0 = time.perf_counter()
        try:
            m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
            del primer_cat  # noqa: keep alive until after fit
            print(f"FIT_OK in {time.perf_counter()-t0:.1f}s -- bug did NOT reproduce",
                  flush=True)
        except BaseException as e:
            print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

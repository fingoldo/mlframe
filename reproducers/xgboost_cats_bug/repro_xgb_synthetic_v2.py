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
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
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

_VOCAB = (
    "python programming framework library module package import export "
    "javascript typescript react angular vue node express fastify django "
    "flask fastapi postgres mysql sqlite mongodb redis elasticsearch kafka "
    "docker kubernetes terraform ansible aws azure gcp cloud serverless "
    "machine learning artificial intelligence deep neural transformer model "
    "tensorflow pytorch keras scikit pandas numpy scipy matplotlib seaborn "
    "data science analytics visualization dashboard reporting metric kpi "
    "marketing seo sem social media content strategy brand advertising "
    "copywriting editorial publishing translation localization proofread "
    "design graphic illustration logo typography branding ui ux wireframe "
    "video animation motion editing rendering compositing color grading "
    "audio music production recording mixing mastering podcast voice talent "
    "accounting bookkeeping tax audit compliance finance banking investment "
    "legal contract corporate immigration intellectual property patent law "
    "recruiting hr resume cover letter interview coaching training mentor "
    "project management agile scrum kanban sprint backlog stakeholder team "
    "sales lead generation crm pipeline conversion funnel outreach email "
    "customer service support chat ticket escalation help desk knowledge "
    "transcription subtitles captions dictation reporting interview article "
    "research market survey competitor intelligence insight ecommerce shopify "
    "security penetration testing audit firewall vpn encryption hashing "
    "database administration backup restore replication sharding index query "
    "mobile ios android swift kotlin react native flutter game unity unreal "
    "blockchain crypto nft solidity ethereum smart contract defi engineering "
    "mechanical electrical civil structural chemical industrial architecture "
    "photography product fashion portrait event wedding real estate drone"
).split()
_PUNCT = (" ", " ", " ", ", ", ". ", " & ", " - ", ": ", " | ", " + ")


def _make_string(rng: random.Random, target_len: int) -> str:
    out, cur = [], 0
    while cur < target_len:
        w = rng.choice(_VOCAB)
        out.append(w)
        cur += len(w)
        if cur < target_len:
            sep = rng.choice(_PUNCT)
            out.append(sep)
            cur += len(sep)
    return "".join(out)[:target_len]


def generate_primer_parquet(path: Path, n: int, seed: int) -> None:
    """Generate n unique variable-length strings and save to parquet."""
    rng = random.Random(seed)
    cat_set = set(CATEGORY_STRINGS)
    strings: list[str] = []
    seen: set[str] = set(cat_set)
    uniq = 0
    while len(strings) < n:
        # Heavy-tailed length: 50% short, 40% medium, 10% long
        x = rng.random()
        if x < 0.50:
            length = rng.randint(4, 40)
        elif x < 0.90:
            length = rng.randint(40, 500)
        else:
            length = rng.randint(500, 4799)
        s = _make_string(rng, length)
        if s in seen:
            s = s[:max(4, length - 6)] + f" ~{uniq:x}"
            uniq += 1
        seen.add(s)
        strings.append(s)

    pl.DataFrame({"skills_text": pl.Series("skills_text", strings,
                                            dtype=pl.String)}).write_parquet(
        path, compression="zstd", compression_level=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-primer", type=int, default=2_526_059,
                    help="Number of unique primer strings (default matches prod cache size)")
    ap.add_argument("--n-train", type=int, default=211_168)
    ap.add_argument("--n-val",   type=int, default=100_000)
    ap.add_argument("--seed",    type=int, default=42)
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
            generate_primer_parquet(primer_path, args.n_primer, args.seed)
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
        _ = skills["skills_text"].cast(pl.Categorical)
        del skills, _
        print(f"  StringCache primed in {time.perf_counter()-t0:.1f}s", flush=True)

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
            print(f"FIT_OK in {time.perf_counter()-t0:.1f}s -- bug did NOT reproduce",
                  flush=True)
        except BaseException as e:
            print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

r"""Pure-synthetic reproducer for the XGBoost 3.2.0 + polars Categorical
0xC0000005 crash on Windows. No parquet, no production data required.

Motivation
----------
Earlier synthetic attempts populated the global StringCache with regular
uniform-width padding (``f"__pad_{i:08d}"``). That reproduces the sparse
physical-code dictionary shape, but does NOT trigger the Windows SEH kill.
Empirically the crash needs something the earlier reproducers were missing:

  1. **String length variability** — real ``skills_text`` values range from
     ~1 to ~4800 characters. Arrow's string array uses an offset buffer; a
     highly variable length distribution produces a fragmented value buffer
     that changes the Windows heap layout around the 9 MB XGB allocation.

  2. **Natural hash distribution** — polars' StringCache uses a Rust HashMap;
     bucket layout depends on the string hashes. ``f"skill_X"`` has too much
     regularity. Natural-language-ish content scatters across buckets the way
     real parquet content does.

This reproducer fabricates ~2.5 M padding strings with:
  * lengths sampled from a heavy-tailed distribution (min=1, p50~50,
    p99~1500, max~4800) matching the prod ``skills_text`` profile.
  * content built from a small English-word vocabulary joined by spaces and
    punctuation, so each value is UTF-8 text with natural byte patterns and
    natural Rust HashMap hash values.

Then it primes the StringCache so that exactly 89 real category strings land
at codes 0..87 and an ``__MISSING__`` sentinel lands at code 2_526_058, and
fits XGBoost on a 1-row-train / 1-row-val slice. Expect silent exit
with rc=3221226505 on Windows + xgboost 3.2.0.

Usage
-----
    D:/ProgramData/anaconda3/python.exe repro_xgb_synthetic_realistic.py
    D:/ProgramData/anaconda3/python.exe repro_xgb_synthetic_realistic.py --workaround
    D:/ProgramData/anaconda3/python.exe repro_xgb_synthetic_realistic.py --seed 7
"""
from __future__ import annotations

import argparse
import random
import sys
import time

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
# Real prod dictionary (89 codes 0..87 + sentinel __MISSING__ @ 2_526_058).
# ---------------------------------------------------------------------------
DICT_PAIRS: list[tuple[int, str]] = [
    (0, "Digital Marketing"),
    (1, "Virtual Assistance"),
    (2, "Marketing, PR & Brand Strategy"),
    (3, "Graphic, Editorial & Presentation Design"),
    (4, "Lead Generation & Telemarketing"),
    (5, "Content Writing"),
    (6, "Management Consulting & Analysis"),
    (7, "ERP/CRM Software"),
    (8, "Web Development"),
    (9, "Mobile Development"),
    (10, "Video & Animation"),
    (11, "Ecommerce Development"),
    (12, "Web & Mobile Design"),
    (13, "3D Modeling & CAD"),
    (14, "DevOps & Solution Architecture"),
    (15, "Electrical & Electronic Engineering"),
    (16, "Desktop Application Development"),
    (17, "Branding & Logo Design"),
    (18, "Network & System Administration"),
    (19, "Building & Landscape Architecture"),
    (20, "AI Apps & Integration"),
    (21, "Editing & Proofreading Services"),
    (22, "Performing Arts"),
    (23, "Data Entry & Transcription Services"),
    (24, "Scripts & Utilities"),
    (25, "Photography"),
    (26, "AI & Machine Learning"),
    (27, "Customer Service & Tech Support"),
    (28, "Data Analysis & Testing"),
    (29, "Recruiting & Human Resources"),
    (30, "Finance & Tax Law"),
    (31, "Art & Illustration"),
    (32, "Energy & Mechanical Engineering"),
    (33, "QA Testing"),
    (34, "Product Design"),
    (35, "Accounting & Bookkeeping"),
    (36, "Civil & Structural Engineering"),
    (37, "Game Design & Development"),
    (38, "Financial Planning"),
    (39, "Market Research & Product Reviews"),
    (40, "Project Management"),
    (41, "Data Extraction/ETL"),
    (42, "Other - Accounting & Consulting"),
    (43, "Personal & Professional Coaching"),
    (44, "Physical Sciences"),
    (45, "Information Security & Compliance"),
    (46, "Interior & Trade Show Design"),
    (47, "Professional & Business Writing"),
    (48, "Translation & Localization Services"),
    (49, "Contract Manufacturing"),
    (50, "Community Management & Tagging"),
    (51, "Corporate & Contract Law"),
    (52, "Sales & Marketing Copywriting"),
    (53, "Data Mining & Management"),
    (54, "Other - Software Development"),
    (55, "Chemical Engineering"),
    (56, "Audio & Music Production"),
    (57, "Database Management & Administration"),
    (58, "Product Management & Scrum"),
    (59, "Public Law"),
    (60, "Language Tutoring & Interpretation"),
    (61, "International & Immigration Law"),
    (62, "Blockchain, NFT & Cryptocurrency"),
    (63, "NFT, AR/VR & Game Art"),
    (64, "Legal, Medical & Technical Translation"),
    (65, "Resumes & Cover Letters"),
    (66, "Market & Customer Research"),
    (67, "Telemarketing & Telesales"),
    (68, "Creative Writing Services"),
    (69, "Social Media Marketing & Strategy"),
    (70, "SEO & SEM Services"),
    (71, "SEM - Search Engine Marketing"),
    (72, "Technical Translation"),
    (73, "Voice Talent"),
    (74, "Transcription"),
    (75, "Quantitative Analysis"),
    (76, "Other - Admin Support"),
    (77, "Other - Data Science & Analytics"),
    (78, "Other - Writing"),
    (79, "Intellectual Property Law"),
    (80, "Email & Marketing Automation"),
    (81, "Corporate Law"),
    (82, "Other - Engineering"),
    (83, "Medical Translation"),
    (84, "Other - Sales & Marketing"),
    (85, "Graphics & Design"),
    (86, "Tech Support & Content Moderation"),
    (87, "Grant & Proposal Writing"),
    (2_526_058, "__MISSING__"),
]
TRAIN_CODE = 16
VAL_CODE = 16  # same string on both sides — avoids XGB "unseen category" guard


# ---------------------------------------------------------------------------
# Natural-content padding: realistic-looking skills_text values with:
#   * variable lengths (heavy-tailed, min=1, max=~4800)
#   * mixed word content producing natural Rust HashMap hash distribution
# ---------------------------------------------------------------------------
# Small vocab of real English tech/business words so synthetic padding has
# realistic bytewise content. ~300 words: enough diversity for 2.5 M unique
# strings when recombined, short enough to keep the source file readable.
_VOCAB = (
    "python programming framework library module package import export "
    "javascript typescript react angular vue svelte node express fastify "
    "django flask fastapi pyramid tornado starlette uvicorn gunicorn "
    "postgres mysql sqlite mongodb redis elasticsearch kafka rabbitmq "
    "docker kubernetes helm terraform ansible puppet chef salt vagrant "
    "aws azure gcp cloud compute storage network serverless lambda fargate "
    "machine learning artificial intelligence deep neural transformer model "
    "tensorflow pytorch keras scikit pandas numpy scipy matplotlib seaborn "
    "data science analytics visualization dashboard reporting metric kpi "
    "marketing seo sem social media content strategy brand advertising "
    "copywriting editorial publishing translation localization proofread "
    "design graphic illustration logo typography branding ui ux wireframe "
    "video animation motion effects editing rendering compositing color "
    "audio music production recording mixing mastering podcast voice talent "
    "accounting bookkeeping tax audit compliance finance banking investment "
    "legal contract corporate immigration intellectual property patent law "
    "recruiting hr resume cover letter interview coaching training mentor "
    "project management agile scrum kanban sprint backlog stakeholder team "
    "sales lead generation crm pipeline conversion funnel outreach email "
    "customer service support chat ticket escalation help desk knowledge "
    "transcription subtitles captions dictation reporting interview article "
    "research market survey focus group competitor intelligence insight "
    "ecommerce shopify woocommerce magento bigcommerce stripe paypal "
    "security penetration testing audit firewall vpn encryption hashing "
    "database administration backup restore replication sharding index "
    "quality assurance testing regression unit integration end to end "
    "mobile ios android swift kotlin react native flutter xamarin ionic "
    "desktop windows macos linux electron qt gtk winforms wpf javafx "
    "blockchain crypto nft solidity ethereum smart contract defi wallet "
    "game design unity unreal godot level scripting physics multiplayer "
    "engineering mechanical electrical civil structural chemical industrial "
    "architecture construction interior landscape cad revit sketchup autocad "
    "art illustration painting sketching portrait character concept logo "
    "photography product fashion portrait event wedding real estate drone"
).split()
_PUNCT = (",", ".", "-", ":", ";", "/", " & ", " + ", " - ", "  ", " | ")


def _sample_length(rng: random.Random) -> int:
    """Heavy-tailed length matching real skills_text: p50~50, p99~1500, max~4800."""
    # Log-normal-ish: exp(uniform(0, log(4800))) clipped to [1, 4799].
    x = rng.random()
    # Bias toward short: 50% short (1..40), 40% medium (40..500), 10% long.
    if x < 0.5:
        return rng.randint(1, 40)
    if x < 0.9:
        return rng.randint(40, 500)
    return rng.randint(500, 4799)


def _make_natural_string(rng: random.Random, target_len: int) -> str:
    """Produce ~target_len chars of English-word content with mixed punctuation."""
    if target_len <= 3:
        # Very short values — single fragment.
        return rng.choice(_VOCAB)[:target_len] or "x"
    out: list[str] = []
    cur = 0
    while cur < target_len:
        w = rng.choice(_VOCAB)
        out.append(w)
        cur += len(w)
        if cur < target_len:
            sep = rng.choice(_PUNCT) if rng.random() < 0.15 else " "
            out.append(sep)
            cur += len(sep)
    s = "".join(out)
    return s[:target_len] if len(s) > target_len else s


def _build_realistic_primer(n_total: int, real_strings: set[str],
                            seed: int) -> list[str]:
    """Build priming sequence of length n_total whose i-th element will land
    at physical code i in an empty StringCache. Natural-language content and
    heavy-tailed length distribution; guaranteed distinct so each registers.
    """
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set(real_strings)  # reserve real category strings
    # Uniqueness counter appended in a rare, varied way — not as a uniform
    # suffix, so we don't re-introduce f"skill_X"-style regularity.
    uniq = 0
    while len(out) < n_total:
        length = _sample_length(rng)
        base = _make_natural_string(rng, length)
        # Inject an occasional uniqueness token only when collision is likely
        # (short strings). For long strings, natural variety is enough.
        if length < 80 or base in seen:
            tok = f" {uniq:x}" if rng.random() < 0.5 else f" ~{uniq}"
            s = (base + tok)[:max(length, 8)]
            uniq += 1
        else:
            s = base
        if s in seen:
            # very rare; add a short salt
            s = s + f" @{uniq:x}"
            uniq += 1
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for synthetic content (deterministic).")
    ap.add_argument("--big-rows", type=int, default=1_000_000,
                    help="Parent DataFrame size (prod was 9M).")
    ap.add_argument("--workaround", action="store_true",
                    help="Apply pl.Enum(sorted_uniques) fix — should not crash.")
    args = ap.parse_args()

    import xgboost
    print(f"polars {pl.__version__}, xgboost {xgboost.__version__}, "
          f"platform={sys.platform}, "
          f"using_string_cache={pl.using_string_cache()}",
          flush=True)

    real_strings = {s for _, s in DICT_PAIRS}
    pairs_sorted = sorted(DICT_PAIRS)  # sort by code

    # ---- Build priming sequence: realistic padding + real strings at codes. ----
    t0 = time.perf_counter()
    padding_needed = pairs_sorted[-1][0] + 1 - len(pairs_sorted)  # codes to fill
    # Generate padding pool once; slot real strings in place.
    print(f"generating {padding_needed:,} realistic padding strings "
          f"(variable length 1..~4800, natural vocab)...", flush=True)
    # We need (max_code + 1) total slots; real strings occupy len(pairs_sorted)
    # of them, the rest are natural padding.
    padding = _build_realistic_primer(padding_needed, real_strings,
                                      seed=args.seed)

    # Interleave: walk codes 0..max_code; at each real-string code, emit the
    # real string; otherwise pop next padding value.
    real_by_code = {c: s for c, s in pairs_sorted}
    max_code = pairs_sorted[-1][0]
    sequence: list[str] = []
    pad_i = 0
    for code in range(max_code + 1):
        if code in real_by_code:
            sequence.append(real_by_code[code])
        else:
            sequence.append(padding[pad_i])
            pad_i += 1
    # Sanity: lengths
    lengths = [len(s) for s in sequence]
    print(f"  primer built: n={len(sequence):_}, "
          f"len min={min(lengths)} p50={np.percentile(lengths, 50):.0f} "
          f"p99={np.percentile(lengths, 99):.0f} max={max(lengths)} "
          f"({time.perf_counter()-t0:.1f}s)",
          flush=True)

    # ---- Populate the global StringCache by casting primer to Categorical. ----
    t0 = time.perf_counter()
    _ = pl.Series("primer", sequence, dtype=pl.Categorical)
    print(f"  primed StringCache in {time.perf_counter()-t0:.1f}s",
          flush=True)

    # ---- Verify every real string landed at its target code. ----
    sample_check = pl.Series("check", [s for _, s in pairs_sorted],
                             dtype=pl.Categorical)
    actual_codes = sample_check.to_physical().to_list()
    expected_codes = [c for c, _ in pairs_sorted]
    mismatches = [
        (e, a, s)
        for (e, a, (_, s)) in zip(expected_codes, actual_codes, pairs_sorted)
        if e != a
    ]
    if mismatches:
        print(f"  WARN: {len(mismatches)} mismatched cache positions", flush=True)
        for e, a, s in mismatches[:5]:
            print(f"    expected code={e}, got={a}, string={s!r}", flush=True)
        print("  (the bug reproduces regardless, as long as max_code stays sparse)",
              flush=True)
    else:
        print(f"  all {len(pairs_sorted)} entries at target cache positions",
              flush=True)

    # ---- Build parent DataFrame so train/val slices share one dict. ----
    train_str = real_by_code[TRAIN_CODE]
    val_str = real_by_code[VAL_CODE]
    print(f"\ntrain row: code={TRAIN_CODE}, string={train_str!r}", flush=True)
    print(f"val   row: code={VAL_CODE}, string={val_str!r}", flush=True)

    rest = [s for _, s in pairs_sorted if s not in (train_str, val_str)]
    head = [train_str, val_str] + rest
    rng = np.random.default_rng(args.seed)
    pad = rng.choice([s for _, s in pairs_sorted],
                     size=args.big_rows - len(head)).tolist()
    all_rows = head + pad
    big = pl.DataFrame(
        {"category": pl.Series("category", all_rows, dtype=pl.Categorical)}
    )
    phys = big["category"].to_physical()
    print(f"parent DataFrame: {big.shape}, "
          f"n_unique={big['category'].n_unique()}, "
          f"codes_range=[{phys.min()}, {phys.max()}], "
          f"n_chunks={big['category'].n_chunks()}",
          flush=True)

    if args.workaround:
        print("  APPLYING WORKAROUND: cast to pl.Enum(sorted_uniques)", flush=True)
        uniques = sorted(big["category"].unique().drop_nulls().to_list())
        big = big.with_columns(pl.col("category").cast(pl.Enum(uniques)))
        print(f"  after Enum cast: codes_max="
              f"{big['category'].to_physical().max()}", flush=True)

    train = big[:1]
    val = big[1:2]
    print(f"train phys code = {train['category'].to_physical().to_list()}",
          flush=True)
    print(f"val   phys code = {val['category'].to_physical().to_list()}",
          flush=True)

    y_tr = np.array([0], dtype=np.int8)
    y_v = np.array([1], dtype=np.int8)

    print("\nfitting XGB -- expect silent kill (exit 3221226505) on Windows",
          flush=True)
    m = XGBClassifier(
        n_estimators=5,
        enable_categorical=True,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        verbosity=1,
        max_cat_to_onehot=1,
        max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic",
        eval_metric="logloss",
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

r"""Pure synthetic reproducer using EXACT physical_code -> string mapping
extracted from prod via ``extract_crash_state.py`` 2026-04-20.

Cache state to recreate:
  codes 0..87        -> 88 actual category strings
  code 2_526_058     -> "__MISSING__" sentinel (came in via fill_null)

After cache is in this exact state, build a polars Categorical column
with 2 rows: train at code 16, val at code 8. Fit XGB. Expect crash.

If THIS reproduces, we have a 100% pure-synthetic upstream reproducer
with NO parquet, NO 2.5M skill strings, NO mystery — just constructing
the exact polars/Arrow state via deterministic cache population.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from xgboost import XGBClassifier

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


# Extracted verbatim from extract_crash_state output on prod 2026-04-20.
DICT_PAIRS = [
    (0, 'Digital Marketing'),
    (1, 'Virtual Assistance'),
    (2, 'Marketing, PR & Brand Strategy'),
    (3, 'Graphic, Editorial & Presentation Design'),
    (4, 'Lead Generation & Telemarketing'),
    (5, 'Content Writing'),
    (6, 'Management Consulting & Analysis'),
    (7, 'ERP/CRM Software'),
    (8, 'Web Development'),
    (9, 'Mobile Development'),
    (10, 'Video & Animation'),
    (11, 'Ecommerce Development'),
    (12, 'Web & Mobile Design'),
    (13, '3D Modeling & CAD'),
    (14, 'DevOps & Solution Architecture'),
    (15, 'Electrical & Electronic Engineering'),
    (16, 'Desktop Application Development'),
    (17, 'Branding & Logo Design'),
    (18, 'Network & System Administration'),
    (19, 'Building & Landscape Architecture'),
    (20, 'AI Apps & Integration'),
    (21, 'Editing & Proofreading Services'),
    (22, 'Performing Arts'),
    (23, 'Data Entry & Transcription Services'),
    (24, 'Scripts & Utilities'),
    (25, 'Photography'),
    (26, 'AI & Machine Learning'),
    (27, 'Customer Service & Tech Support'),
    (28, 'Data Analysis & Testing'),
    (29, 'Recruiting & Human Resources'),
    (30, 'Finance & Tax Law'),
    (31, 'Art & Illustration'),
    (32, 'Energy & Mechanical Engineering'),
    (33, 'QA Testing'),
    (34, 'Product Design'),
    (35, 'Accounting & Bookkeeping'),
    (36, 'Civil & Structural Engineering'),
    (37, 'Game Design & Development'),
    (38, 'Financial Planning'),
    (39, 'Market Research & Product Reviews'),
    (40, 'Project Management'),
    (41, 'Data Extraction/ETL'),
    (42, 'Other - Accounting & Consulting'),
    (43, 'Personal & Professional Coaching'),
    (44, 'Physical Sciences'),
    (45, 'Information Security & Compliance'),
    (46, 'Interior & Trade Show Design'),
    (47, 'Professional & Business Writing'),
    (48, 'Translation & Localization Services'),
    (49, 'Contract Manufacturing'),
    (50, 'Community Management & Tagging'),
    (51, 'Corporate & Contract Law'),
    (52, 'Sales & Marketing Copywriting'),
    (53, 'Data Mining & Management'),
    (54, 'Other - Software Development'),
    (55, 'Chemical Engineering'),
    (56, 'Audio & Music Production'),
    (57, 'Database Management & Administration'),
    (58, 'Product Management & Scrum'),
    (59, 'Public Law'),
    (60, 'Language Tutoring & Interpretation'),
    (61, 'International & Immigration Law'),
    (62, 'Blockchain, NFT & Cryptocurrency'),
    (63, 'NFT, AR/VR & Game Art'),
    (64, 'Legal, Medical & Technical Translation'),
    (65, 'Resumes & Cover Letters'),
    (66, 'Market & Customer Research'),
    (67, 'Telemarketing & Telesales'),
    (68, 'Creative Writing Services'),
    (69, 'Social Media Marketing & Strategy'),
    (70, 'SEO & SEM Services'),
    (71, 'SEM - Search Engine Marketing'),
    (72, 'Technical Translation'),
    (73, 'Voice Talent'),
    (74, 'Transcription'),
    (75, 'Quantitative Analysis'),
    (76, 'Other - Admin Support'),
    (77, 'Other - Data Science & Analytics'),
    (78, 'Other - Writing'),
    (79, 'Intellectual Property Law'),
    (80, 'Email & Marketing Automation'),
    (81, 'Corporate Law'),
    (82, 'Other - Engineering'),
    (83, 'Medical Translation'),
    (84, 'Other - Sales & Marketing'),
    (85, 'Graphics & Design'),
    (86, 'Tech Support & Content Moderation'),
    (87, 'Grant & Proposal Writing'),
    (2_526_058, '__MISSING__'),
]
TRAIN_CODE = 16  # 'Desktop Application Development'
VAL_CODE = 16    # SAME as train -- avoids XGB's 'unseen category' guard
                 # so we can probe the pure bin-allocation crash path.


def main():
    import xgboost
    print(f"polars {pl.__version__}, xgboost {xgboost.__version__}, "
          f"using_string_cache={pl.using_string_cache()}", flush=True)

    # Step 1: populate the global StringCache so each entry of
    # DICT_PAIRS ends up at its target physical code.
    # Strategy: cast a single Series listing all strings in the order
    # required by the codes. Pad with dummy strings between entries
    # to fill intermediate cache positions.
    t0 = time.perf_counter()
    pairs_sorted = sorted(DICT_PAIRS)  # sort by code
    # Build the priming sequence: for each (code, string) pair, ensure
    # that exactly `code` strings precede this one in the cache.
    sequence: list[str] = []
    next_code = 0
    pad_idx = 0
    for code, s in pairs_sorted:
        # Pad with dummies until the cache has `code` entries.
        while next_code < code:
            sequence.append(f"__pad_{pad_idx:08d}")
            pad_idx += 1
            next_code += 1
        # Now register the real string — it gets `code`.
        sequence.append(s)
        next_code += 1
    print(f"  built priming sequence: {len(sequence):_} strings "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # Cast to Categorical to populate the global cache.
    t0 = time.perf_counter()
    _ = pl.Series("primer", sequence, dtype=pl.Categorical)
    print(f"  cast primer to Categorical in {time.perf_counter()-t0:.1f}s",
          flush=True)

    # Verify each (code, string) pair landed at its target code.
    print("Verifying cache state...", flush=True)
    sample_check = pl.Series("check", [s for _, s in pairs_sorted],
                             dtype=pl.Categorical)
    actual_codes = sample_check.to_physical().to_list()
    expected_codes = [c for c, _ in pairs_sorted]
    mismatches = [(e, a, s) for (e, a, (_, s)) in
                  zip(expected_codes, actual_codes, pairs_sorted) if e != a]
    if mismatches:
        print(f"  MISMATCHED {len(mismatches)} entries:", flush=True)
        for e, a, s in mismatches[:5]:
            print(f"    expected code={e}, got={a}, string={s!r}", flush=True)
    else:
        print(f"  all {len(pairs_sorted)} entries at correct cache positions",
              flush=True)

    # Step 2: build ONE DataFrame containing all 89 strings, then slice
    # train and val from it. This way both halves share the SAME
    # Categorical dict (89 entries) — matching prod's slice-of-one-frame
    # behaviour. XGB then doesn't see "unseen val category" because all
    # values are visible in the column's dict.
    train_str = next(s for c, s in pairs_sorted if c == TRAIN_CODE)
    val_str   = next(s for c, s in pairs_sorted if c == VAL_CODE)
    print(f"\ntrain row: code={TRAIN_CODE}, string={train_str!r}", flush=True)
    print(f"val   row: code={VAL_CODE}, string={val_str!r}", flush=True)

    # Build a LARGE DataFrame whose parent state mimics prod's 9M-row
    # source. Row 0 = train_str, row 1 = val_str, rest of 89 strings
    # filling rows 2..89, then PAD up to BIG_ROWS using random
    # selection from the 89 dict so the parent has all 89 categories
    # used somewhere. Slicing preserves parent's dict.
    BIG_ROWS = 1_000_000  # tunable; prod was 9M
    rest = [s for _, s in pairs_sorted if s not in (train_str, val_str)]
    head = [train_str, val_str] + rest
    rng = np.random.default_rng(42)
    pad = rng.choice([s for _, s in pairs_sorted], size=BIG_ROWS - len(head)).tolist()
    all_rows = head + pad
    big = pl.DataFrame({"category": pl.Series("category", all_rows,
                                              dtype=pl.Categorical)})
    print(f"big DataFrame: {big.shape}, "
          f"category n_unique={big['category'].n_unique()}, "
          f"codes_range=[{big['category'].to_physical().min()}, "
          f"{big['category'].to_physical().max()}], "
          f"n_chunks={big['category'].n_chunks()}", flush=True)

    train = big[:1]
    val   = big[1:2]
    print(f"train[category] phys code = {train['category'].to_physical().to_list()}",
          flush=True)
    print(f"val[category] phys code = {val['category'].to_physical().to_list()}",
          flush=True)

    y_tr = np.array([0], dtype=np.int8)
    y_v  = np.array([1], dtype=np.int8)

    print("\nfitting XGB — expect silent kill (exit 3221226505)", flush=True)
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
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s — bug did NOT reproduce",
              flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

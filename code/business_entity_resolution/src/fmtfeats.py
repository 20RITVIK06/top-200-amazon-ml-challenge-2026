"""Raw-format features of the Source-2/3 record (label-free).

Normalisation deliberately removes surface formatting, but the generator's noise pipeline
leaves artefacts (case changes, doubled spaces, leading zeros, '#', NULL tokens, junk prefixes)
whose frequencies differ between records of real Source-1 entities and distractor records
(e.g. an empty address almost never occurs on a distractor: P(matched | empty) = 0.978).

usage: python fmtfeats.py <data_dir> <cand_dir> <split> <out_parquet>
"""
import sys
import time

import numpy as np
import pandas as pd


def record_formats(d):
    n, a = d.business_name.fillna(""), d.business_address.fillna("")
    f = pd.DataFrame({
        "fm_name_upper": (n.str.upper() == n) & n.str.contains(r"[A-Za-z]", regex=True),
        "fm_name_lower": (n.str.lower() == n) & n.str.contains(r"[A-Za-z]", regex=True),
        "fm_name_dbl": n.str.contains("  ", regex=False),
        "fm_name_lead_punct": n.str.match(r"^[^\w\s]"),
        "fm_name_trail_punct": n.str.contains(r"[^\w\s\)\]\.]$", regex=True),
        "fm_name_brackets": n.str.contains(r"[\[\]\(\)]", regex=True),
        "fm_addr_upper": (a.str.upper() == a) & (a.str.len() > 0),
        "fm_addr_empty": a.str.len() == 0,
        "fm_addr_dbl": a.str.contains("  ", regex=False),
        "fm_addr_lead0": a.str.contains(r"(?:^|[ ,#])0\d", regex=True),
        "fm_addr_hash": a.str.contains("#", regex=False),
        "fm_addr_null": a.str.contains(r"(?i)\bnull\b|n/a|<null>", regex=True),
        "fm_addr_pobox": a.str.contains(r"(?i)\b(?:po box|pmb)\b", regex=True),
    }).astype(np.float32)
    f["fm_addr_ncomma"] = a.str.count(",").astype(np.float32)
    f["fm_name_len"] = n.str.len().astype(np.float32)
    f["fm_addr_len"] = a.str.len().astype(np.float32)
    return f


def main():
    data_dir, cand_dir, split, out = sys.argv[1:5]
    t0 = time.time()
    d = pd.concat([pd.read_parquet(f"{data_dir}/{split}_s{k}.parquet", columns=["business_name", "business_address"])
                   for k in (2, 3)], ignore_index=True)
    R = record_formats(d)
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet", columns=["j"])
    F = R.iloc[cand.j.values].reset_index(drop=True)
    F.to_parquet(out, index=False)
    print(f"{split}: format feats {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

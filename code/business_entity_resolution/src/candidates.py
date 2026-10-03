"""Stage A: build the raw candidate pair table for a split (train or test).

usage: python candidates.py <norm_dir> <split> <out_dir>

Output parquet: one row per unique (S1 record, S2/S3 record) pair with channel ranks
and exact TF-IDF cosines on the full (unpruned) vectors: name / address / combined.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from blocking import CFG, block_country
from features import rowdot


def main():
    norm_dir, split, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    cols = ["entity_id", "country", "nc", "at"]
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s2_norm.parquet", columns=cols),
                     pd.read_parquet(f"{norm_dir}/{split}_s3_norm.parquet", columns=cols)], ignore_index=True)
    out = []
    for country in sorted(set(s1.country) | set(s23.country)):
        a1 = np.where(s1.country.values == country)[0]
        a23 = np.where(s23.country.values == country)[0]
        print(f"== {split} {country}: S1 {len(a1)}  S2/3 {len(a23)}", flush=True)
        if len(a1) == 0 or len(a23) == 0:
            continue
        t = time.time()
        cand, V = block_country(s1.iloc[a1].reset_index(drop=True), s23.iloc[a23].reset_index(drop=True), CFG)
        I, J = cand.i1.values, cand.j.values
        cand["cos_name"] = rowdot(V["N1"], V["N23"], I, J)
        cand["cos_addr"] = rowdot(V["A1"], V["A23"], I, J)
        cand["cos_comb"] = rowdot(V["C1"], V["C23"], I, J)
        cand["i1"] = a1[I]
        cand["j"] = a23[J]
        out.append(cand)
        print(f"   {country}: {len(cand)} unique pairs ({len(cand) / len(a23):.1f}/record) {time.time() - t:.0f}s", flush=True)
    cand = pd.concat(out, ignore_index=True)
    cand["s1_id"] = s1.entity_id.values[cand.i1.values]
    cand["r_id"] = s23.entity_id.values[cand.j.values]
    cand.to_parquet(f"{out_dir}/{split}_cand_raw.parquet", index=False)
    print("saved", len(cand), flush=True)


if __name__ == "__main__":
    main()

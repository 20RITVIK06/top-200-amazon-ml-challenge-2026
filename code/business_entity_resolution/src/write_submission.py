"""Write matching_results.tsv and candidate_pairs.tsv for the test set.

usage: python write_submission.py <data_dir> <cand_dir> <prob.npy> <out_dir> <mode> [param]

mode 'threshold' : arg-max assignment per record, keep if prob >= param
mode 'expf'      : arg-max assignment per record, then per-entity expected-F0.5 selection
"""
import os
import sys

import numpy as np
import pandas as pd

from expected_f import select
from postproc import argmax_assign


def write_lists(path, s1_ids, pairs_s1, pairs_r, col):
    lists = pd.Series(pairs_r).groupby(pairs_s1).agg(lambda x: ",".join(sorted(set(x))))
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False)
    return out


def main():
    data_dir, cand_dir, prob_path, out_dir, mode = sys.argv[1:6]
    param = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    os.makedirs(out_dir, exist_ok=True)
    s1 = pd.read_parquet(f"{data_dir}/test_s1.parquet", columns=["entity_id"])
    cand = pd.read_parquet(f"{cand_dir}/test_cand.parquet", columns=["s1_id", "r_id", "i1", "j"])
    p = np.load(prob_path).astype(np.float64)
    assert len(p) == len(cand)
    # candidate set (exactly the pairs scored by the matcher)
    c = write_lists(f"{out_dir}/candidate_pairs.tsv", s1.entity_id.values, cand.s1_id.values, cand.r_id.values,
                    "candidate_entity_ids")
    m = argmax_assign(cand.i1.values, cand.j.values, p)
    if mode == "threshold":
        keep = m & (p >= param)
    else:
        idx = np.where(m)[0]
        sel = select(cand.i1.values[idx], p[idx], int(cand.i1.max()) + 1, beta=0.5)
        keep = np.zeros(len(p), bool)
        keep[idx[sel]] = True
    r = write_lists(f"{out_dir}/matching_results.tsv", s1.entity_id.values, cand.s1_id.values[keep], cand.r_id.values[keep],
                    "matched_entity_ids")
    n_match = (r.matched_entity_ids != "").mean()
    print(f"wrote {len(r)} rows; entities with >=1 match: {n_match:.4f}; matched pairs: {keep.sum()}; "
          f"candidate rows non-empty: {(c.candidate_entity_ids != '').mean():.4f}")


if __name__ == "__main__":
    main()

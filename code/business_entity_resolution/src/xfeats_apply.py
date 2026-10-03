"""Compute token log-odds / number-relation features for another train candidate set, using the
fold-aware tables learned on the reference train candidates (so test features stay consistent).

usage: python xfeats_apply.py <norm_dir> <ref_cand_dir> <target_cand_dir> <data_dir> <out_parquet>
"""
import sys
import time

import numpy as np
import pandas as pd

import extra_feats as XF


def main():
    norm_dir, ref_dir, tgt_dir, data_dir, out = sys.argv[1:6]
    t0 = time.time()
    gt = pd.read_parquet(f"{data_dir}/train_gt.parquet")
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    ref = pd.read_parquet(f"{ref_dir}/train_cand.parquet", columns=["i1", "j", "s1_id", "r_id"])
    Y = (ref.r_id.map(truth).values == ref.s1_id.values).astype(np.int8)
    half = (pd.util.hash_pandas_object(ref.s1_id, index=False).values % 2).astype(np.int8)
    s1 = pd.read_parquet(f"{norm_dir}/train_s1_norm.parquet", columns=["nc"])
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/train_s{k}_norm.parquet", columns=["nc"]) for k in (2, 3)], ignore_index=True)
    nc1, nc23 = s1.nc.tolist(), s23.nc.tolist()
    I, J = ref.i1.values, ref.j.values
    tables = [XF.learn_logodds(nc1, nc23, I[half == h], J[half == h], Y[half == h]) for h in (0, 1)]
    print(f"tables learned {time.time() - t0:.0f}s", flush=True)
    tgt = pd.read_parquet(f"{tgt_dir}/train_cand.parquet", columns=["s1_id"])
    th = (pd.util.hash_pandas_object(tgt.s1_id, index=False).values % 2).astype(np.int8)
    F = XF.compute("train", norm_dir, tgt_dir, tables, np.where(th == 0, 1, 0).astype(np.int8))
    F.to_parquet(out, index=False)
    print(f"saved {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

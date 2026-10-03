"""Shift-aware training set: simulate the test regime on train.

The test set behaves like train with ~18% of Source-1 entities removed: their records become
distractors ("orphans") that look exactly like real-entity records (realistic noise, sibling
records that agree with each other).  Here we drop a random `frac` of train S1 entities,
remove their pairs, and recompute every feature that depends on the set of competing
candidates (context features).  Stage-0 derived features (p0, blocking ranks) cannot be
recomputed faithfully without re-running blocking, so the shift-aware models do not use them.

usage: python sa_prepare.py <work_dir> <out_cand_dir> <out_feats_dir> <frac> <extra1.parquet> ...
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from build_features import context_features

STALE = ["p0", "rk_comb", "rk_name", "rk_addr", "rk_rev"]


def main():
    W, cand_out, feats_out, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    extras = sys.argv[5:]
    t0 = time.time()
    os.makedirs(cand_out, exist_ok=True)
    os.makedirs(feats_out, exist_ok=True)
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    h = pd.util.hash_pandas_object(gt.source1_entity_id, index=False).values
    drop = ((h >> np.uint64(7)) % np.uint64(1000)) < frac * 1000
    kept_ids = set(gt.source1_entity_id.values[~drop])
    gt[~drop].reset_index(drop=True).to_parquet(f"{cand_out}/train_gt_sa.parquet", index=False)
    s1 = pd.read_parquet(f"{W}/data/train_s1.parquet", columns=["entity_id"])
    s1[s1.entity_id.isin(kept_ids)].reset_index(drop=True).to_parquet(f"{cand_out}/train_s1_sa.parquet", index=False)
    cand = pd.read_parquet(f"{W}/cand/train_cand.parquet")
    keep = cand.s1_id.isin(kept_ids).values
    c = cand[keep].reset_index(drop=True)
    c.to_parquet(f"{cand_out}/train_cand.parquet", index=False)
    for k in ("name", "addr", "comb"):
        src = os.path.abspath(f"{W}/cand/train_recvec_{k}.npz")
        dst = f"{cand_out}/train_recvec_{k}.npz"
        if not os.path.exists(dst):
            os.symlink(src, dst)
    src = os.path.abspath(f"{W}/cand/train_recnn.parquet")
    if not os.path.exists(f"{cand_out}/train_recnn.parquet"):
        os.symlink(src, f"{cand_out}/train_recnn.parquet")
    print(f"dropped {drop.sum()} S1 ({drop.mean():.3f}); pairs kept {len(c)} / {len(cand)}  {time.time() - t0:.0f}s", flush=True)
    X = pd.read_parquet(f"{W}/feats3/train_feats.parquet")[keep].reset_index(drop=True)
    for k, v in context_features(c).items():
        X[k] = v
    X = X.drop(columns=[s for s in STALE if s in X.columns])
    X.to_parquet(f"{feats_out}/train_feats.parquet", index=False)
    for e in extras:
        pd.read_parquet(e)[keep].reset_index(drop=True).to_parquet(f"{feats_out}/{os.path.basename(e)}", index=False)
    print(f"saved shift-aware features {X.shape} + {len(extras)} extras  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

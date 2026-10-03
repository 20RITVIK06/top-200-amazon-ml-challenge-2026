"""Compute stage-2 collective features for a split from stage-1 probabilities.

usage: python make_stage2_feats.py <norm_dir> <cand_dir> <split> <p1.npy> <out_parquet>
       python make_stage2_feats.py prepare <norm_dir> <cand_dir> <split>   (cache record vectors)
"""
import os

import scipy.sparse as sp
import sys
import time

import numpy as np
import pandas as pd

from stage2 import collective_features, record_vectors


def cached_vectors(norm_dir, cand_dir, split):
    keys = ("name", "addr", "comb")
    paths = {k: f"{cand_dir}/{split}_recvec_{k}.npz" for k in keys}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: sp.load_npz(paths[k]).tocsr() for k in keys}
    V = record_vectors(norm_dir, split)
    for k in keys:
        sp.save_npz(paths[k], V[k])
    return V


def main():
    if sys.argv[1] == "prepare":
        norm_dir, cand_dir, split = sys.argv[2:5]
        cached_vectors(norm_dir, cand_dir, split)
        print("cached", split)
        return
    norm_dir, cand_dir, split, p1_path, out = sys.argv[1:6]
    t0 = time.time()
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet", columns=["i1", "j", "r_id"])
    cand["src3"] = cand.r_id.str.startswith("S3-").astype(np.int8)
    p1 = np.load(p1_path).astype(np.float64)
    assert len(p1) == len(cand)
    V = cached_vectors(norm_dir, cand_dir, split)
    print(f"vectors {time.time() - t0:.0f}s", flush=True)
    F = collective_features(cand, p1, V)
    F.to_parquet(out, index=False)
    print(f"saved {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

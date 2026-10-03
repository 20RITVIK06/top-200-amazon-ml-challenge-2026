"""Full-fidelity shift-aware candidate set.

Re-creates, on train, the test regime where ~frac of Source-1 entities are absent: their
pairs are removed from the RAW blocking candidates, per-channel blocking ranks are recomputed
among the remaining S1 entities, the stage-0 pre-ranker is re-applied (same model) and the
same pruning rule (top-M per record, p0 >= floor) yields the candidate set.  All downstream
features are then rebuilt on it, so orphaned records look exactly as they do on test.

usage: python sa_full.py <work_dir> <norm_s0_dir> <out_cand_dir> <frac> [M] [floor]
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from stage0 import meta, prune, s0_features


def main():
    W, norm_s0, out, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    M = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    floor = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-4
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    h = pd.util.hash_pandas_object(gt.source1_entity_id, index=False).values
    dropped = set(gt.source1_entity_id.values[((h >> np.uint64(7)) % np.uint64(1000)) < frac * 1000])
    raw = pd.read_parquet(f"{W}/cand/train_cand_raw.parquet")
    raw = raw[~raw.s1_id.isin(dropped)].reset_index(drop=True)
    print(f"raw pairs after dropping {len(dropped)} S1: {len(raw)}  {time.time() - t0:.0f}s", flush=True)
    # recompute per-record channel ranks among the remaining S1 entities (reverse ranks are per S1: unchanged)
    for col in ("rk_comb", "rk_name", "rk_addr"):
        v = raw[col].values
        m = v < 99
        r = pd.Series(v[m]).groupby(raw.j.values[m]).rank(method="first").values - 1
        nv = v.copy()
        nv[m] = r.astype(np.int16)
        raw[col] = nv
    m1, m23 = meta(norm_s0, "train")
    X = s0_features(raw, m1, m23).values
    bst = lgb.Booster(model_file=f"{W}/cand/stage0.txt")
    p0 = bst.predict(X, num_threads=int(os.environ.get("LGB_THREADS", "96")))
    del X
    kept = prune(raw, p0, M, floor)
    kept = kept.sort_values(["j", "i1"]).reset_index(drop=True)
    kept.to_parquet(f"{out}/train_cand.parquet", index=False)
    for f in ("train_recvec_name.npz", "train_recvec_addr.npz", "train_recvec_comb.npz", "train_recnn.parquet"):
        dst = f"{out}/{f}"
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f"{W}/cand/{f}"), dst)
    print(f"shift-aware candidate set: {len(kept)} pairs ({len(kept) / kept.j.nunique():.2f}/record)  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Stage B: cheap cascade pruning of the raw candidate set.

A small LightGBM trained on blocking-level features (channel cosines and ranks, plus
competition context) scores every raw candidate pair.  Per Source-2/3 record we keep
its top-M pairs by this score that also clear a small probability floor.  On train the
scores are 2-fold out-of-fold (grouped by S1 entity); the test scores come from a model
fitted on all train pairs.  The kept set is the candidate set fed to the full matcher
(and therefore what is written to candidate_pairs.tsv).

usage: python stage0.py train <norm_dir> <cand_dir> <gt_parquet> [M] [floor]
       python stage0.py test  <norm_dir> <cand_dir> [M] [floor]
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

S0_FEATS = ["cos_name", "cos_addr", "cos_comb", "rk_comb", "rk_name", "rk_addr", "rk_rev"]


def s0_features(c, s1_meta, s23_meta):
    X = pd.DataFrame({k: c[k].values.astype(np.float32) for k in S0_FEATS})
    for col in ["cos_comb", "cos_name", "cos_addr"]:
        g = c.groupby("j")[col]
        X[f"{col}_jmax_gap"] = (g.transform("max") - c[col]).values.astype(np.float32)
        X[f"{col}_jrank"] = g.rank(ascending=False, method="min").values.astype(np.float32)
        g1 = c.groupby("i1")[col]
        X[f"{col}_imax_gap"] = (g1.transform("max") - c[col]).values.astype(np.float32)
    X["j_n"] = c.groupby("j").j.transform("size").values.astype(np.float32)
    X["i_n"] = c.groupby("i1").i1.transform("size").values.astype(np.float32)
    X["r_addr_empty"] = s23_meta["aempty"][c.j.values]
    X["r_name_len"] = s23_meta["nlen"][c.j.values]
    X["s1_name_len"] = s1_meta["nlen"][c.i1.values]
    X["r_name_freq_s1"] = s23_meta["nfreq_s1"][c.j.values]
    X["s1_name_freq"] = s1_meta["nfreq_s1"][c.i1.values]
    return X


def meta(norm_dir, split):
    cols = ["country", "nc", "at"]
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s{k}_norm.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    cnt = s1.groupby(["country", "nc"]).size()
    m1 = {"nlen": s1.nc.str.len().values.astype(np.float32),
          "nfreq_s1": cnt.reindex(pd.MultiIndex.from_arrays([s1.country, s1.nc])).values.astype(np.float32)}
    m23 = {"nlen": s23.nc.str.len().values.astype(np.float32),
           "aempty": (s23["at"].str.len() == 0).values.astype(np.float32),
           "nfreq_s1": cnt.reindex(pd.MultiIndex.from_arrays([s23.country, s23.nc])).fillna(0).values.astype(np.float32)}
    return m1, m23


PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.5, bagging_freq=1, num_threads=96, verbose=-1)


def prune(c, p, M, floor):
    c = c.assign(p0=p.astype(np.float32))
    rk = pd.Series(p).groupby(c.j.values).rank(ascending=False, method="first").values
    keep = (rk <= M) & (p >= floor)
    return c[keep].reset_index(drop=True)


def test_features(norm_dir, cand_dir):
    te = pd.read_parquet(f"{cand_dir}/test_cand_raw.parquet")
    t1, t23 = meta(norm_dir, "test")
    X = s0_features(te, t1, t23).values
    np.save(f"{cand_dir}/test_s0X.npy", X)
    return te, X


def run_test(norm_dir, cand_dir, M, floor):
    import os
    t0 = time.time()
    bst = lgb.Booster(model_file=f"{cand_dir}/stage0.txt")
    if os.path.exists(f"{cand_dir}/test_s0X.npy"):
        te = pd.read_parquet(f"{cand_dir}/test_cand_raw.parquet")
        X = np.load(f"{cand_dir}/test_s0X.npy")
    else:
        te, X = test_features(norm_dir, cand_dir)
    pt = bst.predict(X, num_threads=96)
    kept = prune(te, pt, M, floor)
    kept.to_parquet(f"{cand_dir}/test_cand.parquet", index=False)
    print(f"test raw {len(te)} kept {len(kept)} ({len(kept) / te.j.nunique():.2f}/record)  {time.time() - t0:.0f}s", flush=True)


def main():
    if sys.argv[1] == "testfeats":
        return test_features(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "test":
        norm_dir, cand_dir = sys.argv[2:4]
        M = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        floor = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
        return run_test(norm_dir, cand_dir, M, floor)
    norm_dir, cand_dir, gt_path = sys.argv[2:5]
    M = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    floor = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-4
    t0 = time.time()
    gt = pd.read_parquet(gt_path)
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    tr = pd.read_parquet(f"{cand_dir}/train_cand_raw.parquet")
    y = (tr.r_id.map(truth).values == tr.s1_id.values).astype(np.int8)
    m1, m23 = meta(norm_dir, "train")
    X = s0_features(tr, m1, m23)
    fnames = list(X.columns)
    X = X.values
    print(f"train raw pairs {len(tr)} pos {y.sum()}  feats {time.time() - t0:.0f}s", flush=True)
    h = pd.util.hash_pandas_object(tr.s1_id, index=False).values % 2
    p = np.zeros(len(tr), np.float32)
    for k in range(2):
        a, b = np.where(h != k)[0], np.where(h == k)[0]
        a = a[np.random.default_rng(k).random(len(a)) < 0.3]
        bst = lgb.train(PARAMS, lgb.Dataset(X[a], y[a], feature_name=fnames), num_boost_round=300)
        p[b] = bst.predict(X[b], num_threads=96)
        print(f"  fold {k} done {time.time() - t0:.0f}s", flush=True)
    total_true = len(truth)
    rk = pd.Series(p).groupby(tr.j.values).rank(ascending=False, method="first").values
    for MM in (5, 8, 10, 12, 15):
        for fl in (1e-4, 1e-3, 1e-2):
            keep = (rk <= MM) & (p >= fl)
            print(f"   M={MM:2d} floor={fl:g}: kept {keep.sum()} ({keep.mean():.3f})  true kept {y[keep].sum()} / raw {y.sum()} "
                  f"(loss {1 - y[keep].sum() / y.sum():.5f})  recall vs all truth {y[keep].sum() / total_true:.5f}", flush=True)
    kept = prune(tr, p, M, floor)
    kept.to_parquet(f"{cand_dir}/train_cand.parquet", index=False)
    print(f"train kept {len(kept)}  {time.time() - t0:.0f}s", flush=True)
    sub = np.random.default_rng(7).random(len(tr)) < 0.3
    bst = lgb.train(PARAMS, lgb.Dataset(X[sub], y[sub], feature_name=fnames), num_boost_round=300)
    bst.save_model(f"{cand_dir}/stage0.txt")
    print(f"stage0 model saved {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

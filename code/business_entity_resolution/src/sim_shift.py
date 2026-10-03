"""Simulate the test regime on train: drop a random fraction of S1 entities (their records
become distractors, as the test's higher records-per-S1 ratio implies), recompute every
competition-dependent feature without them, re-score each pair with its out-of-fold models
(stage 1 then stage 2) and compare decision rules.

usage: python sim_shift.py <work_dir> <s1_prefix> <s2_prefix> <drop_frac> [extra_train_feats.parquet]
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_features import context_features
from expected_f import select
from metric import f05_vectorized
from postproc import argmax_assign
from stage2 import collective_features


def oof_predict(prefix, X, fold, k=4):
    p = np.zeros(len(X))
    for f in range(k):
        m = fold == f
        bst = lgb.Booster(model_file=f"{prefix}_fold{f}.txt")
        p[m] = bst.predict(X.loc[m, bst.feature_name()].values.astype(np.float32), num_threads=96)
    return p


def main():
    W, s1p, s2p, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    extra = sys.argv[5] if len(sys.argv) > 5 else None
    t0 = time.time()
    cand = pd.read_parquet(f"{W}/cand/train_cand.parquet")
    X = pd.read_parquet(f"{W}/{__import__("os").environ.get("FEATS_DIR", "feats")}/train_feats.parquet")
    if extra:
        X = pd.concat([X, pd.read_parquet(extra)], axis=1)
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    h = pd.util.hash_pandas_object(gt.source1_entity_id, index=False).values
    drop = set(gt.source1_entity_id.values[((h >> np.uint64(7)) % np.uint64(1000)) < frac * 1000])
    keep = ~cand.s1_id.isin(drop).values
    cand, X = cand[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    gt = gt[~gt.source1_entity_id.isin(drop)].reset_index(drop=True)
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    y = (cand.r_id.map(truth).values == cand.s1_id.values).astype(float)
    print(f"dropped {len(drop)} S1 ({frac:.2f}); pairs {len(cand)}; pos rate {y.mean():.4f}  {time.time() - t0:.0f}s", flush=True)
    # recompute competition-dependent context features on the reduced candidate table
    for k, v in context_features(cand).items():
        X[k] = v
    fold = (pd.util.hash_pandas_object(cand.s1_id, index=False).values % 4).astype(int)
    p1 = oof_predict(s1p, X, fold)
    V = {k: sp.load_npz(f"{W}/cand/train_recvec_{k}.npz").tocsr() for k in ("name", "addr", "comb")}
    c2 = cand[["i1", "j"]].copy()
    c2["src3"] = cand.r_id.str.startswith("S3-").values.astype(np.int8)
    F2 = collective_features(c2, p1, V)
    X2 = pd.concat([X, F2], axis=1)
    p2 = oof_predict(s2p, X2, fold)
    print(f"re-scored {time.time() - t0:.0f}s  mean p2 {p2.mean():.4f}", flush=True)
    np.save(f"{W}/models/sim_p2_{int(frac * 100)}.npy", p2)
    # evaluation over the remaining S1 entities
    ids = gt.source1_entity_id.values
    pos = pd.Series(np.arange(len(ids)), index=ids)
    s1i = pos.reindex(cand.s1_id).values
    rj = pd.factorize(cand.r_id)[0]
    ntrue = gt.matched_entity_ids.map(lambda l: len(l.split(",")) if l else 0).values.astype(float)
    n = len(ids)
    m = argmax_assign(s1i, rj, p2)
    idx = np.where(m)[0]

    def score(sel_idx):
        tp = np.bincount(s1i[sel_idx], weights=y[sel_idx], minlength=n)
        npred = np.bincount(s1i[sel_idx], minlength=n)
        return f05_vectorized(tp, npred, ntrue).mean()

    res = {}
    for th in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        res[f"th{th}"] = score(idx[p2[idx] >= th])
    for odds in (1.0, 0.8, 0.6, 0.5, 0.4):
        q = p2[idx] * odds / (p2[idx] * odds + 1 - p2[idx])
        res[f"expf_odds{odds}"] = score(idx[select(s1i[idx], q, n, beta=0.5)])
    for k, v in res.items():
        print(f"  {k:16s} {v:.5f}")


if __name__ == "__main__":
    main()

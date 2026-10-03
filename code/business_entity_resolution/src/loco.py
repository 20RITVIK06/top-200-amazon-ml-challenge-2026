"""Leave-one-country-out proxy for an unseen test country.

Train a stage-1 model on one country only (e.g. US) using label-free features, score the
held-out country (India) and measure its macro-F0.5 with true labels; then test
self-training: add pseudo-labelled pairs of the held-out country (confident predictions of
the source model) to the training data, retrain, and re-evaluate.

usage: python loco.py <work_dir> <src_country> <tgt_country>
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from expected_f import select
from metric import f05_vectorized
from postproc import argmax_assign

LABEL_DEPENDENT = ["loc_alias_hits"]  # alias tables are learned from labels -> excluded


def score_country(cand, p, y, s1_country, ntrue, country):
    n = len(s1_country)
    I, J = cand.i1.values, cand.j.values
    m = argmax_assign(I, J, p)
    idx = np.where(m)[0]
    sel = idx[select(I[idx], p[idx], n)]
    tp = np.bincount(I[sel], weights=y[sel].astype(float), minlength=n)
    npred = np.bincount(I[sel], minlength=n)
    f = f05_vectorized(tp, npred, ntrue)
    return float(f[s1_country == country].mean())


def main():
    W, src, tgt = sys.argv[1:4]
    t0 = time.time()
    cand = pd.read_parquet(f"{W}/cand/train_cand.parquet", columns=["s1_id", "r_id", "i1", "j"])
    X = pd.read_parquet(f"{W}/feats3/train_feats.parquet")
    X = X.drop(columns=[c for c in LABEL_DEPENDENT if c in X.columns])
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    y = (cand.r_id.map(truth).values == cand.s1_id.values).astype(np.int8)
    s1 = pd.read_parquet(f"{W}/data/train_s1.parquet", columns=["entity_id", "country"])
    s1_country = s1.country.values
    cnt = {a: (len(l.split(",")) if l else 0) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    ntrue = np.array([cnt[e] for e in s1.entity_id], float)
    pc = s1_country[cand.i1.values]
    A, B = np.where(pc == src)[0], np.where(pc == tgt)[0]
    Xv = X.values.astype(np.float32)
    names = list(X.columns)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, num_threads=64, verbose=-1)
    print(f"loaded {time.time() - t0:.0f}s  src pairs {len(A)}  tgt pairs {len(B)}", flush=True)
    bst = lgb.train(params, lgb.Dataset(Xv[A], y[A], feature_name=names), num_boost_round=800)
    p = np.zeros(len(cand))
    p[B] = bst.predict(Xv[B], num_threads=64)
    base = score_country(cand, p, y, s1_country, ntrue, tgt)
    print(f"[{src}->{tgt}] source-only model: {tgt} macro-F0.5 = {base:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    for hi, lo in ((0.95, 0.05), (0.9, 0.1)):
        pos = B[p[B] >= hi]
        neg = B[p[B] <= lo]
        pl = np.concatenate([pos, neg])
        yl = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
        acc = (y[pl] == yl).mean()
        Xtr = np.vstack([Xv[A], Xv[pl]])
        ytr = np.concatenate([y[A], yl])
        bst2 = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names), num_boost_round=800)
        p2 = np.zeros(len(cand))
        p2[B] = bst2.predict(Xv[B], num_threads=64)
        s = score_country(cand, p2, y, s1_country, ntrue, tgt)
        print(f"[{src}->{tgt}] self-training (pseudo p>={hi} / <={lo}, {len(pl)} pairs, pseudo-label acc {acc:.4f}): "
              f"{tgt} macro-F0.5 = {s:.5f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""LOCO diagnostics: calibration vs feature-shift for an unseen country.

1. source-only model, evaluated on the target with odds multipliers (calibration test);
2. target predicted-empty rate vs source OOF (moment signal);
3. per-country rank (percentile) normalisation of all features, then source-only training.

usage: python loco2.py <work_dir> <src> <tgt>
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from expected_f import select
from metric import f05_vectorized
from postproc import argmax_assign


def decide(cand, p, n):
    I, J = cand.i1.values, cand.j.values
    m = argmax_assign(I, J, p)
    idx = np.where(m)[0]
    return idx[select(I[idx], p[idx], n)]


def fscore(cand, sel, y, ntrue, mask_s1):
    n = len(ntrue)
    I = cand.i1.values
    tp = np.bincount(I[sel], weights=y[sel].astype(float), minlength=n)
    npred = np.bincount(I[sel], minlength=n)
    f = f05_vectorized(tp, npred, ntrue)
    empty = (npred == 0)
    return float(f[mask_s1].mean()), float(empty[mask_s1].mean()), float(npred[mask_s1].mean())


def pct_norm(X, groups):
    """rank-normalise every column within each group (country) to [0, 1]."""
    out = np.empty_like(X, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        sub = pd.DataFrame(X[m])
        out[m] = sub.rank(pct=True, method="average").values.astype(np.float32)
    return out


def main():
    W, src, tgt = sys.argv[1:4]
    t0 = time.time()
    cand = pd.read_parquet(f"{W}/cand/train_cand.parquet", columns=["s1_id", "r_id", "i1", "j"])
    X = pd.read_parquet(f"{W}/feats3/train_feats.parquet").drop(columns=["loc_alias_hits"])
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    y = (cand.r_id.map(truth).values == cand.s1_id.values).astype(np.int8)
    s1 = pd.read_parquet(f"{W}/data/train_s1.parquet", columns=["entity_id", "country"])
    sc = s1.country.values
    cnt = {a: (len(l.split(",")) if l else 0) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    ntrue = np.array([cnt[e] for e in s1.entity_id], float)
    n = len(sc)
    pc = sc[cand.i1.values]
    A, B = np.where(pc == src)[0], np.where(pc == tgt)[0]
    names = list(X.columns)
    Xv = X.values.astype(np.float32)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, num_threads=64, verbose=-1)
    print(f"true singleton rate {tgt}: {np.mean(ntrue[sc == tgt] == 0):.4f}  mean true matches {ntrue[sc == tgt].mean():.3f}", flush=True)
    bst = lgb.train(params, lgb.Dataset(Xv[A], y[A], feature_name=names), num_boost_round=800)
    p = np.zeros(len(cand))
    p[B] = bst.predict(Xv[B], num_threads=64)
    for odds in (0.25, 0.5, 1.0, 2.0, 4.0):
        q = p * odds / (p * odds + 1 - p)
        f, e, mm = fscore(cand, decide(cand, q, n), y, ntrue, sc == tgt)
        print(f"  odds {odds:4}: {tgt} F0.5 {f:.5f}  empty {e:.4f}  mean pred matches {mm:.3f}", flush=True)
    # precision / recall of the base decision
    sel = decide(cand, p, n)
    sel = sel[pc[sel] == tgt]
    print(f"  base: predicted {len(sel)}  precision {y[sel].mean():.4f}  recall {y[sel].sum() / ntrue[sc == tgt].sum():.4f}", flush=True)
    # per-country rank normalisation
    Xn = pct_norm(Xv, pc)
    bst = lgb.train(params, lgb.Dataset(Xn[A], y[A], feature_name=names), num_boost_round=800)
    p = np.zeros(len(cand))
    p[B] = bst.predict(Xn[B], num_threads=64)
    f, e, mm = fscore(cand, decide(cand, p, n), y, ntrue, sc == tgt)
    print(f"  per-country rank-normalised features: {tgt} F0.5 {f:.5f}  empty {e:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

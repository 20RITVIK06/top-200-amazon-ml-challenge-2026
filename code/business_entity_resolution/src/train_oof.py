"""K-fold (grouped by Source-1 entity) LightGBM matcher with out-of-fold evaluation.

usage: python train_oof.py <cand_parquet> <feats_parquet> <gt_parquet> <s1_norm_parquet> <out_prefix>
       [--folds 2] [--rounds 3000] [--lr 0.05] [--extra extra_feats.parquet ...]

Writes <out_prefix>_oof.npy (OOF probabilities aligned with the candidate table) and
prints the macro-F0.5 after arg-max assignment for a sweep of thresholds.
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from expected_f import select
from metric import f05_vectorized
from postproc import argmax_assign, ceiling, evaluate


def eval_expf(ev, p, truth_cnt, n_s1):
    """Arg-max assignment then per-entity expected-F0.5 selection; returns macro F0.5."""
    s1i, rj, y = ev.s1i.values, ev.rj.values, ev.y.values.astype(float)
    m = argmax_assign(s1i, rj, p)
    idx = np.where(m)[0]
    sel = select(s1i[idx], p[idx], n_s1, beta=0.5)
    k = idx[sel]
    tp = np.bincount(s1i[k], weights=y[k], minlength=n_s1)
    npred = np.bincount(s1i[k], minlength=n_s1)
    return float(f05_vectorized(tp, npred, truth_cnt).mean())


def labels(cand, gt):
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    t = cand.r_id.map(truth)
    return (t.values == cand.s1_id.values).astype(np.int8)


def fold_of(ids, k, seed=0):
    h = pd.util.hash_pandas_object(pd.Series(ids), index=False).values
    return ((h + np.uint64(seed)) % np.uint64(k)).astype(np.int8)


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=96, verbose=-1,
              min_sum_hessian_in_leaf=1e-3)
import os as _os
PARAMS["num_threads"] = int(_os.environ.get("LGB_THREADS", "96"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("feats")
    ap.add_argument("gt")
    ap.add_argument("s1norm")
    ap.add_argument("out")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--min_leaf", type=int, default=100)
    ap.add_argument("--ff", type=float, default=0.7)
    ap.add_argument("--neg_keep", type=float, default=1.0, help="fraction of easy negatives kept in training")
    ap.add_argument("--hard_p0", type=float, default=0.01, help="negatives with p0 above this are always kept")
    args = ap.parse_args()
    t0 = time.time()
    cand = pd.read_parquet(args.cand, columns=["s1_id", "r_id"])
    X = pd.read_parquet(args.feats)
    for e in args.extra:
        X = pd.concat([X, pd.read_parquet(e)], axis=1)
    X = X.drop(columns=[c for c in args.drop if c in X.columns])
    gt = pd.read_parquet(args.gt)
    y = labels(cand, gt)
    s1 = pd.read_parquet(args.s1norm, columns=["entity_id"])
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    cnt = {a: (len(l.split(",")) if l else 0) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    truth_cnt = np.array([cnt[e] for e in s1.entity_id], dtype=float)
    ev = pd.DataFrame({"s1i": s1_pos.reindex(cand.s1_id).values, "rj": pd.factorize(cand.r_id)[0], "y": y})
    print(f"pairs {len(X)} feats {X.shape[1]} pos {y.sum()} ({y.mean():.4f})  load {time.time() - t0:.0f}s", flush=True)
    print(f"blocking ceiling macro-F0.5: {ceiling(ev, truth_cnt, len(s1)):.5f}", flush=True)
    fold = fold_of(cand.s1_id.values, args.folds)
    oof = np.zeros(len(X), dtype=np.float32)
    params = dict(PARAMS, learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
                  feature_fraction=args.ff)
    Xv = X.values.astype(np.float32)
    names = list(X.columns)
    imp = np.zeros(len(names))
    for k in range(args.folds):
        tr = np.where(fold != k)[0]
        va = np.where(fold == k)[0]
        w = None
        if args.neg_keep < 1.0:
            rng = np.random.default_rng(k)
            hard = X["p0"].values[tr] >= args.hard_p0 if "p0" in X.columns else np.zeros(len(tr), bool)
            keep = (y[tr] == 1) | hard | (rng.random(len(tr)) < args.neg_keep)
            w = np.where((y[tr] == 1) | hard, 1.0, 1.0 / args.neg_keep)[keep].astype(np.float32)
            tr = tr[keep]
        dtr = lgb.Dataset(Xv[tr], y[tr], weight=w, feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(Xv[va], y[va], reference=dtr, feature_name=names)
        t = time.time()
        bst = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        oof[va] = bst.predict(Xv[va], num_iteration=bst.best_iteration)
        imp += bst.feature_importance("gain")
        bst.save_model(f"{args.out}_fold{k}.txt", num_iteration=bst.best_iteration)
        print(f"fold {k}: best_iter {bst.best_iteration}  {time.time() - t:.0f}s", flush=True)
    np.save(f"{args.out}_oof.npy", oof)
    best, th, curve = evaluate(ev, oof.astype(float), truth_cnt, len(s1))
    print(f"OOF macro-F0.5 (argmax+threshold): {best:.5f} @ th={th}", flush=True)
    print("  curve:", {k: round(v, 5) for k, v in curve.items() if abs(k - th) <= 0.2})
    print(f"OOF macro-F0.5 (argmax+expected-F): {eval_expf(ev, oof.astype(float), truth_cnt, len(s1)):.5f}", flush=True)
    order = np.argsort(-imp)
    print("top features:", [(names[i], int(imp[i])) for i in order[:40]])


if __name__ == "__main__":
    main()

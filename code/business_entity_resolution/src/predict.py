"""Average the fold models of a stage over a feature matrix.

usage: python predict.py <model_prefix> <n_folds> <feats_parquet> <out_npy> [--extra f.parquet ...] [--drop cols ...]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("folds", type=int)
    ap.add_argument("feats")
    ap.add_argument("out")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--drop", nargs="*", default=[])
    args = ap.parse_args()
    X = pd.read_parquet(args.feats)
    for e in args.extra:
        X = pd.concat([X, pd.read_parquet(e)], axis=1)
    X = X.drop(columns=[c for c in args.drop if c in X.columns])
    p = np.zeros(len(X))
    for k in range(args.folds):
        bst = lgb.Booster(model_file=f"{args.prefix}_fold{k}.txt")
        names = bst.feature_name()
        assert set(names) <= set(X.columns), set(names) - set(X.columns)
        p += bst.predict(X[names].values.astype(np.float32), num_threads=int(os.environ.get("LGB_THREADS", "96")))
    p /= args.folds
    np.save(args.out, p.astype(np.float32))
    print(f"saved {args.out}: n={len(p)} mean={p.mean():.4f} >0.5: {(p > 0.5).mean():.4f}")


if __name__ == "__main__":
    main()

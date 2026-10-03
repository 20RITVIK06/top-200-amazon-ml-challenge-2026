"""Convert the challenge TSV files to parquet (strict tab parsing, no quoting, no NA coercion).

usage: python convert.py <dataset_dir> <out_dir>
"""
import csv
import os
import sys

import pandas as pd


def main():
    base, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            p = f"{base}/{split}/{split}_source{s}.tsv"
            df = pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False, dtype=str)
            df.to_parquet(f"{out}/{split}_s{s}.parquet", index=False)
            print(split, s, df.shape, flush=True)
    gt = pd.read_csv(f"{base}/train/train_ground_truth.tsv", sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False,
                     na_filter=False, dtype=str)
    gt.to_parquet(f"{out}/train_gt.parquet", index=False)
    print("gt", gt.shape)


if __name__ == "__main__":
    main()

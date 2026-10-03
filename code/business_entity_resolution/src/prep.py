"""Normalise all records of a split (train/test) in parallel and store as parquet.

usage: python prep.py <data_dir> <translit.json> <split> <out_dir> [n_jobs] [drop_locality 0/1] [country_token 0/1]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from textnorm import Normalizer

_N = None


def _init(path, drop_locality=True, country_token=True):
    global _N
    _N = Normalizer(path, drop_locality, country_token)


def _work(chunk):
    names, addrs, countries = chunk
    out = []
    for n, a, c in zip(names, addrs, countries):
        nf, nc, na, nw, nl, nflag = _N.name(n)
        at, an, ast, ac, aind = _N.address(a, c)
        out.append((nf, nc, na, nw, nl, nflag, at, an, ast, ac, aind))
    return out


def normalise_frame(df, translit, n_jobs, drop_locality=True, country_token=True):
    idx = np.array_split(np.arange(len(df)), max(1, n_jobs * 8))
    chunks = [(df.business_name.values[i].tolist(), df.business_address.values[i].tolist(), df.country.values[i].tolist()) for i in idx]
    with Pool(n_jobs, initializer=_init, initargs=(translit, drop_locality, country_token)) as pool:
        res = pool.map(_work, chunks, chunksize=1)
    rows = [r for part in res for r in part]
    cols = ["nf", "nc", "na", "nw", "nl", "nflag", "at", "an", "ast", "ac", "aind"]
    out = pd.DataFrame(rows, columns=cols)
    out.insert(0, "entity_id", df.entity_id.values)
    out.insert(1, "country", df.country.values)
    out["nflag"] = out.nflag.astype(np.int16)
    out["aind"] = out.aind.astype(bool)
    return out


def main():
    data_dir, translit, split, out_dir = sys.argv[1:5]
    n_jobs = int(sys.argv[5]) if len(sys.argv) > 5 else os.cpu_count()
    drop_locality = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
    country_token = bool(int(sys.argv[7])) if len(sys.argv) > 7 else True
    os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2, 3):
        t = time.time()
        df = pd.read_parquet(f"{data_dir}/{split}_s{s}.parquet")
        out = normalise_frame(df, translit, n_jobs, drop_locality, country_token)
        out["src"] = np.int8(s)
        out.to_parquet(f"{out_dir}/{split}_s{s}_norm.parquet", index=False)
        print(f"{split} s{s}: {len(out)} rows in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()

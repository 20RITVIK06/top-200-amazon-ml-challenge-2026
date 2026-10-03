"""Compute the pair feature matrix for a candidate table.

usage: python build_features.py <norm_dir> <cand_dir> <split> <out_dir> [alias_json]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import json

from features import STR_FEATS, idf_weighted, overlap_feats, records, set_alias, string_features


def context_features(c):
    """Features describing a pair relative to competing candidates (no labels used)."""
    f = {}
    for col in ["cos_comb", "cos_name", "cos_addr"]:
        g = c.groupby("j")[col]
        mx = g.transform("max").values
        f[f"{col}_rmax_gap"] = (mx - c[col].values).astype(np.float32)
        f[f"{col}_rrank"] = g.rank(ascending=False, method="min").values.astype(np.float32)
        # second best for this record (margin of the best one)
        srt = c[["j", col]].sort_values(["j", col], ascending=[True, False])
        second = srt.groupby("j")[col].nth(1)
        sec = pd.Series(second.values, index=srt.loc[second.index, "j"].values)
        f[f"{col}_r2nd"] = c.j.map(sec).fillna(0).values.astype(np.float32)
        g1 = c.groupby("i1")[col]
        f[f"{col}_s1max_gap"] = (g1.transform("max").values - c[col].values).astype(np.float32)
        f[f"{col}_s1rank"] = g1.rank(ascending=False, method="min").values.astype(np.float32)
    f["r_ncand"] = c.groupby("j").j.transform("size").values.astype(np.float32)
    f["s1_ncand"] = c.groupby("i1").i1.transform("size").values.astype(np.float32)
    return f


def main():
    norm_dir, cand_dir, split, out_dir = sys.argv[1:5]
    alias = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet")
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s2_norm.parquet"),
                     pd.read_parquet(f"{norm_dir}/{split}_s3_norm.parquet")], ignore_index=True)
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet")
    cand = cand.sort_values(["j", "i1"]).reset_index(drop=True)
    cand.to_parquet(f"{cand_dir}/{split}_cand.parquet", index=False)  # canonical order shared with features
    print(f"loaded {len(cand)} pairs {time.time() - t0:.0f}s", flush=True)
    feats = {}
    # --- name / address frequency statistics (per country, unsupervised)
    s1_nc_cnt = s1.groupby(["country", "nc"]).nc.transform("size").values.astype(np.float32)
    all_nc = pd.concat([s1[["country", "nc"]], s23[["country", "nc"]]])
    nc_cnt_all = all_nc.groupby(["country", "nc"]).size()
    key23 = pd.MultiIndex.from_arrays([s23.country.values, s23.nc.values])
    s23_nc_in_s1 = pd.Series(s1.groupby(["country", "nc"]).size()).reindex(key23).fillna(0).values.astype(np.float32)
    s23_nc_all = nc_cnt_all.reindex(key23).fillna(0).values.astype(np.float32)
    I, J = cand.i1.values, cand.j.values
    feats["s1_name_freq"] = s1_nc_cnt[I]
    feats["r_name_freq_s1"] = s23_nc_in_s1[J]
    feats["r_name_freq_all"] = s23_nc_all[J]
    # co-located businesses: S1 records sharing the exact canonical address
    s1_at_cnt = s1.groupby(["country", "at"]).at.transform("size").values.astype(np.float32)
    feats["s1_addr_freq"] = s1_at_cnt[I]
    at_cnt = s1.groupby(["country", "at"]).size()
    feats["r_addr_freq_s1"] = at_cnt.reindex(pd.MultiIndex.from_arrays([s23.country.values, s23["at"].values])).fillna(0).values.astype(np.float32)[J]
    # pseudo / generated names: fraction of record core tokens never seen in any S1 name of the country
    oov = np.zeros(len(s23), np.float32)
    for country in set(s1.country):
        vocab = set(t for x in s1.nc.values[s1.country.values == country] for t in x.split())
        idx = np.where(s23.country.values == country)[0]
        oov[idx] = [(sum(t not in vocab for t in x.split()) / max(1, len(x.split()))) for x in s23.nc.values[idx]]
    feats["r_name_oov"] = oov[J]
    # --- sparse IDF overlaps per country
    ov = {}
    for country in sorted(set(s1.country)):
        m = (s1.country.values[I] == country)
        if not m.any():
            continue
        i1_idx = np.where(s1.country.values == country)[0]
        j_idx = np.where(s23.country.values == country)[0]
        pos1 = np.full(len(s1), -1, np.int64)
        pos1[i1_idx] = np.arange(len(i1_idx))
        pos23 = np.full(len(s23), -1, np.int64)
        pos23[j_idx] = np.arange(len(j_idx))
        for field, pre in (("nc", "on"), ("at", "oa")):
            texts = np.concatenate([s1[field].values[i1_idx], s23[field].values[j_idx]])
            cv, idf, X = idf_weighted(texts)
            X = X.tocsr()
            B1, B23 = X[:len(i1_idx)], X[len(i1_idx):]
            o = overlap_feats(B1, B23, idf, pos1[I[m]], pos23[J[m]], pre)
            for k, v in o.items():
                if k not in ov:
                    ov[k] = np.full(len(cand), np.nan, np.float32)
                ov[k][m] = v
        print(f"  overlaps {country} {time.time() - t0:.0f}s", flush=True)
    feats.update(ov)
    # --- string features
    R1, R23 = records(s1), records(s23)
    table = None
    if alias is not None:
        # _ALIAS[0] is empty; [1]=fold0 table, [2]=fold1 table, [3]=all-train table
        set_alias([alias["fold0"], alias["fold1"], alias["all"]])
        if split == "train":
            f = (pd.util.hash_pandas_object(cand.s1_id, index=False).values % 2).astype(np.int8)
            table = np.where(f == 0, 2, 1).astype(np.int8)  # use the table learned on the OTHER fold
        else:
            table = np.full(len(cand), 3, np.int8)
    SF = string_features(R1, R23, I, J, table)
    for k, name in enumerate(STR_FEATS):
        feats[name] = SF[:, k]
    print(f"  string feats {time.time() - t0:.0f}s", flush=True)
    feats.update(context_features(cand))
    print(f"  context feats {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(feats)
    for c in ["rk_name", "rk_addr", "rk_comb", "rk_rev", "cos_name", "cos_addr", "cos_comb"]:
        out[c] = cand[c].values.astype(np.float32)
    out["src3"] = cand.r_id.str.startswith("S3-").values.astype(np.float32)
    out["p0"] = cand.p0.values.astype(np.float32)
    out.to_parquet(f"{out_dir}/{split}_feats.parquet", index=False)
    print(f"saved {out.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

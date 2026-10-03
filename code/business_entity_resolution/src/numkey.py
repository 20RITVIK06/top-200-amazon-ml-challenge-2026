"""Supplementary blocking channel: (house number, locality token) keys ranked by name similarity.

Many true pairs are missed by TF-IDF blocking when the record's address is truncated to
"house number, city, state" (common in India): each token alone is too frequent to retrieve the
right S1 entity, and generic names are shared by many S1 entities.  Here every record contributes
keys "<first number>|<alphabetic address token>", every S1 entity "<number>|<token>" for up to
three numbers; S1 entities sharing a key (keys shared by <= max_freq S1 entities) are ranked by
name TF-IDF cosine and the top-k with cosine >= min_cos are added to the candidate set
(channel rank features are set to "not retrieved", p0 to missing).

usage: python numkey.py <norm_block_dir> <cand_dir> <split> <out_cand_dir> [k] [min_cos] [max_freq]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from blocking import comb_matrix, fit_vectors, CFG
from features import rowdot


def keys(at_list, an_list, nc_list, idx, first_only):
    """keys = number | locality token | first core-name token (rare even in big cities)."""
    ks, ids = [], []
    for i, a, n, nc in zip(idx, at_list, an_list, nc_list):
        nums = n.split()
        w = nc.split()
        if not nums or not w:
            continue
        nums = nums[:1] if first_only else nums[:3]
        al = {t for t in a.split() if not t.isdigit() and not t.startswith("s_") and len(t) > 2}
        for x in nums:
            for t in al:
                ks.append(x + "|" + t + "|" + w[0])
                ids.append(i)
    return pd.DataFrame({"key": ks, "id": np.array(ids, dtype=np.int64)})


def main():
    norm_dir, cand_dir, split, out_dir = sys.argv[1:5]
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    min_cos = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    max_freq = int(sys.argv[7]) if len(sys.argv) > 7 else 50
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    cols = ["entity_id", "country", "nc", "at", "an"]
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s{q}_norm.parquet", columns=cols) for q in (2, 3)], ignore_index=True)
    old = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet")
    new_parts = []
    for country in sorted(set(s1.country)):
        a1 = np.where(s1.country.values == country)[0]
        a23 = np.where(s23.country.values == country)[0]
        if len(a1) == 0 or len(a23) == 0:
            continue
        K1 = keys(s1["at"].values[a1], s1["an"].values[a1], s1["nc"].values[a1], a1, False)
        f = K1.groupby("key").id.transform("size")
        K1 = K1[f.values <= max_freq]
        K2 = keys(s23["at"].values[a23], s23["an"].values[a23], s23["nc"].values[a23], a23, True)
        P = K2.merge(K1, on="key", suffixes=("_j", "_i")).rename(columns={"id_j": "j", "id_i": "i1"})[["j", "i1"]].drop_duplicates()
        # name / address / comb cosines in this country's TF-IDF space (same as blocking)
        N1, N23, A1, A23 = fit_vectors(s1.iloc[a1].reset_index(drop=True), s23.iloc[a23].reset_index(drop=True))
        p1 = pd.Series(np.arange(len(a1)), index=a1)
        p23 = pd.Series(np.arange(len(a23)), index=a23)
        li, lj = p1.reindex(P.i1.values).values, p23.reindex(P.j.values).values
        P["cos_name"] = rowdot(N1, N23, li, lj)
        P = P[P.cos_name.values >= min_cos]
        P = P.sort_values(["j", "cos_name"], ascending=[True, False])
        P = P[P.groupby("j").cumcount().values < k]
        li, lj = p1.reindex(P.i1.values).values, p23.reindex(P.j.values).values
        P["cos_addr"] = rowdot(A1, A23, li, lj)
        C1, C23 = comb_matrix(N1, A1, CFG["w_name"]), comb_matrix(N23, A23, CFG["w_name"])
        P["cos_comb"] = rowdot(C1, C23, li, lj)
        new_parts.append(P)
        print(f"  {split} {country}: key pairs kept {len(P)}  {time.time() - t0:.0f}s", flush=True)
    new = pd.concat(new_parts, ignore_index=True)
    key_old = pd.Series(1, index=pd.MultiIndex.from_arrays([old.j.values, old.i1.values]))
    is_old = key_old.reindex(pd.MultiIndex.from_arrays([new.j.values, new.i1.values])).notna().values
    new = new[~is_old].reset_index(drop=True)
    for c in ("rk_comb", "rk_name", "rk_addr", "rk_rev"):
        new[c] = np.int16(99)
    new["p0"] = np.float32(np.nan)
    new["s1_id"] = s1.entity_id.values[new.i1.values]
    new["r_id"] = s23.entity_id.values[new.j.values]
    union = pd.concat([old, new[old.columns]], ignore_index=True).sort_values(["j", "i1"]).reset_index(drop=True)
    union.to_parquet(f"{out_dir}/{split}_cand.parquet", index=False)
    print(f"{split}: added {len(new)} new pairs -> {len(union)} total  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

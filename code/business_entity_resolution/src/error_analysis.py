"""OOF error analysis after post-processing.

usage: python error_analysis.py <data_dir> <norm_dir> <cand_dir> <oof.npy> <threshold> [n_examples]
"""
import sys

import numpy as np
import pandas as pd

from postproc import argmax_assign


def main():
    data_dir, norm_dir, cand_dir, oof_path, th = sys.argv[1:6]
    th = float(th)
    nex = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    cand = pd.read_parquet(f"{cand_dir}/train_cand.parquet", columns=["s1_id", "r_id", "i1", "j"])
    p = np.load(oof_path).astype(float)
    gt = pd.read_parquet(f"{data_dir}/train_gt.parquet")
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    y = (cand.r_id.map(truth).values == cand.s1_id.values)
    m = argmax_assign(cand.i1.values, cand.j.values, p) & (p >= th)
    raw = pd.concat([pd.read_parquet(f"{data_dir}/train_s{k}.parquet") for k in (1, 2, 3)]).set_index("entity_id")
    nrm = pd.concat([pd.read_parquet(f"{norm_dir}/train_s{k}_norm.parquet", columns=["entity_id", "nc", "at", "country"]) for k in (1, 2, 3)]).set_index("entity_id")
    s1n = nrm[nrm.index.str.startswith("S1-")]
    nc_cnt = s1n.groupby(["country", "nc"]).size()
    # false positives: predicted pairs that are wrong
    fp = cand[m & ~y].copy()
    fp["p"] = p[m & ~y]
    # false negatives: true pairs not predicted (in candidates or not)
    pred_pairs = set(zip(cand.s1_id.values[m], cand.r_id.values[m]))
    fn_ids = [(a, b) for b, a in truth.items() if (a, b) not in pred_pairs]
    in_cand = set(zip(cand.s1_id.values, cand.r_id.values))
    fn = pd.DataFrame(fn_ids, columns=["s1_id", "r_id"])
    fn["in_cand"] = [(a, b) in in_cand for a, b in fn_ids]
    for name, d in (("FP", fp), ("FN", fn)):
        d["country"] = nrm.country.reindex(d.s1_id).values
        d["src"] = d.r_id.str[:2]
        d["r_addr_empty"] = (nrm["at"].reindex(d.r_id).fillna("").str.len() == 0).values
        rk = pd.MultiIndex.from_arrays([d.country.values, nrm.nc.reindex(d.r_id).fillna("").values])
        d["r_name_s1freq"] = nc_cnt.reindex(rk).fillna(0).values
        print(f"===== {name}: {len(d)}")
        print(d.groupby(["country", "src"]).size().to_string())
        print(f"  record address empty: {d.r_addr_empty.mean():.3f}")
        print(f"  record core name shared by >=2 S1: {(d.r_name_s1freq >= 2).mean():.3f}   unseen in S1: {(d.r_name_s1freq == 0).mean():.3f}")
        if name == "FN":
            print(f"  FN inside candidate set: {d.in_cand.mean():.3f}")
    print(f"\ntotal predicted {m.sum()}  TP {(m & y).sum()}  FP {len(fp)}  FN {len(fn)}  (true pairs {len(truth)})")
    rng = np.random.default_rng(0)
    print("\n##### FP examples (predicted S1 | record | true S1 of record)")
    for i in rng.choice(len(fp), min(nex, len(fp)), replace=False):
        r = fp.iloc[i]
        a, b = raw.loc[r.s1_id], raw.loc[r.r_id]
        t = truth.get(r.r_id)
        tt = raw.loc[t] if t else None
        print(f"p={r.p:.3f}  S1: {a.business_name} | {a.business_address}\n          R : {b.business_name} | {b.business_address}")
        print(f"          T : {(tt.business_name + ' | ' + tt.business_address) if t else '(unmatched record)'}")
    print("\n##### FN examples inside candidates (non-trivial: record has an address)")
    sub = fn[fn.in_cand & ~fn.r_addr_empty]
    pmap = dict(zip(zip(cand.s1_id.values, cand.r_id.values), p))
    for i in rng.choice(len(sub), min(nex, len(sub)), replace=False):
        r = sub.iloc[i]
        a, b = raw.loc[r.s1_id], raw.loc[r.r_id]
        print(f"p={pmap[(r.s1_id, r.r_id)]:.3f}  S1: {a.business_name} | {a.business_address}\n          R : {b.business_name} | {b.business_address}")


if __name__ == "__main__":
    main()

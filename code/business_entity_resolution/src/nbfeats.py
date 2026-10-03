"""Neighbourhood-consensus features for candidate pairs (label-free).

The generator's clone entities have their *own* several records, which agree with each other
on the clone's house number / name; a true record carrying number or name noise is a lone
outlier whose sibling records agree with the Source-1 entity instead.  For a pair (S1 a,
record r) we therefore look at r's nearest records (recknn.py) and measure whether they agree
with r or with a on the first house number and on the core name.

usage: python nbfeats.py <norm_dir> <cand_dir> <split> <out_parquet>
"""
import sys
import time

import numpy as np
import pandas as pd


def first_num(s):
    return s.str.split(" ", n=1).str[0].fillna("")


def main():
    norm_dir, cand_dir, split, out = sys.argv[1:5]
    t0 = time.time()
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet", columns=["nc", "an"])
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s{k}_norm.parquet", columns=["nc", "an"]) for k in (2, 3)],
                    ignore_index=True)
    n1 = len(s1)
    num_codes, _ = pd.factorize(pd.concat([first_num(s1["an"]), first_num(s23["an"])], ignore_index=True))
    core_codes, _ = pd.factorize(pd.concat([s1.nc, s23.nc], ignore_index=True))
    empty_num = pd.concat([first_num(s1["an"]), first_num(s23["an"])], ignore_index=True).values == ""
    num_codes = np.where(empty_num, -1, num_codes)
    num1, num23 = num_codes[:n1], num_codes[n1:]
    core1, core23 = core_codes[:n1], core_codes[n1:]
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet", columns=["i1", "j"])
    nn = pd.read_parquet(f"{cand_dir}/{split}_recnn.parquet")
    pairs = pd.DataFrame({"row": np.arange(len(cand), dtype=np.int64), "j": cand.j.values, "i1": cand.i1.values})
    m = pairs.merge(nn, on="j", how="inner")
    print(f"{split}: pairs {len(cand)} joined neighbour rows {len(m)}  {time.time() - t0:.0f}s", flush=True)
    nbn, nbc = num23[m.nb.values], core23[m.nb.values]
    rn, an_ = num23[m.j.values], num1[m.i1.values]
    rc, ac = core23[m.j.values], core1[m.i1.values]
    hi = m.sim.values >= 0.8
    has = nbn >= 0
    df = pd.DataFrame({
        "row": m.row.values,
        "w": m.sim.values,
        "hi": hi.astype(np.float32),
        "has": has.astype(np.float32),
        "n_eq_r": ((nbn == rn) & (rn >= 0) & has).astype(np.float32),
        "n_eq_a": ((nbn == an_) & (an_ >= 0) & has).astype(np.float32),
        "c_eq_r": (nbc == rc).astype(np.float32),
        "c_eq_a": (nbc == ac).astype(np.float32),
    })
    for c in ("n_eq_r", "n_eq_a", "c_eq_r", "c_eq_a"):
        df[c + "_hi"] = df[c] * df["hi"]
    g = df.groupby("row")
    agg = g.agg(nb_n=("w", "size"), nb_hi=("hi", "sum"), nb_has=("has", "sum"), nb_wsum=("w", "sum"),
                n_eq_r=("n_eq_r", "sum"), n_eq_a=("n_eq_a", "sum"), c_eq_r=("c_eq_r", "sum"), c_eq_a=("c_eq_a", "sum"),
                n_eq_r_hi=("n_eq_r_hi", "sum"), n_eq_a_hi=("n_eq_a_hi", "sum"),
                c_eq_r_hi=("c_eq_r_hi", "sum"), c_eq_a_hi=("c_eq_a_hi", "sum"))
    F = pd.DataFrame(index=np.arange(len(cand)))
    F = F.join(agg).fillna(0).astype(np.float32)
    eps = 1e-6
    out_df = pd.DataFrame({
        "nb_n": F.nb_n.values,
        "nb_hi": F.nb_hi.values,
        "nb_num_eq_r": F.n_eq_r.values / (F.nb_has.values + eps),
        "nb_num_eq_a": F.n_eq_a.values / (F.nb_has.values + eps),
        "nb_num_dir": (F.n_eq_a.values - F.n_eq_r.values) / (F.nb_has.values + eps),
        "nb_core_eq_r": F.c_eq_r.values / (F.nb_n.values + eps),
        "nb_core_eq_a": F.c_eq_a.values / (F.nb_n.values + eps),
        "nb_core_dir": (F.c_eq_a.values - F.c_eq_r.values) / (F.nb_n.values + eps),
        "nb_hi_num_eq_r": F.n_eq_r_hi.values,
        "nb_hi_num_eq_a": F.n_eq_a_hi.values,
        "nb_hi_core_eq_r": F.c_eq_r_hi.values,
        "nb_hi_core_eq_a": F.c_eq_a_hi.values,
        "r_num_eq_a": ((num23[cand.j.values] == num1[cand.i1.values]) & (num1[cand.i1.values] >= 0)).astype(np.float32),
    }).astype(np.float32)
    out_df.to_parquet(out, index=False)
    print(f"{split}: saved {out_df.shape}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

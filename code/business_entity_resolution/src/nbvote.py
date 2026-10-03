"""Neighbour-vote features (stage-2, from stage-1 probabilities).

For a pair (S1 a, record r): r's nearest records r' (recknn.py) each "vote" for their own best
Source-1 entity under stage-1 (arg-max probability).  Votes are weighted by record similarity
and by the neighbour's confidence.  vote_a = weighted share of neighbours whose best S1 is a;
vote_other = weighted share voting for a different S1.  Near-duplicate records of one real
entity should agree, so this propagates confident decisions to ambiguous records.

usage: python nbvote.py <cand_dir> <split> <p1.npy> <out_parquet>
"""
import sys
import time

import numpy as np
import pandas as pd


def main():
    cand_dir, split, p1_path, out = sys.argv[1:5]
    t0 = time.time()
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet", columns=["i1", "j"])
    p = np.load(p1_path).astype(np.float32)
    # best S1 per record under stage 1
    order = np.lexsort((-p, cand.j.values))
    js = cand.j.values[order]
    first = np.ones(len(order), bool)
    first[1:] = js[1:] != js[:-1]
    best = pd.DataFrame({"nb": js[first], "nb_best_i1": cand.i1.values[order][first], "nb_best_p": p[order][first]})
    nn = pd.read_parquet(f"{cand_dir}/{split}_recnn.parquet")
    nn = nn.merge(best, on="nb", how="left")
    nn["nb_best_i1"] = nn.nb_best_i1.fillna(-1).astype(np.int64)
    nn["nb_best_p"] = nn.nb_best_p.fillna(0).astype(np.float32)
    pairs = pd.DataFrame({"row": np.arange(len(cand), dtype=np.int64), "j": cand.j.values, "i1": cand.i1.values})
    m = pairs.merge(nn, on="j", how="inner")
    w = m.sim.values
    same = (m.nb_best_i1.values == m.i1.values)
    has = m.nb_best_i1.values >= 0
    df = pd.DataFrame({"row": m.row.values, "w": w,
                       "va": w * m.nb_best_p.values * same,
                       "vo": w * m.nb_best_p.values * (has & ~same),
                       "ca": same.astype(np.float32), "co": (has & ~same).astype(np.float32),
                       "pa_max": np.where(same, m.nb_best_p.values, 0).astype(np.float32),
                       "po_max": np.where(has & ~same, m.nb_best_p.values, 0).astype(np.float32)})
    g = df.groupby("row").agg(w=("w", "sum"), va=("va", "sum"), vo=("vo", "sum"), ca=("ca", "sum"), co=("co", "sum"),
                              pa_max=("pa_max", "max"), po_max=("po_max", "max"))
    F = pd.DataFrame(index=np.arange(len(cand))).join(g).fillna(0)
    eps = 1e-6
    out_df = pd.DataFrame({
        "nv_vote_a": F.va.values / (F.w.values + eps),
        "nv_vote_other": F.vo.values / (F.w.values + eps),
        "nv_cnt_a": F.ca.values, "nv_cnt_other": F.co.values,
        "nv_pmax_a": F.pa_max.values, "nv_pmax_other": F.po_max.values,
        "nv_margin": (F.va.values - F.vo.values) / (F.w.values + eps),
    }).astype(np.float32)
    out_df.to_parquet(out, index=False)
    print(f"{split}: neighbour-vote feats {out_df.shape}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

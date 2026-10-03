"""Stage-2 collective features built from stage-1 pair probabilities.

For a pair (S1 entity a, record r) with stage-1 probability p:
  competition   : best / second-best probability of r with other S1 entities, rank of a
  entity context: how many records are already strongly linked to a (by source), mass
  siblings      : similarity (TF-IDF cosines over name / address / combined) between r and
                  the records strongly linked to a, and to the records strongly linked to
                  r's best competing S1 entity.  Captures transitivity: records of one real
                  entity look alike, so r resembling a's confirmed records supports (a, r).
All inputs are unsupervised or out-of-fold predictions; no labels are used.
"""
import numpy as np
import pandas as pd

from features import rowdot


def _grp_max_excl(key, val):
    """For each row: max of val over other rows with the same key (0 if none)."""
    df = pd.DataFrame({"k": key, "v": val, "i": np.arange(len(key))})
    df = df.sort_values(["k", "v"], ascending=[True, False])
    first = ~df.k.duplicated()
    top = df[first].set_index("k").v
    sec = df[~first].groupby("k").v.first()  # rows are sorted desc within key: first non-top = 2nd max
    t = top.reindex(key).values
    s2 = sec.reindex(key).fillna(0).values
    is_top = np.zeros(len(key), bool)
    is_top[df[first].i.values] = True
    return np.where(is_top, s2, t)


def collective_features(c, p, V23, strong=0.5):
    """c: candidate table with i1, j, src3 (0/1); p: stage-1 probs; V23: dict of record matrices."""
    i1, j = c.i1.values, c.j.values
    f = {"p1": p.astype(np.float32)}
    # ---- competition on the record side
    f["p1_r_other_max"] = _grp_max_excl(j, p).astype(np.float32)
    f["p1_r_margin"] = (p - f["p1_r_other_max"]).astype(np.float32)
    f["p1_r_rank"] = pd.Series(p).groupby(j).rank(ascending=False, method="min").values.astype(np.float32)
    f["p1_r_sum"] = pd.Series(p).groupby(j).transform("sum").values.astype(np.float32)
    # ---- entity side
    ps = pd.Series(p)
    f["p1_a_sum_excl"] = (ps.groupby(i1).transform("sum").values - p).astype(np.float32)
    strong_m = (p >= strong).astype(np.float32)
    f["p1_a_nstrong_excl"] = (pd.Series(strong_m).groupby(i1).transform("sum").values - strong_m).astype(np.float32)
    src3 = c.src3.values.astype(np.float32)
    s3s = strong_m * src3
    s2s = strong_m * (1 - src3)
    f["p1_a_nstrong_s3_excl"] = (pd.Series(s3s).groupby(i1).transform("sum").values - s3s).astype(np.float32)
    f["p1_a_nstrong_s2_excl"] = (pd.Series(s2s).groupby(i1).transform("sum").values - s2s).astype(np.float32)
    f["p1_a_max_excl"] = _grp_max_excl(i1, p).astype(np.float32)
    f["p1_a_rank"] = ps.groupby(i1).rank(ascending=False, method="min").values.astype(np.float32)
    # ---- siblings: pairs (r, r') where r' is strongly linked to the same a
    idx = np.arange(len(c))
    sm = p >= strong
    S = pd.DataFrame({"i1": i1[sm], "rs": j[sm], "ps": p[sm]})
    A = pd.DataFrame({"i1": i1, "r": j, "row": idx})
    pairs = A.merge(S, on="i1")
    pairs = pairs[pairs.r.values != pairs.rs.values]
    for name, M in V23.items():
        sim = rowdot(M, M, pairs.r.values, pairs.rs.values)
        g = pd.DataFrame({"row": pairs.row.values, "s": sim, "sw": sim * pairs.ps.values})
        agg = g.groupby("row").agg(smax=("s", "max"), smean=("s", "mean"), swmax=("sw", "max"))
        for k in agg.columns:
            v = np.zeros(len(c), np.float32)
            v[agg.index.values] = agg[k].values
            f[f"sib_{name}_{k}"] = v
    # sibling support of the best competitor: r-level max over other candidates
    for name in V23:
        key = f"sib_{name}_smax"
        f[f"sib_{name}_rival_max"] = _grp_max_excl(j, f[key]).astype(np.float32)
    f["sib_n"] = np.bincount(pairs.row.values, minlength=len(c)).astype(np.float32)
    return pd.DataFrame(f)


def record_vectors(norm_dir, split, w_name=0.6):
    """Global (row-aligned with S2/S3 table) TF-IDF matrices: name, addr, comb.

    Countries are fitted separately (as in blocking) and placed in disjoint column blocks.
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize

    from blocking import comb_matrix, fit_vectors

    cols = ["country", "nc", "at"]
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s{k}_norm.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    blocks = {"name": [], "addr": [], "comb": []}
    order = []
    for country in sorted(set(s23.country)):
        a1 = np.where(s1.country.values == country)[0]
        a23 = np.where(s23.country.values == country)[0]
        if len(a1) == 0:  # records of a country absent from S1 can never match: empty rows
            for k in blocks:
                blocks[k].append(sp.csr_matrix((len(a23), 1), dtype=np.float32))
            order.append(a23)
            continue
        N1, N23, A1, A23 = fit_vectors(s1.iloc[a1], s23.iloc[a23])
        blocks["name"].append(N23)
        blocks["addr"].append(A23)
        blocks["comb"].append(comb_matrix(N23, A23, w_name))
        order.append(a23)
    order = np.concatenate(order)
    inv = np.empty(len(s23), np.int64)
    inv[order] = np.arange(len(order))
    out = {}
    for k, bl in blocks.items():
        M = sp.block_diag(bl, format="csr")
        out[k] = normalize(M[inv]).tocsr() if k != "comb" else M[inv].tocsr()
    return out

"""Pair-level raw-string features (before normalisation).

True-match records are produced from *this* Source-1 record by the noise pipeline, whereas
clone/distractor records are produced from another base record; raw formatting relations
(case-insensitive exact equality, raw similarity, punctuation signature, component order)
carry part of that provenance.  Label-free and pair-local (independent of the distractor mix).

usage: python rawpair.py <data_dir> <cand_dir> <split> <out_parquet>
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

_G = {}
PUNCT = set(",.&'()-/#[]|@*+")


def _sig(s):
    return "".join(sorted(c for c in s if c in PUNCT))


def _order(a, b):
    """share of b's comma components found in a, and whether their order agrees."""
    ca = [x.strip() for x in a.split(",") if x.strip()]
    cb = [x.strip() for x in b.split(",") if x.strip()]
    if not ca or not cb:
        return -1.0, -1.0
    pos = [ca.index(x) for x in cb if x in ca]
    share = len(pos) / len(cb)
    if len(pos) < 2:
        return share, -1.0
    inc = sum(1 for u, v in zip(pos, pos[1:]) if v > u) / (len(pos) - 1)
    return share, inc


import re as _re
_TOK = _re.compile(r"[A-Za-zÀ-ÿ0-9]+")


def _acr(n1, n2):
    """record name is an acronym of the S1 name: exact initials (1.0), prefix of initials (0.5), else 0."""
    r = "".join(_TOK.findall(n2))
    if not (2 <= len(r) <= 6) or not r.isalpha() or len(_TOK.findall(n2)) != 1:
        return 0.0, 0.0
    toks = _TOK.findall(n1)
    ini_all = "".join(t[0] for t in toks).lower()
    ini_cap = "".join(t[0] for t in toks if t[0].isupper()).lower()
    rl = r.lower()
    if rl == ini_all or rl == ini_cap:
        return 1.0, 1.0
    if ini_all.startswith(rl) or ini_cap.startswith(rl):
        return 0.5, 1.0
    return 0.0, 1.0


def _work(args):
    I, J = args
    n1, a1, n2, a2 = _G["n1"], _G["a1"], _G["n2"], _G["a2"]
    out = np.empty((len(I), 12), np.float32)
    for k, (i, j) in enumerate(zip(I, J)):
        x, y = n1[i], n2[j]
        xa, ya = a1[i], a2[j]
        xl, yl = x.lower(), y.lower()
        xal, yal = xa.lower(), ya.lower()
        sh, inc = _order(xal, yal)
        out[k] = (float(" ".join(xl.split()) == " ".join(yl.split())),
                  float(" ".join(xal.split()) == " ".join(yal.split())) if ya else -1.0,
                  fuzz.ratio(xl, yl), fuzz.ratio(xal, yal) if ya else -1.0,
                  float(x == y),
                  float(_sig(x) == _sig(y)),
                  float(len(y) - len(x)),
                  float(len(ya) - len(xa)) if ya else 0.0,
                  sh, inc) + _acr(x, y)
    return out


NAMES = ["rp_name_eq_ci", "rp_addr_eq_ci", "rp_name_ratio", "rp_addr_ratio", "rp_name_eq_cs", "rp_punct_eq",
         "rp_name_dlen", "rp_addr_dlen", "rp_comp_share", "rp_comp_order", "rp_acronym", "rp_short_name"]


def main():
    data_dir, cand_dir, split, out = sys.argv[1:5]
    t0 = time.time()
    s1 = pd.read_parquet(f"{data_dir}/{split}_s1.parquet", columns=["business_name", "business_address"])
    s23 = pd.concat([pd.read_parquet(f"{data_dir}/{split}_s{k}.parquet", columns=["business_name", "business_address"])
                     for k in (2, 3)], ignore_index=True)
    _G.update(n1=s1.business_name.tolist(), a1=s1.business_address.tolist(),
              n2=s23.business_name.tolist(), a2=s23.business_address.tolist())
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet", columns=["i1", "j"])
    I, J = cand.i1.values, cand.j.values
    chunks = np.array_split(np.arange(len(I)), os.cpu_count() * 4)
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        parts = pool.map(_work, [(I[c], J[c]) for c in chunks])
    F = pd.DataFrame(np.vstack(parts), columns=NAMES)
    F.to_parquet(out, index=False)
    print(f"{split}: raw pair feats {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Token-level raw provenance features for candidate pairs (label-free, pair-local).

Complements rawpair.py: agreement of the raw token order and of per-token casing between the
S1 name and the record name, raw first-token and raw legal-suffix equality, and equality of the
raw first / last address component and of the component counts.

usage: python rawtok.py <data_dir> <cand_dir> <split> <out_parquet>
"""
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

_G = {}
TOK = re.compile(r"[^\W_]+")
LEGAL = {"inc", "inc.", "llc", "l.l.c.", "ltd", "ltd.", "limited", "corp", "corp.", "corporation", "co", "co.",
         "pvt", "private", "lp", "llp", "pllc", "pc", "p.c.", "sarl", "sas", "sasu", "eurl", "sci", "sa", "ei"}


def _order(ta, tb):
    la = [t.lower() for t in ta]
    lb = [t.lower() for t in tb]
    pos = [la.index(t) for t in lb if t in la]
    if len(pos) < 2:
        return -1.0
    return sum(1 for u, v in zip(pos, pos[1:]) if v > u) / (len(pos) - 1)


def _case(ta, tb):
    da = {t.lower(): t for t in ta}
    sh = [t for t in tb if t.lower() in da]
    if not sh:
        return -1.0
    return sum(1 for t in sh if da[t.lower()] == t) / len(sh)


def _legal(s):
    w = s.split()
    return w[-1].lower().strip("()[]") if w and w[-1].lower().strip("()[]") in LEGAL else ""


def _work(args):
    I, J = args
    n1, a1, n2, a2 = _G["n1"], _G["a1"], _G["n2"], _G["a2"]
    out = np.empty((len(I), 8), np.float32)
    for k, (i, j) in enumerate(zip(I, J)):
        x, y = n1[i], n2[j]
        ta, tb = TOK.findall(x), TOK.findall(y)
        ca = [c.strip().lower() for c in a1[i].split(",") if c.strip()]
        cb = [c.strip().lower() for c in a2[j].split(",") if c.strip()]
        la, lb = _legal(x), _legal(y)
        out[k] = (_order(ta, tb), _case(ta, tb),
                  float(bool(ta) and bool(tb) and ta[0] == tb[0]),
                  -1.0 if not (la and lb) else float(la == lb),
                  float(len(tb) - len(ta)),
                  -1.0 if not cb else float(bool(ca) and ca[0] == cb[0]),
                  -1.0 if not cb else float(bool(ca) and ca[-1] == cb[-1]),
                  float(len(cb) - len(ca)) if cb else 0.0)
    return out


NAMES = ["rt_tok_order", "rt_case_same", "rt_first_tok_eq", "rt_legal_raw_eq", "rt_ntok_diff",
         "rt_addr_first_eq", "rt_addr_last_eq", "rt_ncomp_diff"]


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
    print(f"{split}: raw token feats {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

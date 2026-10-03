"""Targeted features for the generator's planted hard negatives.

1. Token log-odds (learned, fold-aware).  For a pair, 'extra' tokens are core-name tokens of
   the record absent from the S1 core name; 'missing' tokens the reverse.  Generator add-ons
   such as 'services' / 'center' are harmless, whereas branch words such as 'westgate' /
   'southside' mark a *different* (clone) entity.  For every token we learn the log-odds of a
   true match when it appears as extra / missing, from labelled training candidates.  Train
   pairs only use statistics learned on the other half of S1 entities (same hash split as
   the alias tables), test pairs use all training pairs.
2. House-number relation flags between the S1 number (first and longest) and the record's
   numbers: equal, record-truncated (drop last digit), S1-truncated, other prefix, small
   offset (<=10), single-digit substitution, same length different.

usage: python extra_feats.py <norm_dir> <cand_dir> <data_dir> <out_dir>
"""
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

_G = {}


def _tok_diff(a, b):
    ta, tb = set(a.split()), set(b.split())
    return tb - ta, ta - tb  # extra (in record only), missing (in S1 only)


def _count_worker(args):
    I, J, Y = args
    nc1, nc23 = _G["nc1"], _G["nc23"]
    ce_p, ce_n, cm_p, cm_n = Counter(), Counter(), Counter(), Counter()
    for i, j, y in zip(I, J, Y):
        e, m = _tok_diff(nc1[i], nc23[j])
        if y:
            ce_p.update(e)
            cm_p.update(m)
        else:
            ce_n.update(e)
            cm_n.update(m)
    return ce_p, ce_n, cm_p, cm_n


def learn_logodds(nc1, nc23, I, J, Y, n_jobs=96):
    _G["nc1"], _G["nc23"] = nc1, nc23
    chunks = np.array_split(np.arange(len(I)), n_jobs * 4)
    with Pool(n_jobs) as pool:
        parts = pool.map(_count_worker, [(I[c], J[c], Y[c]) for c in chunks])
    ce_p, ce_n, cm_p, cm_n = Counter(), Counter(), Counter(), Counter()
    for a, b, c, d in parts:
        ce_p.update(a)
        ce_n.update(b)
        cm_p.update(c)
        cm_n.update(d)
    prior = np.log((Y.sum() + 1) / (len(Y) - Y.sum() + 1))
    k = 5.0  # smoothing pseudo-count

    def lo(p, n):
        return {t: float(np.log((p.get(t, 0) + k * np.exp(prior) / (1 + np.exp(prior))) /
                                (n.get(t, 0) + k / (1 + np.exp(prior)))) - prior)
                for t in set(p) | set(n) if p.get(t, 0) + n.get(t, 0) >= 3}
    return lo(ce_p, ce_n), lo(cm_p, cm_n)


def _num_rel(a, b):
    """relation flags between two digit strings."""
    if not a or not b:
        return (0, 0, 0, 0, 0, 0, 0)
    if a == b:
        return (1, 0, 0, 0, 0, 0, 0)
    tr_rec = int(len(a) == len(b) + 1 and a.startswith(b))
    tr_s1 = int(len(b) == len(a) + 1 and b.startswith(a))
    pref = int(not tr_rec and not tr_s1 and (a.startswith(b) or b.startswith(a)))
    try:
        off = int(abs(int(a[:12]) - int(b[:12])) <= 10)
    except ValueError:
        off = 0
    sub = int(len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1)
    samelen = int(len(a) == len(b) and not sub)
    return (0, tr_rec, tr_s1, pref, off, sub, samelen)


def _feat_worker(args):
    I, J, T = args
    nc1, nc23, an1, an23 = _G["nc1"], _G["nc23"], _G["an1"], _G["an23"]
    tables = _G["tables"]
    out = np.zeros((len(I), 8 + 2 * 7 + 2), np.float32)
    for r, (i, j, t) in enumerate(zip(I, J, T)):
        le, lm = tables[t]
        e, m = _tok_diff(nc1[i], nc23[j])
        ve = [le.get(x, 0.0) for x in e]
        vm = [lm.get(x, 0.0) for x in m]
        out[r, 0] = min(ve) if ve else 0.0
        out[r, 1] = max(ve) if ve else 0.0
        out[r, 2] = sum(ve)
        out[r, 3] = len(e)
        out[r, 4] = min(vm) if vm else 0.0
        out[r, 5] = max(vm) if vm else 0.0
        out[r, 6] = sum(vm)
        out[r, 7] = len(m)
        n1, n2 = an1[i].split(), an23[j].split()
        f1 = n1[0] if n1 else ""
        lg = max(n1, key=len) if n1 else ""
        best_first = max((_num_rel(f1, x) for x in n2[:4]), default=(0,) * 7, key=lambda z: (z[0], sum(z)))
        best_long = max((_num_rel(lg, x) for x in n2[:4]), default=(0,) * 7, key=lambda z: (z[0], sum(z)))
        out[r, 8:15] = best_first
        out[r, 15:22] = best_long
        out[r, 22] = float(len(set(n1) & set(n2)) > 0)
        out[r, 23] = float(bool(n1) and not n2)
    return out


NAMES = ["xe_lo_min", "xe_lo_max", "xe_lo_sum", "xe_n", "xm_lo_min", "xm_lo_max", "xm_lo_sum", "xm_n"] + \
        [f"nrf_{k}" for k in ("eq", "trrec", "trs1", "pref", "off", "sub", "samelen")] + \
        [f"nrl_{k}" for k in ("eq", "trrec", "trs1", "pref", "off", "sub", "samelen")] + ["num_any_shared", "rec_no_num"]


def compute(split, norm_dir, cand_dir, tables, table_idx, n_jobs=96):
    cols = ["nc", "an"]
    s1 = pd.read_parquet(f"{norm_dir}/{split}_s1_norm.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/{split}_s{k}_norm.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    cand = pd.read_parquet(f"{cand_dir}/{split}_cand.parquet", columns=["i1", "j"])
    _G.update(nc1=s1.nc.tolist(), nc23=s23.nc.tolist(), an1=s1["an"].tolist(), an23=s23["an"].tolist(), tables=tables)
    I, J = cand.i1.values, cand.j.values
    chunks = np.array_split(np.arange(len(I)), n_jobs * 8)
    with Pool(n_jobs) as pool:
        parts = pool.map(_feat_worker, [(I[c], J[c], table_idx[c]) for c in chunks])
    return pd.DataFrame(np.vstack(parts), columns=NAMES)


def main():
    norm_dir, cand_dir, data_dir, out_dir = sys.argv[1:5]
    t0 = time.time()
    gt = pd.read_parquet(f"{data_dir}/train_gt.parquet")
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    cand = pd.read_parquet(f"{cand_dir}/train_cand.parquet", columns=["i1", "j", "s1_id", "r_id"])
    Y = (cand.r_id.map(truth).values == cand.s1_id.values).astype(np.int8)
    half = (pd.util.hash_pandas_object(cand.s1_id, index=False).values % 2).astype(np.int8)
    s1 = pd.read_parquet(f"{norm_dir}/train_s1_norm.parquet", columns=["nc"])
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/train_s{k}_norm.parquet", columns=["nc"]) for k in (2, 3)], ignore_index=True)
    nc1, nc23 = s1.nc.tolist(), s23.nc.tolist()
    I, J = cand.i1.values, cand.j.values
    lo = []
    for h in (0, 1):
        m = half == h
        lo.append(learn_logodds(nc1, nc23, I[m], J[m], Y[m]))
    lo_all = learn_logodds(nc1, nc23, I, J, Y)
    print(f"log-odds tables learned {time.time() - t0:.0f}s  sizes {[len(x[0]) for x in lo]} all {len(lo_all[0])}", flush=True)
    # show a few informative extra tokens
    top = sorted(lo_all[0].items(), key=lambda kv: kv[1])[:15]
    print("most negative extra tokens:", [(k, round(v, 2)) for k, v in top], flush=True)
    tables = [lo[0], lo[1], lo_all]
    # train: pairs of half h use the table learned on the other half
    F = compute("train", norm_dir, cand_dir, tables, np.where(half == 0, 1, 0).astype(np.int8))
    F.to_parquet(f"{out_dir}/train_xfeats.parquet", index=False)
    print(f"train extra feats {F.shape} {time.time() - t0:.0f}s", flush=True)
    tc = pd.read_parquet(f"{cand_dir}/test_cand.parquet", columns=["i1"])
    F = compute("test", norm_dir, cand_dir, tables, np.full(len(tc), 2, np.int8))
    F.to_parquet(f"{out_dir}/test_xfeats.parquet", index=False)
    print(f"test extra feats {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

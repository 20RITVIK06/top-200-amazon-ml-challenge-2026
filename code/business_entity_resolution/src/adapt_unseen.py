"""Unsupervised adaptation of the name-token log-odds tables to countries unseen in training.

Weak supervision from addresses: among candidate pairs whose core names share the first
token and whose street words overlap (>=2 shared alphabetic address tokens), pairs whose
first house numbers agree are mostly true matches, pairs whose numbers differ are mostly
clones (the generator's clone entities change the number).  Token log-odds computed from
this address-only signal are independent of the name model, and on train they rank tokens
like the true (label-based) log-odds (Pearson 0.67; clone markers such as 'group',
'holdings', 'north' strongly negative, add-ons such as 'services', 'center' positive).
A monotone (isotonic) map fitted on train converts pseudo log-odds to the true scale; it is
applied to every test country absent from train, and the resulting tables replace the
token log-odds features for that country's pairs.

usage: python adapt_unseen.py <norm_dir> <cand_dir> <data_dir> <feats_dir>
"""
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import extra_feats as XF


def pseudo_counts(nc1, nc23, an1, an23, at1, at23, I, J, Y=None):
    ep, en, mp, mn = Counter(), Counter(), Counter(), Counter()   # address-pseudo
    tep, ten, tmp, tmn = Counter(), Counter(), Counter(), Counter()  # true labels (train only)
    for k in range(len(I)):
        i, j = I[k], J[k]
        a, b = nc1[i].split(), nc23[j].split()
        if not a or not b or a[0] != b[0]:
            continue
        ex, mi = set(b) - set(a), set(a) - set(b)
        if not ex and not mi:
            continue
        if Y is not None:
            if Y[k]:
                tep.update(ex)
                tmp.update(mi)
            else:
                ten.update(ex)
                tmn.update(mi)
        x1, x2 = an1[i].split()[:1], an23[j].split()[:1]
        w1 = {t for t in at1[i].split() if not t.isdigit()}
        w2 = {t for t in at23[j].split() if not t.isdigit()}
        if not x1 or not x2 or len(w1 & w2) < 2:
            continue
        if x1 == x2:
            ep.update(ex)
            mp.update(mi)
        else:
            en.update(ex)
            mn.update(mi)
    return (ep, en, mp, mn), (tep, ten, tmp, tmn)


def lo(p, n, t):
    return float(np.log((p.get(t, 0) + 1.0) / (n.get(t, 0) + 1.0)))


def fit_map(pseudo_p, pseudo_n, true_p, true_n, min_count=100):
    toks = [t for t in set(true_p) | set(true_n) if true_p.get(t, 0) + true_n.get(t, 0) >= min_count]
    x = np.array([lo(pseudo_p, pseudo_n, t) for t in toks])
    y = np.array([lo(true_p, true_n, t) for t in toks])
    w = np.array([true_p.get(t, 0) + true_n.get(t, 0) for t in toks], float)
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y, sample_weight=np.sqrt(w))
    return iso, np.corrcoef(x, y)[0, 1]


def main():
    norm_dir, cand_dir, data_dir, feats_dir = sys.argv[1:5]
    t0 = time.time()
    cols = ["nc", "an", "at", "country"]
    tr1 = pd.read_parquet(f"{norm_dir}/train_s1_norm.parquet", columns=cols)
    tr23 = pd.concat([pd.read_parquet(f"{norm_dir}/train_s{k}_norm.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    gt = pd.read_parquet(f"{data_dir}/train_gt.parquet")
    truth = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in (l.split(",") if l else [])}
    tc = pd.read_parquet(f"{cand_dir}/train_cand.parquet", columns=["i1", "j", "s1_id", "r_id"])
    Y = (tc.r_id.map(truth).values == tc.s1_id.values)
    sel = np.random.default_rng(0).choice(len(tc), min(len(tc), 6_000_000), replace=False)
    (ep, en, mp, mn), (tep, ten, tmp, tmn) = pseudo_counts(
        tr1.nc.values, tr23.nc.values, tr1["an"].values, tr23["an"].values, tr1["at"].values, tr23["at"].values,
        tc.i1.values[sel], tc.j.values[sel], Y[sel])
    iso_e, r_e = fit_map(ep, en, tep, ten)
    iso_m, r_m = fit_map(mp, mn, tmp, tmn)
    print(f"train calibration: extra corr {r_e:.3f}  missing corr {r_m:.3f}  {time.time() - t0:.0f}s", flush=True)
    train_countries = set(tr1.country)
    te1 = pd.read_parquet(f"{norm_dir}/test_s1_norm.parquet", columns=cols)
    te23 = pd.concat([pd.read_parquet(f"{norm_dir}/test_s{k}_norm.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    cand = pd.read_parquet(f"{cand_dir}/test_cand.parquet", columns=["i1", "j"])
    ctry = te1.country.values[cand.i1.values]
    # base tables learned on all training pairs with true labels (as used for test)
    base_e, base_m = XF.learn_logodds(tr1.nc.tolist(), tr23.nc.tolist(), tc.i1.values, tc.j.values, Y.astype(np.int8))
    tables = [(base_e, base_m)]
    table_idx = np.zeros(len(cand), np.int8)
    for c in sorted(set(te1.country) - train_countries):
        m = np.where(ctry == c)[0]
        (cp, cn, cmp_, cmn), _ = pseudo_counts(te1.nc.values, te23.nc.values, te1["an"].values, te23["an"].values,
                                               te1["at"].values, te23["at"].values, cand.i1.values[m], cand.j.values[m])
        toks_e = [t for t in set(cp) | set(cn) if cp.get(t, 0) + cn.get(t, 0) >= 20]
        toks_m = [t for t in set(cmp_) | set(cmn) if cmp_.get(t, 0) + cmn.get(t, 0) >= 20]
        ce = dict(base_e)
        ce.update(zip(toks_e, iso_e.predict(np.array([lo(cp, cn, t) for t in toks_e]))))
        cm = dict(base_m)
        cm.update(zip(toks_m, iso_m.predict(np.array([lo(cmp_, cmn, t) for t in toks_m]))))
        tables.append((ce, cm))
        table_idx[m] = len(tables) - 1
        show = sorted(((t, ce[t]) for t in toks_e), key=lambda kv: kv[1])
        print(f"{c}: {len(toks_e)} extra / {len(toks_m)} missing tokens adapted; most negative extra: "
              f"{[(t, round(v, 1)) for t, v in show[:12]]}; most positive: {[(t, round(v, 1)) for t, v in show[-8:]]}", flush=True)
    F = XF.compute("test", norm_dir, cand_dir, tables, table_idx)
    F.to_parquet(f"{feats_dir}/test_xfeats_adapted.parquet", index=False)
    print(f"saved adapted test token features {F.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

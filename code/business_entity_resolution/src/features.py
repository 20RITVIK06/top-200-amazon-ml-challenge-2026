"""Pairwise feature computation for (Source-1 record, Source-2/3 record) candidate pairs.

Two kinds of features:
  * vectorised sparse features (TF-IDF cosines, IDF-weighted token overlaps) computed
    with scipy on whole pair arrays;
  * string-similarity features computed per pair with rapidfuzz in a fork-based
    process pool (the record tables are shared copy-on-write with the workers).
"""
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

# ----------------------------------------------------------------------------- sparse features


def rowdot(X, Y, I, J, chunk=5_000_000):
    """sum_k X[I[p],k] * Y[J[p],k] for every pair p."""
    out = np.empty(len(I), dtype=np.float32)
    for s in range(0, len(I), chunk):
        e = min(s + chunk, len(I))
        out[s:e] = np.asarray(X[I[s:e]].multiply(Y[J[s:e]]).sum(axis=1)).ravel()
    return out


def binarize(X):
    B = X.copy().tocsr()
    B.data[:] = 1.0
    return B


def idf_weighted(texts_all, min_df=1):
    """Return (vocab dict, idf array) for whitespace tokens over the given texts (set semantics)."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer=str.split, lowercase=False, binary=True, min_df=min_df, dtype=np.float32)
    X = cv.fit_transform(texts_all)
    df = np.asarray(X.sum(axis=0)).ravel()
    idf = np.log((X.shape[0] + 1) / (df + 1)).astype(np.float32) + 1.0
    return cv, idf, X


def overlap_feats(B1, B23, idf, I, J, prefix):
    """IDF-weighted and count overlaps between token sets (binary CSR matrices)."""
    W1 = B1 @ sp.diags(idf)
    W23 = B23 @ sp.diags(idf)
    inter_w = rowdot(W1, B23, I, J)
    inter_c = rowdot(B1, B23, I, J)
    w1 = np.asarray(W1.sum(axis=1)).ravel()[I]
    w2 = np.asarray(W23.sum(axis=1)).ravel()[J]
    c1 = np.asarray(B1.sum(axis=1)).ravel()[I]
    c2 = np.asarray(B23.sum(axis=1)).ravel()[J]
    eps = 1e-6
    f = {
        f"{prefix}_iw_jac": inter_w / (w1 + w2 - inter_w + eps),
        f"{prefix}_iw_cov1": inter_w / (w1 + eps),
        f"{prefix}_iw_cov2": inter_w / (w2 + eps),
        f"{prefix}_c_jac": inter_c / (c1 + c2 - inter_c + eps),
        f"{prefix}_c_inter": inter_c,
        f"{prefix}_c_n1": c1,
        f"{prefix}_c_n2": c2,
        f"{prefix}_iw_miss1": w1 - inter_w,
        f"{prefix}_iw_miss2": w2 - inter_w,
    }
    return {k: v.astype(np.float32) for k, v in f.items()}


# ----------------------------------------------------------------------------- string features

_G = {}
_ALIAS = [set()]


def set_alias(tables):
    """tables: list of alias-pair lists; pair p uses tables[fold[p]] (see string_features)."""
    _ALIAS.clear()
    _ALIAS.append(set())  # index 0: no alias knowledge
    for pairs in tables:
        _ALIAS.append({(a, b) for a, b, *_ in pairs})


def _alias_feats(ac1, ac2, alias):
    c1 = [c for c in ac1.split("|") if c and not any(ch.isdigit() for ch in c)]
    c2 = [c for c in ac2.split("|") if c and not any(ch.isdigit() for ch in c)]
    if not c1 or not c2:
        return [-1.0, -1.0, -1.0]
    u2 = [x for x in c2 if max(fuzz.ratio(x, y) for y in c1) < 80]
    u1 = [y for y in c1 if max(fuzz.ratio(x, y) for x in c2) < 80]
    hits = 0
    for x in u2:
        for y in u1:
            if (y, x) in alias:
                hits += 1
                break
    return [float(len(u2)), float(len(u1)), float(hits)]


def _num_sim(x, y):
    if x == y:
        return 1.0
    return 1.0 - Levenshtein.normalized_distance(x, y)


def _monge_elkan(ta, tb):
    if not ta or not tb:
        return 0.0
    s = 0.0
    for x in ta:
        best = 0.0
        for y in tb:
            v = JaroWinkler.similarity(x, y)
            if v > best:
                best = v
                if best == 1.0:
                    break
        s += best
    return s / len(ta)


def _pair_feats(a, b, alias=frozenset()):
    """a, b: tuples of record fields -> list of floats."""
    (nf1, nc1, na1, nw1, nl1, fl1, at1, an1, st1, ac1) = a
    (nf2, nc2, na2, nw2, nl2, fl2, at2, an2, st2, ac2) = b
    f = []
    # ---- name
    f.append(fuzz.ratio(nc1, nc2))
    f.append(fuzz.token_sort_ratio(nc1, nc2))
    f.append(fuzz.token_set_ratio(nc1, nc2))
    f.append(fuzz.partial_ratio(nc1, nc2) if nc1 and nc2 else 0.0)
    f.append(JaroWinkler.similarity(nc1, nc2))
    f.append(fuzz.ratio(nf1, nf2))
    cc1, cc2 = nc1.replace(" ", ""), nc2.replace(" ", "")
    f.append(fuzz.ratio(cc1, cc2))
    # common prefix of concatenated core names relative to shorter
    m = 0
    for x, y in zip(cc1, cc2):
        if x != y:
            break
        m += 1
    f.append(m / max(1, min(len(cc1), len(cc2))))
    f.append(m)
    t1, t2 = nc1.split(), nc2.split()
    f.append(_monge_elkan(t1, t2))
    f.append(_monge_elkan(t2, t1))
    f.append(1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0)
    f.append(JaroWinkler.similarity(t1[0], t2[0]) if (t1 and t2) else 0.0)
    f.append(1.0 if nc1 == nc2 else 0.0)
    f.append(1.0 if sorted(t1) == sorted(t2) else 0.0)
    # web / domain stem vs concatenated S1 name
    if nw2:
        f.append(fuzz.ratio(nw2, cc1))
        f.append(fuzz.partial_ratio(nw2, cc1))
        mm = 0
        for x, y in zip(nw2, cc1):
            if x != y:
                break
            mm += 1
        f.append(mm / max(1, min(len(nw2), len(cc1))))
    else:
        f.extend([-1.0, -1.0, -1.0])
    # alias part of S2/3 name vs S1 core
    f.append(fuzz.token_set_ratio(nc1, na2) if na2 else -1.0)
    # legal forms
    l1, l2 = set(nl1.split()), set(nl2.split())
    f.append(-1.0 if not (l1 and l2) else (1.0 if l1 == l2 else (0.5 if l1 & l2 else 0.0)))
    f.append(len(nc1))
    f.append(len(nc2))
    f.append(len(t1))
    f.append(len(t2))
    f.append(fl2 & 1)
    f.append((fl2 >> 1) & 1)
    f.append((fl2 >> 2) & 1)
    f.append((fl2 >> 3) & 1)
    f.append((fl2 >> 4) & 1)
    # ---- address
    e2 = 1.0 if not at2 else 0.0
    f.append(e2)
    if at2:
        f.append(fuzz.token_set_ratio(at1, at2))
        f.append(fuzz.token_sort_ratio(at1, at2))
        f.append(fuzz.partial_token_set_ratio(at1, at2))
    else:
        f.extend([-1.0, -1.0, -1.0])
    # numbers
    n1, n2 = an1.split(), an2.split()
    if n1 and n2:
        s1n, s2n = set(n1), set(n2)
        f.append(1.0 if n1[0] == n2[0] else 0.0)
        f.append(len(s1n & s2n) / len(s1n | s2n))
        f.append(len(s1n & s2n))
        f.append(1.0 if n1[0] in s2n else 0.0)
        f.append(1.0 if n2[0] in s1n else 0.0)
        f.append(_num_sim(n1[0], n2[0]))
        best = 0.0
        for x in n1[:4]:
            for y in n2[:4]:
                v = _num_sim(x, y)
                if v > best:
                    best = v
        f.append(best)
        # numeric closeness of first numbers
        try:
            x, y = int(n1[0][:9]), int(n2[0][:9])
            f.append(abs(x - y) / max(1, max(x, y)))
        except ValueError:
            f.append(-1.0)
        # longest number of S1 (usually the house number) present in S2/3
        lg = max(n1, key=len)
        f.append(1.0 if lg in s2n else 0.0)
    else:
        f.extend([-1.0] * 9)
    f.append(len(n1))
    f.append(len(n2))
    # state
    if st1 and st2:
        f.append(1.0 if st1 == st2 else 0.0)
    else:
        f.append(-1.0)
    # alpha (non-number, non-state) tokens soft match
    w1 = [t for t in at1.split() if not t.isdigit() and not t.startswith("s_") and len(t) > 1]
    w2 = [t for t in at2.split() if not t.isdigit() and not t.startswith("s_") and len(t) > 1]
    if w1 and w2:
        f.append(_monge_elkan(w2, w1))
        f.append(_monge_elkan(w1, w2))
    else:
        f.extend([-1.0, -1.0])
    # component level best matches
    c1 = [c for c in ac1.split("|") if c]
    c2 = [c for c in ac2.split("|") if c]
    if c1 and c2:
        bests = []
        for x in c2:
            bb = 0.0
            for y in c1:
                v = fuzz.ratio(x, y)
                if v > bb:
                    bb = v
            bests.append(bb)
        f.append(sum(bests) / len(bests))
        f.append(min(bests))
        f.append(max(bests))
    else:
        f.extend([-1.0, -1.0, -1.0])
    f.extend(_alias_feats(ac1, ac2, alias))
    return f


STR_FEATS = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "nf_ratio", "ncat_ratio", "ncat_pref", "ncat_pref_len",
    "n_me12", "n_me21", "n_first_eq", "n_first_jw", "n_exact", "n_sorted_eq",
    "web_ratio", "web_partial", "web_pref", "alias_tset", "legal_cmp", "n_len1", "n_len2", "n_ntok1", "n_ntok2",
    "fl_alias", "fl_domain", "fl_indic", "fl_web", "fl_handle",
    "a_empty2", "a_tset", "a_tsort", "a_ptset",
    "num_first_eq", "num_jac", "num_inter", "num_f1_in2", "num_f2_in1", "num_first_sim", "num_best_sim",
    "num_first_reldiff", "num_long_in2", "num_n1", "num_n2", "state_eq",
    "aw_me21", "aw_me12", "ac_mean", "ac_min", "ac_max",
    "loc_unm2", "loc_unm1", "loc_alias_hits",
]


def _worker(args):
    I, J, T = args
    R1, R23 = _G["R1"], _G["R23"]
    out = np.empty((len(I), len(STR_FEATS)), dtype=np.float32)
    for k in range(len(I)):
        out[k] = _pair_feats(R1[I[k]], R23[J[k]], _ALIAS[T[k]])
    return out


def string_features(R1, R23, I, J, table=None, n_jobs=None, chunk=20000):
    """R1/R23: lists of field tuples; I/J: pair index arrays; table: alias table index per pair."""
    _G["R1"], _G["R23"] = R1, R23
    n_jobs = n_jobs or os.cpu_count()
    if table is None:
        table = np.zeros(len(I), np.int8)
    tasks = [(I[s:s + chunk], J[s:s + chunk], table[s:s + chunk]) for s in range(0, len(I), chunk)]
    with Pool(n_jobs) as pool:
        parts = pool.map(_worker, tasks, chunksize=1)
    _G.clear()
    return np.vstack(parts) if parts else np.zeros((0, len(STR_FEATS)), np.float32)


FIELDS = ["nf", "nc", "na", "nw", "nl", "nflag", "at", "an", "ast", "ac"]


def records(df):
    return list(zip(*[df[c].tolist() for c in FIELDS]))

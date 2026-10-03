"""Candidate generation (blocking).

For every country (an open set of labels) we build TF-IDF representations of the
records and retrieve, for each Source-2/3 record, its top-K most similar Source-1
records under several complementary channels:

  comb : name char-4-gram TF-IDF (x w_name) concatenated with address unigram+bigram
         TF-IDF, L2-normalised.  Disambiguates common names ('Primary Care Group') by
         address and survives heavy address noise through the name.  (main channel)
  name : name char-4-gram TF-IDF only (records with missing / corrupted address)
  addr : address unigram+bigram TF-IDF only (records whose name was replaced / DBA)
  rev  : the reverse direction on the comb channel: every S1 retrieves its top-K
         records (protects S1 entities whose records are crowded out elsewhere).

Very frequent features (document frequency above a cap within the S1 index) are
dropped from the retrieval vectors: they carry little ranking information but
dominate the cost of the sparse top-K product (20x speed-up, <0.1% recall change).
Exact cosines on the full vectors are recomputed later for all kept pairs.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn


def addr_bigrams(s):
    t = s.split()
    return t + [t[i] + "_" + t[i + 1] for i in range(len(t) - 1)]


def name_text(df):
    return (" " + df.nc.fillna("") + " ").values


def addr_text(df):
    return df["at"].fillna("").values


def fit_vectors(s1, s23):
    """Per-country TF-IDF fitted on S1+S2+S3 (unsupervised). Returns name/addr matrices."""
    alln = np.concatenate([name_text(s1), name_text(s23)])
    alla = np.concatenate([addr_text(s1), addr_text(s23)])
    vn = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), sublinear_tf=True, min_df=2, dtype=np.float32, lowercase=False)
    Xn = vn.fit_transform(alln).tocsr()
    va = TfidfVectorizer(analyzer=addr_bigrams, sublinear_tf=True, min_df=1, dtype=np.float32, lowercase=False)
    Xa = va.fit_transform(alla).tocsr()
    n1 = len(s1)
    return Xn[:n1], Xn[n1:], Xa[:n1], Xa[n1:]


def prune_cols(X1, X23, cap):
    df = np.diff(X1.tocsc().indptr)
    keep = np.where(df <= cap)[0]
    return normalize(X1[:, keep]).tocsr(), normalize(X23[:, keep]).tocsr()


def topk(Q, I, k, threads=96, threshold=0.0):
    """Top-k cosine neighbours of rows of Q among rows of I (both L2-normalised CSR)."""
    C = sp_matmul_topn(Q, I.T.tocsr(), top_n=k, threshold=threshold, sort=True, n_threads=threads)
    C = C.tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)


def comb_matrix(N, A, w_name):
    return normalize(sp.hstack([N * w_name, A], format="csr")).tocsr()


CFG = dict(k_comb=30, k_name=15, k_addr=10, k_rev=10, cap_name=20000, cap_addr=5000, w_name=0.6)


def block_country(s1, s23, cfg=CFG, threads=96, verbose=True):
    """Return candidate DataFrame (i1 row in s1, j row in s23, channel ranks) and full vectors."""
    t0 = time.time()
    N1, N23, A1, A23 = fit_vectors(s1, s23)
    pN1, pN23 = prune_cols(N1, N23, cfg["cap_name"])
    pA1, pA23 = prune_cols(A1, A23, cfg["cap_addr"])
    pC1, pC23 = comb_matrix(pN1, pA1, cfg["w_name"]), comb_matrix(pN23, pA23, cfg["w_name"])
    if verbose:
        print(f"   vectors {time.time() - t0:.0f}s", flush=True)
    parts = []
    for tag, Q, I, k in (("comb", pC23, pC1, cfg["k_comb"]), ("name", pN23, pN1, cfg["k_name"]), ("addr", pA23, pA1, cfg["k_addr"])):
        t = time.time()
        r, c, v = topk(Q, I, k, threads)
        d = pd.DataFrame({"j": r, "i1": c})
        d["rk_" + tag] = d.groupby("j").cumcount().astype(np.int16)
        parts.append(d)
        if verbose:
            print(f"   {tag}: {len(d)} pairs {time.time() - t:.0f}s", flush=True)
    t = time.time()
    r, c, v = topk(pC1, pC23, cfg["k_rev"], threads)
    d = pd.DataFrame({"i1": r, "j": c})
    d["rk_rev"] = d.groupby("i1").cumcount().astype(np.int16)
    parts.append(d)
    if verbose:
        print(f"   rev: {len(d)} pairs {time.time() - t:.0f}s", flush=True)
    cand = parts[0]
    for d in parts[1:]:
        cand = cand.merge(d, on=["j", "i1"], how="outer")
    for c in ["rk_comb", "rk_name", "rk_addr", "rk_rev"]:
        cand[c] = cand[c].fillna(99).astype(np.int16)
    C1, C23 = comb_matrix(N1, A1, cfg["w_name"]), comb_matrix(N23, A23, cfg["w_name"])
    return cand, dict(N1=N1, N23=N23, A1=A1, A23=A23, C1=C1, C23=C23)

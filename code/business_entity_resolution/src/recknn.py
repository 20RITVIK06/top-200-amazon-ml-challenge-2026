"""Record-to-record nearest neighbours (Source-2/3 records of the same split and country).

Uses the cached combined TF-IDF record vectors (name char-4-grams + address uni/bi-grams,
country blocks are disjoint so neighbours never cross countries).  Very frequent columns
are pruned for speed, then a multithreaded sparse top-K self-join is run; the record
itself is removed from its neighbour list.

usage: python recknn.py <cand_dir> <split> [k] [min_sim]
output: <cand_dir>/<split>_recnn.parquet  (j, nb, sim)
"""
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn


def main():
    cand_dir, split = sys.argv[1:3]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    min_sim = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    t0 = time.time()
    C = sp.load_npz(f"{cand_dir}/{split}_recvec_comb.npz").tocsr()
    df = np.diff(C.tocsc().indptr)
    keep = np.where(df <= int(sys.argv[5]) if len(sys.argv) > 5 else df <= 3000)[0]
    Cp = normalize(C[:, keep]).tocsr().astype(np.float32)
    print(f"{split}: {Cp.shape} pruned cols {C.shape[1] - len(keep)}  {time.time() - t0:.0f}s", flush=True)
    R = sp_matmul_topn(Cp, Cp.T.tocsr(), top_n=k + 1, threshold=min_sim, sort=True, n_threads=96).tocoo()
    d = pd.DataFrame({"j": R.row.astype(np.int64), "nb": R.col.astype(np.int64), "sim": R.data.astype(np.float32)})
    d = d[d.j.values != d.nb.values].reset_index(drop=True)
    d.to_parquet(f"{cand_dir}/{split}_recnn.parquet", index=False)
    print(f"{split}: neighbour pairs {len(d)} ({len(d) / C.shape[0]:.2f}/record)  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

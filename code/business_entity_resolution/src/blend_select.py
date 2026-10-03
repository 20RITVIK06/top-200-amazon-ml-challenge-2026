"""Choose the final probability blend by out-of-fold macro-F0.5 and write the test submission.

usage: python blend_select.py <work_dir> <out_dir> name1:oof1.npy:test1.npy [name2:oof2.npy:test2.npy ...]

Evaluates every single model and every equal-weight blend of 2..k models on OOF
(arg-max assignment + expected-F0.5 decisions), then applies the best blend to the
test predictions of the same models (same weights) and writes the submission files.
"""
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from train_oof import eval_expf, labels


def main():
    W, out_dir = sys.argv[1], sys.argv[2]
    specs = [a.split(":") for a in sys.argv[3:]]
    cand = pd.read_parquet(f"{W}/cand/train_cand.parquet", columns=["s1_id", "r_id"])
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    y = labels(cand, gt)
    s1 = pd.read_parquet(f"{W}/data/train_s1.parquet", columns=["entity_id"])
    pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    cnt = {a: (len(l.split(",")) if l else 0) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    tc = np.array([cnt[e] for e in s1.entity_id], float)
    ev = pd.DataFrame({"s1i": pos.reindex(cand.s1_id).values, "rj": pd.factorize(cand.r_id)[0], "y": y})
    oof = {n: np.load(o).astype(float) for n, o, _ in specs}
    res = {}
    for k in range(1, len(specs) + 1):
        for combo in itertools.combinations([s[0] for s in specs], k):
            p = np.mean([oof[c] for c in combo], axis=0)
            res[combo] = eval_expf(ev, p, tc, len(s1))
            print(f"{'+'.join(combo):30s} OOF expF {res[combo]:.5f}", flush=True)
    best = max(res, key=res.get)
    print(f"BEST: {'+'.join(best)}  {res[best]:.5f}", flush=True)
    test = {n: np.load(t).astype(float) for n, _, t in specs}
    pt = np.mean([test[c] for c in best], axis=0).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/final_test_prob.npy", pt)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, f"{here}/write_submission.py", f"{W}/data", f"{W}/cand", f"{out_dir}/final_test_prob.npy",
                    out_dir, "expf"], check=True)


if __name__ == "__main__":
    main()

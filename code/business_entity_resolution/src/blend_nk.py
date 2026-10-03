"""Blend the enlarged-candidate model with models trained on the original candidate set.

Pairs present in both candidate tables get the average of the selected models' probabilities;
pairs added by the number-key channel keep the enlarged-candidate model's probability.
Blend chosen by OOF macro-F0.5 (arg-max + expected-F0.5), then applied to test.

usage: python blend_nk.py <work_dir> <out_dir>
"""
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from train_oof import eval_expf, labels


def aligned(union, old_cand, p_old):
    key_u = union.j.values.astype(np.int64) * 10_000_000 + union.i1.values
    key_o = old_cand.j.values.astype(np.int64) * 10_000_000 + old_cand.i1.values
    s = pd.Series(p_old, index=key_o)
    return s.reindex(key_u).values  # NaN for pairs added by the new channel


def main():
    W, out = sys.argv[1], sys.argv[2]
    M = f"{W}/models"
    gt = pd.read_parquet(f"{W}/data/train_gt.parquet")
    s1 = pd.read_parquet(f"{W}/data/train_s1.parquet", columns=["entity_id"])
    pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    cnt = {a: (len(l.split(",")) if l else 0) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    tc = np.array([cnt[e] for e in s1.entity_id], float)
    res = {}
    preds = {}
    for split in ("train", "test"):
        u = pd.read_parquet(f"{W}/cand_nk/{split}_cand.parquet", columns=["i1", "j", "s1_id", "r_id"])
        o = pd.read_parquet(f"{W}/cand/{split}_cand.parquet", columns=["i1", "j"])
        suf = "oof" if split == "train" else "test"
        base = np.load(f"{M}/n2_{suf}.npy").astype(float)
        P = {"n2": base}
        for name in ("t2", "tb2", "ab2"):
            P[name] = aligned(u, o, np.load(f"{M}/{name}_{suf}.npy").astype(float))
        preds[split] = (u, P)
    u, P = preds["train"]
    y = labels(u, gt)
    ev = pd.DataFrame({"s1i": pos.reindex(u.s1_id).values, "rj": pd.factorize(u.r_id)[0], "y": y})
    others = ["t2", "tb2", "ab2"]
    for k in range(0, len(others) + 1):
        for combo in itertools.combinations(others, k):
            stack = np.vstack([P["n2"]] + [P[c] for c in combo])
            p = np.nanmean(stack, axis=0)
            res[combo] = eval_expf(ev, p, tc, len(s1))
            print(f"n2{'+' if combo else ''}{'+'.join(combo):20s} OOF expF {res[combo]:.5f}", flush=True)
    best = max(res, key=res.get)
    print(f"BEST: n2{'+' if best else ''}{'+'.join(best)}  {res[best]:.5f}", flush=True)
    u, P = preds["test"]
    pt = np.nanmean(np.vstack([P["n2"]] + [P[c] for c in best]), axis=0).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/final_test_prob.npy", pt)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, f"{here}/write_submission.py", f"{W}/data", f"{W}/cand_nk", f"{out}/final_test_prob.npy",
                    out, "expf"], check=True)


if __name__ == "__main__":
    main()

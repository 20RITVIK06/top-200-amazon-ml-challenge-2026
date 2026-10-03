"""Learn locality alias pairs (S1 address component -> record address component) from
labelled training pairs.

The generator frequently replaces a locality by another name of the same place
(neighbourhood, CDP, district, gazetteer alias: 'Phoenix' -> 'Maryvale',
'Boston' -> 'Dorchester', 'Hyderabad' -> 'Sangareddy').  For every true pair we
collect the alphabetic components of the record that fuzzy-match nothing in the S1
address and pair them with the S1 components that are likewise unmatched.  Pairs seen
at least `min_count` times are kept.  Only the provided training data is used.

usage: python learn_alias.py <norm_dir> <gt_parquet> <out_json> [min_count]

Output: {"fold0": [...], "fold1": [...], "all": [...]} where foldF is learned only from the
S1 entities of fold F (so out-of-fold features never see their own labels).
"""
import json
import sys
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz


def alpha(c):
    return not any(ch.isdigit() for ch in c)


def unmatched(ca, cb, thr=80):
    ub = [x for x in cb if max((fuzz.ratio(x, y) for y in ca), default=0) < thr]
    ua = [y for y in ca if max((fuzz.ratio(x, y) for x in cb), default=0) < thr]
    return ua, ub


def main():
    norm_dir, gt_path, out = sys.argv[1:4]
    min_count = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    cols = ["entity_id", "ac"]
    s1 = pd.read_parquet(f"{norm_dir}/train_s1_norm.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(f"{norm_dir}/train_s{k}_norm.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
    ac = dict(zip(s1.entity_id, s1.ac))
    ac.update(zip(s23.entity_id, s23.ac))
    gt = pd.read_parquet(gt_path)
    # fold of each S1 entity (same hash split as the grouped K-fold trainer, k=2)
    fold = (pd.util.hash_pandas_object(gt.source1_entity_id, index=False).values % 2).astype(int)
    cos = [Counter(), Counter()]
    for a, lst, f in zip(gt.source1_entity_id, gt.matched_entity_ids, fold):
        if not lst:
            continue
        co = cos[f]
        ca = [c for c in ac[a].split("|") if c and alpha(c)]
        for b in lst.split(","):
            cb = [c for c in ac[b].split("|") if c and alpha(c)]
            if not cb or not ca:
                continue
            ua, ub = unmatched(ca, cb)
            for x in ub:
                for y in ua:
                    co[(y, x)] += 1
    tot = cos[0] + cos[1]
    tables = {"fold0": [[a, b, n] for (a, b), n in cos[0].items() if n >= min_count],
              "fold1": [[a, b, n] for (a, b), n in cos[1].items() if n >= min_count],
              "all": [[a, b, n] for (a, b), n in tot.items() if n >= min_count]}
    json.dump(tables, open(out, "w"))
    print({k: len(v) for k, v in tables.items()})


if __name__ == "__main__":
    main()

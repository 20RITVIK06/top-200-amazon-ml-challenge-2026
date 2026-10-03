"""Learn Indic-script -> Latin dictionaries from the labelled training pairs.

The data generator transliterates English business names word-by-word into Indic
scripts (Devanagari, Kannada, Telugu, ...), and writes state names in addresses in
native script.  Because the vocabulary is small (~1.3k distinct Indic words), a
dictionary learned from aligned training pairs is far more accurate than any
rule-based transliterator.  Only the provided training data is used.

Output: JSON {"name": {indic_token: latin_token}, "addr_comp": {indic_component: latin_component},
              "addr_tok": {indic_token: latin_token}}
"""
import json
import sys
from collections import Counter, defaultdict

import pandas as pd
import regex

WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def is_indic(w):
    return any(0x0900 <= ord(c) <= 0x0DFF for c in w)


def toks(s):
    return WORD.findall(s.translate(ZW))


def comps(s):
    return [c.strip() for c in s.translate(ZW).split(",") if c.strip()]


def main(data_dir, out_path):
    s1 = pd.read_parquet(f"{data_dir}/train_s1.parquet")
    s2 = pd.read_parquet(f"{data_dir}/train_s2.parquet")
    s3 = pd.read_parquet(f"{data_dir}/train_s3.parquet")
    gt = pd.read_parquet(f"{data_dir}/train_gt.parquet")
    name = dict(zip(s1.entity_id, s1.business_name))
    addr = dict(zip(s1.entity_id, s1.business_address))
    for df in (s2, s3):
        name.update(zip(df.entity_id, df.business_name))
        addr.update(zip(df.entity_id, df.business_address))

    name_co = defaultdict(Counter)
    comp_co = defaultdict(Counter)
    comp_n = Counter()
    tok_co = defaultdict(Counter)
    for a, lst in zip(gt.source1_entity_id, gt.matched_entity_ids):
        if not lst:
            continue
        ta = [t.lower() for t in toks(name[a])]
        ca = [c.lower() for c in comps(addr[a])]
        for b in lst.split(","):
            tb = toks(name[b])
            if any(is_indic(w) for w in tb) and len(ta) == len(tb):
                for x, y in zip(tb, ta):
                    if is_indic(x):
                        name_co[x][y] += 1
            cb = comps(addr[b])
            ind = [c for c in cb if is_indic(c)]
            if ind:
                sa = set(ca)
                for c in ind:
                    comp_n[c] += 1
                    for d in sa:
                        comp_co[c][d] += 1
                    # token level fallback: Indic token vs S1 tokens
                    for w in toks(c):
                        if is_indic(w):
                            for d in {t.lower() for t in toks(addr[a])}:
                                tok_co[w][d] += 1

    name_map = {}
    for w, c in name_co.items():
        best, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= 2 and n / tot >= 0.3:
            name_map[w] = best
    comp_map = {}
    for c, cc in comp_co.items():
        best, n = cc.most_common(1)[0]
        if n >= 3 and n / comp_n[c] >= 0.6:
            comp_map[c] = best
    tok_map = {}
    for w, cc in tok_co.items():
        best, n = cc.most_common(1)[0]
        tot = max(cc.values())
        if n >= 3:
            tok_map[w] = best
    json.dump({"name": name_map, "addr_comp": comp_map, "addr_tok": tok_map}, open(out_path, "w"), ensure_ascii=False, indent=0)
    print(f"name map: {len(name_map)}  addr comp map: {len(comp_map)}  addr tok map: {len(tok_map)}")
    for k in list(comp_map)[:30]:
        print("  ", k, "->", comp_map[k])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

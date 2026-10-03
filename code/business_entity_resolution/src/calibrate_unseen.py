"""Prior matching for countries unseen in training.

The generator's singleton rate is a constant (5.58% for both US and India in train), so the
share of Source-1 entities that receive no match should not depend on the country.  For each
test country absent from train we scale its pair odds by a factor m (q = p·m / (p·m + 1 − p))
and pick m so that the country's predicted-empty rate (after arg-max assignment and
expected-F0.5 selection) equals the mean predicted-empty rate of the countries seen in
training, measured on the same test predictions.

usage: python calibrate_unseen.py <norm_dir> <cand_dir> <prob.npy> <out.npy>
"""
import sys

import numpy as np
import pandas as pd

from expected_f import select
from postproc import argmax_assign


def empty_rates(s1i, rj, p, s1_country, n_s1):
    m = argmax_assign(s1i, rj, p)
    idx = np.where(m)[0]
    sel = idx[select(s1i[idx], p[idx], n_s1, beta=0.5)]
    cnt = np.bincount(s1i[sel], minlength=n_s1)
    return {c: float(np.mean(cnt[s1_country == c] == 0)) for c in np.unique(s1_country)}


def main():
    norm_dir, cand_dir, prob_path, out_path = sys.argv[1:5]
    s1_country = pd.read_parquet(f"{norm_dir}/test_s1_norm.parquet", columns=["country"]).country.values
    train_countries = set(pd.read_parquet(f"{norm_dir}/train_s1_norm.parquet", columns=["country"]).country)
    cand = pd.read_parquet(f"{cand_dir}/test_cand.parquet", columns=["i1", "j"])
    p = np.load(prob_path).astype(np.float64)
    s1i, rj = cand.i1.values, cand.j.values
    n_s1 = len(s1_country)
    pair_c = s1_country[s1i]
    base = empty_rates(s1i, rj, p, s1_country, n_s1)
    seen = [c for c in base if c in train_countries]
    target = float(np.mean([base[c] for c in seen]))
    print("empty rates before:", {k: round(v, 4) for k, v in base.items()}, " target:", round(target, 4), flush=True)
    q = p.copy()
    for c in sorted(set(base) - train_countries):
        mask = pair_c == c
        lo, hi = 0.05, 1.0  # odds multipliers; empty rate decreases monotonically with m
        best_m = 1.0
        if base[c] < target:
            for _ in range(12):
                mid = np.sqrt(lo * hi)
                trial = p.copy()
                trial[mask] = p[mask] * mid / (p[mask] * mid + 1 - p[mask])
                r = empty_rates(s1i, rj, trial, s1_country, n_s1)[c]
                if r > target:
                    lo = mid
                else:
                    hi = mid
            best_m = np.sqrt(lo * hi)
        q[mask] = p[mask] * best_m / (p[mask] * best_m + 1 - p[mask])
        print(f"{c}: odds multiplier {best_m:.3f}", flush=True)
    print("empty rates after:", {k: round(v, 4) for k, v in empty_rates(s1i, rj, q, s1_country, n_s1).items()}, flush=True)
    np.save(out_path, q.astype(np.float32))


if __name__ == "__main__":
    main()

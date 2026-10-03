"""Post-processing: turn pair probabilities into per-S1 match sets.

Structural constraint (verified on train): every Source-2/3 record belongs to at most
one Source-1 entity.  So each record is assigned to its arg-max S1 candidate and kept
only if the probability clears a threshold.  Utilities to evaluate macro-F0.5 quickly
(vectorised) over all S1 entities, including those without any candidate.
"""
import numpy as np
import pandas as pd

from metric import f05_vectorized


def argmax_assign(s1_idx, r_idx, p):
    """Return boolean mask selecting, for every record r, its highest-probability pair."""
    order = np.lexsort((-p, r_idx))
    r_sorted = r_idx[order]
    first = np.ones(len(order), bool)
    first[1:] = r_sorted[1:] != r_sorted[:-1]
    mask = np.zeros(len(p), bool)
    mask[order[first]] = True
    return mask


def score_threshold(s1_idx, y, p, ntrue, thresholds, n_s1):
    """Macro F0.5 for each threshold. s1_idx in [0, n_s1); ntrue: array len n_s1."""
    res = []
    for th in thresholds:
        m = p >= th
        tp = np.bincount(s1_idx[m], weights=y[m], minlength=n_s1)
        npred = np.bincount(s1_idx[m], minlength=n_s1)
        res.append(f05_vectorized(tp, npred, ntrue).mean())
    return np.array(res)


def evaluate(cand, p, truth_cnt, n_s1, thresholds=None, use_argmax=True):
    """cand: DataFrame with s1i (int index of S1), rj (int index of record), y (0/1)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.025), 3)
    s1i, rj, y = cand.s1i.values, cand.rj.values, cand.y.values.astype(float)
    if use_argmax:
        m = argmax_assign(s1i, rj, p)
        s1i, y, p = s1i[m], y[m], p[m]
    sc = score_threshold(s1i, y, p, truth_cnt, thresholds, n_s1)
    b = int(np.argmax(sc))
    return float(sc[b]), float(thresholds[b]), dict(zip(thresholds.tolist(), sc.tolist()))


def ceiling(cand, truth_cnt, n_s1):
    """Macro F0.5 of a perfect matcher restricted to the candidate set."""
    s1i, y = cand.s1i.values, cand.y.values.astype(float)
    tp = np.bincount(s1i, weights=y, minlength=n_s1)
    return float(f05_vectorized(tp, tp, truth_cnt).mean())

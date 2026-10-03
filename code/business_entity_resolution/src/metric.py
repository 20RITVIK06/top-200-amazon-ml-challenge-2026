"""Competition metric: macro F0.5 over Source-1 entities (singletons included)."""
import numpy as np
import pandas as pd


def f05_per_entity(pred_sets, true_sets, beta=0.5):
    """pred_sets/true_sets: dict s1_id -> set. Evaluated over keys of true_sets."""
    b2 = beta * beta
    scores = {}
    for k, t in true_sets.items():
        p = pred_sets.get(k, set())
        if not t and not p:
            scores[k] = 1.0
            continue
        if not t or not p:
            scores[k] = 0.0
            continue
        tp = len(p & t)
        if tp == 0:
            scores[k] = 0.0
            continue
        prec, rec = tp / len(p), tp / len(t)
        scores[k] = (1 + b2) * prec * rec / (b2 * prec + rec)
    return scores


def macro_f05(pred_sets, true_sets):
    s = f05_per_entity(pred_sets, true_sets)
    return float(np.mean(list(s.values()))) if s else 0.0


def f05_vectorized(tp, npred, ntrue, beta=0.5):
    """Vectorised per-entity F-beta given counts (arrays)."""
    b2 = beta * beta
    tp = np.asarray(tp, float)
    npred = np.asarray(npred, float)
    ntrue = np.asarray(ntrue, float)
    out = np.zeros_like(tp)
    both0 = (npred == 0) & (ntrue == 0)
    out[both0] = 1.0
    ok = (tp > 0)
    prec = np.where(ok, tp / np.maximum(npred, 1), 0)
    rec = np.where(ok, tp / np.maximum(ntrue, 1), 0)
    out[ok] = ((1 + b2) * prec[ok] * rec[ok]) / (b2 * prec[ok] + rec[ok])
    return out


def load_truth(gt_df):
    return {a: set(l.split(",")) if l else set() for a, l in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids)}

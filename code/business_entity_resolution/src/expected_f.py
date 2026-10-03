"""Per-entity expected-F_beta maximisation.

For one Source-1 entity with candidate records sorted by calibrated match probability
q_1 >= q_2 >= ... >= q_m (assumed independent Bernoulli), plus optional "unselectable"
probability mass (true matches that can never be predicted, e.g. blocking misses), choose
k in {0..m} (predict the top-k) maximising

    E[F_beta] = sum_{s,t} P(TP_top=s) P(TP_rest=t) F(s, t, k)

with F(s,t,k) = (1+b^2) s / (b^2 (s+t) + k), and F = 1 iff k = 0 and s + t = 0.
Exact Poisson-binomial DP, O(m^2) per k, compiled with numba.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _pb(q):
    """Poisson-binomial pmf of the number of successes of independent Bernoulli(q)."""
    n = len(q)
    d = np.zeros(n + 1)
    d[0] = 1.0
    for i in range(n):
        p = q[i]
        for s in range(i + 1, 0, -1):
            d[s] = d[s] * (1 - p) + d[s - 1] * p
        d[0] = d[0] * (1 - p)
    return d


@njit(cache=True)
def best_k(q, extra, beta2):
    """q: probabilities sorted descending (selectable); extra: probs of unselectable true matches."""
    m = len(q)
    best_val = -1.0
    best = 0
    for k in range(m + 1):
        top = _pb(q[:k])
        rest_q = np.concatenate((q[k:], extra))
        rest = _pb(rest_q)
        val = 0.0
        if k == 0:
            val = rest[0]
        else:
            for s in range(1, k + 1):
                ps = top[s]
                if ps < 1e-12:
                    continue
                for t in range(len(rest)):
                    pt = rest[t]
                    if pt < 1e-12:
                        continue
                    val += ps * pt * (1 + beta2) * s / (beta2 * (s + t) + k)
        if val > best_val + 1e-12:
            best_val = val
            best = k
    return best, best_val


@njit(parallel=True, cache=True)
def decide_all(offsets, q_sorted, extra_offsets, extra_q, beta2):
    """offsets: CSR-style segment starts for each entity into q_sorted (sorted desc within each)."""
    n = len(offsets) - 1
    ks = np.zeros(n, np.int64)
    vals = np.zeros(n)
    for e in prange(n):
        q = q_sorted[offsets[e]:offsets[e + 1]]
        ex = extra_q[extra_offsets[e]:extra_offsets[e + 1]]
        k, v = best_k(q, ex, beta2)
        ks[e] = k
        vals[e] = v
    return ks, vals


def select(s1_idx, q, n_s1, beta=0.5, max_cand=25, min_q=1e-4, extra_s1=None, extra_q=None):
    """Return boolean mask over pairs chosen by expected-F maximisation per entity.

    s1_idx: int entity index per pair; q: calibrated probability per pair.
    Pairs with q < min_q are ignored (treated as not predicted, prob mass kept in extra).
    """
    s1_idx = np.asarray(s1_idx)
    q = np.asarray(q, dtype=np.float64)
    order = np.lexsort((-q, s1_idx))
    s_sorted = s1_idx[order]
    q_sorted = q[order]
    # rank within entity
    starts = np.searchsorted(s_sorted, np.arange(n_s1 + 1))
    rank = np.arange(len(order)) - starts[s_sorted]
    usable = (q_sorted >= min_q) & (rank < max_cand)
    # selectable segments
    sel_idx = np.where(usable)[0]
    seg_s = s_sorted[sel_idx]
    offsets = np.searchsorted(seg_s, np.arange(n_s1 + 1)).astype(np.int64)
    qs = q_sorted[sel_idx]
    # unselectable extra mass: low-prob pairs aggregated + optional external
    ex_mask = ~usable & (q_sorted >= 1e-3)
    ex_s = s_sorted[ex_mask]
    ex_q = q_sorted[ex_mask]
    if extra_s1 is not None:
        ex_s = np.concatenate([ex_s, np.asarray(extra_s1)])
        ex_q = np.concatenate([ex_q, np.asarray(extra_q, dtype=np.float64)])
    o2 = np.argsort(ex_s, kind="stable")
    ex_s, ex_q = ex_s[o2], ex_q[o2]
    ex_off = np.searchsorted(ex_s, np.arange(n_s1 + 1)).astype(np.int64)
    ks, _ = decide_all(offsets, qs, ex_off, ex_q, beta * beta)
    # map back: pair selected iff its within-segment position < k of its entity
    pos_in_seg = np.arange(len(sel_idx)) - offsets[seg_s]
    chosen_sorted = np.zeros(len(order), bool)
    chosen_sorted[sel_idx[pos_in_seg < ks[seg_s]]] = True
    mask = np.zeros(len(q), bool)
    mask[order[chosen_sorted]] = True
    return mask

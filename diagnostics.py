"""Falsifiable diagnostics for the multi-timescale transfer operator.

The manuscript (Results, "Diagnostic criteria for the framework") states four
falsifiable predictions that a candidate state representation either satisfies
or not.  This module ships them as callable checks so they can be evaluated on
*your own* data before any biological interpretation is attempted:

    (i)   spectral gap        -- a clear ratio gap after some M >= 2
    (ii)  participation ratio -- the leading slow eigenvectors are collective
                                 (PR >> 1), not localized on a handful of states
    (iii) simplex geometry    -- cluster centroids align along M radial arms
                                 from a common hub
    (iv)  held-out prediction -- the lumped M-state Markov model beats a
                                 memoryless null on held-out individuals

`run_diagnostics(...)` evaluates all four and prints / returns a pass/fail
table.  `stationarity_drift(...)` is an extra guardrail for the operator's
single-stationary-distribution assumption.

Usage (after running the pipeline + G-PCCA, with a *list* of per-individual
cluster sequences ``states_list``)::

    import diagnostics as dx
    report = dx.run_diagnostics(states_list, chi, eigvals, phi, pi, lag, M)
"""
from __future__ import annotations

import numpy as np

from pipeline import participation_ratio
from gpcca_utils import compute_hub_arms


# ---------------------------------------------------------------------------
# (i) spectral gap
# ---------------------------------------------------------------------------
def spectral_gap(eigvals, M):
    """Ratio gap |lambda_M| / |lambda_{M+1}| at the chosen M, plus the full
    ratio vector.  `eigvals` are the NON-TRIVIAL eigenvalues (the stationary
    lambda = 1 dropped), ordered by descending magnitude."""
    e = np.abs(np.asarray(eigvals))     # non-trivial, descending: e[0] = |lambda_2|
    ratios = e[:-1] / np.maximum(e[1:], 1e-12)
    # gap defining M basins: |lambda_M|/|lambda_{M+1}| = e[M-2]/e[M-1].
    gap = (float(e[M - 2] / max(e[M - 1], 1e-12))
           if (M >= 2 and len(e) >= M) else float('nan'))
    return dict(gap=gap, ratios=ratios)


# ---------------------------------------------------------------------------
# (ii) collectivity / participation ratio
# ---------------------------------------------------------------------------
def collectivity(phi, M):
    """Participation ratio of the leading M-1 non-trivial eigenvectors.

    PR >> 1 means the slow modes are spread over many clusters (collective);
    PR ~ 1 means they spike on a single cluster (localized, the fixed-timescale
    failure mode).  Returns the per-eigenvector PR and its minimum."""
    phi = np.asarray(phi).real
    k = max(M - 1, 1)
    pr = np.atleast_1d(participation_ratio(phi[:, :k]))
    return dict(participation_ratio=pr, min_pr=float(np.min(pr)))


# ---------------------------------------------------------------------------
# (iii) simplex / arms geometry
# ---------------------------------------------------------------------------
def simplex_score(phi, pi, chi):
    """How well cluster centroids lie along M radial arms from the hub.

    For each cluster, take its position relative to the pi-weighted hub in the
    leading (M-1)-dimensional eigenspace and measure the cosine with the
    direction of the arm it is (hard) assigned to.  Clusters sitting on a clean
    radial spoke have |cos| ~ 1; a diffuse cloud gives small cosines.  Returns
    the pi-weighted mean alignment cosine per arm and overall (near 1 =
    simplex-like)."""
    phi = np.asarray(phi).real
    M = chi.shape[1]
    phi = phi[:, :max(M - 1, 1)]
    geo = compute_hub_arms(phi, pi, chi)
    hub, arm_dirs = geo['hub'], geo['arm_dirs']
    assign = chi.argmax(axis=1)
    rel = phi - hub
    norms = np.linalg.norm(rel, axis=1)
    per_arm = np.full(M, np.nan)
    for j in range(M):
        m = (assign == j) & (norms > 1e-12)
        if not np.any(m):
            continue
        cos = (rel[m] @ arm_dirs[j]) / norms[m]
        per_arm[j] = float(np.average(np.abs(cos), weights=pi[m]))
    return dict(per_arm_cosine=per_arm, mean_cosine=float(np.nanmean(per_arm)),
                geometry=geo)


# ---------------------------------------------------------------------------
# (iv) held-out cross-validation vs a memoryless null
# ---------------------------------------------------------------------------
def crossval_vs_markov_null(states_list, chi, lag, smoothing=1.0):
    """Leave-one-individual-out held-out predictive information (bits/transition).

    For each held-out individual, fit the lumped M-state basin Markov model on
    the OTHER individuals (counting transitions within each sequence only),
    then score the held-out individual's basin transitions relative to a
    memoryless (marginal) null:

        mean_t [ log2 T[b_t, b_{t+lag}] - log2 pi_next[b_{t+lag}] ]

    A value > 0 means the Markov model carries genuine predictive information
    beyond the basin marginals on data it was not fit to.  Needs >= 2
    individuals (returns NaN per fold otherwise).  Returns per-fold
    bits/transition, the mean, and ``passes = mean > 0``."""
    seqs = [np.asarray(s, dtype=int) for s in states_list]
    M = chi.shape[1]
    basin_assign = chi.argmax(axis=1)
    basin_seqs = [basin_assign[s] for s in seqs]
    n = len(basin_seqs)
    per_fold = np.full(n, np.nan)
    for h in range(n):
        counts = np.zeros((M, M))
        marg = np.zeros(M)
        for i, bs in enumerate(basin_seqs):
            if i == h:
                continue
            if bs.size > lag:
                np.add.at(counts, (bs[:-lag], bs[lag:]), 1.0)
            np.add.at(marg, bs, 1.0)
        counts += smoothing
        marg += smoothing
        T = counts / counts.sum(axis=1, keepdims=True)
        p_next = marg / marg.sum()
        bs_h = basin_seqs[h]
        if bs_h.size <= lag:
            continue
        a, b = bs_h[:-lag], bs_h[lag:]
        ll_model = np.log2(np.maximum(T[a, b], 1e-12))
        ll_null = np.log2(np.maximum(p_next[b], 1e-12))
        per_fold[h] = float(np.mean(ll_model - ll_null))
    mean = float(np.nanmean(per_fold)) if np.any(np.isfinite(per_fold)) else float('nan')
    return dict(per_fold_bits=per_fold, mean_bits=mean,
                passes=bool(np.isfinite(mean) and mean > 0))


# ---------------------------------------------------------------------------
# the four criteria together
# ---------------------------------------------------------------------------
def run_diagnostics(states_list, chi, eigvals, phi, pi, lag, M,
                    pr_threshold=10.0, gap_threshold=1.05, cos_threshold=0.8,
                    verbose=True):
    """Evaluate all four falsifiable criteria and (optionally) print a table.

    Parameters
    ----------
    states_list : list of (T_i,) int arrays
        Per-individual cluster sequences (pass a list, even for one recording).
    chi : (N, M) ndarray            G-PCCA memberships.
    eigvals : array-like            Non-trivial eigenvalues, descending |.|.
    phi : (N, >=M-1) ndarray        Leading non-trivial right eigenvectors.
    pi : (N,) ndarray               Stationary distribution.
    lag : int                       Working lag in frames.
    M : int                         Number of basins.
    pr_threshold, gap_threshold, cos_threshold : float
        Conventions for the printed PASS/FAIL flags.  The raw numbers are
        always returned so you can apply your own judgement.

    Returns a dict with the per-criterion results and a summary `table`.
    """
    g = spectral_gap(eigvals, M)
    c = collectivity(phi, M)
    s = simplex_score(phi, pi, chi)
    x = crossval_vs_markov_null(states_list, chi, lag)
    table = {
        '(i)  spectral gap':     dict(value=g['gap'],         passes=bool(g['gap'] >= gap_threshold)),
        '(ii) participation':    dict(value=c['min_pr'],      passes=bool(c['min_pr'] > pr_threshold)),
        '(iii) simplex geom.':   dict(value=s['mean_cosine'], passes=bool(s['mean_cosine'] > cos_threshold)),
        '(iv) held-out > null':  dict(value=x['mean_bits'],   passes=bool(x['passes'])),
    }
    if verbose:
        print(f'{"criterion":22s}{"value":>10s}   result')
        print('-' * 46)
        for k, v in table.items():
            val = v['value']
            vs = f'{val:>10.3f}' if np.isfinite(val) else f'{"n/a":>10s}'
            print(f'{k:22s}{vs}   {"PASS" if v["passes"] else "FAIL"}')
    return dict(table=table, spectral_gap=g, collectivity=c,
                simplex=s, crossval=x)


# ---------------------------------------------------------------------------
# extra guardrail: non-stationarity drift
# ---------------------------------------------------------------------------
def stationarity_drift(states_list, chi, n_windows=4):
    """Basin-occupancy drift across the recording (non-stationarity check).

    Splits each individual's sequence into `n_windows` equal time windows,
    computes the M-vector of mean basin occupancy in each window (pooled over
    individuals), and reports the maximum total-variation distance between any
    two windows.  The operator assumes a single stationary distribution; large
    drift (say > 0.1-0.2) flags that this is being violated over the recording
    -- consider per-epoch refitting or a windowed analysis.  Returns the
    per-window occupancy matrix and `max_tv`."""
    seqs = [np.asarray(s, dtype=int) for s in states_list]
    M = chi.shape[1]
    basin_assign = chi.argmax(axis=1)
    occ = np.zeros((n_windows, M))
    cnt = np.zeros(n_windows)
    for s in seqs:
        bs = basin_assign[s]
        L = len(bs)
        if L < n_windows:
            continue
        edges = np.linspace(0, L, n_windows + 1).astype(int)
        for w in range(n_windows):
            seg = bs[edges[w]:edges[w + 1]]
            for j in range(M):
                occ[w, j] += int((seg == j).sum())
            cnt[w] += len(seg)
    occ = occ / np.maximum(cnt[:, None], 1)
    max_tv = 0.0
    for w1 in range(n_windows):
        for w2 in range(w1 + 1, n_windows):
            max_tv = max(max_tv, 0.5 * float(np.abs(occ[w1] - occ[w2]).sum()))
    return dict(window_occupancy=occ, max_tv=max_tv)

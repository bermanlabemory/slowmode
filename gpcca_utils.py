"""G-PCCA basin decomposition and arm-and-hub geometry.

This module wraps `pygpcca` to perform Generalized PCCA on a row-stochastic
transition matrix T(tau) and to extract the arm-and-hub structure described
in the manuscript Sec. "Arm identification".

The two operations are decoupled:

    chi, ... = run_gpcca(T, M)              # soft basin memberships
    hub, arms = compute_hub_arms(phi, pi, chi)   # geometry in eigenspace

`select_M_spectral_gap` picks M from the largest ratio gap |lambda_M| /
|lambda_{M+1}|, the same criterion the paper uses (Methods Sec. "Choosing
the number of basins").
"""
from __future__ import annotations

import numpy as np

try:
    from pygpcca import GPCCA
except ImportError:  # pragma: no cover
    GPCCA = None


def select_M_spectral_gap(eigvals, M_min=2, M_max=8, min_eigval=0.0, warn=True):
    """Pick the number of basins by the largest ratio gap |lambda_M|/|lambda_{M+1}|.

    Parameters
    ----------
    eigvals : array-like of complex or real
        Non-trivial eigenvalues of T (i.e. with the lambda=1 stationary
        eigenvalue dropped), ordered by descending magnitude.
    M_min, M_max : int
        Search range for M.
    min_eigval : float
        Only consider a gap at M if |lambda_M| >= min_eigval.  The bare
        largest-ratio rule can latch onto a spurious spike in the flat, noisy
        tail of the spectrum and return an implausibly large M; a floor (e.g.
        0.3, or a fraction of |lambda_2|) restricts the choice to slow modes
        that are actually long-lived.  Default 0.0 reproduces the old behavior.
    warn : bool
        If True, emit a warning when the choice is ambiguous -- the top two
        eligible gap ratios are within 10% of each other -- so you inspect the
        eigenvalue ladder rather than trusting the arg max.

    Returns
    -------
    M : int
        The M with the largest (eligible) ratio gap.
    gap : float
        Its gap ratio (>1 means lambda_M is meaningfully larger than lambda_{M+1}).
    ratios : (M_max - M_min + 1,) ndarray
        Gap ratios at M = M_min .. M_max (NaN where excluded by `min_eigval`).
    """
    import warnings as _warnings
    e = np.abs(np.asarray(eigvals))     # non-trivial, descending: e[0] = |lambda_2|
    # Candidate M = number of basins; its gap is |lambda_M|/|lambda_{M+1}|,
    # which is e[M-2]/e[M-1] once the stationary lambda=1 has been dropped.
    Ms = np.arange(M_min, min(M_max, len(e)) + 1)
    raw = np.array([e[m - 2] / max(e[m - 1], 1e-12) for m in Ms])
    eligible = np.array([e[m - 2] >= min_eigval for m in Ms])
    if not eligible.any():
        eligible = np.ones_like(raw, dtype=bool)   # floor too high; ignore it
    scored = np.where(eligible, raw, -np.inf)
    bi = int(np.argmax(scored))
    best = int(Ms[bi])
    top = np.sort(raw[eligible])[::-1]
    if warn and len(top) >= 2 and top[0] < 1.10 * top[1]:
        _warnings.warn(
            f'select_M_spectral_gap: ambiguous M -- the top two gap ratios '
            f'({top[0]:.3f}, {top[1]:.3f}) are within 10%. Inspect the '
            f'eigenvalue ladder and consider raising min_eigval.',
            stacklevel=2)
    return best, float(raw[bi]), np.where(eligible, raw, np.nan)


def run_gpcca(T, M, eta=None, method='brandts', z='LM'):
    """Run G-PCCA at fixed M on a row-stochastic T.

    Parameters
    ----------
    T : (N, N) ndarray
        Row-stochastic transition matrix.
    M : int
        Number of metastable basins.
    eta : (N,) ndarray, optional
        Stationary distribution (recomputed from T if omitted).
    method : str
        Schur decomposition method passed to pygpcca.GPCCA.  'brandts' is
        used in the manuscript.
    z : str
        Sort criterion for Schur eigenvalues ('LM' for largest magnitude).

    Returns
    -------
    out : dict
        chi : (N, M) memberships, columns reordered by descending pi-mass;
        assignments : (N,) hard basin assignment = argmax_j chi[:, j];
        crispness : float, optimal crispness from G-PCCA;
        basin_counts : (M,) cluster counts per basin (hard);
        pi_basin : (M,) stationary mass per basin;
        schur_vectors : (N, M) Schur basis;
        pi : (N,) stationary distribution.
    """
    if GPCCA is None:
        raise ImportError('pygpcca is required: `pip install pygpcca`.')
    if eta is None:
        from pipeline import stationary_distribution
        eta = stationary_distribution(T)
    # Floor eta strictly above 0; pyGPCCA rejects zero entries (which happen
    # when a cluster is visited only at the very start of a per-individual
    # sub-sequence and therefore never receives incoming probability mass).
    eta = np.maximum(eta, 1e-12); eta = eta / eta.sum()
    g = GPCCA(T, eta=eta, z=z, method=method)
    try:
        g.optimize(M)
    except ValueError as exc:
        # Common pyGPCCA failure: M splits a complex-conjugate eigenvalue
        # pair.  Standard recipe (cf. Reuter & Weber 2018, sec. 3.1) is to
        # retry at M + 1 and drop the smallest-mass basin.
        if 'complex conjugate' in str(exc).lower():
            g = GPCCA(T, eta=eta, z=z, method=method)
            g.optimize(M + 1)
            chi = g.memberships
            mass = chi.T @ eta
            keep = np.argsort(-mass)[:M]
            chi_keep = chi[:, keep]
            chi_keep = chi_keep / chi_keep.sum(axis=1, keepdims=True)
            pi_basin = chi_keep.T @ eta
            order = np.argsort(-pi_basin)
            chi_keep = chi_keep[:, order]; pi_basin = pi_basin[order]
            assignments = chi_keep.argmax(axis=1)
            basin_counts = np.array([(assignments == j).sum() for j in range(M)])
            return dict(chi=chi_keep, assignments=assignments,
                        crispness=float(g.optimal_crispness),
                        basin_counts=basin_counts, pi_basin=pi_basin,
                        schur_vectors=g.schur_vectors, pi=eta,
                        retry_at_M_plus_1=True)
        raise
    chi = g.memberships
    pi_basin = chi.T @ eta
    order = np.argsort(-pi_basin)
    chi = chi[:, order]
    pi_basin = pi_basin[order]
    assignments = chi.argmax(axis=1)
    basin_counts = np.array([(assignments == j).sum() for j in range(M)])
    return dict(chi=chi, assignments=assignments,
                crispness=float(g.optimal_crispness),
                basin_counts=basin_counts, pi_basin=pi_basin,
                schur_vectors=g.schur_vectors, pi=eta)


def compute_hub_arms(phi, pi, chi, threshold=0.5, fallback_threshold=0.3):
    """Compute the pi-weighted hub and the M arm vectors in eigenspace.

    The hub is the pi-weighted mean cluster position in the displayed
    eigenspace.  Each arm vector points from the hub to the basin centroid,
    where the centroid is the pi-and-chi-weighted mean over the clusters
    with chi_j > threshold.

    Parameters
    ----------
    phi : (N, k) ndarray
        Cluster positions in the leading k non-trivial eigenvectors.
    pi : (N,) ndarray
        Stationary distribution.
    chi : (N, M) ndarray
        G-PCCA memberships.
    threshold : float
        Soft-membership cutoff defining "high-chi" clusters used for the
        centroid.  If fewer than 3 clusters survive, fall back to
        `fallback_threshold`.

    Returns
    -------
    out : dict
        hub : (k,) array;
        arm_dirs : (M, k) unit vectors centroid-minus-hub;
        arm_centroids : (M, k) basin centroids;
        arm_lengths : (M,) Euclidean lengths centroid-minus-hub.
    """
    phi = np.asarray(phi).real
    M = chi.shape[1]
    hub = (phi * pi[:, None]).sum(axis=0) / pi.sum()
    arm_dirs = np.zeros((M, phi.shape[1]))
    arm_centroids = np.zeros((M, phi.shape[1]))
    arm_lengths = np.zeros(M)
    for j in range(M):
        mask = chi[:, j] > threshold
        if mask.sum() < 3:
            mask = chi[:, j] > fallback_threshold
        pts = phi[mask]
        wts = (chi[mask, j] * pi[mask])
        if wts.sum() < 1e-12:
            arm_centroids[j] = hub
            continue
        c = (pts * wts[:, None]).sum(axis=0) / wts.sum()
        arm_centroids[j] = c
        d = c - hub
        L = np.linalg.norm(d)
        arm_lengths[j] = L
        arm_dirs[j] = d / max(L, 1e-12)
    return dict(hub=hub, arm_dirs=arm_dirs, arm_centroids=arm_centroids,
                arm_lengths=arm_lengths)


def lump_to_basin_msm(state_seq, chi, lag, smoothing=1.0):
    """Build the basin-level Markov model from cluster sequence and chi.

    Lumps integer cluster states into basins via argmax_j chi[i, j], then
    estimates the row-stochastic M x M transition matrix at the requested lag
    with Laplace smoothing.  `state_seq` may be a single sequence or a list of
    per-individual sequences; in the list case transitions are counted within
    each sequence only (never across the splice between individuals).

    Returns (T_basin, pi_basin, basin_assignments).
    """
    if isinstance(state_seq, (list, tuple)):
        seqs = [np.asarray(s, dtype=int) for s in state_seq]
    else:
        seqs = [np.asarray(state_seq, dtype=int)]
    M = chi.shape[1]
    basin_assign = chi.argmax(axis=1)
    counts = np.zeros((M, M))
    for s in seqs:
        bs = basin_assign[s]
        if bs.size > lag:
            np.add.at(counts, (bs[:-lag], bs[lag:]), 1.0)
    counts += smoothing
    T_basin = counts / counts.sum(axis=1, keepdims=True)
    vals, vecs = np.linalg.eig(T_basin.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi_basin = np.abs(vecs[:, idx].real); pi_basin /= pi_basin.sum()
    return T_basin, pi_basin, basin_assign

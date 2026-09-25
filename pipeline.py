"""Multi-timescale transfer-operator pipeline.

The functions in this module implement the analysis pipeline described in
Kaur, Jain, & Berman, "Using timescale as a state coordinate reveals the
metastable geometry of behavior" (2026):

    raw signal -> wavelet amplitudes -> PCA -> delay embedding ->
    k-means clustering -> transition matrix T(tau) -> eigendecomposition.

Downstream basin identification (G-PCCA, hub/arm geometry) lives in
gpcca_utils.py.

References to the manuscript are given in each function's docstring.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Morlet wavelet transform (Berman 2014 MotionMapper convention; Methods Sec.
# "Wavelet decomposition").  Identical numerical recipe to find_wavelets.py
# in the original code repository, refactored for clarity and made GPU-free.
# ---------------------------------------------------------------------------
def morlet_wavelet_amplitudes(x, fs, fmin, fmax, n_freqs, omega0=5.0):
    """Morlet wavelet amplitudes of a multivariate time series.

    Parameters
    ----------
    x : (T, d) ndarray
        Multivariate time series.  Columns are channels (e.g. joint angles).
    fs : float
        Sampling frequency in Hz.
    fmin, fmax : float
        Lowest and highest wavelet centre frequencies in Hz.
    n_freqs : int
        Number of dyadically-spaced frequency channels.  Methods uses 25.
    omega0 : float
        Dimensionless Morlet parameter (controls time-frequency tradeoff).
        We use omega0 = 5 throughout.

    Returns
    -------
    amplitudes : (T, d * n_freqs) ndarray
        Wavelet amplitudes |W(t,f)|, channel-major: amplitudes[:, c*n_freqs:(c+1)*n_freqs]
        is the spectrogram of channel c.
    f : (n_freqs,) ndarray
        Centre frequencies (low to high).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    T, d = x.shape
    dt = 1.0 / fs

    # Dyadic frequency spacing (low -> high).
    Tmin, Tmax = 1.0 / fmax, 1.0 / fmin
    Ts = Tmin * (2 ** ((np.arange(n_freqs) * np.log(Tmax / Tmin))
                       / (np.log(2) * (n_freqs - 1))))
    f = (1.0 / Ts)[::-1]

    out = np.zeros((T, d * n_freqs))
    for c in range(d):
        out[:, c * n_freqs:(c + 1) * n_freqs] = _wavelet_one_channel(
            x[:, c], f, dt, omega0).T
    return out, f


def _wavelet_one_channel(x, f, dt, omega0):
    """Per-channel Morlet wavelet amplitudes (frequency x time)."""
    N0 = len(x)
    if N0 % 2 == 1:
        # Append one zero so the padded length is even; the extra sample is
        # trimmed off again before returning.
        x = np.concatenate([x, [0.0]])
    M = len(x)
    # Symmetric zero-padding to reduce edge effects.
    x_pad = np.concatenate([np.zeros(M // 2), x, np.zeros(M // 2)])
    Npad = len(x_pad)
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
    Omega = 2 * np.pi * np.arange(-Npad / 2, Npad / 2) / (Npad * dt)

    xhat = np.fft.fftshift(np.fft.fft(x_pad))
    L = len(f)
    amp = np.zeros((L, M))
    idx = np.arange(M // 2, M // 2 + M).astype(int)
    norm = (np.pi ** -0.25) * np.exp(0.25 * (omega0
                                             - np.sqrt(omega0 ** 2 + 2)) ** 2)
    for i in range(L):
        m = (np.pi ** -0.25) * np.exp(-0.5 * (-Omega * scales[i] - omega0) ** 2)
        q = np.fft.ifft(m * xhat) * np.sqrt(scales[i])
        q = q[idx]
        amp[i, :] = np.abs(q) * norm / np.sqrt(2 * scales[i])
    return amp[:, :N0]


# ---------------------------------------------------------------------------
# PCA on wavelet amplitudes with a temporal-shuffle threshold (Methods Sec.
# "PCA on wavelet amplitudes").  We keep the leading components whose
# eigenvalues exceed the average of the leading eigenvalue across
# temporally-shuffled copies of the wavelet matrix (each feature permuted
# independently in time -- not a phase randomization).
# ---------------------------------------------------------------------------
def pca_with_shuffle_threshold(amplitudes, n_shuffles=10, max_keep=50, seed=0,
                               percentile=None):
    """PCA with shuffle-based component selection.

    Parameters
    ----------
    amplitudes : (T, F) ndarray
        Wavelet amplitudes from morlet_wavelet_amplitudes (or any non-negative
        spectrogram-like matrix).
    n_shuffles : int
        Number of independent temporal shuffles for the threshold estimate.
    max_keep : int
        Maximum number of components to keep regardless of threshold.
    percentile : float or None
        How to reduce the `n_shuffles` shuffled leading-eigenvalues to a single
        threshold.  None (default) uses their mean (the manuscript recipe); a
        value in (0, 100] uses that percentile instead -- e.g. 95 is more
        conservative (keeps fewer PCs), 50 is the median.

    Returns
    -------
    projections : (T, k) ndarray
        Projection of `amplitudes` onto the leading k principal components.
    n_kept : int
        Number of components above threshold.
    eigvals : (max_keep,) ndarray
        Top-`max_keep` eigenvalues of the data covariance.
    threshold : float
        The lambda_1-mean of the shuffled covariance.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(amplitudes, dtype=float)
    pca = PCA(n_components=min(max_keep, A.shape[1]))
    pca.fit(A)
    eigvals = pca.explained_variance_

    # Threshold: mean of lambda_1 across phase-shuffled wavelet matrices.
    shuf_lambdas = np.zeros(n_shuffles)
    for s in range(n_shuffles):
        # Shuffle each column independently across time: preserves each
        # feature's marginal while destroying temporal and cross-feature
        # structure (a plain permutation, not a phase randomization).
        Ash = A.copy()
        for j in range(Ash.shape[1]):
            Ash[:, j] = rng.permutation(Ash[:, j])
        pca_sh = PCA(n_components=1)
        pca_sh.fit(Ash)
        shuf_lambdas[s] = pca_sh.explained_variance_[0]
    if percentile is None:
        threshold = float(shuf_lambdas.mean())
    else:
        threshold = float(np.percentile(shuf_lambdas, percentile))
    n_kept = int(np.sum(eigvals > threshold))
    projections = pca.transform(A)[:, :n_kept]
    return projections, n_kept, eigvals, threshold


# ---------------------------------------------------------------------------
# Exact nearest neighbours, used by Cao's E_1 below.  A k-d tree is fast in a
# few dimensions but degrades towards single-threaded brute force as the
# dimension grows (e.g. the 100-800 of 50 PCs x d); there a blocked matrix
# product (BLAS, multithreaded) is much faster and just as exact.
# ---------------------------------------------------------------------------
_KDTREE_MAX_DIM = 16    # method='auto' uses the k-d tree up to this dimension


def nearest_neighbours(X, query_idx, method='auto'):
    """Exact nearest neighbour of selected rows of X among all other rows.

    Parameters
    ----------
    X : (n, D) ndarray
        Points (e.g. delay vectors), one per row.
    query_idx : (m,) int array
        Rows of X to find neighbours for.  A query's own row is excluded; an
        exact duplicate of it elsewhere in X is a valid neighbour at distance 0.
    method : {'auto', 'blas', 'kdtree'}
        'kdtree' queries a scipy cKDTree.  'blas' ranks every row by
        |x_j|^2 - 2 x_q.x_j, one matrix product per block, then re-ranks with
        exact differences the rare queries whose two best candidates are closer
        than that formula's rounding error, so the answer is exact, not
        approximately nearest.  'auto' uses 'kdtree' up to 16 dimensions and
        'blas' above.  The methods agree except, at most, in which of several
        exactly equidistant neighbours is returned ('blas' returns the lowest
        index).  Both use as many threads as BLAS is allowed (OMP_NUM_THREADS,
        threadpoolctl).

    Returns
    -------
    nn : (m,) int ndarray
        Row index of each query's nearest neighbour.
    dist : (m,) ndarray
        Euclidean distance to it.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    query_idx = np.asarray(query_idx, dtype=np.intp)
    if X.shape[0] < 2:
        raise ValueError('nearest_neighbours needs at least two points')
    if method == 'auto':
        method = 'kdtree' if X.shape[1] <= _KDTREE_MAX_DIM else 'blas'
    if method == 'kdtree':
        return _nn_kdtree(X, query_idx)
    if method == 'blas':
        return _nn_blas(X, query_idx)
    raise ValueError(f"method must be 'auto', 'blas' or 'kdtree', not {method!r}")


def _n_threads():
    """Thread budget: the current BLAS thread limit, else the CPU count."""
    try:
        from threadpoolctl import threadpool_info
        n = [i['num_threads'] for i in threadpool_info()
             if i.get('user_api') == 'blas']
    except Exception:  # pragma: no cover  (threadpoolctl ships with sklearn)
        n = []
    return max(1, min(n)) if n else (os.cpu_count() or 1)


def _nn_kdtree(X, query_idx):
    from scipy.spatial import cKDTree
    dists, nbrs = cKDTree(X).query(X[query_idx], k=2, workers=_n_threads())
    # Column 0 is the query itself unless it has an exact duplicate, in which
    # case the two can come back in either order.
    col = np.where(nbrs[:, 1] == query_idx, 0, 1)
    rows = np.arange(len(query_idx))
    return nbrs[rows, col], dists[rows, col]


def _nn_blas(X, query_idx, block_queries=256, block_rows=8192, n_threads=None):
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, D = X.shape
    sq = np.einsum('ij,ij->i', X, X)
    # One product [-2 x_q, 1] . [x_j, |x_j|^2] = |x_j|^2 - 2 x_q.x_j orders
    # the rows of X exactly as |x_q - x_j|^2 does (|x_q|^2 is constant).
    Xa = np.empty((n, D + 1))
    Xa[:, :D] = X
    Xa[:, D] = sq
    # Bound (x2 margin) on the rounding error of those values: candidates
    # closer than `band` may be mis-ordered, so such queries are re-ranked
    # with exact differences.
    eps = np.finfo(np.float64).eps
    x_max = np.sqrt(sq.max())

    def run(q):
        m = len(q)
        rows = np.arange(m)
        Qa = np.empty((m, D + 1))
        Qa[:, :D] = -2.0 * X[q]
        Qa[:, D] = 1.0
        best = np.full(m, np.inf)       # smallest value so far
        second = np.full(m, np.inf)     # runner-up, at any other row
        arg = np.zeros(m, dtype=np.intp)
        for s in range(0, n, block_rows):
            G = Qa @ Xa[s:s + block_rows].T
            own = (q >= s) & (q < s + G.shape[1])
            G[rows[own], q[own] - s] = np.inf            # exclude self
            j = np.argmin(G, axis=1)
            v1 = G[rows, j]
            G[rows, j] = np.inf
            v2 = G.min(axis=1)
            second = np.minimum(np.maximum(best, v1), np.minimum(second, v2))
            better = v1 < best          # strict: earlier block wins ties
            arg[better] = j[better] + s
            best[better] = v1[better]
        band = 4.0 * (D + 2) * eps * (np.sqrt(sq[q]) + x_max) ** 2
        for r in np.flatnonzero(second - best <= band):
            g = Xa @ Qa[r]
            g[q[r]] = np.inf
            cand = np.flatnonzero(g <= g.min() + band[r])
            d2 = ((X[cand] - X[q[r]]) ** 2).sum(axis=1)
            arg[r] = cand[np.argmin(d2)]
        return arg

    n_threads = _n_threads() if n_threads is None else n_threads
    # Smaller query blocks when there are too few to keep every thread busy.
    block_queries = min(block_queries,
                        max(32, -(-len(query_idx) // n_threads)))
    chunks = [query_idx[a:a + block_queries]
              for a in range(0, len(query_idx), block_queries)]
    nn = np.concatenate([np.zeros(0, dtype=np.intp)]
                        + _map_threads(run, chunks, n_threads))
    dist = np.sqrt(((X[nn] - X[query_idx]) ** 2).sum(axis=1))
    return nn, dist


def _map_threads(fn, items, n_threads):
    """[fn(x) for x in items] on a thread pool, with BLAS single-threaded
    inside it so the pool, not BLAS, spreads the work over the cores."""
    if n_threads > 1 and len(items) > 1:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # pragma: no cover
            threadpool_limits = None
        if threadpool_limits is not None:
            from concurrent.futures import ThreadPoolExecutor
            with threadpool_limits(limits=1, user_api='blas'), \
                    ThreadPoolExecutor(max_workers=n_threads) as pool:
                return list(pool.map(fn, items))
    return [fn(x) for x in items]


# ---------------------------------------------------------------------------
# Cao's E_1(d) saturation criterion for embedding dimension (Cao 1997).
# Used to choose d in Sec. "Delay embedding and state space construction".
# ---------------------------------------------------------------------------
def cao_e1(X, max_d=20, tau=1, n_samples=20000, seed=0, nn='auto'):
    """Cao's E_1(d) statistic for delay embedding.

    Parameters
    ----------
    X : (T, p) ndarray, or list of (T_i, p) ndarrays
        Multivariate time series in which to embed (e.g. PCA projections).
        A list is treated as separate recording segments (e.g. the clean
        stretches between artifacts): delay vectors are formed within each
        segment, so none spans a boundary, and neighbours are searched among
        the vectors of all segments.
    max_d : int
        Largest embedding dimension to try.
    tau : int
        Delay (in frames).  Default 1, matching the manuscript.
    n_samples : int
        Random subset size (Cao's statistic depends only on nearest-neighbour
        structure; subsampling keeps cost manageable for long series).
    nn : {'auto', 'blas', 'kdtree'}
        Nearest-neighbour search; see `nearest_neighbours`.  All three find
        the same neighbours.  'kdtree' can be fastest at any dimension for
        long, densely sampled trajectories whose neighbours are very close
        (e.g. the Lorenz and worm data); 'blas' is far faster for
        high-dimensional, noise-like features such as EEG band amplitudes.

    Returns
    -------
    E1 : (max_d-1,) ndarray
        E_1(d) for d = 1..max_d-1.  Saturation indicates the embedding
        dimension; first d at which E_1 plateaus is the manuscript's choice.
        The last entry (d = max_d - 1) is not computed and is left at 0.
    """
    segs = [np.asarray(s, dtype=float)
            for s in (X if isinstance(X, (list, tuple)) else [X])]
    segs = [s[:, None] if s.ndim == 1 else s for s in segs]
    lens = np.array([len(s) for s in segs])
    seg_start = np.cumsum(lens) - lens
    X_all = segs[0] if len(segs) == 1 else np.concatenate(segs)

    # Query points: the delay vectors starting at t < T_i - max_d*tau - 2 in
    # each segment, which exist at every d tried.
    n_query = np.maximum(lens - max_d * tau - 2, 0)
    n_pop = int(n_query.sum())
    if n_pop == 0:
        raise ValueError('cao_e1: no segment is longer than max_d * tau + 2')
    rng = np.random.default_rng(seed)
    # The first test is the original condition (so a single series draws the
    # same subsample as before); the second keeps it from asking for more
    # points than exist.
    if lens.sum() > n_samples and n_pop >= n_samples:
        pick = rng.choice(n_pop, size=n_samples, replace=False)
    else:
        pick = np.arange(n_pop)
    q_seg = np.searchsorted(np.cumsum(n_query), pick, side='right')
    q_t = pick - (np.cumsum(n_query) - n_query)[q_seg]

    E1 = np.zeros(max_d - 1)
    a_prev = None
    for d in range(1, max_d):
        Ed, row_start, row_end = _embed_segments(segs, d, tau)
        q_row = row_start[q_seg] + q_t
        # Nearest neighbour in d-dim (excluding self).
        nn_row, dnn = nearest_neighbours(Ed, q_row, method=nn)
        nn_seg = np.searchsorted(row_end, nn_row, side='right')
        nn_t = nn_row - row_start[nn_seg]
        # Keep queries whose neighbour also has a (d+1)-dim vector, i.e. is not
        # among the last tau vectors of its segment.
        valid = nn_t < lens[nn_seg] - d * tau
        # Distance after promoting both to d+1 dims (append x(t + d*tau)).
        next_q = X_all[seg_start[q_seg[valid]] + q_t[valid] + d * tau]
        next_n = X_all[seg_start[nn_seg[valid]] + nn_t[valid] + d * tau]
        dnp = np.linalg.norm(np.hstack([Ed[q_row[valid]], next_q])
                             - np.hstack([Ed[nn_row[valid]], next_n]), axis=1)
        a_d = np.mean(dnp / np.maximum(dnn[valid], 1e-12))
        if a_prev is not None:
            E1[d - 2] = a_d / a_prev
        a_prev = a_d
    return E1


def _embed_segments(segs, d, tau):
    """Delay-embed each segment separately (as delay_embed) and stack.

    Returns the stacked vectors and each segment's first and one-past-last
    row; segments shorter than (d-1)*tau + 1 contribute no rows."""
    p = segs[0].shape[1]
    n_rows = np.array([max(len(s) - (d - 1) * tau, 0) for s in segs])
    row_end = np.cumsum(n_rows)
    row_start = row_end - n_rows
    E = np.empty((int(row_end[-1]), d * p))
    for s, r0, l in zip(segs, row_start, n_rows):
        for k in range(d):
            E[r0:r0 + l, k * p:(k + 1) * p] = s[k * tau:k * tau + l]
    return E, row_start, row_end


# ---------------------------------------------------------------------------
# Delay embedding (Methods Sec. "Delay embedding").
# ---------------------------------------------------------------------------
def delay_embed(X, d, tau=1):
    """Delay-embed a (T, p) time series into (T - (d-1)*tau, d*p)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, p = X.shape
    l = T - (d - 1) * tau
    out = np.zeros((l, d * p))
    for k in range(d):
        out[:, k * p:(k + 1) * p] = X[k * tau:k * tau + l]
    return out


# ---------------------------------------------------------------------------
# K-means partition (Methods Sec. "K-means partitioning").  Uses
# MiniBatchKMeans with n_init=20 to match the analyses in the paper.
# ---------------------------------------------------------------------------
def kmeans_partition(X, N, batch_size=None, n_init=20, seed=None,
                     return_centers=False):
    """MiniBatchKMeans partition into N clusters.

    Returns the (T,) sequence of integer cluster labels, and optionally the
    (N, d) array of cluster centers.
    """
    if batch_size is None:
        batch_size = N * 5
    km = MiniBatchKMeans(n_clusters=N, batch_size=batch_size, n_init=n_init,
                         random_state=seed, init='random').fit(X)
    if return_centers:
        return km.labels_, km.cluster_centers_
    return km.labels_


# ---------------------------------------------------------------------------
# Entropy-gap criterion for choosing N (Methods Sec. "Choosing the number of
# clusters").  We compare the per-step Markov entropy of the true sequence to
# that of a Shannon-shuffled surrogate; the optimal N maximises the gap.
# All three functions accept either one sequence or a list of per-segment
# sequences (as make_transition_matrix does), for recordings broken into
# segments -- transitions are never counted across a segment boundary.
# ---------------------------------------------------------------------------
def markov_entropy(states, lag, framerate=1.0, n_states=None):
    """Shannon entropy rate (nats / time-unit) of the lag-tau Markov model.

    `states` is one integer sequence or a list of per-segment sequences.
    (Natural log, matching the entropy-gap figures, which are labelled in
    nats.)  Row i of T is weighted by the empirical occupation of state i --
    its share of the counted transitions -- rather than by the stationary
    vector of T, which is ill-conditioned when short segments make the chain
    nearly reducible.  For one long sequence the two agree to O(lag / T).
    """
    seqs, _ = _as_sequences(states)
    if n_states is None:
        n_states = _n_states_of(seqs)
    C = _transition_counts(seqs, lag, n_states)
    if not C.any():
        raise ValueError(f'markov_entropy: no transitions at lag {lag}')
    # -sum_i w_i sum_j T_ij log T_ij with w_i = n_i / n and T_ij = C_ij / n_i
    # (n_i transitions out of i, n in all) = -sum_ij C_ij log(C_ij / n_i) / n.
    n_out = np.broadcast_to(C.sum(axis=1, keepdims=True), C.shape)
    m = C > 0
    h = -np.sum(C[m] * np.log(C[m] / n_out[m])) / C.sum()
    return float(h) * framerate


def shannon_shuffle(states, seed=None):
    """Shannon shuffle: preserve the empirical pair frequency p(s_{t+1}|s_t)
    on aggregate but break long-range correlations.  Used as an entropy
    surrogate for choosing N.

    `states` may be a list of per-segment sequences.  Successors are then
    drawn from within-segment pairs only, and one surrogate chain of the total
    length is cut back into the original segment lengths (returned as a list),
    so it is scored with exactly the segment structure of the data.

    A closed set of states -- one the chain can enter but never leave, such
    as a cluster seen only in the final run of the data, whose only observed
    successor is itself -- would absorb the surrogate for good.  Unless the
    only closed set is the bulk of the data, a closed set's segment-final
    occurrences therefore count as draws that restart the chain (at a random
    position, as for a state that never has a successor), so the surrogate
    leaves such a set at the rate the data does.  Other draws are unchanged.
    """
    rng = np.random.default_rng(seed)
    seqs, is_list = _as_sequences(states)
    lens = np.array([len(s) for s in seqs])
    states = np.concatenate(seqs)
    L = len(states)
    # Positions with a successor: all but the last of each segment.
    has_next = np.ones(L, dtype=bool)
    has_next[(np.cumsum(lens) - 1)[lens > 0]] = False
    starts = np.flatnonzero(has_next)
    if len(starts) == 0:
        raise ValueError('shannon_shuffle: no transitions to resample')
    # positions[v]: the positions of value v that have a successor, ascending.
    positions = _positions_by_value(states, starts)
    stuck = _states_in_absorbing_sets(states, starts)
    if len(stuck):
        every = _positions_by_value(states, np.arange(L))
        for v in stuck:
            positions[v] = every[v]
    none = starts[:0]
    out = np.empty(L, dtype=int)
    out[0] = states[starts[rng.integers(0, len(starts))]]
    for i in range(1, L):
        nxt_pool = positions.get(out[i - 1], none)
        if len(nxt_pool) == 0:
            out[i] = states[starts[rng.integers(0, len(starts))]]
        else:
            j = rng.choice(nxt_pool)
            if has_next[j]:
                out[i] = states[j + 1]
            else:            # a segment-final occurrence: restart
                out[i] = states[starts[rng.integers(0, len(starts))]]
    if is_list:
        return np.split(out, np.cumsum(lens)[:-1])
    return out


def _positions_by_value(states, idx):
    """{v: the positions in `idx` (ascending) at which states == v}."""
    order = np.argsort(states[idx], kind='stable')
    vals, first = np.unique(states[idx][order], return_index=True)
    return dict(zip(vals, np.split(idx[order], first[1:])))


def _states_in_absorbing_sets(states, starts):
    """Values in the closed sets of the successor graph (i -> states[i+1] for
    i in `starts`) that would absorb a Shannon-shuffle chain: every closed
    set, unless the only one is the largest strongly connected set.  Values
    that never have a successor are left out; the chain restarts from them
    anyway."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    vals, code = np.unique(states, return_inverse=True)
    src, dst = code[starts], code[starts + 1]
    graph = csr_matrix((np.ones(len(src)), (src, dst)),
                       shape=(len(vals), len(vals)))
    n_comp, comp = connected_components(graph, directed=True,
                                        connection='strong')
    c_src, c_dst = comp[src], comp[dst]
    n_out = np.bincount(c_src, minlength=n_comp)     # transitions out of a set
    leaks = np.zeros(n_comp, dtype=bool)
    leaks[c_src[c_src != c_dst]] = True
    closed = (n_out > 0) & ~leaks
    main = np.argmax(n_out)
    if not closed.any() or (closed.sum() == 1 and closed[main]):
        return vals[:0]
    return vals[closed[comp]]


def entropy_gap(X_embedded, N_values, lag, framerate=1.0, seed=0,
                n_init=20, verbose=False):
    """Compute Delta H(N) = H_shuf(N) - H(N) for each N in N_values.

    `X_embedded` is one delay-embedded time series, or a list of per-segment
    delay-embedded arrays (embed each segment separately, e.g. with
    `delay_embed`, so that no delay vector spans a boundary).  K-means runs on
    the pooled rows; the labels are then split back into segments, and H, the
    Shannon-shuffled surrogate and H_shuf are all computed within segments.

    Returns N_values (ndarray), H (ndarray), H_shuf (ndarray)."""
    if isinstance(X_embedded, (list, tuple)):
        cuts = np.cumsum([len(x) for x in X_embedded])[:-1]
        X_embedded = np.concatenate(X_embedded)
    else:
        cuts = None
    rng_seed = seed
    H = np.zeros(len(N_values))
    H_shuf = np.zeros(len(N_values))
    for k, N in enumerate(N_values):
        labels = kmeans_partition(X_embedded, N, n_init=n_init, seed=rng_seed)
        if cuts is not None:
            labels = np.split(labels, cuts)
        H[k] = markov_entropy(labels, lag, framerate=framerate, n_states=N)
        labels_shuf = shannon_shuffle(labels, seed=rng_seed)
        H_shuf[k] = markov_entropy(labels_shuf, lag, framerate=framerate,
                                   n_states=N)
        if verbose:
            print(f'  N={N}: H={H[k]:.3f}  H_shuf={H_shuf[k]:.3f}  '
                  f'DeltaH={H_shuf[k] - H[k]:.3f}')
    return np.array(N_values), H, H_shuf


# ---------------------------------------------------------------------------
# Transition matrix and eigendecomposition (Methods Sec. "Transfer operator
# estimation").  Returns the row-stochastic T(tau).
# ---------------------------------------------------------------------------
def make_transition_matrix(states, lag, n_states=None):
    """Row-stochastic transition matrix at the given lag.

    The element T[i, j] is the empirical probability of being in state j at
    time t + lag conditional on being in state i at time t.

    `states` may be a single 1-D integer sequence, OR a list/tuple of
    per-individual sequences.  When a list is given, transitions are counted
    *within* each sequence only and never across the boundary between
    individuals (or recording segments) -- pooling with ``np.concatenate``
    instead would inject ``lag`` spurious splice transitions per boundary, so
    always pass a list when you have more than one recording.

    Rows for states that never appear are set to a uniform distribution to
    keep T well-defined (downstream G-PCCA rejects all-zero rows).
    """
    seqs, _ = _as_sequences(states)
    if n_states is None:
        n_states = _n_states_of(seqs)
    F = _transition_counts(seqs, lag, n_states)
    sums = F.sum(axis=1, keepdims=True)
    # Empty rows (state never appears as a "from") get a uniform distribution
    # rather than all zeros, so T remains a valid (row-stochastic) transition
    # matrix.  Without this, downstream G-PCCA rejects T with "not a transition
    # matrix" when running on short per-individual sub-sequences.
    empty = (sums.ravel() == 0)
    sums[empty, 0] = 1.0
    F[empty] = 1.0 / n_states
    return F / sums


def _as_sequences(states):
    """One integer sequence, or a list/tuple of per-segment sequences, as
    (list of 1-D int arrays, whether a list of sequences was given).  A plain
    list of scalars is one sequence."""
    if (isinstance(states, (list, tuple)) and len(states)
            and np.ndim(states[0]) > 0):
        return [np.asarray(s, dtype=int) for s in states], True
    return [np.asarray(states, dtype=int)], False


def _n_states_of(seqs):
    return int(max(int(s.max()) for s in seqs if s.size)) + 1


def _transition_counts(seqs, lag, n_states):
    """C[i, j] = number of i -> j transitions at `lag`, counted within each
    sequence only."""
    a = np.arange(n_states + 1)
    F = np.zeros((n_states, n_states))
    for s in seqs:
        if s.size > lag:
            Fi, _, _ = np.histogram2d(s[:-lag], s[lag:], bins=[a, a])
            F += Fi
    return F


def stationary_distribution(T):
    """Stationary distribution of a row-stochastic T (left eigenvector at 1)."""
    vals, vecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(vecs[:, idx].real)
    pi /= pi.sum()
    return pi


def leading_eigvecs(T, k=10, drop_stationary=True):
    """Leading right eigenvectors of T, ordered by descending |lambda|.

    Returns
    -------
    eigvals : (k,) complex ndarray
        Eigenvalues, ordered by descending magnitude (with the stationary
        eigenvalue at 1 dropped if drop_stationary).
    eigvecs : (N, k) complex ndarray
        Right eigenvectors, columns matching `eigvals`.
    """
    vals, vecs = np.linalg.eig(T)
    order = np.argsort(np.abs(vals))[::-1]
    keep_v, keep_idx = [], []
    for idx in order:
        v = vals[idx]
        if drop_stationary and np.isclose(v, 1.0):
            continue
        keep_idx.append(idx); keep_v.append(v)
        if len(keep_idx) >= k:
            break
    return np.array(keep_v), vecs[:, keep_idx]


# ---------------------------------------------------------------------------
# Implied timescales t_k(tau) = -tau / ln|lambda_k(tau)|, used to choose the
# working lag (Methods Sec. "Choosing the lag").  With short segments a rare
# cluster can be entered but never left, or left but never re-entered; T is
# then reducible and such a cluster contributes an eigenvalue near 1 -- a
# spurious slow mode (|lambda| ~ 0.999 at lag 1 is a ~1000-frame timescale).
# The standard remedy is to estimate T on the "active set", the largest
# strongly connected set of the transition-count graph.
# ---------------------------------------------------------------------------
def largest_connected_set(states, lag, n_states=None):
    """States in the largest strongly connected set of the lag-tau count graph.

    i -> j is an edge when that transition is observed at least once (within
    a segment, if `states` is a list of per-segment sequences).  Components
    are ranked by the number of transitions leaving their states.  Returns
    the sorted state indices.
    """
    seqs, _ = _as_sequences(states)
    if n_states is None:
        n_states = _n_states_of(seqs)
    return _largest_connected(_transition_counts(seqs, lag, n_states))


def _largest_connected(C):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    _, comp = connected_components(csr_matrix(C > 0), directed=True,
                                   connection='strong')
    size = np.bincount(comp, weights=C.sum(axis=1))
    return np.flatnonzero(comp == np.argmax(size))


def implied_timescales(states, lags, k=5, n_states=None, framerate=1.0,
                       active_set=False):
    """Implied timescales of T(tau) over a range of lags.

    Parameters
    ----------
    states : (T,) int array, or list of per-segment int arrays
        Cluster sequence(s); transitions are counted within segments only.
    lags : sequence of int
        Lags tau, in frames.
    k : int
        Number of non-trivial eigenvalues (lambda_2 .. lambda_{k+1}).
    framerate : float
        Timescales are returned in units of 1 / framerate (default: frames).
    active_set : bool
        If True, at each lag restrict T(tau) to the largest strongly connected
        set of the count graph (`largest_connected_set`), renormalising its
        rows, before the eigendecomposition.  Recommended for short or
        fragmented recordings, where otherwise a rare, nearly absorbing
        cluster gives a spurious |lambda| ~ 1 mode.

    Returns
    -------
    timescales : (len(lags), k) ndarray
        -tau / ln|lambda| / framerate; inf where |lambda| = 1, NaN where T has
        fewer than k + 1 eigenvalues or there are no transitions at tau.
    eigvals : (len(lags), k) ndarray
        |lambda_2| .. |lambda_{k+1}|, in descending order (the leading
        eigenvalue, 1, is dropped; any further eigenvalue at 1 is kept).
    n_active : (len(lags),) int ndarray
        Number of states in T(tau) (n_states unless `active_set`).
    """
    seqs, _ = _as_sequences(states)
    if n_states is None:
        n_states = _n_states_of(seqs)
    timescales = np.full((len(lags), k), np.nan)
    eigvals = np.full((len(lags), k), np.nan)
    n_active = np.zeros(len(lags), dtype=int)
    for i, lag in enumerate(lags):
        C = _transition_counts(seqs, lag, n_states)
        if not C.any():
            continue
        if active_set:
            keep = _largest_connected(C)
            C = C[np.ix_(keep, keep)]
            T = C / C.sum(axis=1, keepdims=True)
        else:
            keep = np.arange(n_states)
            T = make_transition_matrix(seqs, lag, n_states=n_states)
        a = np.sort(np.abs(np.linalg.eigvals(T)))[::-1][1:k + 1]
        eigvals[i, :len(a)] = a
        with np.errstate(divide='ignore'):
            timescales[i, :len(a)] = np.where(
                a < 1.0, -lag / np.log(a), np.inf) / framerate
        n_active[i] = len(keep)
    return timescales, eigvals, n_active


def phi2_correlation_preet(states, h, fs=100.0, savgol_window=150,
                           savgol_polyorder=5):
    """Pearson |r(phi_2(t), h(t))| with the per-seed characteristic-timescale
    lag and Savitzky-Golay smoothing used in the manuscript Fig. 2D sweep.

    Recipe (Methods Sec. "Lorenz panel D"; equivalent to Preet's
    `compute_lorenz_corr_seeds.py` in the original repo):

        1. Build T(lag = 1) and compute t_c = -1 / log|lambda_2| (in frames).
        2. Build T(lag = round(t_c)).
        3. Sort the eigenvalues of T by descending real part; the leading
           non-trivial eigenvector is at index 1 (index 0 is the trivial
           lambda = 1 mode).
        4. Project per-frame as `phi_2(t) = phi_2_cluster[states]` and smooth
           with `savgol_filter(window_length=150, polyorder=5)`.
        5. Return |Pearson r| against the (length-aligned) hidden driver h.

    Returns (r_pearson, lag_used_in_frames, t_c_in_frames).
    Returns (np.nan, np.nan, np.nan) if t_c is not finite.
    """
    from scipy.signal import savgol_filter
    from scipy.stats import pearsonr
    states = np.asarray(states, dtype=int)
    n_states_ = int(states.max()) + 1

    # Step 1: characteristic timescale from a lag=1 transition matrix.
    T1 = make_transition_matrix(states, lag=1, n_states=n_states_)
    eigvals1 = np.linalg.eig(T1)[0]
    abs_sorted = np.sort(np.abs(eigvals1))[::-1]
    lam2 = abs_sorted[1] if len(abs_sorted) > 1 else np.nan
    if not (0 < lam2 < 1):
        return np.nan, np.nan, np.nan
    t_c = -1.0 / np.log(lam2)
    lag = max(int(round(t_c)), 1)

    # Step 2-3: eigenvectors at lag = round(t_c), sort by real eigvalue.
    Tlag = make_transition_matrix(states, lag=lag, n_states=n_states_)
    eigvals, eigvecs = np.linalg.eig(Tlag)
    sorted_idx = np.argsort(eigvals.real)[::-1]
    phi2_cluster = eigvecs[:, sorted_idx[1]].real

    # Step 4-5: project, smooth, correlate.
    phi2_t = phi2_cluster[states]
    h_trim = h[: len(phi2_t)]
    phi2_sm = savgol_filter(phi2_t, savgol_window, savgol_polyorder)
    r, _ = pearsonr(phi2_sm, h_trim)
    return abs(r), lag, t_c


def participation_ratio(phi):
    """Participation ratio of one or more eigenvectors.

    PR = (sum_i phi_i^2)^2 / sum_i phi_i^4.

    Near 1 when phi is concentrated on a single state, near N when uniform.
    """
    phi = np.asarray(phi)
    if phi.ndim == 1:
        phi = phi[:, None]
    s2 = (phi.real ** 2).sum(axis=0)
    s4 = (phi.real ** 4).sum(axis=0)
    return s2 ** 2 / np.maximum(s4, 1e-30)


# ---------------------------------------------------------------------------
# Multi-timescale wavelet pipeline as a single convenience call.
# ---------------------------------------------------------------------------
def multi_timescale_pipeline(x, fs, fmin, fmax, n_freqs, n_pcs=None,
                             d=None, tau_embed=1, N=None, lag=None,
                             omega0=5.0, seed=None, verbose=False):
    """Run the full multi-timescale pipeline on a multivariate signal.

    Returns a dict with intermediate products: amplitudes, projections,
    n_pcs_kept, X_embedded, states, T, eigvals, eigvecs, pi.
    Each of these is also obtainable individually via the lower-level
    functions above.
    """
    if verbose:
        print(f'  computing wavelets ({n_freqs} bands {fmin:.2f}-{fmax:.2f} Hz)')
    amps, f = morlet_wavelet_amplitudes(x, fs, fmin, fmax, n_freqs, omega0)
    if verbose:
        print(f'  PCA on amplitudes (shape {amps.shape})')
    if n_pcs is None:
        proj, n_kept, eigvals, thresh = pca_with_shuffle_threshold(amps, seed=seed or 0)
    else:
        pca = PCA(n_components=n_pcs).fit(amps)
        proj = pca.transform(amps)
        n_kept = n_pcs; eigvals = pca.explained_variance_; thresh = None
    if verbose:
        print(f'  kept {n_kept} PCs')
    if d is None:
        raise ValueError('d (embedding dimension) must be provided')
    X_embed = delay_embed(proj, d, tau_embed)
    if verbose:
        print(f'  delay embedding -> {X_embed.shape}')
    if N is not None:
        states = kmeans_partition(X_embed, N, seed=seed)
    else:
        states = None
    out = dict(amplitudes=amps, frequencies=f, projections=proj,
               n_pcs_kept=n_kept, eigvals_pca=eigvals, shuffle_threshold=thresh,
               X_embedded=X_embed, states=states)
    if states is not None and lag is not None:
        T = make_transition_matrix(states, lag)
        evals, evecs = leading_eigvecs(T, k=10)
        pi = stationary_distribution(T)
        out.update(T=T, eigvals=evals, eigvecs=evecs, pi=pi)
    return out


# ---------------------------------------------------------------------------
# Metastable residence times (Methods Sec. "Dwell-time distributions").
# Smooth the per-frame soft membership chi[states] with a Delta-second moving
# average, assign each frame to its dominant basin (argmax), and take the
# durations of maximal same-basin runs.  This single definition is used for
# every dwell-time figure (main Fig. 5D; Supp. Figs. S4, S5, S8, S10, S11),
# so they are mutually consistent.
# ---------------------------------------------------------------------------
def residences_per_individual(states, chi, framerate, delta=2.0):
    """Per-individual metastable residence times, in seconds.

    Parameters
    ----------
    states : list of (T_i,) int arrays
        Per-individual cluster-index sequences (one per worm/fly).
    chi : (N_clusters, M) ndarray
        Soft basin memberships from G-PCCA.
    framerate : float
        Sampling rate in Hz.
    delta : float
        Membership smoothing window in seconds (boxcar moving average).
        delta = 0 uses the raw per-frame argmax (sub-second flicker); the
        manuscript working value is 2 s.

    Returns
    -------
    list (len = n_individuals) of lists (len = M) of float arrays
        residences[i][b] = basin-b residence durations (s) for individual i.
    """
    from scipy.ndimage import uniform_filter1d
    chi = np.asarray(chi)
    M = chi.shape[1]
    w = max(1, int(round(delta * framerate)))
    per_indiv = []
    for s in states:
        s = np.asarray(s, dtype=int)
        m = chi[s]
        if delta > 0:
            m = uniform_filter1d(m, size=w, axis=0, mode='nearest')
        lab = np.argmax(m, axis=1)
        runs = [[] for _ in range(M)]
        cur = int(lab[0]); n = 1
        for a in lab[1:]:
            if a == cur:
                n += 1
            else:
                runs[cur].append(n); cur = int(a); n = 1
        runs[cur].append(n)
        per_indiv.append([np.asarray(r, float) / framerate for r in runs])
    return per_indiv


def metastable_residences(states, chi, framerate, delta=2.0):
    """Per-basin metastable residence times (seconds), pooled over individuals.

    Thin wrapper over `residences_per_individual` that concatenates each
    basin's residences across individuals.  Returns a list of M float arrays.
    """
    M = np.asarray(chi).shape[1]
    per_indiv = residences_per_individual(states, chi, framerate, delta=delta)
    pooled = []
    for b in range(M):
        parts = [pi[b] for pi in per_indiv if len(pi[b])]
        pooled.append(np.concatenate(parts) if parts else np.array([], float))
    return pooled

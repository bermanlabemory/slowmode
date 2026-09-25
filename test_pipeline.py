"""Regression tests for pipeline.py and gpcca_utils.py.

Wavelets: the transform must accept both odd- and even-length inputs, and the
even-length path -- the one used for every published figure -- must keep
returning exactly the values it returned before the odd-length fix.

Short, fragmented recordings: the exact BLAS nearest-neighbour search must
agree with cKDTree and leave Cao's E_1 unchanged; cao_e1, markov_entropy,
shannon_shuffle and entropy_gap must accept per-segment lists without looking
across a boundary; implied_timescales(active_set=True) must drop a rare,
nearly absorbing cluster; and run_gpcca must survive pygpcca's missing
`warnings` import.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.spatial import cKDTree

import gpcca_utils as gu
import pipeline as pp
from pipeline import morlet_wavelet_amplitudes


# Golden even-length output, captured from the implementation used for the
# manuscript figures: N0 = 10, d = 1, fs = 10 Hz, 0.5-4 Hz, 3 frequencies,
# x = np.random.default_rng(1234).standard_normal((10, 1)).
GOLDEN_SMALL = np.array([
    [0.05649627720031784, 0.10334836437799978, 0.12631512667754113],
    [0.05649634234982334, 0.11250911500950729, 0.09762435486262408],
    [0.05649639402145984, 0.11956535873092186, 0.09737690234946865],
    [0.056496427157248105, 0.1241003265934221, 0.17603358501161104],
    [0.05649643851362517, 0.12580675499324567, 0.2645629000893884],
    [0.0564964269789474, 0.12451865812149487, 0.31832269970025256],
    [0.056496393682306634, 0.12023856916809592, 0.3232456851505189],
    [0.0564963418830081, 0.11315271519003044, 0.29261788397964594],
    [0.056496276651527434, 0.10363268553126334, 0.24830046485403498],
    [0.0564962043731776, 0.09223047679188022, 0.19721506797571112],
])

# Same, for a larger and more realistic even-length case: N0 = 256, d = 2,
# fs = 25 Hz, 0.5-8 Hz, 5 frequencies, seed 7.  Probed at scattered
# (time, feature) entries rather than stored in full.
GOLDEN_BIG_PROBES = [
    ((0, 0), 0.04868226078566352),
    ((1, 3), 0.053661235204552224),
    ((37, 9), 0.16367681377214724),
    ((128, 5), 0.04934588846920182),
    ((200, 2), 0.09995509578673306),
    ((255, 7), 0.03529771906120341),
]


@pytest.mark.parametrize("n_samples", [1000, 1001, 256, 257, 8, 9])
@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_shape_is_preserved_for_odd_and_even_lengths(n_samples, n_channels):
    """Odd lengths used to raise; both parities must return (N0, d*n_freqs)."""
    n_freqs = 25
    x = np.random.default_rng(0).standard_normal((n_samples, n_channels))

    amp, f = morlet_wavelet_amplitudes(x, fs=25.0, fmin=0.1, fmax=8.0,
                                       n_freqs=n_freqs)

    assert amp.shape == (n_samples, n_channels * n_freqs)
    assert f.shape == (n_freqs,)
    assert np.all(np.isfinite(amp))
    assert np.all(amp >= 0.0)


def test_odd_length_1d_input():
    """The 1-D convenience path also has to survive an odd length."""
    x = np.random.default_rng(0).standard_normal(1001)

    amp, _ = morlet_wavelet_amplitudes(x, fs=25.0, fmin=0.1, fmax=8.0,
                                       n_freqs=25)

    assert amp.shape == (1001, 25)
    assert np.all(np.isfinite(amp))


def test_even_length_amplitudes_unchanged():
    """The even-length path must reproduce the published numbers exactly."""
    x = np.random.default_rng(1234).standard_normal((10, 1))

    amp, f = morlet_wavelet_amplitudes(x, fs=10.0, fmin=0.5, fmax=4.0,
                                       n_freqs=3)

    np.testing.assert_allclose(amp, GOLDEN_SMALL, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(f, [0.5, np.sqrt(2.0), 4.0], rtol=1e-12)


def test_even_length_amplitudes_unchanged_realistic_case():
    """Same guard at a figure-like size, spot-checked at scattered entries."""
    x = np.random.default_rng(7).standard_normal((256, 2))

    amp, _ = morlet_wavelet_amplitudes(x, fs=25.0, fmin=0.5, fmax=8.0,
                                       n_freqs=5)

    assert amp.shape == (256, 10)
    for (t, c), expected in GOLDEN_BIG_PROBES:
        np.testing.assert_allclose(amp[t, c], expected, rtol=1e-10, atol=0.0)


def test_odd_length_matches_even_away_from_the_trailing_edge():
    """Dropping the last sample must not move the amplitudes in the bulk.

    An odd-length signal is zero-extended by one sample, so relative to the
    even-length transform of the untruncated signal the only perturbation is
    a single-sample impulse at the very end.  Its effect is confined to the
    trailing edge, decaying over roughly the width of the widest wavelet, so
    the two agree closely everywhere before that.
    """
    fs = 25.0
    rng = np.random.default_rng(3)
    t = np.arange(1000) / fs
    x = np.sin(2 * np.pi * 1.5 * t)[:, None] + 0.3 * rng.standard_normal((1000, 2))

    kwargs = dict(fs=fs, fmin=0.1, fmax=8.0, n_freqs=25)
    amp_even, _ = morlet_wavelet_amplitudes(x, **kwargs)
    amp_odd, _ = morlet_wavelet_amplitudes(x[:-1], **kwargs)

    assert amp_even.shape == (1000, 50)
    assert amp_odd.shape == (999, 50)

    # Compare over the overlapping region, excluding the trailing quarter
    # where the missing sample legitimately changes the answer.
    keep = 750
    diff = np.abs(amp_even[:keep] - amp_odd[:keep])
    peak = np.abs(amp_even).max()
    assert diff.max() <= 1e-2 * peak

    # The disagreement really is an edge effect: it grows towards the end.
    assert (np.abs(amp_even[:keep] - amp_odd[:keep]).max()
            < np.abs(amp_even[keep:999] - amp_odd[keep:]).max())


# ---------------------------------------------------------------------------
# Exact nearest neighbours (the search behind cao_e1)
# ---------------------------------------------------------------------------
def _kdtree_reference(X, q):
    dist, idx = cKDTree(X).query(X[q], k=2)
    assert np.all(idx[:, 0] == q)            # no duplicates: self comes first
    return idx[:, 1], dist[:, 1]


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e8])
@pytest.mark.parametrize("dim", [1, 3, 40])
def test_blas_nearest_neighbours_match_ckdtree(dim, offset):
    """Same neighbours and distances as cKDTree -- also far from the origin,
    where |x|^2 - 2 x.y loses (at 1e8, all) precision and the exact
    re-ranking has to find the answer."""
    rng = np.random.default_rng(dim)
    X = rng.standard_normal((700, dim)) + offset
    q = rng.choice(700, size=200, replace=False)
    nn_ref, dist_ref = _kdtree_reference(X, q)

    nn, dist = pp.nearest_neighbours(X, q, method='blas')

    np.testing.assert_array_equal(nn, nn_ref)
    np.testing.assert_allclose(dist, dist_ref, rtol=1e-12, atol=0.0)


def test_blas_nearest_neighbours_do_not_depend_on_blocks_or_threads():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    q = rng.choice(300, size=90, replace=False)
    nn_ref, dist_ref = pp._nn_blas(X, q, n_threads=1)

    for bq, br, nt in [(7, 13, 1), (7, 13, 4), (256, 8192, 4)]:
        nn, dist = pp._nn_blas(X, q, block_queries=bq, block_rows=br,
                               n_threads=nt)
        np.testing.assert_array_equal(nn, nn_ref)
        np.testing.assert_array_equal(dist, dist_ref)


@pytest.mark.parametrize("method", ["blas", "kdtree"])
def test_nearest_neighbours_exclude_only_the_query_itself(method):
    """An exact duplicate is a neighbour at distance 0; the query never is."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 3))
    X = np.vstack([X, X[:10]])               # rows 60..69 repeat rows 0..9
    q = np.arange(70)

    nn, dist = pp.nearest_neighbours(X, q, method=method)

    assert np.all(nn != q)
    np.testing.assert_array_equal(nn[:10], np.arange(60, 70))
    np.testing.assert_array_equal(nn[60:], np.arange(10))
    assert np.all(dist[:10] == 0.0) and np.all(dist[60:] == 0.0)


@pytest.mark.parametrize("dim", [4, 30])      # below / above the k-d tree cutoff
def test_nearest_neighbours_methods_agree(dim):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, dim))
    q = rng.choice(400, size=100, replace=False)

    nn_auto, dist_auto = pp.nearest_neighbours(X, q)

    for method in ("blas", "kdtree"):
        nn, dist = pp.nearest_neighbours(X, q, method=method)
        np.testing.assert_array_equal(nn, nn_auto)
        np.testing.assert_allclose(dist, dist_auto, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Cao's E_1: unchanged results, and per-segment lists
# ---------------------------------------------------------------------------
# E_1 from the cKDTree implementation used for the manuscript (pipeline.cao_e1
# at 05083f3) for _cao_input(), max_d=7, n_samples=800, seed=3.  Embedding
# dimensions run from 4 to 24, so nn='auto' uses the k-d tree for d <= 4 and
# the BLAS search for d = 5, 6.
GOLDEN_CAO = [0.5800600163043221, 0.8600798261577739, 0.9367589509943408,
              0.9675980073817396, 0.9760634669745181, 0.0]


def _cao_input(T=3000):
    rng = np.random.default_rng(0)
    t = np.arange(T) * 0.05
    X = np.column_stack([np.sin(t), np.cos(1.3 * t), np.sin(0.7 * t + 1),
                         np.cos(2.1 * t)])
    return X + 0.05 * rng.standard_normal((T, 4))


@pytest.mark.parametrize("nn", ["auto", "blas", "kdtree"])
def test_cao_e1_unchanged(nn):
    E1 = pp.cao_e1(_cao_input(), max_d=7, n_samples=800, seed=3, nn=nn)

    np.testing.assert_allclose(E1, GOLDEN_CAO, rtol=1e-12, atol=0.0)


def test_cao_e1_one_segment_list_is_the_array():
    X = _cao_input(T=1500)

    np.testing.assert_array_equal(pp.cao_e1([X], max_d=6, n_samples=400),
                                  pp.cao_e1(X, max_d=6, n_samples=400))


def _cao_e1_brute_force(segs, max_d, tau):
    """Cao's E_1 with every segment embedded on its own, all distances
    computed directly and every admissible query point used."""
    a = []
    for d in range(1, max_d):
        vec, nxt, query = [], [], []
        for s in segs:
            for t in range(len(s) - (d - 1) * tau):
                vec.append(np.concatenate([s[t + k * tau] for k in range(d)]))
                nxt.append(s[t + d * tau] if t + d * tau < len(s) else None)
                query.append(t < len(s) - max_d * tau - 2)
        V = np.array(vec)
        D2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(axis=-1)
        np.fill_diagonal(D2, np.inf)
        ratios = []
        for i in np.flatnonzero(query):
            j = int(np.argmin(D2[i]))
            if nxt[j] is None:               # neighbour has no (d+1)-dim vector
                continue
            dnp = np.linalg.norm(np.r_[V[i], nxt[i]] - np.r_[V[j], nxt[j]])
            ratios.append(dnp / max(np.sqrt(D2[i, j]), 1e-12))
        a.append(np.mean(ratios))
    a = np.array(a)
    return np.r_[a[1:] / a[:-1], 0.0]


@pytest.mark.parametrize("tau", [1, 2])
def test_cao_e1_segments_never_span_a_boundary(tau):
    """A list of segments gives the brute-force per-segment answer, which
    differs from the answer for the spliced series.  The third segment is too
    short to hold a query but still supplies neighbours at small d."""
    rng = np.random.default_rng(4)
    segs = [np.cumsum(rng.standard_normal((n, 2)), axis=0) + offset
            for n, offset in [(70, 0.0), (55, 40.0), (6, -40.0)]]
    max_d = 5

    E1 = pp.cao_e1(segs, max_d=max_d, tau=tau, n_samples=10 ** 6)

    np.testing.assert_allclose(E1, _cao_e1_brute_force(segs, max_d, tau),
                               rtol=1e-10)
    spliced = pp.cao_e1(np.concatenate(segs), max_d=max_d, tau=tau,
                        n_samples=10 ** 6)
    assert not np.allclose(E1, spliced, rtol=1e-3)


def test_cao_e1_uses_every_point_when_fewer_than_n_samples():
    """T > n_samples with fewer admissible query points than n_samples used
    to raise inside rng.choice; now every admissible point is used."""
    X = _cao_input(T=520)                    # 520 - 6 - 2 = 512 query points

    np.testing.assert_array_equal(pp.cao_e1(X, max_d=6, n_samples=515),
                                  pp.cao_e1(X, max_d=6, n_samples=10 ** 6))


# ---------------------------------------------------------------------------
# Entropy gap: occupation-weighted entropy rate, segment lists
# ---------------------------------------------------------------------------
def test_markov_entropy_weights_rows_by_occupation():
    """h = -sum_ij C_ij log(C_ij / n_i) / n, worked by hand."""
    s = np.array([0, 0, 1, 0, 1, 1])         # 0->0, 0->1, 1->0, 0->1, 1->1
    expected = -(np.log(1 / 3) + 2 * np.log(2 / 3) + 2 * np.log(1 / 2)) / 5

    assert pp.markov_entropy(s, 1) == pytest.approx(expected, rel=1e-14)
    assert pp.markov_entropy(s, 1, framerate=10.0) == pytest.approx(
        10 * expected, rel=1e-14)


def test_markov_entropy_counts_transitions_within_segments_only():
    segs = [np.array([0, 0, 1, 1]), np.array([2, 2, 3])]
    # 0->0, 0->1, 1->1 | 2->2, 2->3; the splice 1->2 is not a transition.
    expected = 4 * np.log(2) / 5

    assert pp.markov_entropy(segs, 1) == pytest.approx(expected, rel=1e-14)
    assert pp.markov_entropy(np.concatenate(segs), 1) == pytest.approx(
        np.log(2), rel=1e-14)


def test_markov_entropy_ignores_rows_without_transitions():
    """State 2 is entered but never left, so its row of T is a uniform
    placeholder; weighting by the stationary vector would count it."""
    segs = [np.array([0, 1, 0, 1]), np.array([1, 2])]
    # 0->1, 1->0, 0->1, 1->2
    assert pp.markov_entropy(segs, 1) == pytest.approx(np.log(2) / 2,
                                                       rel=1e-14)
    # Never-visited states make no difference either.
    assert pp.markov_entropy(segs, 1, n_states=10) == pytest.approx(
        np.log(2) / 2, rel=1e-14)


# shannon_shuffle(SHUFFLE_SEQ, seed=11) from the manuscript implementation
# (05083f3), and the same for SHUFFLE_SEQ + [7], which ends in a label seen
# nowhere else (the chain restarts whenever it reaches it).
SHUFFLE_SEQ = [2, 3, 0, 3, 1, 2, 2, 1, 3, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 3,
               0, 2, 3, 0, 1, 1, 1, 3, 0, 3, 3, 3, 0, 1, 2, 1, 2, 2, 2, 0]
GOLDEN_SHUFFLE = [2, 2, 2, 3, 0, 3, 3, 0, 0, 1, 0, 1, 1, 2, 3, 1, 3, 0, 1, 1,
                  2, 1, 3, 0, 0, 2, 0, 0, 3, 1, 2, 2, 1, 1, 0, 0, 1, 2, 2, 3]
GOLDEN_SHUFFLE_SINK = [2, 2, 2, 3, 0, 2, 2, 3, 0, 1, 0, 7, 2, 3, 0, 1, 3, 0,
                       7, 1, 2, 1, 3, 0, 0, 2, 0, 0, 1, 3, 0, 3, 0, 1, 0, 3,
                       0, 3, 3, 0, 7]


def test_shannon_shuffle_single_sequence_unchanged():
    out = pp.shannon_shuffle(np.array(SHUFFLE_SEQ), seed=11)
    np.testing.assert_array_equal(out, GOLDEN_SHUFFLE)
    np.testing.assert_array_equal(
        pp.shannon_shuffle(np.array(SHUFFLE_SEQ + [7]), seed=11),
        GOLDEN_SHUFFLE_SINK)
    # A plain list of labels is still one sequence ...
    np.testing.assert_array_equal(pp.shannon_shuffle(SHUFFLE_SEQ, seed=11),
                                  GOLDEN_SHUFFLE)
    # ... and a one-segment list gives the same draw, as a list.
    out = pp.shannon_shuffle([np.array(SHUFFLE_SEQ)], seed=11)
    assert isinstance(out, list) and len(out) == 1
    np.testing.assert_array_equal(out[0], GOLDEN_SHUFFLE)


def test_shannon_shuffle_segments_resample_within_segment_pairs_only():
    segs = [np.array([0, 0, 1, 1]), np.array([3, 3, 0, 1, 2, 2, 3, 0]),
            np.array([2, 3])]
    seen = {(a, b) for s in segs for a, b in zip(s[:-1], s[1:])}
    for seed in range(20):
        out = pp.shannon_shuffle(segs, seed=seed)
        assert [len(o) for o in out] == [4, 8, 2]
        # Every step of the surrogate chain, even across the cuts, is a pair
        # seen inside a segment -- never the splices 1->3 or 0->2.
        chain = np.concatenate(out).tolist()
        assert set(zip(chain[:-1], chain[1:])) <= seen


def test_shannon_shuffle_is_not_absorbed_by_a_closed_final_cluster():
    """Cluster 2 occurs only in the final run, so its only observed successor
    is itself.  The manuscript implementation, once in it, never left (in 15
    of these 20 seeds, for up to 98% of the surrogate); now its last
    occurrence restarts the chain, so it stays about as rare as in the data
    (0.5%)."""
    s = np.r_[np.tile([0, 1], 500), [2] * 5]

    share = [np.mean(pp.shannon_shuffle(s, seed=seed) == 2)
             for seed in range(20)]

    assert max(share) < 0.05


def test_shannon_shuffle_segments_visit_disjoint_closed_sets():
    """Segments that live in two disjoint sets of clusters: without restarts
    at segment-final occurrences the chain would stay in whichever set it
    started in."""
    rng = np.random.default_rng(10)
    segs = [np.where(rng.random(40) < 0.5, *((0, 1) if k % 2 == 0 else (2, 3)))
            for k in range(10)]

    for seed in range(20):
        chain = np.concatenate(pp.shannon_shuffle(segs, seed=seed))
        assert 0.0 < np.mean(chain <= 1) < 1.0


def test_entropy_gap_one_segment_list_is_the_array():
    E = np.random.default_rng(7).standard_normal((400, 3))

    a = pp.entropy_gap(E, [3, 5], lag=1, n_init=2)
    b = pp.entropy_gap([E], [3, 5], lag=1, n_init=2)

    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_entropy_gap_scores_segments_separately():
    rng = np.random.default_rng(8)
    segs = [rng.standard_normal((n, 3)) for n in (150, 90, 160)]

    _, H, H_shuf = pp.entropy_gap(segs, [4], lag=2, seed=1, n_init=2)

    labels = pp.kmeans_partition(np.concatenate(segs), 4, n_init=2, seed=1)
    parts = np.split(labels, [150, 240])
    assert H[0] == pytest.approx(pp.markov_entropy(parts, 2, n_states=4))
    shuffled = pp.shannon_shuffle(parts, seed=1)
    assert H_shuf[0] == pytest.approx(pp.markov_entropy(shuffled, 2,
                                                        n_states=4))


# ---------------------------------------------------------------------------
# Implied timescales on the active set
# ---------------------------------------------------------------------------
def _chain_with_transient_cluster():
    """A mixing 4-state chain, plus a short segment that starts in a fifth
    cluster, stays there 999 frames, leaves, and never returns."""
    rng = np.random.default_rng(9)
    P = np.full((4, 4), 0.1) + 0.6 * np.eye(4)
    s = [0]
    for _ in range(3000):
        s.append(rng.choice(4, p=P[s[-1]]))
    return [np.array(s), np.array([4] * 999 + [0, 1, 2])]


def test_largest_connected_set_drops_a_transient_cluster():
    np.testing.assert_array_equal(
        pp.largest_connected_set(_chain_with_transient_cluster(), 1),
        [0, 1, 2, 3])


def test_active_set_removes_the_spurious_slow_mode():
    seqs = _chain_with_transient_cluster()

    t_all, lam_all, n_all = pp.implied_timescales(seqs, [1], k=2)
    t_act, lam_act, n_act = pp.implied_timescales(seqs, [1], k=2,
                                                  active_set=True)

    # Without the restriction, T[4, 4] = 998/999 is lambda_2: ~1000 frames.
    assert lam_all[0, 0] == pytest.approx(998 / 999, rel=1e-12)
    assert t_all[0, 0] > 900
    # On the active set, lambda_2 is the chain's own (0.6 in expectation).
    assert lam_act[0, 0] < 0.7 and t_act[0, 0] < 3
    assert n_all[0] == 5 and n_act[0] == 4


def test_implied_timescales_match_the_spectrum_of_T():
    seqs = _chain_with_transient_cluster()
    lags = [1, 3]

    t, lam, _ = pp.implied_timescales(seqs, lags, k=3, framerate=10.0)

    for i, lag in enumerate(lags):
        T = pp.make_transition_matrix(seqs, lag)
        ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1][1:4]
        np.testing.assert_allclose(lam[i], ev, rtol=1e-12)
        np.testing.assert_allclose(t[i], -lag / np.log(ev) / 10.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# G-PCCA: pygpcca's missing `warnings` import, and the M - 1 fallback
# ---------------------------------------------------------------------------
# Two metastable states and a third that is left but never re-entered
# (pi = 0): pygpcca finds the start simplex ill-conditioned at M = 2 and
# warns -- or, when its `warnings` import was skipped, raises NameError.
T_TRANSIENT = np.array([[0.98, 0.02, 0.0],
                        [0.02, 0.98, 0.0],
                        [0.001, 0.0, 0.999]])

T_TWO_BLOCKS = np.array([[0.9, 0.1, 0.0, 0.0],
                         [0.1, 0.899, 0.001, 0.0],
                         [0.0, 0.001, 0.899, 0.1],
                         [0.0, 0.0, 0.1, 0.9]])


def test_run_gpcca_survives_pygpcca_missing_warnings(monkeypatch):
    import pygpcca._gpcca
    monkeypatch.delattr(pygpcca._gpcca, 'warnings', raising=False)

    with pytest.warns(UserWarning, match='condition number'):
        out = gu.run_gpcca(T_TRANSIENT, 2)

    assert out['chi'].shape == (3, 2)
    assert pygpcca._gpcca.warnings is warnings


def test_gpcca_utils_fixes_pygpcca_under_warning_options():
    """The real trigger: a process started with -W or PYTHONWARNINGS -- and
    importing pygpcca sets PYTHONWARNINGS for every child process."""
    code = ('import numpy as np, gpcca_utils as gu; '
            f'gu.run_gpcca(np.array({T_TRANSIENT.tolist()}), 2); print("ok")')
    res = subprocess.run([sys.executable, '-W', 'ignore', '-c', code],
                         cwd=os.path.dirname(os.path.abspath(__file__)),
                         capture_output=True, text=True, timeout=300)

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == 'ok'


def _gpcca_failing_above(M_max):
    """pygpcca.GPCCA, except that optimize(m) fails for m > M_max, as a
    degenerate Schur basis would."""
    real = gu.GPCCA

    class Failing:
        def __init__(self, *args, **kwargs):
            self._g = real(*args, **kwargs)

        def optimize(self, m):
            if m > M_max:
                raise ValueError('degenerate Schur basis (test stand-in)')
            return self._g.optimize(m)

        def __getattr__(self, name):
            return getattr(self._g, name)

    return Failing


def test_run_gpcca_falls_back_to_fewer_basins_on_request(monkeypatch):
    monkeypatch.setattr(gu, 'GPCCA', _gpcca_failing_above(2))

    with pytest.raises(ValueError, match='degenerate'):
        gu.run_gpcca(T_TWO_BLOCKS, 3)
    with pytest.warns(UserWarning, match='retrying at M=2'):
        out = gu.run_gpcca(T_TWO_BLOCKS, 3, fallback=True)

    assert out['M'] == 2 and out['M_requested'] == 3
    assert out['chi'].shape == (4, 2)


def test_run_gpcca_fallback_stops_at_two_basins(monkeypatch):
    monkeypatch.setattr(gu, 'GPCCA', _gpcca_failing_above(1))

    with pytest.warns(UserWarning, match='retrying at M=2'), \
            pytest.raises(ValueError, match='degenerate'):
        gu.run_gpcca(T_TWO_BLOCKS, 3, fallback=True)

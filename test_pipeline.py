"""Regression tests for pipeline.morlet_wavelet_amplitudes.

Focus: the transform must accept both odd- and even-length inputs, and the
even-length path -- the one used for every published figure -- must keep
returning exactly the values it returned before the odd-length fix.
"""
import numpy as np
import pytest

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

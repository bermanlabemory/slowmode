"""Figure 2D: Time-series of the fixed-timescale slow eigenvector
projection, the multi-timescale slow eigenvector projection, and the
hidden driver h(t) for the driven Lorenz system at beta = 0.5.

We simulate the system, run both the multi-timescale (wavelet -> PCA ->
delay-embed -> k-means -> T(tau) -> eigendecomposition) and the
fixed-timescale (raw xyz -> delay-embed -> k-means -> T(tau) ->
eigendecomposition) pipelines, then display the 4000-s window with the
greatest number of basin switches in h(t).  The bottom row, h(t), is the
hidden-driver basin position that defines ground truth.

Runtime: ~3--5 minutes on a 2024 laptop (clustering on ~5e5 frames).
Reduce ``N_CLUSTERS`` for a faster (lower-resolution) preview.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

import lorenz_simulation as ls
import pipeline as pp

# ----- Simulation parameters -----
BETA       = 0.50
T_SIM      = 5000.0   # total seconds simulated
DISCARD    = 200.0
FS         = 100.0
SEED       = 0

# ----- Pipeline parameters (paper convention; both pipelines use x(t) only) -----
N_CLUSTERS = 800      # paper uses 1300; smaller for runtime, basins still resolve
D_MULTI    = 7
D_FIXED    = 8
N_FREQS    = 25
FMIN, FMAX = 0.5, 25.0
LAG_FRAMES = 309      # 3.09 s working lag at beta = 0.5

# ----- Display window -----
T_WINDOW_S = 4000     # seconds of trajectory shown in the figure
SMOOTH_WIN = 150      # moving-average window (frames) for display only;
                      # matches the original Lorenz notebook convention.


def moving_average(arr, window):
    """Centred boxcar moving average (mode='same'), NaN-safe."""
    a = np.asarray(arr, dtype=float)
    mask = np.isnan(a)
    a = np.where(mask, 0.0, a)
    kernel = np.ones(window) / window
    smoothed = np.convolve(a, kernel, mode='same')
    norm = np.convolve((~mask).astype(float), kernel, mode='same')
    with np.errstate(invalid='ignore'):
        return np.where(norm > 0, smoothed / norm, np.nan)


# ===================================================================
# 1. Simulate
# ===================================================================
print(f'Simulating Lorenz at beta = {BETA}, T = {T_SIM:.0f} s, fs = {FS} Hz ...')
t, xyz, h = ls.simulate(beta=BETA, T=T_SIM, discard=DISCARD, fs=FS, seed=SEED)
x_only = xyz[:, 0]
print(f'  trajectory shape = {xyz.shape}; '
      f'mean dwell = {ls.mean_dwell_time(h, fs=FS):.1f} s')


# ===================================================================
# 2. Multi-timescale pipeline: wavelet of x(t), delay-embed in wavelet
#    space, k-means, T(tau), eigendecomposition.
# ===================================================================
print('Multi-timescale pipeline ...')
amps_mt, _ = pp.morlet_wavelet_amplitudes(x_only, fs=FS, fmin=FMIN, fmax=FMAX,
                                           n_freqs=N_FREQS)
X_emb_m = pp.delay_embed(amps_mt, d=D_MULTI, tau=1)
states_m = pp.kmeans_partition(X_emb_m, N=N_CLUSTERS, n_init=20, seed=SEED)
T_m = pp.make_transition_matrix(states_m, lag=LAG_FRAMES, n_states=N_CLUSTERS)
eigvals_m, eigvecs_m = pp.leading_eigvecs(T_m, k=5)
print(f'  multi |lambda_k| = {np.abs(eigvals_m)[:5].round(4)}')

# Project per-frame; pad the delay-embedding offset with NaN.
phi2_clusters_m = eigvecs_m[:, 0].real
phi2_m = np.full(len(xyz), np.nan)
phi2_m[D_MULTI - 1:D_MULTI - 1 + len(states_m)] = phi2_clusters_m[states_m]


# ===================================================================
# 3. Fixed-timescale pipeline: delay-embed raw x(t) at d=8, cluster,
#    transition matrix, eigendecomposition.
# ===================================================================
print('Fixed-timescale pipeline ...')
X_emb_f = pp.delay_embed(x_only.reshape(-1, 1), d=D_FIXED, tau=1)
states_f = pp.kmeans_partition(X_emb_f, N=N_CLUSTERS, n_init=20, seed=SEED)
T_f = pp.make_transition_matrix(states_f, lag=LAG_FRAMES, n_states=N_CLUSTERS)
eigvals_f, eigvecs_f = pp.leading_eigvecs(T_f, k=5)
print(f'  fixed |lambda_k| = {np.abs(eigvals_f)[:5].round(4)}')

phi2_clusters_f = eigvecs_f[:, 0].real
phi2_f = np.full(len(xyz), np.nan)
phi2_f[D_FIXED - 1:D_FIXED - 1 + len(states_f)] = phi2_clusters_f[states_f]


# ===================================================================
# 4. Find the T_WINDOW_S window with the most basin switches in h(t)
# ===================================================================
sgn = np.sign(h); sgn[sgn == 0] = 1
switches = np.where(np.diff(sgn) != 0)[0]      # frame indices of sign flips
W = int(T_WINDOW_S * FS)
if W >= len(h):
    best_start = 0
    best_n_sw = len(switches)
else:
    step = int(FS)                              # slide in 1-s strides
    best_n_sw = -1
    best_start = 0
    for start in range(0, len(h) - W + 1, step):
        n_sw = int(np.sum((switches >= start) & (switches < start + W)))
        if n_sw > best_n_sw:
            best_n_sw = n_sw
            best_start = start
print(f'Display window: starts at t = {best_start / FS:.0f} s, '
      f'{best_n_sw} basin switches in {T_WINDOW_S:.0f} s')

sl = slice(best_start, best_start + W)
t_w     = t[sl] - t[best_start]                 # zero-base the time axis
phi2_f_w = phi2_f[sl]
phi2_m_w = phi2_m[sl]
h_w      = h[sl]

# Moving-average smoothing of the eigenvector projections for display
# (matches the original Lorenz notebook convention).  This averages out
# the high-frequency Lorenz oscillation and leaves the slow-mode envelope.
phi2_f_w = moving_average(phi2_f_w, SMOOTH_WIN)
phi2_m_w = moving_average(phi2_m_w, SMOOTH_WIN)

# Sign-align phi_2 with the sign of h for visual coherence (the eigenvector
# sign is arbitrary).
if np.nansum(phi2_f_w * h_w) < 0:
    phi2_f_w = -phi2_f_w
if np.nansum(phi2_m_w * h_w) < 0:
    phi2_m_w = -phi2_m_w


# ===================================================================
# 5. Plot
# ===================================================================
fig, axes = plt.subplots(3, 1, figsize=(8.5, 4.6), sharex=True,
                         gridspec_kw=dict(hspace=0.45))

axes[0].plot(t_w, phi2_f_w, color='0.15', lw=1.0)
axes[0].set_ylabel(r'$\phi_2$', fontsize=11, fontweight='bold')
axes[0].set_title(rf'Fixed-timescale ($\beta = {BETA:.2f}$)',
                  fontsize=10, fontweight='bold')

axes[1].plot(t_w, phi2_m_w, color='0.15', lw=1.0)
axes[1].set_ylabel(r'$\phi_2$', fontsize=11, fontweight='bold')
axes[1].set_title(rf'Multi-timescale ($\beta = {BETA:.2f}$)',
                  fontsize=10, fontweight='bold')

axes[2].plot(t_w, h_w, color='0.15', lw=1.0)
axes[2].set_xlabel('time (s)', fontsize=11, fontweight='bold')
axes[2].set_ylabel(r'$h(t)$', fontsize=11, fontweight='bold')
axes[2].set_title(rf'Hidden driver ($\beta = {BETA:.2f}$)',
                  fontsize=10, fontweight='bold')

for ax in axes:
    ax.tick_params(labelsize=8)

save(fig, 'figure_2D')
plt.close(fig)

"""Fig 5 (fly biology) — slow modes recover coarse-grained states and reveal
long-timescale organization.

  A: chi-weighted behavior-map enrichment per arm (Berman 2014 map)
  B: apparent decay rate r_k(tau) vs lag (non-Markovian slowdown)
  C: predictive mutual information I(Arm(t); Arm(t+tau)), data vs Markov
  D: pooled per-arm dwell-time CCDFs with truncated-power-law fits

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys, pickle, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
import powerlaw


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save
from pipeline import metastable_residences
ARM_LABELS = [f'arm {j+1}' for j in range(4)]

OUT = FLIES_DATA  # data location
fr = 100; M = 4

ARM_LABELS_CAP = [f'Arm {j+1}' for j in range(4)]

ARM_TITLES = [
    'Arm 1\nIdle & Slow',
    'Arm 2\nAnterior Movements',
    'Arm 3\nPosterior & Wing Movements',
    'Arm 4\nLocomotion',
]

# ---- Load caches ----
dyn = np.load(os.path.join(OUT, 'arm_dynamics_results_tau2s.npz'),
              allow_pickle=True)
bdn_path = os.path.join(OUT, 'behavior_density_chi_tau2s.npz')
have_density = os.path.exists(bdn_path)
if have_density:
    bdn = np.load(bdn_path)

lags_s     = dyn['lags_s'] / fr
apparent_r = dyn['apparent_r_hz']
mi_emp     = dyn['mi_emp']
mi_markov  = dyn['mi_markov']

# Metastable residences for panel D, computed live from states + memberships
# with Delta=2 s smoothing (pipeline.metastable_residences; Methods Sec.
# "Dwell-time distributions").
with open(os.path.join(OUT, 'states_flies.pkl'), 'rb') as f:
    _sd = pickle.load(f)
fly_states = [_sd[i].astype(int) for i in sorted(_sd)]
chi_fly = np.load(os.path.join(OUT, 'gpcca_flies_M4_tau2s.npz'))['chi']
fly_dwells = metastable_residences(fly_states, chi_fly, framerate=fr, delta=2.0)

# ---- Build figure ----
fig = plt.figure(figsize=(11.5, 8.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.32,
              height_ratios=[0.85, 1.0])

# Panel A: chi-weighted behavior density
gs_A = gs[0, :].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.12)
if have_density:
    cond = bdn['cond']; gx = bdn['x_edges']; gy = bdn['y_edges']
    cond_mean = np.nanmean(cond, axis=0, keepdims=True)
    dev = cond - cond_mean
    vmax = np.nanpercentile(np.abs(dev), 99)
    for j in range(M):
        axA = fig.add_subplot(gs_A[0, j])
        im = axA.pcolormesh(gx, gy, dev[j], cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, shading='auto',
                            rasterized=True)
        axA.set_aspect('equal')
        axA.set_title(ARM_TITLES[j], color=ARM_PALETTE[j],
                      fontsize=12, fontweight='bold', pad=4)
        axA.set_xticks([]); axA.set_yticks([])
        for sp in axA.spines.values(): sp.set_visible(False)
        if j == 0:
            axA.text(-0.02, 1.22, 'A', transform=axA.transAxes,
                     fontsize=15, fontweight='bold', ha='left', va='top')
    axA_cb = fig.add_subplot(gs_A[0, 4])
    cb = fig.colorbar(im, cax=axA_cb)
    cb.set_label(r'$\chi_j$ Enrichment Above Uniform',
                 fontsize=10, fontweight='bold')
    cb.ax.tick_params(labelsize=7)

# Panel B: apparent decay rate
axB = fig.add_subplot(gs[1, 0])
for k in range(3):
    axB.loglog(lags_s, apparent_r[:, k], 'o-', color=ARM_PALETTE[k+1],
               ms=3, lw=1.0, alpha=0.9, label=rf'$r_{k+2}$')
axB.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axB.set_ylabel(r'apparent decay rate $-\log|\lambda_k|/\tau$ (s$^{-1}$)',
               fontsize=11, fontweight='bold')
axB.legend(prop={'size': 9, 'weight': 'bold'}, ncol=2, loc='lower left')
axB.text(-0.18, 1.04, 'B', transform=axB.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

# Panel C: predictive MI
axC = fig.add_subplot(gs[1, 1])
mask_c = lags_s >= 1.0
axC.semilogx(lags_s[mask_c], mi_emp[mask_c], 'o-', color='0.15',
             ms=3.5, lw=1.2, label='Data')
axC.semilogx(lags_s[mask_c], mi_markov[mask_c], 's--', color='crimson',
             ms=3, lw=1.0, label='Markov')
axC.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axC.set_ylabel(r'$I(\mathrm{Arm}(t);\,\mathrm{Arm}(t+\tau))$  (bits)',
               fontsize=11, fontweight='bold')
leg_C = axC.legend(prop={'size': 9, 'weight': 'bold'}, loc='upper right',
                   frameon=True, facecolor='white', edgecolor='0.3',
                   framealpha=1.0)
leg_C.get_frame().set_linewidth(0.8)
axC.text(-0.18, 1.04, 'C', transform=axC.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

# Panel D: pooled per-arm dwell-time CCDFs with truncated-power-law fits
axD = fig.add_subplot(gs[1, 2])
D_handles = []
for j in range(M):
    d_arm = np.sort(fly_dwells[j])
    ccdf = 1.0 - np.arange(len(d_arm)) / len(d_arm)
    axD.loglog(d_arm[:-1], ccdf[:-1], color=ARM_PALETTE[j], lw=1.6, alpha=0.95)
    fit = powerlaw.Fit(d_arm, discrete=False, verbose=False)
    xmin = fit.power_law.xmin
    alpha_j = fit.truncated_power_law.alpha
    lam_j = fit.truncated_power_law.parameter2
    idx_xmin = np.searchsorted(d_arm, xmin)
    if idx_xmin < len(d_arm):
        ccdf_at_xmin = 1.0 - idx_xmin / len(d_arm)
        tau_fit = np.logspace(np.log10(xmin), np.log10(d_arm.max() * 1.2), 80)
        f_tp = tau_fit ** (-alpha_j) * np.exp(-lam_j * tau_fit)
        cdf_tp = np.cumsum(f_tp[:-1] * np.diff(tau_fit))
        if cdf_tp[-1] > 0:
            cdf_tp = cdf_tp / cdf_tp[-1]
            ccdf_tp = ccdf_at_xmin * (1.0 - cdf_tp)
            axD.loglog(tau_fit[:-1], ccdf_tp, color=ARM_PALETTE[j], lw=1.0,
                       ls='--', alpha=0.9)
    D_handles.append(Line2D([0], [0], color=ARM_PALETTE[j], lw=1.8,
                            label=fr'{ARM_LABELS_CAP[j]} ($\alpha={alpha_j:.1f}$)'))
D_handles.append(Line2D([0], [0], color='0.4', lw=1.0, ls='--',
                        label='truncated power-law'))
axD.set_xlabel(r'metastable dwell time $\tau$ (s)',
               fontsize=11, fontweight='bold')
axD.set_ylabel(r'CCDF $P(T \geq \tau)$', fontsize=11, fontweight='bold')
axD.set_ylim(5e-4, 1.3)
leg_D = axD.legend(handles=D_handles, loc='lower left',
                   prop={'size': 8, 'weight': 'bold'}, frameon=False,
                   handlelength=1.2, labelspacing=0.3)
axD.text(-0.18, 1.04, 'D', transform=axD.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

save(plt.gcf(), 'figure_5')
